"""Fused residual + LayerNorm + modulation for Qwen-Image 2.1 blocks.

The eager block path materializes per-token scale/gate as full (L, D) tensors
(torch.where in _select_modulation_rows), then runs separate tanh/mul/add/
norm kernels over them -- ~1 GB of traffic per block per step. These ops keep
the modulation in its small (B+1, D) table and select rows inside the kernel:

    fused_norm_mod:        mod = LayerNorm(x) * (1 + scale)
    fused_res_norm_mod:    h   = x + tanh(gate) * res
                           mod = LayerNorm(h) * (1 + scale)
    fused_residual:        h   = x + tanh(gate) * res

Batch-1 inference path (the decode/prefill glue); B>1 keeps the eager route.
"""
from __future__ import annotations

import os

import torch
from torch.library import custom_op

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    if not hasattr(tl, "fp8e4nv"):
        tl.fp8e4nv = tl.float8e4nv
    _TRITON_OK = True
except ImportError:  # pragma: no cover
    _TRITON_OK = False

_ENABLED = os.environ.get("AITK_FUSED_RNM", "1") == "1"


@triton.jit
def _rnm_kernel(
    X, RES, GATE, SCALE, MASK, XNEW, MOD,
    eps,
    G_STRIDE,
    S_STRIDE,
    D: tl.constexpr,
    HAS_RES: tl.constexpr,
    NORM: tl.constexpr,
    HAS_MASK: tl.constexpr,
    ZERO_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + cols).to(tl.float32)
    sel = 0
    if HAS_MASK:
        sel = tl.where(tl.load(MASK + row) != 0, 0, ZERO_ROW)
    if HAS_RES:
        r = tl.load(RES + row * D + cols).to(tl.float32)
        g = tl.load(GATE + sel * G_STRIDE + cols).to(tl.float32)
        x = x + libdevice.tanh(g) * r
        tl.store(XNEW + row * D + cols, x.to(tl.bfloat16))
    if NORM:
        mean = tl.sum(x, axis=0) / D
        xc = x - mean
        var = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        s = tl.load(SCALE + sel * S_STRIDE + cols).to(tl.float32)
        mod = xc * rstd * (1.0 + s)
        tl.store(MOD + row * D + cols, mod.to(tl.bfloat16))


def _launch(x, res, gate, scale, mask, eps, norm):
    L, D = x.shape
    assert D == 4096, D
    x = x.contiguous()
    if res is not None:
        res = res.contiguous()
    has_res = res is not None
    xnew = torch.empty_like(x) if has_res else x
    mod = torch.empty_like(x) if norm else x
    _rnm_kernel[(L,)](
        x, res if has_res else x, gate if gate is not None else scale,
        scale, mask if mask is not None else scale, xnew, mod,
        float(eps),
        gate.stride(0) if gate is not None else scale.stride(0),
        scale.stride(0),
        D=D,
        HAS_RES=has_res, NORM=norm, HAS_MASK=mask is not None,
        ZERO_ROW=scale.shape[0] - 1, BLOCK=D, num_warps=8,
    )
    return xnew, mod


@custom_op("aitk::fused_norm_mod", mutates_args=(), device_types="cuda")
def _fused_norm_mod_op(x: torch.Tensor, scale: torch.Tensor,
                       mask: torch.Tensor | None, eps: float) -> torch.Tensor:
    return _launch(x, None, None, scale, mask, eps, True)[1]


@_fused_norm_mod_op.register_fake
def _(x, scale, mask, eps):
    return torch.empty_like(x)


@custom_op("aitk::fused_res_norm_mod", mutates_args=(), device_types="cuda")
def _fused_res_norm_mod_op(x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor,
                           scale: torch.Tensor, mask: torch.Tensor | None,
                           eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    h, mod = _launch(x, res, gate, scale, mask, eps, True)
    return h, mod


@_fused_res_norm_mod_op.register_fake
def _(x, res, gate, scale, mask, eps):
    return torch.empty_like(x), torch.empty_like(x)


@custom_op("aitk::fused_residual_gate", mutates_args=(), device_types="cuda")
def _fused_residual_op(x: torch.Tensor, res: torch.Tensor, gate: torch.Tensor,
                       mask: torch.Tensor | None) -> torch.Tensor:
    return _launch(x, res, gate, gate, mask, 0.0, False)[0]


@_fused_residual_op.register_fake
def _(x, res, gate, mask):
    return torch.empty_like(x)


def fused_norm_mod_available(x: torch.Tensor) -> bool:
    return (
        _ENABLED
        and _TRITON_OK
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.shape[0] == 1
        and x.shape[-1] == 4096
    )


def fused_norm_mod(x, scale, mask, eps):
    """x (1, L, D); scale (B+1, D); mask (L,) bool or None."""
    return _fused_norm_mod_op(x[0], scale, mask, eps).unsqueeze(0)


def fused_res_norm_mod(x, res, gate, scale, mask, eps):
    h, mod = _fused_res_norm_mod_op(x[0], res[0], gate, scale, mask, eps)
    return h.unsqueeze(0), mod.unsqueeze(0)


def fused_residual_gate(x, res, gate, mask):
    return _fused_residual_op(x[0], res[0], gate, mask).unsqueeze(0)
