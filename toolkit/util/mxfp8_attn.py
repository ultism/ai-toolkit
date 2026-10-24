"""MXFP8 (block-scaled FP8) attention forward+backward for sm_120, wired as a
torch.autograd.Function. Forward = flashinfer-style ragged mxfp8 kernel
(prebuilt mxfp8_ragged_mx.so). Backward = dv64/dk64/dq_ws kernels
(fa-blackwell/kernels/attn_sm120/tests/{s3b64,s3b_dq_ws}_kernel.cuh).

Contract: q/k/v [H, L, 128] (batch must be 1), non-causal. L is zero-padded to a
multiple of 128 internally. Saved-for-backward: e4m3 data + ue8m0 SF for
Q/K/V (natural) and Q/K (transposed); dO is quantized in backward.

Enable via AITK_MXFP8_ATTN=1 (see patch point in flux2 model.py attention()).

Also provides a fused front-end for Qwen-Image 2.1 (`mxfp8_attention_fused` /
`mxfp8_attention_fused_dec`): one Triton kernel per tensor does
RMSNorm + RoPE + mxfp8 quant (natural + transposed + packed scale frames)
straight off the to_q/k/v GEMM outputs, bit-compatible with fwd_prep.
"""
import importlib.util
import math
import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("AITK_MXFP8_ATTN", "0") == "1"
_DK_WS = os.environ.get("AITK_DK_WS", "1") == "1"
# bwd kernel mode: 1 = dv_ws+dk_ws split (default), 3 = dvdk2 fused (faster)
_DK_WS_MODE = int(os.environ.get("AITK_DK_WS_MODE", "1")) if "AITK_DK_WS_MODE" in os.environ else (1 if _DK_WS else 0)
_FWD_SO = "/root/.cache/torch_extensions/py312_cu130/mxfp8_ragged_mx/mxfp8_ragged_mx.so"
_BWD_SRC = [
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_bwd_ext.cpp",
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_bwd_kernel.cu",
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_ragged_mx_kernel.cu",
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_bc_kernel.cu",
]
_BWD_INC = ["/root/fa-blackwell/tmp/cutlass/include", "/root/fa-blackwell/include",
            "/root/fa-blackwell/kernels/attn_sm120/tests"]

_fwd = None
_bwd = None


def enabled() -> bool:
    return _ENABLED


def _load_fwd():
    global _fwd
    if _fwd is None:
        spec = importlib.util.spec_from_file_location("mxfp8_ragged_mx", _FWD_SO)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _fwd = mod
    return _fwd


def _load_bwd():
    global _bwd
    if _bwd is None:
        from torch.utils.cpp_extension import load
        _bwd = load(
            name="mxfp8_bwd_ext" + ("_dbg" if os.environ.get("AITK_DVDK_DEBUG") == "1" else "") + ("_li" if os.environ.get("AITK_LINEINFO") == "1" else ""),
            sources=_BWD_SRC,
            extra_include_paths=_BWD_INC,
            extra_cuda_cflags=[
                "-O2", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-gencode", "arch=compute_120a,code=sm_120a",
            ] + (["-DDVDK_DEBUG"] if os.environ.get("AITK_DVDK_DEBUG") == "1" else []) + (["-lineinfo"] if os.environ.get("AITK_LINEINFO") == "1" else []),
            extra_cflags=["-O2"],
            verbose=os.environ.get("AITK_MXFP8_ATTN_VERBOSE", "0") == "1",
        )
    return _bwd


def _e8m0(amax: torch.Tensor) -> torch.Tensor:
    """amax [..] fp32 -> ue8m0 byte tensor (uint8), matches kernel e8m0_byte."""
    b = torch.ceil(torch.log2(amax.clamp_min(2.0**-126))) - 8 + 127
    return b.clamp(1, 254).to(torch.uint8)


def _pack_nat(b: torch.Tensor) -> torch.Tensor:
    """raw sf [H, S, 4] -> packed [H, S//128, 512]: off(r,kb) = (r//128)*512 + 16*(r%32) + 4*((r%128)//32) + kb"""
    H, S, KD = b.shape
    sf = b.reshape(H, S // 128, 4, 32, KD)
    return sf.permute(0, 1, 3, 2, 4).reshape(H, S // 128, 512).contiguous()


def _quant_nat(x: torch.Tensor):
    """x [H, S, 128] -> data uint8 [H, S, 128], raw sf uint8 [H, S, 4]."""
    H, S, D = x.shape
    xf = x.float().reshape(H, S, D // 32, 32)
    b = _e8m0(xf.abs().amax(-1))                          # [H, S, 4]
    scale = torch.exp2(b.float() - 127).unsqueeze(-1)     # [H, S, 4, 1]
    q = (xf / scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(H, S, D).contiguous()
    return q, b


def _pack_trn(b: torch.Tensor) -> torch.Tensor:
    """raw sf [H, 128, S//32] -> packed [H, S//128, 512]: off(d,kb) = (kb//4)*512 + 16*(d%32) + 4*(d//32) + (kb%4)"""
    H, D, KB = b.shape
    sf = b.reshape(H, 4, 32, KB // 4, 4)                  # [H, dhi, dlo, t, kblo]
    return sf.permute(0, 3, 2, 1, 4).reshape(H, KB // 4, 512).contiguous()


def _quant_trn(x: torch.Tensor):
    """x [H, S, 128] -> data uint8 [H, 128, S] (d-major), raw sf uint8 [H, 128, S//32].
    Scales along the token axis (32-token blocks per (feature, block))."""
    H, S, D = x.shape
    xt = x.transpose(1, 2).contiguous().float().reshape(H, D, S // 32, 32)
    b = _e8m0(xt.abs().amax(-1))                          # [H, D, S/32]
    scale = torch.exp2(b.float() - 127).unsqueeze(-1)
    q = (xt / scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(H, D, S).contiguous()
    return q, b


_META_CACHE = {}


def _ragged_meta(dev, S: int, L: int):
    key = (dev.index, S, L)
    t = _META_CACHE.get(key)
    if t is None:
        indptr = torch.tensor([0, S], dtype=torch.int32, device=dev)  # padded coords
        lens = torch.tensor([L], dtype=torch.int32, device=dev)       # true lengths
        t = (indptr, indptr.clone(), lens, lens.clone())
        _META_CACHE[key] = t
    return t


# Registered as a proper custom op so torch.compile keeps ONE graph (no graph
# break + per-segment guard re-eval that @torch.compiler.disable caused).
# Returns [out bf16 [H,L,D], qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o]
# -- the trailing tensors are the backward saves, kept alive by the graph.
from torch.library import custom_op


@custom_op("aitk::mxfp8_attn", mutates_args=(), device_types="cuda")
def _mx8_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> list[torch.Tensor]:
    bwd = _load_bwd()
    H, L, D = q.shape
    S = (L + 127) // 128 * 128
    if os.environ.get("AITK_MXFP8_SHAPES") == "1":
        print(f"[mxfp8] fwd H={H} L={L} S={S}", flush=True)
    qdT, kdT, vt, rq, rk, rvt, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, sfvt = bwd.fwd_prep(q, k, v)
    o, lse = bwd.fwd_attn(qdT, kdT, vt, sfq, sfk, sfvt, L, float(sm_scale))  # o [S,H,D] bf16 (S15)
    out = o.permute(1, 0, 2)[:, :L].contiguous()  # already bf16; inductor wants contiguous
    return [out, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o]


@_mx8_op.register_fake
def _(q, k, v, sm_scale):
    H, L, D = q.shape
    S = (L + 127) // 128 * 128
    u8 = dict(dtype=torch.uint8, device=q.device)
    f32 = dict(dtype=torch.float32, device=q.device)
    return [q.new_empty(H, L, D),
            q.new_empty(H, S, D, **u8), q.new_empty(H, S, D, **u8), q.new_empty(H, S, D, **u8),
            q.new_empty(H, D, S, **u8), q.new_empty(H, D, S, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S, **f32), q.new_empty(S, H, D)]   # lse fp32, o bf16 (S15)


def _mx8_setup(ctx, inputs, output):
    q, k, v, sm_scale = inputs
    ctx.save_for_backward(*output[1:])
    ctx.sm_scale = sm_scale


# Backward body as its own opaque op: AOTAutograd traces the custom-op backward
# into the compiled graph, so the pybind ext must hide behind an op boundary too.
@custom_op("aitk::mxfp8_attn_bwd", mutates_args=(), device_types="cuda")
def _mx8_bwd_op(do: torch.Tensor, qd: torch.Tensor, kd: torch.Tensor, vd: torch.Tensor,
                qt: torch.Tensor, kt: torch.Tensor, sfq: torch.Tensor, sfk: torch.Tensor,
                sfv: torch.Tensor, sfqt: torch.Tensor, sfkt: torch.Tensor,
                lse: torch.Tensor, o: torch.Tensor, sm_scale: float, mode: int) -> list[torch.Tensor]:
    bwd = _load_bwd()
    return bwd.bwd_full(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
                        lse, o, sm_scale, mode)


@_mx8_bwd_op.register_fake
def _(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o, sm_scale, mode):
    H, L, D = do.shape
    return [do.new_empty(H, L, D), do.new_empty(H, L, D), do.new_empty(H, L, D)]


def _mx8_bwd(ctx, grad_out, *grad_rest):
    if isinstance(grad_out, (list, tuple)):   # multi-output op: grads arrive as one list
        grad_out = grad_out[0]
    qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o = ctx.saved_tensors
    do = grad_out.contiguous()
    dq, dk, dv = torch.ops.aitk.mxfp8_attn_bwd(
        do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
        lse, o, ctx.sm_scale, _DK_WS_MODE)
    return dq, dk, dv, None


_mx8_op.register_autograd(_mx8_bwd, setup_context=_mx8_setup)


def mxfp8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v [B=1, H, L, 128] -> [B, H, L, 128], bf16 in/out."""
    B, H, L, D = q.shape
    x = torch.ops.aitk.mxfp8_attn(q[0], k[0], v[0], 1.0 / math.sqrt(D))[0]
    return x.unsqueeze(0)


def available(q: torch.Tensor) -> bool:
    return (_ENABLED and q.is_cuda and q.dim() == 4 and q.shape[0] == 1
            and q.shape[-1] == 128 and q.dtype == torch.bfloat16)


# ===================== fused norm+rope+quant front-end =====================
# For models whose attention is [QK-RMSNorm -> RoPE] -> quantize -> kernel
# (Qwen-Image 2.1). One Triton kernel per tensor reads the RAW to_q/k/v GEMM
# output (L, H*128) and emits exactly what fwd_prep produces, replacing:
#   inductor norm/rope kernel + (B,H,L,D) transpose copy + fwd_prep's internal
#   .contiguous() x3 + quant_nat x3 + quant_trn x3
# with a single pass (~16.7ms -> ~3ms at L=16k on a 5060 Ti).
#
# Bit-compatibility with mxfp8_bwd_kernel.cu's quant kernels:
#   e8m0: b = clamp(ceil(log2(max(amax, 2^-126))) - 8 + 127, 1, 254), done with
#         integer exponent math (exact for normals; ceil(log2(2^e)) = e)
#   e4m3: RNE-saturating (tl .to(float8e4nv) == cvt.rn.satfinite.e4m3x2.f32)
#   the quantized value is the bf16-rounded post-norm/rope tensor, pad rows
#   (s >= L) write qdata 0 and scale byte 1 -- same as quant_nat/quant_trn.
# RMSNorm mean and the rope complex mul run fp32 in a different reduction order
# than the eager reference, so a handful of q/k bytes (~30 per 67M) can differ
# by one e4m3 ulp; scale bytes match exactly.

try:
    import triton
    import triton.language as tl
    # torch 2.14's inductor rewrites float8e4nv -> fp8e4nv in generated
    # sources, but triton 3.8 only exposes the old name; alias it so
    # recompiled kernels resolve in the compile workers too
    if not hasattr(tl, "fp8e4nv"):
        tl.fp8e4nv = tl.float8e4nv
    _TRITON_OK = True
except ImportError:  # pragma: no cover - triton ships with cuda torch
    _TRITON_OK = False

if _TRITON_OK:
    # NB: fp8 dtype is referenced inline in the kernels (tl.float8e4nv), never
    # through a module-level global -- inductor's generated sources rebuild
    # globals via the NEW triton name (fp8e4nv), which triton 3.8 lacks
    pass

    @triton.jit
    def _e8m0_byte(amax):
        a = tl.maximum(amax, 1.1754943508222875e-38)  # 2^-126
        bits = a.to(tl.int32, bitcast=True)
        ceil_log2 = ((bits >> 23) - 127) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
        return tl.minimum(tl.maximum(ceil_log2 + 119, 1), 254)

    @triton.jit
    def _fused_qkv_quant_kernel(
        X,            # (L, H*128) bf16 GEMM output
        W,            # (128,) bf16 norm weight (unused if not APPLY_NORM)
        COS, SIN,     # (L, 64) fp32 rope freqs (unused if not APPLY_ROPE)
        QD,           # out (H, S, 128) u8, natural layout
        QDT,          # out (S, H, 128) u8, token-major dual write
        QT,           # out (H, 128, S) u8, feature-major (transposed)
        SFP,          # out (H, S//128, 512) u8, packed nat scales
        SFPT,         # out (H, S//128, 512) u8, packed trn scales
        L, S, H, EPS,
        APPLY_NORM: tl.constexpr, APPLY_ROPE: tl.constexpr,
        WRITE_QDT: tl.constexpr,
        # in-place cache writes (mxfp8 KV cache): outputs are rows
        # [ROW_OFF, ROW_OFF + S) of buffers padded to S_OUT. ROW_OFF must be a
        # multiple of 32 so the packed-SF 32-row groups never straddle the write
        # boundary. WRITE_QD/WRITE_QT skip unused layouts (saves bandwidth).
        ROW_OFF: tl.constexpr = 0, S_OUT: tl.constexpr = 0,
        WRITE_QD: tl.constexpr = True, WRITE_QT: tl.constexpr = True,
        WRITE_SFP: tl.constexpr = True, WRITE_SFPT: tl.constexpr = True,
    ):
        h = tl.program_id(0)
        t0 = tl.program_id(1) * 32
        rows = t0 + tl.arange(0, 32)
        cols = tl.arange(0, 128)
        mrow = rows < L

        x = tl.load(
            X + rows[:, None] * (H * 128) + h * 128 + cols[None, :],
            mask=mrow[:, None], other=0.0,
        ).to(tl.float32)

        if APPLY_NORM:
            # diffusers RMSNorm: fp32 variance, fp32 mul, round to bf16,
            # then the bf16 weight mul (fp32 opmath, bf16 round)
            var = tl.sum(x * x, axis=1) / 128.0
            rrms = tl.math.rsqrt(var + EPS)
            x = (x * rrms[:, None]).to(tl.bfloat16).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            x = (x * w[None, :]).to(tl.bfloat16).to(tl.float32)

        if APPLY_ROPE:
            # adjacent (even, odd) complex pairs, same freqs for all heads;
            # fp32 math then the model's .type_as bf16 round
            a, b = tl.split(tl.reshape(x, (32, 64, 2)))
            pc = tl.arange(0, 64)
            c = tl.load(COS + rows[:, None] * 64 + pc[None, :], mask=mrow[:, None], other=0.0)
            s = tl.load(SIN + rows[:, None] * 64 + pc[None, :], mask=mrow[:, None], other=0.0)
            e = a * c - b * s
            o = a * s + b * c
            x = tl.reshape(tl.join(e, o), (32, 128)).to(tl.bfloat16).to(tl.float32)

        # natural quant: per (token, 32-feature group)
        SO: tl.constexpr = S if S_OUT == 0 else S_OUT   # output buffers' padded seq len
        g = tl.reshape(x, (32, 4, 32))
        amax = tl.max(tl.abs(g), axis=2)                       # (32, 4)
        eb = _e8m0_byte(amax)
        scale = (eb << 23).to(tl.float32, bitcast=True)        # 2^(eb-127), exact
        qb = tl.reshape((g / scale[:, :, None]).to(tl.float8e4nv), (32, 128)).to(tl.uint8, bitcast=True)
        if WRITE_QD:
            tl.store(QD + h * SO * 128 + (ROW_OFF + rows[:, None]) * 128 + cols[None, :], qb)
        if WRITE_QDT:
            tl.store(QDT + (ROW_OFF + rows[:, None]) * (H * 128) + h * 128 + cols[None, :], qb)
        kb = tl.arange(0, 4)
        rl = tl.arange(0, 32)
        sfp_off = (h * (SO // 128) * 512 + ((ROW_OFF + t0) // 128) * 512 + 16 * rl[:, None]
                   + 4 * (((ROW_OFF + t0) // 32) % 4) + kb[None, :])
        if WRITE_SFP:
            tl.store(SFP + sfp_off, eb.to(tl.uint8))

        # transposed quant: per (feature, 32-token group)
        amax_t = tl.max(tl.abs(x), axis=0)                     # (128,)
        ebt = _e8m0_byte(amax_t)
        scale_t = (ebt << 23).to(tl.float32, bitcast=True)
        qtb = tl.trans((x / scale_t[None, :]).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
        if WRITE_QT:
            tl.store(QT + (h * 128 + cols[:, None]) * SO + (ROW_OFF + rows)[None, :], qtb)
        sfpt_off = (h * (SO // 128) * 512 + ((ROW_OFF + t0) // 128) * 512 + 16 * (cols % 32)
                    + 4 * (cols // 32) + (((ROW_OFF + t0) // 32) % 4))
        if WRITE_SFPT:
            tl.store(SFPT + sfpt_off, ebt.to(tl.uint8))


def _fused_quant_one(x: torch.Tensor, w, cos, sin, eps: float, norm_rope: bool):
    """x (L, H*128) bf16 contiguous -> dict of the five quantized buffers.
    norm_rope=False makes it a pure quant (v, or pre-roped cached k/v)."""
    L, HD = x.shape
    H = HD // 128
    S = (L + 127) // 128 * 128
    u8 = dict(dtype=torch.uint8, device=x.device)
    qd = torch.empty(H, S, 128, **u8)
    qdT = torch.empty(S, H, 128, **u8)
    qt = torch.empty(H, 128, S, **u8)
    sfp = torch.empty(H, S // 128, 512, **u8)
    sfpt = torch.empty(H, S // 128, 512, **u8)
    _fused_qkv_quant_kernel[(H, S // 32)](
        x, w, cos, sin, qd, qdT, qt, sfp, sfpt, L, S, H, eps,
        norm_rope, norm_rope, True, num_warps=4,
    )
    return {"qd": qd, "qdT": qdT, "qt": qt, "sfp": sfp, "sfpt": sfpt}


def _fused_quant_into(x: torch.Tensor, w, cos, sin, eps: float, norm_rope: bool,
                      row_off: int, s_out: int, *,
                      qd=None, qdt=None, qt=None, sfp=None, sfpt=None):
    """In-place variant of _fused_quant_one: quantize x (L, H*128) into rows
    [row_off, row_off + ceil128(L)) of caller-owned buffers padded to s_out.
    row_off must be a multiple of 32 (packed-SF group alignment). Only the
    outputs passed as non-None are written."""
    L, HD = x.shape
    H = HD // 128
    S = (L + 127) // 128 * 128
    dummy = x  # never dereferenced when the matching WRITE flag is off
    _fused_qkv_quant_kernel[(H, S // 32)](
        x, w, cos, sin,
        qd if qd is not None else dummy,
        qdt if qdt is not None else dummy,
        qt if qt is not None else dummy,
        sfp if sfp is not None else dummy,
        sfpt if sfpt is not None else dummy,
        L, S, H, eps,
        norm_rope, norm_rope, qdt is not None,
        row_off, s_out,
        qd is not None, qt is not None,
        sfp is not None, sfpt is not None,
        num_warps=4,
    )


class Mx8KVLayerCache:
    """Prefix KV cache stored directly in mxfp8 kernel buffers (decode path).

    Halves cache memory (e4m3 + per-32 e8m0 ~ 1.03 B/elem vs 2 B bf16) and
    removes the per-step prefix re-quantization of the bf16 path entirely.
    Layout:
      k_t  (Sk, H, 128) u8  token-major   (the fwd kernel's KdT)
      k_sf (H, Sk//128, 512) u8 packed natural-layout scales
      v_t  (H, 128, Sk) u8  transposed    (the fwd kernel's Vt)
      v_sf (H, Sk//128, 512) u8 packed transposed scales
    Rows [0, floor32(Lpre)) hold the prefix, quantized once at extraction.
    Rows [floor32(Lpre), Sk) are rewritten every decode step by quantizing
    [boundary stash, current k/v] in place (ROW_OFF = floor32(Lpre), 32-row
    aligned, so the packed scale frames never straddle the splice).
    k_stash/v_stash hold the <=31 bf16 boundary rows. Same kernel + same inputs
    as the bf16 path's per-step quant -> the decode output is bit-identical.
    """

    __slots__ = ("k_t", "k_sf", "v_t", "v_sf", "k_stash", "v_stash", "lpre", "sk")

    def __init__(self):
        self.k_t = self.k_sf = self.v_t = self.v_sf = None
        self.k_stash = self.v_stash = None
        self.lpre = 0
        self.sk = 0

    def store(self, k: torch.Tensor, v: torch.Tensor, total_kv: int):
        """k/v: (B=1, Lpre, H, 128) or (Lpre, H*128) bf16 post-rope prefix
        projections. total_kv: full joint sequence length this cache will
        serve (prefix + target)."""
        if k.dim() == 4:
            k = k[0].reshape(k.shape[1], -1)
            v = v[0].reshape(v.shape[1], -1)
        Lpre, HD = k.shape
        H = HD // 128
        nb = Lpre % 32
        # per-step writes cover [floor32(Lpre), floor32(Lpre) + ceil128(nb+Lq)),
        # which can exceed ceil128(Lpre+Lq) by up to 96 rows -- size for that,
        # and keep the buffers 128-row aligned (packed SF atom structure).
        # The attention never reads past ceil128(Lpre+Lq) rows, so the rounded
        # tail is never observed.
        Sk = ((Lpre - nb) + (nb + (total_kv - Lpre) + 127) // 128 * 128 + 127) // 128 * 128
        u8 = dict(dtype=torch.uint8, device=k.device)
        self.k_t = torch.empty(Sk, H, 128, **u8)
        self.k_sf = torch.empty(H, Sk // 128, 512, **u8)
        self.v_t = torch.empty(H, 128, Sk, **u8)
        self.v_sf = torch.empty(H, Sk // 128, 512, **u8)
        nb = Lpre % 32
        self.k_stash = k[Lpre - nb:].contiguous()
        self.v_stash = v[Lpre - nb:].contiguous()
        self.lpre = Lpre
        self.sk = Sk
        # rows [0, ceil128(Lpre)) get written here; only [0, Lpre-nb) survives
        # (the rest is rewritten by every decode step anyway)
        _fused_quant_into(k, None, None, None, 0.0, False, 0, Sk,
                          qdt=self.k_t, sfp=self.k_sf)
        _fused_quant_into(v, None, None, None, 0.0, False, 0, Sk,
                          qt=self.v_t, sfpt=self.v_sf)


@custom_op("aitk::mxfp8_attn_dec_q8", mutates_args=("k_cache_t", "k_cache_sf",
                                                    "v_cache_t", "v_cache_sf"),
           device_types="cuda")
def _mx8_dec_q8_op(q: torch.Tensor, k_cur: torch.Tensor, v_cur: torch.Tensor,
                   k_stash: torch.Tensor, v_stash: torch.Tensor,
                   k_cache_t: torch.Tensor, k_cache_sf: torch.Tensor,
                   v_cache_t: torch.Tensor, v_cache_sf: torch.Tensor,
                   wq: torch.Tensor, wk: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor,
                   eps: float, sm_scale: float, lpre: int) -> torch.Tensor:
    """Decode step against an Mx8KVLayerCache: quantize q (fused norm+rope),
    quantize [stash, current] k/v INTO the quantized cache, full-attention
    fwd over the whole cache. Inference-only. Returns o (Sq,H,D) bf16."""
    bwd = _load_bwd()
    Lq = q.shape[0]
    nb = k_stash.shape[0]
    row_off = lpre - nb                     # floor32(lpre)
    qq = _fused_quant_one(q, wq, cos, sin, eps, True)
    kc = _rmsnorm_rope_torch(k_cur, wk, cos, sin, eps)
    k_chunk = torch.cat([k_stash, kc], dim=0) if nb else kc
    v_chunk = torch.cat([v_stash, v_cur], dim=0) if nb else v_cur
    Sk = k_cache_t.shape[0]
    _fused_quant_into(k_chunk, wk, cos, sin, eps, False, row_off, Sk,
                      qdt=k_cache_t, sfp=k_cache_sf)
    _fused_quant_into(v_chunk, wq, cos, sin, eps, False, row_off, Sk,
                      qt=v_cache_t, sfpt=v_cache_sf)
    Lkv = lpre + Lq
    o, _ = bwd.fwd_attn(qq["qdT"], k_cache_t, v_cache_t,
                        qq["sfp"], k_cache_sf, v_cache_sf, Lq, float(sm_scale), Lkv)
    return o


@_mx8_dec_q8_op.register_fake
def _(q, k_cur, v_cur, k_stash, v_stash, k_cache_t, k_cache_sf, v_cache_t,
      v_cache_sf, wq, wk, cos, sin, eps, sm_scale, lpre):
    Lq, HD = q.shape
    H = HD // 128
    Sq = (Lq + 127) // 128 * 128
    return q.new_empty(Sq, H, 128)


def mxfp8_attention_fused_dec_q8(q, k_cur, v_cur, cache: Mx8KVLayerCache,
                                 wq, wk, cos, sin, eps: float, sm_scale: float) -> torch.Tensor:
    """Decode step with the quantized prefix cache. q/k_cur/v_cur: (L, H*128)
    raw GEMM outputs. Returns (L, H*128) bf16."""
    L = q.shape[0]
    o = torch.ops.aitk.mxfp8_attn_dec_q8(
        q, k_cur, v_cur, cache.k_stash, cache.v_stash,
        cache.k_t, cache.k_sf, cache.v_t, cache.v_sf,
        wq, wk, cos, sin, float(eps), float(sm_scale), cache.lpre)
    return o[:L].reshape(L, -1)


def _rmsnorm_rope_torch(x: torch.Tensor, w, cos, sin, eps: float) -> torch.Tensor:
    """Reference norm+rope in plain torch ops (used inside the decode op for the
    current-token k only; the fused kernel handles everything in the equal-L op).
    Mirrors diffusers RMSNorm + apply_rotary_emb_qwen(use_real=False) exactly."""
    L, HD = x.shape
    H = HD // 128
    x = x.view(L, H, 128)
    var = x.float().pow(2).mean(-1, keepdim=True)
    t = (x.float() * torch.rsqrt(var + eps)).to(torch.bfloat16) * w
    pairs = t.float().reshape(L, H, 64, 2)
    a, b = pairs.unbind(-1)
    c = cos.view(L, 1, 64)
    s = sin.view(L, 1, 64)
    out = torch.stack([a * c - b * s, a * s + b * c], dim=-1)
    return out.reshape(L, HD).to(torch.bfloat16)


@custom_op("aitk::mxfp8_attn_fused", mutates_args=(), device_types="cuda")
def _mx8_fused_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  wq: torch.Tensor, wk: torch.Tensor,
                  cos: torch.Tensor, sin: torch.Tensor,
                  eps: float, sm_scale: float) -> list[torch.Tensor]:
    """Full fused attention forward: q/k/v raw GEMM outputs (L, H*128) bf16,
    norm weights + rope cos/sin -> norm+rope+quant+kernel. Returns
    [o (S,H,D) bf16 padded, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse]."""
    bwd = _load_bwd()
    L = q.shape[0]
    qq = _fused_quant_one(q, wq, cos, sin, eps, True)
    kk = _fused_quant_one(k, wk, cos, sin, eps, True)
    vv = _fused_quant_one(v, wq, cos, sin, eps, False)
    o, lse = bwd.fwd_attn(qq["qdT"], kk["qdT"], vv["qt"],
                          qq["sfp"], kk["sfp"], vv["sfpt"], L, float(sm_scale))
    return [o, qq["qd"], kk["qd"], vv["qd"], qq["qt"], kk["qt"],
            qq["sfp"], kk["sfp"], vv["sfp"], qq["sfpt"], kk["sfpt"], lse]


@_mx8_fused_op.register_fake
def _(q, k, v, wq, wk, cos, sin, eps, sm_scale):
    L, HD = q.shape
    H = HD // 128
    S = (L + 127) // 128 * 128
    u8 = dict(dtype=torch.uint8, device=q.device)
    f32 = dict(dtype=torch.float32, device=q.device)
    return [q.new_empty(S, H, 128),
            q.new_empty(H, S, 128, **u8), q.new_empty(H, S, 128, **u8),
            q.new_empty(H, S, 128, **u8), q.new_empty(H, 128, S, **u8),
            q.new_empty(H, 128, S, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S, **f32)]


def _rope_rmsnorm_bwd(g: torch.Tensor, x_raw: torch.Tensor,
                      w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      eps: float) -> torch.Tensor:
    """g: (H, L, D) grad wrt the post-rope bf16 tensor; x_raw: (L, H*D) bf16 norm
    input. Returns (L, H*D) bf16 grad wrt x_raw. rope^T then RMSNorm backward,
    fp32 math, mirroring what autograd does on the unfused graph."""
    H, L, D = g.shape
    g = g.float().transpose(0, 1).reshape(L, H, D // 2, 2)
    ge, go = g.unbind(-1)
    c = cos.view(L, 1, D // 2)
    s = sin.view(L, 1, D // 2)
    g = torch.stack([ge * c + go * s, go * c - ge * s], dim=-1).reshape(L, H, D)
    x = x_raw.view(L, H, D).float()
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    dt = g * w.float()
    dx = rrms * dt - x * (rrms.pow(3) / D) * (dt * x).sum(-1, keepdim=True)
    return dx.reshape(L, H * D).to(torch.bfloat16)


def _mx8_fused_setup(ctx, inputs, output):
    q, k, v, wq, wk, cos, sin, eps, sm_scale = inputs
    # output[0] (the padded o) is a backward save too - bwd_full needs it for delta
    ctx.save_for_backward(*output[1:], output[0], q, k, wq, wk, cos, sin)
    ctx.L = q.shape[0]
    ctx.eps = eps
    ctx.sm_scale = sm_scale


def _mx8_fused_bwd(ctx, grad_o, *grad_rest):
    if isinstance(grad_o, (list, tuple)):   # multi-output op: grads arrive as one list
        grad_o = grad_o[0]
    (qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o,
     q, k, wq, wk, cos, sin) = ctx.saved_tensors
    L = ctx.L
    do = grad_o[:L].transpose(0, 1).contiguous()      # (H, L, D) bf16
    dq, dk, dv = torch.ops.aitk.mxfp8_attn_bwd(
        do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
        lse, o, ctx.sm_scale, _DK_WS_MODE)
    gv = dv.transpose(0, 1).reshape(L, -1)            # v: no norm/rope
    gq = _rope_rmsnorm_bwd(dq, q, wq, cos, sin, ctx.eps)
    gk = _rope_rmsnorm_bwd(dk, k, wk, cos, sin, ctx.eps)
    return gq, gk, gv, None, None, None, None, None, None


_mx8_fused_op.register_autograd(_mx8_fused_bwd, setup_context=_mx8_fused_setup)


@custom_op("aitk::mxfp8_attn_fused_dec", mutates_args=(), device_types="cuda")
def _mx8_fused_dec_op(q: torch.Tensor, k_cur: torch.Tensor, v_cur: torch.Tensor,
                      k_pre: torch.Tensor, v_pre: torch.Tensor,
                      wq: torch.Tensor, wk: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor,
                      eps: float, sm_scale: float) -> list[torch.Tensor]:
    """Cached decode: q/k_cur/v_cur (L_tgt, H*128) raw GEMM outputs; k_pre/v_pre
    (L_pre, H*128) post-rope cached prefix (L_pre may be 0). q gets fused
    norm+rope+quant; current k gets torch norm+rope, then the concatenated k/v
    are quantized in one pass each. Inference-only (no autograd registered).
    Returns [o (S_tgt, H, D) bf16 padded]."""
    bwd = _load_bwd()
    Lq = q.shape[0]
    Lpre = k_pre.shape[0]
    Lkv = Lpre + Lq
    qq = _fused_quant_one(q, wq, cos, sin, eps, True)
    kc = _rmsnorm_rope_torch(k_cur, wk, cos, sin, eps)
    k_full = torch.cat([k_pre, kc], dim=0) if Lpre else kc.contiguous()
    kk = _fused_quant_one(k_full, wk, cos, sin, eps, False)
    v_full = torch.cat([v_pre, v_cur], dim=0) if Lpre else v_cur
    vv = _fused_quant_one(v_full, wk, cos, sin, eps, False)
    o, lse = bwd.fwd_attn(qq["qdT"], kk["qdT"], vv["qt"],
                          qq["sfp"], kk["sfp"], vv["sfpt"],
                          Lq, float(sm_scale), Lkv)
    return [o]


@_mx8_fused_dec_op.register_fake
def _(q, k_cur, v_cur, k_pre, v_pre, wq, wk, cos, sin, eps, sm_scale):
    Lq, HD = q.shape
    H = HD // 128
    Sq = (Lq + 127) // 128 * 128
    return [q.new_empty(Sq, H, 128)]


def fused_decode_available(hidden_states: torch.Tensor, attn) -> bool:
    """Gate for the fused decode path: env flag, triton, batch 1, bf16,
    head_dim 128, cuda."""
    return (_ENABLED and _TRITON_OK and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[0] == 1
            and attn.norm_q.weight.numel() == 128
            and hidden_states.shape[-1] == attn.heads * 128)


def mxfp8_attention_fused(q, k, v, wq, wk, cos, sin, eps: float, sm_scale: float) -> torch.Tensor:
    """Equal-length fused attention. q/k/v: (L, H*128) bf16 raw GEMM outputs
    (batch 1). Returns (L, H*128) bf16. Full autograd (dgrad only)."""
    L = q.shape[0]
    o = torch.ops.aitk.mxfp8_attn_fused(q, k, v, wq, wk, cos, sin,
                                        float(eps), float(sm_scale))[0]
    return o[:L].reshape(L, -1)


def mxfp8_attention_fused_dec(q, k_cur, v_cur, k_pre, v_pre, wq, wk, cos, sin,
                              eps: float, sm_scale: float) -> torch.Tensor:
    """Cached-decode fused attention. q/k_cur/v_cur: (L_tgt, H*128) raw GEMM
    outputs; k_pre/v_pre: (L_pre, H*128) post-rope cached prefix. Returns
    (L_tgt, H*128) bf16. Inference-only."""
    L = q.shape[0]
    o = torch.ops.aitk.mxfp8_attn_fused_dec(q, k_cur, v_cur, k_pre, v_pre,
                                            wq, wk, cos, sin,
                                            float(eps), float(sm_scale))[0]
    return o[:L].reshape(L, -1)


# ===================== block-causal (masked) mxfp8 attention =====================
# Training prefill path: the whole joint sequence with Qwen-Image 2.1's
# block-causal mask, expressed as a per-row key limit (row r attends [0, limit[r]):
# text rows limit = r+1, image rows limit = block end, pad rows limit = 0) and a
# persistent-scheduler plan whose per-tile kv_len carries the tile's max limit.
# The S3M kernels (s3m_*.cuh, new files forked from the tuned s3/s3b set) mask
# per-element; fwd and the mode-1 bwd kernels are covered. dvdk2/s3b64 modes are
# NOT masked -- the bc backward always uses the mode-1 dv/dk/dq_ws kernels.


class Mxfp8BCPlan:
    """One forward's block-causal work plan, shared by all layers. Built once per
    (seq_len, segments) -- see build_block_causal_plan. Passed to the processor
    through `attention_mask` (the same slot the flex BlockMask uses)."""

    __slots__ = (
        "work_indptr", "head_indices", "qo_tile_indices", "qo_indptr",
        "kv_indptr", "qo_lens", "kv_lens", "batch_indices", "row_kv_limit",
    )


_BC_PLAN_CACHE = {}


def build_block_causal_plan(segments, prefix_len: int, seq_len: int,
                            heads: int, device) -> Mxfp8BCPlan:
    """segments: [(start, end, is_text)] over the prefix; the target image block
    is [prefix_len, seq_len). Returns the cached plan for this exact layout."""
    key = (seq_len, prefix_len, tuple(segments), heads, device.index)
    plan = _BC_PLAN_CACHE.get(key)
    if plan is not None:
        return plan

    L, S = seq_len, (seq_len + 127) // 128 * 128
    limit = torch.zeros(S, dtype=torch.int64)
    for start, end, is_text in segments:
        if is_text:
            limit[start:end] = torch.arange(start + 1, end + 1, dtype=torch.int64)
        else:
            limit[start:end] = end
    limit[prefix_len:L] = L          # target block: attends everything
    # pad rows [L, S) stay 0 -> never attend -> contribute exactly 0

    # per 128-q-tile max limit (last real row; real rows are non-decreasing)
    n_tiles = (L + 127) // 128
    tile_kv = []
    for t in range(n_tiles):
        last = min(t * 128 + 127, L - 1)
        tile_kv.append(int(limit[last]))

    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    works = []
    for hq in range(heads):
        for qt in range(n_tiles):
            works.append((hq, qt, (tile_kv[qt] + 63) // 64))
    works.sort(key=lambda w: -w[2])          # stable desc, mirrors the C++ builder
    load = [0] * num_sm
    cta = [[] for _ in range(num_sm)]
    for hq, qt, cost in works:
        c = min(range(num_sm), key=lambda i: load[i])
        cta[c].append((hq, qt))
        load[c] += cost

    work_indptr, head_i, qtile_i = [0], [], []
    for c in range(num_sm):
        for hq, qt in cta[c]:
            head_i.append(hq)
            qtile_i.append(qt)
        work_indptr.append(len(head_i))
    n_work = len(head_i)

    i32 = dict(dtype=torch.int32, device=device)
    plan = Mxfp8BCPlan()
    plan.work_indptr = torch.tensor(work_indptr, **i32)
    plan.head_indices = torch.tensor(head_i, **i32)
    plan.qo_tile_indices = torch.tensor(qtile_i, **i32)
    plan.qo_indptr = torch.zeros(n_work, **i32)
    plan.kv_indptr = torch.zeros(n_work, **i32)
    plan.qo_lens = torch.full((n_work,), L, **i32)
    plan.kv_lens = torch.tensor([tile_kv[t] for t in qtile_i], **i32)
    plan.batch_indices = torch.zeros(n_work, **i32)
    plan.row_kv_limit = limit.to(**i32)
    _BC_PLAN_CACHE[key] = plan
    return plan


def bc_plan_available() -> bool:
    return _ENABLED and _TRITON_OK and torch.cuda.is_available()


@custom_op("aitk::mxfp8_attn_fused_bc", mutates_args=(), device_types="cuda")
def _mx8_bc_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               wq: torch.Tensor, wk: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor,
               eps: float, sm_scale: float,
               work_indptr: torch.Tensor, head_indices: torch.Tensor,
               qo_tile_indices: torch.Tensor, qo_indptr: torch.Tensor,
               kv_indptr: torch.Tensor, qo_lens: torch.Tensor,
               kv_lens: torch.Tensor, batch_indices: torch.Tensor,
               row_kv_limit: torch.Tensor) -> list[torch.Tensor]:
    """Block-causal fused prefill: fused norm+rope+quant -> S3M masked kernel.
    Returns [o (S,H,D) bf16 padded, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse]."""
    bwd = _load_bwd()
    L = q.shape[0]
    qq = _fused_quant_one(q, wq, cos, sin, eps, True)
    kk = _fused_quant_one(k, wk, cos, sin, eps, True)
    vv = _fused_quant_one(v, wq, cos, sin, eps, False)
    o, lse = bwd.fwd_attn_bc(qq["qdT"], kk["qdT"], vv["qt"],
                             qq["sfp"], kk["sfp"], vv["sfpt"], L, float(sm_scale),
                             work_indptr, head_indices, qo_tile_indices, qo_indptr,
                             kv_indptr, qo_lens, kv_lens, batch_indices, row_kv_limit)
    return [o, qq["qd"], kk["qd"], vv["qd"], qq["qt"], kk["qt"],
            qq["sfp"], kk["sfp"], vv["sfp"], qq["sfpt"], kk["sfpt"], lse]


@_mx8_bc_op.register_fake
def _(q, k, v, wq, wk, cos, sin, eps, sm_scale,
      work_indptr, head_indices, qo_tile_indices, qo_indptr, kv_indptr,
      qo_lens, kv_lens, batch_indices, row_kv_limit):
    L, HD = q.shape
    H = HD // 128
    S = (L + 127) // 128 * 128
    u8 = dict(dtype=torch.uint8, device=q.device)
    f32 = dict(dtype=torch.float32, device=q.device)
    return [q.new_empty(S, H, 128),
            q.new_empty(H, S, 128, **u8), q.new_empty(H, S, 128, **u8),
            q.new_empty(H, S, 128, **u8), q.new_empty(H, 128, S, **u8),
            q.new_empty(H, 128, S, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S, **f32)]


@custom_op("aitk::mxfp8_attn_bc_bwd", mutates_args=(), device_types="cuda")
def _mx8_bc_bwd_op(do: torch.Tensor, qd: torch.Tensor, kd: torch.Tensor,
                   vd: torch.Tensor, qt: torch.Tensor, kt: torch.Tensor,
                   sfq: torch.Tensor, sfk: torch.Tensor, sfv: torch.Tensor,
                   sfqt: torch.Tensor, sfkt: torch.Tensor,
                   lse: torch.Tensor, o: torch.Tensor, sm_scale: float,
                   row_kv_limit: torch.Tensor) -> list[torch.Tensor]:
    bwd = _load_bwd()
    return bwd.bwd_full_bc(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
                           lse, o, sm_scale, row_kv_limit)


@_mx8_bc_bwd_op.register_fake
def _(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o, sm_scale,
      row_kv_limit):
    H, L, D = do.shape
    out = do.new_empty(H, L, D)
    return [out, out.clone(), out.clone()]


def _mx8_bc_setup(ctx, inputs, output):
    (q, k, v, wq, wk, cos, sin, eps, sm_scale, work_indptr, head_indices,
     qo_tile_indices, qo_indptr, kv_indptr, qo_lens, kv_lens, batch_indices,
     row_kv_limit) = inputs
    ctx.save_for_backward(*output[1:], output[0], q, k, wq, wk, cos, sin, row_kv_limit)
    ctx.L = q.shape[0]
    ctx.eps = eps
    ctx.sm_scale = sm_scale


def _mx8_bc_bwd(ctx, grad_o, *grad_rest):
    if isinstance(grad_o, (list, tuple)):   # multi-output op: grads arrive as one list
        grad_o = grad_o[0]
    (qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o,
     q, k, wq, wk, cos, sin, row_kv_limit) = ctx.saved_tensors
    L = ctx.L
    do = grad_o[:L].transpose(0, 1).contiguous()      # (H, L, D) bf16
    dq, dk, dv = torch.ops.aitk.mxfp8_attn_bc_bwd(
        do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
        lse, o, ctx.sm_scale, row_kv_limit)
    gv = dv.transpose(0, 1).reshape(L, -1)            # v: no norm/rope
    gq = _rope_rmsnorm_bwd(dq, q, wq, cos, sin, ctx.eps)
    gk = _rope_rmsnorm_bwd(dk, k, wk, cos, sin, ctx.eps)
    return (gq, gk, gv) + (None,) * 15


_mx8_bc_op.register_autograd(_mx8_bc_bwd, setup_context=_mx8_bc_setup)


def mxfp8_attention_fused_bc(q, k, v, wq, wk, cos, sin, eps: float, sm_scale: float,
                             plan: Mxfp8BCPlan) -> torch.Tensor:
    """Block-causal prefill attention. q/k/v: (L, H*128) bf16 raw GEMM outputs
    (batch 1); plan from build_block_causal_plan. Returns (L, H*128) bf16.
    Full autograd (masked mode-1 bwd kernels)."""
    L = q.shape[0]
    o = torch.ops.aitk.mxfp8_attn_fused_bc(
        q, k, v, wq, wk, cos, sin, float(eps), float(sm_scale),
        plan.work_indptr, plan.head_indices, plan.qo_tile_indices, plan.qo_indptr,
        plan.kv_indptr, plan.qo_lens, plan.kv_lens, plan.batch_indices,
        plan.row_kv_limit)[0]
    return o[:L].reshape(L, -1)
