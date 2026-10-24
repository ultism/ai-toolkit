"""Serialization for static-mxfp8-quantized modules (the quantized-weight cache).

The stock comfy-format save path dequantizes MXFp8StaticFrozenWeight back to
bf16 (it only knows how to keep OstrisLinear's storage). For mxfp8 we instead
store the *quantized payload itself*: every static weight is flattened to its
raw components (qdata + e8m0 scale frames, uint8 views) plus a small metadata
record, so a later load rebuilds the weight with **zero quantization work and
zero GPU involvement**, bit-identical to the first run.

Layout (one directory per quantized component):
    manifest.json        format_version, qtype, source, per-weight metadata
    weights.safetensors  plain tensors as-is + flattened mxfp8 components

Only the payload format lives here; wiring it into component loading is a
separate step (validated on GPU).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch

from toolkit.util.mxfp8_patch import MXFp8StaticFrozenWeight

FORMAT_VERSION = 1

_QDATA = ".__qdata__"
_SCALE = ".__scale__"
_QDATA1 = ".__qdata1__"
_SCALE1 = ".__scale1__"
_SF_RT = ".__sf_rt__"


def flatten_state_dict(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """(tensors for safetensors, per-weight mxfp8 metadata)."""
    tensors: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, MXFp8StaticFrozenWeight):
            tensors[key] = value.detach().cpu().contiguous()
            continue
        w = value
        m0 = w._w_dim0
        record = {
            "kind": "mxfp8_static",
            "shape": list(w.shape),
            "orig_dtype": str(w._orig_dtype).replace("torch.", ""),
            "elem_dtype": str(m0.elem_dtype).replace("torch.", ""),
            "block_size": m0.block_size,
            "is_swizzled": bool(m0.is_swizzled_scales),
            "qdata_shape": list(m0.qdata.shape),
            "scale_shape": list(m0.scale.shape),
        }
        tensors[key + _QDATA] = m0.qdata.view(torch.uint8).cpu().contiguous()
        tensors[key + _SCALE] = m0.scale.view(torch.uint8).cpu().contiguous()
        if w._w_dim1 is not None:
            m1 = w._w_dim1
            record["qdata1_shape"] = list(m1.qdata.shape)
            record["scale1_shape"] = list(m1.scale.shape)
            tensors[key + _QDATA1] = m1.qdata.view(torch.uint8).cpu().contiguous()
            tensors[key + _SCALE1] = m1.scale.view(torch.uint8).cpu().contiguous()
        if w._sf_rt is not None:
            record["sf_rt_shape"] = list(w._sf_rt.shape)
            tensors[key + _SF_RT] = w._sf_rt.view(torch.uint8).cpu().contiguous()
        meta[key] = record
    return tensors, meta


def unflatten_state_dict(
    tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Rebuild MXFp8StaticFrozenWeight objects from the flattened payload."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )

    def _mx(qd_flat, sc_flat, record, suffix=""):
        qd = qd_flat.view(getattr(torch, record["elem_dtype"]))
        qd = qd.view(record[f"qdata{suffix}_shape"])
        sc = sc_flat.view(torch.float8_e8m0fnu).view(record[f"scale{suffix}_shape"])
        return MXTensor(
            qd, sc, getattr(torch, record["elem_dtype"]), record["block_size"],
            getattr(torch, record["orig_dtype"]), KernelPreference.AUTO, None,
            record["is_swizzled"],
        )

    out: Dict[str, Any] = {}
    for key, record in meta.items():
        w0 = _mx(tensors[key + _QDATA], tensors[key + _SCALE], record)
        w1 = (
            _mx(tensors[key + _QDATA1], tensors[key + _SCALE1], record, "1")
            if key + _QDATA1 in tensors
            else None
        )
        sf_rt = (
            tensors[key + _SF_RT].view(torch.float8_e8m0fnu).view(record["sf_rt_shape"])
            if key + _SF_RT in tensors
            else None
        )
        w = MXFp8StaticFrozenWeight(
            w0, w1, tuple(record["shape"]), getattr(torch, record["orig_dtype"]),
            w0.qdata.device, sf_rt=sf_rt,
        )
        out[key] = w
    for key, value in tensors.items():
        if key.split(".__")[0] in meta:
            continue
        out[key] = value
    return out


def save_quantized(model, directory: str, qtype: str, source: str = "") -> None:
    """Save a module carrying static mxfp8 weights (and its plain tensors)."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    tensors, meta = flatten_state_dict(state_dict)
    os.makedirs(directory, exist_ok=True)
    save_file(tensors, os.path.join(directory, "weights.safetensors"))
    manifest = {
        "format_version": FORMAT_VERSION,
        "qtype": qtype,
        "source": source,
        "num_mxfp8_weights": len(meta),
        "mxfp8": meta,
    }
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"[mxfp8_io] saved {len(meta)} mxfp8 weights "
        f"({len(state_dict) - len(meta)} plain tensors) -> {directory}",
        flush=True,
    )


def load_quantized(directory: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """-> (state_dict with rebuilt static weights, manifest)."""
    from safetensors.torch import load_file

    with open(os.path.join(directory, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported mxfp8 cache format {manifest['format_version']} "
            f"(expected {FORMAT_VERSION}); re-run the converter"
        )
    tensors = load_file(os.path.join(directory, "weights.safetensors"), device="cpu")
    return unflatten_state_dict(tensors, manifest["mxfp8"]), manifest
