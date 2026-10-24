#!/usr/bin/env python
"""Batch inference for Qwen-Image 2.1 / 2.1-Turbo on a single small-VRAM
Blackwell card (sm_120, 16 GB).

Built around three ideas:

1. Phased offload
   The mxfp8 DiT (~7 GB) and the int8 Qwen3-VL text encoder (~8 GB) cannot be
   resident at the same time, so the run is split into phases and components
   are moved between CPU and GPU at phase boundaries:
     ENCODE   text encoder on GPU, DiT parked on CPU (mxfp8 static weights
              move CPU<->GPU as fp8 bytes -- verified bit-identical)
     IMAGES   VAE encodes all condition images (VAE is small, stays resident)
     SAMPLE   DiT on GPU for the rest of the run; VAE decodes per job
   `low_vram=True` on the ModelConfig parks DiT+TE on CPU right after load.

2. Caches
   - prompt-embed cache: (prompt, condition images, TE-affecting LoRAs) ->
     CPU embeds; N jobs sharing a prompt pay one TE pass. Optional on-disk
     persistence (--prompt-cache-dir).
   - condition-image cache: (path, mtime, budget) -> prepared tensor + packed
     VAE latents; shared by every job referencing the image.
   - prefix KV cache: step 0 ("extract") prefills text+condition K/V once,
     later steps decode only the target tokens. With --prefix-cache-reuse a
     finished cache is kept (LRU, byte budget) and jobs with the same
     prompt+images+resolution+LoRAs skip the prefill entirely -- exact, the
     prefix activations are timestep-independent (causal_condition=True).
   - with AITK_MXFP8_ATTN=1 (default) the cache is stored as mxfp8 kernel
     buffers (~2x compression) and decode steps run the fused mxfp8 kernel.

3. LoRA
   Per-job `loras: [{path, strength}]`. Jobs are grouped by LoRA signature;
   hooks (toolkit.inference_lora, dynamic strength) are attached/detached per
   group. Hook mode composes with the mxfp8 static weights (the weight
   subclass is untouched; the residual runs in bf16). `--lora-mode merge`
   instead folds the delta into the mxfp8 weights (dequantize -> add ->
   requantize) for zero per-step overhead when one LoRA covers the whole
   batch, and is the only mode compatible with --compile.

Usage:
    python scripts/qwen21_batch_infer.py --jobs jobs.jsonl --out-dir out/
    python scripts/qwen21_batch_infer.py --prompt "a cat" --num-per-prompt 4

Job file: JSON list, JSONL, or {"defaults": {...}, "jobs": [...]}:
    {"prompt": "...", "width": 1024, "height": 1024, "steps": 8, "seed": 1,
     "num_images": 1, "condition_images": ["ref.png"],
     "loras": [{"path": "...", "strength": 0.8}], "output": "name.png"}
Turbo needs no CFG; guidance>1 is supported for the base model (the negative
prompt is then encoded with the same condition images, so its token layout
matches). Defaults: steps 8 for *turbo* models, else 28; guidance 1.0.

Sampling modes:
    --sample-batch-size 1 (default) "stream": every DiT call is batch 1, so
        the fused mxfp8 attention + quantized KV cache stay engaged. Fastest
        per image on this card at >=1K resolutions.
    --sample-batch-size N>1: pads prompts into real batches. Attention falls
        back to SDPA (the fused kernel is batch-1) and the KV cache is bf16;
        mxfp8 GEMMs still apply. Prefix-cache reuse is stream-mode only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# mxfp8 attention is read from the env when toolkit.util.mxfp8_attn is first
# imported (during model load), so setting it here before main() is enough.
os.environ.setdefault("AITK_MXFP8_ATTN", "1")

# runnable from anywhere: the repo root holds `toolkit` and `extensions_built_in`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
@dataclass
class Job:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 8
    guidance: float = 1.0
    seed: int = 42
    condition_images: Tuple[str, ...] = ()
    loras: Tuple[Tuple[str, float], ...] = ()  # (path, strength), sorted
    output: Optional[str] = None
    index: int = 0

    @property
    def lora_sig(self) -> Tuple[Tuple[str, float], ...]:
        return self.loras

    def group_key(self, cond_shapes: Tuple[Tuple[int, int], ...]):
        """Jobs may share a sampling group iff this matches: everything that
        changes a DiT call's shapes or weights."""
        return (
            self.lora_sig,
            self.width,
            self.height,
            self.steps,
            self.guidance,
            cond_shapes,
        )


def load_jobs(path: str) -> List[dict]:
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            defaults = data.get("defaults", {})
            jobs = data["jobs"]
        else:
            defaults, jobs = {}, data
    except json.JSONDecodeError:
        defaults, jobs = {}, []
        for line in text.splitlines():
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return [dict(defaults, **job) for job in jobs]


def expand_jobs(raw_jobs: List[dict], cli_defaults: dict) -> List[Job]:
    jobs = []
    for raw in raw_jobs:
        spec = dict(cli_defaults, **raw)
        n = int(spec.pop("num_images", 1))
        loras = tuple(
            sorted(
                (str(l["path"]), float(l.get("strength", 1.0)))
                for l in (spec.get("loras") or [])
            )
        )
        for k in range(n):
            jobs.append(
                Job(
                    prompt=spec["prompt"],
                    negative_prompt=spec.get("negative_prompt", "") or "",
                    width=int(spec["width"]),
                    height=int(spec["height"]),
                    steps=int(spec["steps"]),
                    guidance=float(spec.get("guidance", 1.0)),
                    seed=int(spec.get("seed", 42)) + k,
                    condition_images=tuple(spec.get("condition_images") or ()),
                    loras=loras,
                    output=spec.get("output") if n == 1 else None,
                    index=len(jobs),
                )
            )
    return jobs


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------
class CondImageCache:
    """(path, mtime, budget) -> (prepared (1,C,H,W) [0,1] cpu tensor, PIL)."""

    def __init__(self, budget_pixels: int, rgba: bool):
        self.budget = budget_pixels
        self.rgba = rgba
        self._cache: Dict[Tuple, Tuple[Any, Any]] = {}

    def key_for(self, path: str) -> Tuple:
        st = os.stat(path)  # deliberate early failure for a bad path
        return (os.path.abspath(path), int(st.st_mtime), self.budget, self.rgba)

    def get(self, path: str):
        import numpy as np
        from PIL import Image

        from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
            prepare_condition_image,
            tensor_to_pil,
        )

        key = self.key_for(path)
        hit = self._cache.get(key)
        if hit is None:
            mode = "RGBA" if self.rgba else "RGB"
            tensor = (
                torch.from_numpy(
                    np.array(Image.open(path).convert(mode), dtype=np.float32) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            prepared = prepare_condition_image(tensor, self.budget)
            hit = (prepared, tensor_to_pil(prepared))
            self._cache[key] = hit
        return key, hit[0], hit[1]


class PromptEmbedCache:
    """(prompt, cond keys, TE-LoRA sig) -> per-sample CPU embeds/mask/slots.

    Optionally persisted as one safetensors file per key under `disk_dir`."""

    def __init__(self, disk_dir: Optional[str] = None):
        self.disk_dir = disk_dir
        self._cache: Dict[Tuple, Tuple[Any, Any, Any]] = {}
        if disk_dir:
            os.makedirs(disk_dir, exist_ok=True)

    def _path(self, key: Tuple) -> str:
        import hashlib

        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        return os.path.join(self.disk_dir, f"pe_{digest}.safetensors")

    def get(self, key: Tuple):
        if key in self._cache:
            return self._cache[key]
        if self.disk_dir:
            path = self._path(key)
            if os.path.exists(path):
                from safetensors.torch import load_file

                sd = load_file(path, device="cpu")
                hit = (sd["embeds"], sd["mask"], sd["slots"].bool())
                self._cache[key] = hit
                return hit
        return None

    def put(self, key: Tuple, value: Tuple[Any, Any, Any]):
        embeds, mask, slots = value
        value = (embeds.cpu(), mask.cpu(), slots.cpu().bool())
        self._cache[key] = value
        if self.disk_dir:
            from safetensors.torch import save_file

            save_file(
                {
                    "embeds": value[0],
                    "mask": value[1],
                    "slots": value[2].to(torch.uint8),
                },
                self._path(key),
            )


class PrefixKVStore:
    """LRU over finished prefix KV caches, keyed by the full prefix identity
    (prompt, condition images, resolution, LoRA signature, cfg stream).

    A hit lets a job skip the step-0 prefill entirely: the prefix K/V are
    timestep-independent, so decode-from-cache is exact for every step.
    """

    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self._store: Dict[Tuple, Tuple[Any, int]] = {}  # key -> (cache, nbytes)
        self._tick = 0
        self._lru: Dict[Tuple, int] = {}
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _nbytes(cache) -> int:
        total = 0
        for layer in cache.layer_caches:
            values = (
                vars(layer).values()
                if hasattr(layer, "__dict__")
                else (getattr(layer, s) for s in layer.__slots__)
            )
            for value in values:
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def get(self, key: Tuple):
        entry = self._store.get(key)
        self._tick += 1
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._lru[key] = self._tick
        return entry[0]

    def put(self, key: Tuple, cache):
        if key in self._store:
            return
        nbytes = self._nbytes(cache)
        if nbytes > self.budget:
            return  # too big to be worth evicting everything
        self._tick += 1
        while self.bytes_used + nbytes > self.budget and self._store:
            victim = min(self._lru, key=self._lru.get)
            self.bytes_used -= self._store.pop(victim)[1]
            self._lru.pop(victim)
        self._store[key] = (cache, nbytes)
        self._lru[key] = self._tick
        self.bytes_used += nbytes
        print(
            f"[prefix-store] put {nbytes / 2**30:.2f} GiB "
            f"({len(self._store)} entries, {self.bytes_used / 2**30:.2f} GiB)",
            flush=True,
        )


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------
class LoRAManager:
    """One InferenceLoRA per unique path (loaded once), attached/detached per
    job group. Hook mode is the default; merge mode folds the delta into the
    mxfp8 weights and restores the originals on detach."""

    def __init__(self, holder, mode: str = "hook"):
        from toolkit.inference_lora import InferenceLoRA

        self.holder = holder
        self.mode = mode
        self._by_path: Dict[str, InferenceLoRA] = {}
        self._active: List[InferenceLoRA] = []
        self._merged_weights: List[Tuple[Any, Any]] = []  # (module, orig param)
        self._te_sig_cache: Dict[Tuple, bool] = {}
        self._InferenceLoRA = InferenceLoRA

    def _load(self, path: str):
        lora = self._by_path.get(path)
        if lora is None:
            lora = self._InferenceLoRA(path).load(self.holder)
            print(f"[lora] {lora.report_line()}", flush=True)
            self._by_path[path] = lora
        return lora

    def has_te_entries(self, lora_sig: Tuple) -> bool:
        cached = self._te_sig_cache.get(lora_sig)
        if cached is not None:
            return cached
        te_ids = {id(m) for te in self.holder.text_encoder for m in te.modules()}
        result = False
        for path, _ in lora_sig:
            if any(id(e.module) in te_ids for e in self._load(path).entries):
                result = True
                break
        self._te_sig_cache[lora_sig] = result
        return result

    def attach(self, lora_sig: Tuple):
        self.detach()
        for path, strength in lora_sig:
            lora = self._load(path)
            lora.strength = strength
            if self.mode == "merge":
                self._merge(lora)
            else:
                lora.attach()
            self._active.append(lora)

    def detach(self):
        for lora in self._active:
            lora.detach()
        self._active = []
        self._unmerge()

    @torch.no_grad()
    def _merge(self, lora):
        """Fold strength*delta into the weights. mxfp8 static weights:
        dequantize -> add fp32 delta -> requantize. Plain floating weights:
        direct add. Originals kept for restore."""
        from toolkit.util.mxfp8_patch import (
            MXFp8StaticFrozenWeight,
            make_mxfp8_static_frozen_weight,
        )

        for e in lora.entries:
            w = getattr(e.module, "weight", None)
            if w is None:
                continue
            dw, db = e.delta(lora.strength, w.device)
            if dw is not None:
                self._merged_weights.append((e.module, e.module.weight))
                if isinstance(w, MXFp8StaticFrozenWeight):
                    fp = w.dequantize().to(torch.float32) + dw.reshape(w.shape)
                    e.module.weight = torch.nn.Parameter(
                        make_mxfp8_static_frozen_weight(fp.to(w._orig_dtype)),
                        requires_grad=False,
                    )
                elif torch.is_tensor(w) and w.is_floating_point():
                    e.module.weight = torch.nn.Parameter(
                        (w.data.to(torch.float32) + dw.reshape(w.shape)).to(w.dtype),
                        requires_grad=False,
                    )
                else:
                    self._merged_weights.pop()
                    print(
                        f"[lora] merge: skipping {e.name} "
                        f"(unsupported weight type {type(w).__name__})",
                        flush=True,
                    )
            if db is not None and getattr(e.module, "bias", None) is not None:
                e.module.bias.data.add_(db.to(e.module.bias.dtype))

    def _unmerge(self):
        for module, orig in self._merged_weights:
            module.weight = orig
        self._merged_weights = []


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class BatchEngine:
    def __init__(self, args):
        self.args = args
        self.sd = None
        self.prompt_cache = PromptEmbedCache(args.prompt_cache_dir or None)
        self.cond_cache: Optional[CondImageCache] = None
        self.prefix_store = PrefixKVStore(int(args.prefix_cache_budget_gb * 2**30))
        self.lora_manager: Optional[LoRAManager] = None
        self._dit_on_gpu = False

    # ---- loading ----
    def load(self):
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        args = self.args
        model_path = resolve_model_path(args.model)
        model_config = ModelConfig(
            arch="qwen_image_2",
            dtype="bf16",
            name_or_path=model_path,
            extras_name_or_path=model_path,  # TE/VAE/processor from the same repo
            quantize=args.qtype != "bf16",
            qtype=args.qtype,
            quantize_te=True,
            qtype_te=args.te_qtype,
            # parks DiT + TE on CPU after load; phases pull them up on demand
            low_vram=True,
            model_kwargs={
                "sampling_kv_cache": args.kv_cache,
                "control_image_max_pixels": args.control_max_pixels,
            },
        )
        ModelClass = get_model_class(model_config)
        self.sd = ModelClass(
            device="cuda",
            model_config=model_config,
            dtype="bf16",
            noise_scheduler=ModelClass.get_train_scheduler(),
        )
        t0 = time.time()
        self.sd.load_model()
        print(f"[load] {time.time() - t0:.1f}s | {gpu_mem()}", flush=True)
        self.cond_cache = CondImageCache(args.control_max_pixels, self.sd.load_rgba)
        self.lora_manager = LoRAManager(self.sd, mode=args.lora_mode)
        # sampling uses the repo's own scheduler config (the training scheduler
        # forces dynamic shifting, which conflicts with fixed distilled sigmas)
        self._scheduler_proto = None
        try:
            from toolkit.samplers.custom_flowmatch_sampler import (
                CustomFlowMatchEulerDiscreteScheduler,
            )

            self._scheduler_proto = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )
        except Exception as e:
            print(f"[load] repo scheduler unavailable ({e}); using train scheduler", flush=True)
        # a distilled repo ships its exact sampling schedule (fixed sigmas, no
        # mu-shift) in model_index.json -- use it whenever steps matches
        self.sample_sigmas = None
        if not args.ignore_repo_sigmas:
            index_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    self.sample_sigmas = json.load(f).get("sample_sigmas")
                if self.sample_sigmas:
                    print(
                        f"[load] repo ships {len(self.sample_sigmas)}-sigma sample schedule",
                        flush=True,
                    )

    def _make_scheduler(self, steps: int, lat_tokens: int):
        """Per-job scheduler. Prefers the repo's shipped sample_sigmas when the
        step count matches (distilled models are tuned to that exact schedule);
        otherwise linspace + mu-shift as in training-time sampling."""
        import numpy as np

        from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
            calculate_shift,
        )

        scheduler = self.sd.get_train_scheduler()
        if self._scheduler_proto is not None:
            import copy

            scheduler = copy.deepcopy(self._scheduler_proto)
        if self.sample_sigmas and steps == len(self.sample_sigmas):
            scheduler.set_timesteps(
                sigmas=np.array(self.sample_sigmas), device=self.sd.device_torch
            )
        else:
            sigmas = np.linspace(1.0, 1 / steps, steps)
            mu = calculate_shift(lat_tokens, 256, 8192, 0.5, 0.9)
            scheduler.set_timesteps(
                sigmas=sigmas, device=self.sd.device_torch, mu=mu
            )
        scheduler.set_begin_index(0)
        return scheduler

    # ---- component moves (the phase scheduler) ----
    def _move(self, module, device, what):
        if module is None or module.device == torch.device(device):
            return
        module.to(device)
        torch.cuda.empty_cache()
        print(f"[phase] {what} -> {device} | {gpu_mem()}", flush=True)

    def te_up(self):
        self._move(self.sd.text_encoder[0], "cuda", "text encoder")

    def te_down(self):
        self._move(self.sd.text_encoder[0], "cpu", "text encoder")

    def dit_up(self):
        if self._dit_on_gpu:
            return
        self._move(self.sd.model, "cuda", "transformer")
        self._dit_on_gpu = True
        if self.args.compile:
            print("[phase] compiling transformer blocks (dynamic)", flush=True)
            for i, block in enumerate(self.sd.model.transformer_blocks):
                self.sd.model.transformer_blocks[i] = torch.compile(
                    block, dynamic=True
                )

    # ---- phase 1: text ----
    @torch.no_grad()
    def encode_prompts(self, jobs: List[Job]):
        """Fill prompt_cache for every unique (prompt, conds, TE-LoRA sig)."""
        # group jobs by TE-affecting LoRA signature so TE LoRAs hook the encode
        te_groups: Dict[Tuple, List[Job]] = {}
        for job in jobs:
            te_sig = (
                job.lora_sig
                if job.lora_sig and self.lora_manager.has_te_entries(job.lora_sig)
                else ()
            )
            te_groups.setdefault(te_sig, []).append(job)

        self.te_up()
        try:
            for te_sig, group_jobs in te_groups.items():
                self.lora_manager.attach(te_sig)
                try:
                    todo: Dict[Tuple, Tuple[str, Tuple[str, ...]]] = {}
                    for job in group_jobs:
                        pairs = [(job.prompt, job.condition_images)]
                        if job.guidance > 1.0:
                            # the uncond stream runs the same joint sequence
                            # layout, so it is encoded with the same images
                            pairs.append((job.negative_prompt, job.condition_images))
                        for prompt, images in pairs:
                            cond_keys = tuple(
                                self.cond_cache.key_for(p) for p in images
                            )
                            key = (prompt, cond_keys, te_sig)
                            if self.prompt_cache.get(key) is None:
                                todo[key] = (prompt, images)
                    items = list(todo.items())
                    chunk = self.args.encode_batch_size
                    for start in range(0, len(items), chunk):
                        part = items[start : start + chunk]
                        pils = [
                            [self.cond_cache.get(p)[2] for p in paths]
                            for _, (_, paths) in part
                        ]
                        embeds, masks, slots = self.sd.prompt_encoder.encode(
                            [prompt for _, (prompt, _) in part],
                            images=pils,
                            device=self.sd.device_torch,
                        )
                        for (key, _), e, m, s in zip(part, embeds, masks, slots):
                            self.prompt_cache.put(
                                key, (e.to(self.sd.torch_dtype), m, s.bool())
                            )
                        print(
                            f"[encode] {min(start + chunk, len(items))}/{len(items)} "
                            f"prompts | {gpu_mem()}",
                            flush=True,
                        )
                finally:
                    self.lora_manager.detach()
        finally:
            self.te_down()

    # ---- phase 2: condition-image latents ----
    @torch.no_grad()
    def encode_condition_images(self, jobs: List[Job]) -> Dict[Tuple, Tuple[Any, Any]]:
        """Unique prepared condition image -> (packed (1,N,C) cpu latent, (h,w))."""
        from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
            pack_latents,
        )

        unique: Dict[Tuple, Any] = {}
        for job in jobs:
            for path in job.condition_images:
                key, prepared, _ = self.cond_cache.get(path)
                unique.setdefault(key, prepared)
        latents: Dict[Tuple, Tuple[Any, Any]] = {}
        for i, (key, prepared) in enumerate(unique.items()):
            latent = self.sd.encode_images(
                [prepared[0].to(self.sd.device_torch) * 2 - 1],
                device=self.sd.device_torch,
                dtype=self.sd.torch_dtype,
            )
            latents[key] = (
                pack_latents(latent).cpu(),
                (latent.shape[2], latent.shape[3]),
            )
            print(f"[vae-encode] {i + 1}/{len(unique)} | {gpu_mem()}", flush=True)
        return latents

    # ---- phase 3: sample ----
    def _make_kv_cache(self, quantized: bool):
        from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import (
            QwenImage21KVCache,
            QwenImage21KVLayerCache,
        )

        if quantized:
            # env-gated: Mx8KVLayerCache (mxfp8 kernel buffers) when the fused
            # attention path is on, plain bf16 otherwise
            return QwenImage21KVCache(self.sd.model.config.num_layers)
        # plain bf16 layers regardless of the env (batch mode: the mx8 cache
        # is batch-1 only)
        cache = QwenImage21KVCache.__new__(QwenImage21KVCache)
        cache.layer_caches = [
            QwenImage21KVLayerCache() for _ in range(self.sd.model.config.num_layers)
        ]
        return cache

    def _prefix_key(self, job: Job, cond_keys: Tuple, stream: str) -> Tuple:
        prompt = job.prompt if stream == "cond" else job.negative_prompt
        return (
            prompt,
            cond_keys,  # both streams carry the same image slots
            job.width,
            job.height,
            job.lora_sig,
            stream,
        )

    def _job_prompt_embeds(self, job: Job, cond_keys: Tuple) -> Dict[str, Tuple]:
        """(embeds, mask, slots) triples per stream, CPU, from the cache."""
        te_sig = (
            job.lora_sig
            if job.lora_sig and self.lora_manager.has_te_entries(job.lora_sig)
            else ()
        )
        out = {"cond": self.prompt_cache.get((job.prompt, cond_keys, te_sig))}
        if job.guidance > 1.0:
            out["uncond"] = self.prompt_cache.get(
                (job.negative_prompt, cond_keys, te_sig)
            )
        return out

    def _job_condition_latents(self, job: Job, cond_latents, cond_keys: Tuple):
        if not cond_keys:
            return None, []
        rows, shapes = zip(*[cond_latents[k] for k in cond_keys])
        return torch.cat(list(rows), dim=1).to(self.sd.device_torch), list(shapes)

    @torch.no_grad()
    def _sample_stream(self, job: Job, cond_latents, cond_keys):
        """One prompt stream, batch 1: the fused mxfp8 path stays engaged."""

        from diffusers.utils.torch_utils import randn_tensor

        from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
            pad_prompt_batch,
            run_transformer,
        )

        sd = self.sd
        device, dtype = sd.device_torch, sd.torch_dtype
        transformer = sd.model
        lat_h, lat_w = job.height // 16, job.width // 16
        channels = transformer.config.in_channels

        scheduler = self._make_scheduler(job.steps, lat_h * lat_w)

        latents = randn_tensor(
            (1, channels, lat_h, lat_w),
            generator=torch.Generator().manual_seed(job.seed),
            device=device,
            dtype=torch.float32,
        ).to(dtype)

        condition_latents, condition_shapes = self._job_condition_latents(
            job, cond_latents, cond_keys
        )
        embeds = self._job_prompt_embeds(job, cond_keys)
        streams = [("cond", embeds["cond"])]
        if "uncond" in embeds:
            streams.append(("uncond", embeds["uncond"]))

        use_kv = self.args.kv_cache and bool(transformer.config.causal_condition)
        reuse = self.args.prefix_cache_reuse and use_kv

        padded = {
            name: pad_prompt_batch([pe[0]], [pe[1]], [pe[2]], device, dtype)
            for name, pe in streams
        }
        # stream -> (prefix key, cache, already prefilled)
        state = {}
        for name, _ in streams:
            if not use_kv:
                state[name] = (None, None, False)
                continue
            key = self._prefix_key(job, cond_keys, name)
            cache = self.prefix_store.get(key) if reuse else None
            prefilled = cache is not None
            if cache is None:
                cache = self._make_kv_cache(quantized=True)
            state[name] = (key, cache, prefilled)
            if use_kv:
                print(
                    f"[prefix] job {job.index} {name}: "
                    f"{'hit' if prefilled else 'miss'} "
                    f"(store {len(self.prefix_store._store)} entries, "
                    f"{self.prefix_store.bytes_used / 2**30:.2f} GiB)",
                    flush=True,
                )

        step_times = []
        profiler = None
        if os.environ.get("AITK_PROFILE_STEPS") == "1":
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler.__enter__()
        for step_index, timestep in enumerate(scheduler.timesteps):
            t_step = time.time()
            t = timestep.expand(1).to(device, dtype=dtype) / 1000
            preds = {}
            for name, _ in streams:
                _, cache, prefilled = state[name]
                kwargs = {}
                if use_kv:
                    kwargs = {
                        "kv_cache": cache,
                        # fresh cache: step 0 extracts; a store hit decodes
                        # straight from the reused prefix on every step
                        "kv_cache_mode": (
                            "cached" if (prefilled or step_index > 0) else "extract"
                        ),
                    }
                preds[name] = run_transformer(
                    transformer,
                    latents,
                    t,
                    *padded[name],
                    condition_latents=condition_latents,
                    condition_shapes=condition_shapes,
                    **kwargs,
                )
            noise_pred = preds["cond"]
            if "uncond" in preds:
                noise_pred = preds["uncond"] + job.guidance * (
                    preds["cond"] - preds["uncond"]
                )
            latents = scheduler.step(
                noise_pred.to(torch.float32),
                timestep,
                latents.to(torch.float32),
                return_dict=False,
            )[0].to(dtype)
            torch.cuda.synchronize()
            step_times.append(time.time() - t_step)

        if profiler is not None:
            profiler.__exit__(None, None, None)
            print(
                profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25),
                flush=True,
            )

        if step_times:
            rest = step_times[1:]
            print(
                f"[steps] job {job.index}: step0 {step_times[0]:.2f}s"
                + (
                    f" | next {len(rest)}x avg {sum(rest)/len(rest):.2f}s"
                    if rest
                    else ""
                ),
                flush=True,
            )

        if use_kv and reuse:
            for name, _ in streams:
                key, cache, prefilled = state[name]
                if not prefilled:
                    self.prefix_store.put(key, cache)
        return latents

    @torch.no_grad()
    def _sample_batch(self, jobs: List[Job], cond_latents):
        """Padded micro-batch. Attention is SDPA here (fused path is batch-1);
        KV cache layers are bf16. Prefix-cache reuse is stream-mode only."""

        from diffusers.utils.torch_utils import randn_tensor

        from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
            pad_prompt_batch,
            run_transformer,
        )

        sd = self.sd
        device, dtype = sd.device_torch, sd.torch_dtype
        transformer = sd.model
        job0 = jobs[0]
        B = len(jobs)
        lat_h, lat_w = job0.height // 16, job0.width // 16
        channels = transformer.config.in_channels

        scheduler = self._make_scheduler(job0.steps, lat_h * lat_w)

        latents = torch.cat(
            [
                randn_tensor(
                    (1, channels, lat_h, lat_w),
                    generator=torch.Generator().manual_seed(job.seed),
                    device=device,
                    dtype=torch.float32,
                )
                for job in jobs
            ],
            dim=0,
        ).to(dtype)

        embeds = [
            self._job_prompt_embeds(
                job, tuple(self.cond_cache.key_for(p) for p in job.condition_images)
            )
            for job in jobs
        ]
        cond = pad_prompt_batch(
            [e["cond"][0] for e in embeds],
            [e["cond"][1] for e in embeds],
            [e["cond"][2] for e in embeds],
            device,
            dtype,
        )
        do_cfg = job0.guidance > 1.0
        uncond = None
        if do_cfg:
            uncond = pad_prompt_batch(
                [e["uncond"][0] for e in embeds],
                [e["uncond"][1] for e in embeds],
                [e["uncond"][2] for e in embeds],
                device,
                dtype,
            )

        condition_latents, condition_shapes = None, []
        if job0.condition_images:
            per_job = []
            for job in jobs:
                keys = tuple(self.cond_cache.key_for(p) for p in job.condition_images)
                rows, shapes = zip(*[cond_latents[k] for k in keys])
                per_job.append(torch.cat(list(rows), dim=1))
                condition_shapes = list(shapes)
            condition_latents = torch.cat(per_job, dim=0).to(device)

        use_kv = self.args.kv_cache and bool(transformer.config.causal_condition)
        cond_cache = self._make_kv_cache(quantized=False) if use_kv else None
        uncond_cache = self._make_kv_cache(quantized=False) if use_kv and do_cfg else None

        for step_index, timestep in enumerate(scheduler.timesteps):
            t = timestep.expand(B).to(device, dtype=dtype) / 1000
            cache_kwargs = (
                {"kv_cache_mode": "extract" if step_index == 0 else "cached"}
                if use_kv
                else {}
            )
            noise_pred = run_transformer(
                transformer,
                latents,
                t,
                *cond,
                condition_latents=condition_latents,
                condition_shapes=condition_shapes,
                kv_cache=cond_cache,
                **cache_kwargs,
            )
            if do_cfg:
                uncond_pred = run_transformer(
                    transformer,
                    latents,
                    t,
                    *uncond,
                    condition_latents=condition_latents,
                    condition_shapes=condition_shapes,
                    kv_cache=uncond_cache,
                    **cache_kwargs,
                )
                noise_pred = uncond_pred + job0.guidance * (noise_pred - uncond_pred)
            latents = scheduler.step(
                noise_pred.to(torch.float32),
                timestep,
                latents.to(torch.float32),
                return_dict=False,
            )[0].to(dtype)
        return latents

    @torch.no_grad()
    def run(self, jobs: List[Job]):
        args = self.args
        os.makedirs(args.out_dir, exist_ok=True)
        t_start = time.time()
        torch.cuda.reset_peak_memory_stats()

        print(f"[phase] ENCODE: {len(jobs)} jobs", flush=True)
        t0 = time.time()
        self.encode_prompts(jobs)
        print(
            f"[phase] ENCODE done in {time.time() - t0:.1f}s | {gpu_mem()}",
            flush=True,
        )

        t0 = time.time()
        cond_latents = self.encode_condition_images(jobs)
        if cond_latents:
            print(
                f"[phase] IMAGE-ENCODE done in {time.time() - t0:.1f}s | {gpu_mem()}",
                flush=True,
            )

        # group jobs: same LoRA set + shape + condition layout
        groups: Dict[Tuple, List[Job]] = {}
        for job in jobs:
            shapes = tuple(
                cond_latents[self.cond_cache.key_for(p)][1]
                for p in job.condition_images
            )
            groups.setdefault(job.group_key(shapes), []).append(job)
        print(f"[phase] SAMPLE: {len(groups)} group(s)", flush=True)

        self.dit_up()
        # sample everything first, latents parked on cpu (131 KB @1024^2):
        # VAE decode gets its own phase below with the DiT off the card, so
        # the decode activation spike never stacks on DiT + KV caches
        pending = []  # (job, cpu latents, per-job sample seconds)
        bs = max(1, args.sample_batch_size)
        for group_key, group_jobs in groups.items():
            self.lora_manager.attach(group_key[0])
            try:
                for start in range(0, len(group_jobs), bs):
                    chunk = group_jobs[start : start + bs]
                    t_chunk = time.time()
                    if bs > 1 and len(chunk) > 1:
                        latents = self._sample_batch(chunk, cond_latents)
                        rows = [(job, latents[i : i + 1]) for i, job in enumerate(chunk)]
                    else:
                        rows = []
                        for job in chunk:
                            keys = tuple(
                                self.cond_cache.key_for(p)
                                for p in job.condition_images
                            )
                            rows.append(
                                (job, self._sample_stream(job, cond_latents, keys))
                            )
                    sample_s = (time.time() - t_chunk) / len(rows)
                    for job, job_latents in rows:
                        pending.append((job, job_latents.cpu(), sample_s))
                        print(
                            f"[sample {len(pending)}/{len(jobs)}] {job.index}: "
                            f"{sample_s:.1f}s | {gpu_mem()}",
                            flush=True,
                        )
                    del rows
            finally:
                self.lora_manager.detach()

        # ---- phase 4: decode (DiT parked, VAE has the card to itself) ----
        self._move(self.sd.model, "cpu", "transformer")
        self._dit_on_gpu = False
        print(f"[phase] DECODE: {len(pending)} images", flush=True)
        t_dec = time.time()
        report = []
        for job, job_latents, sample_s in pending:
            t0 = time.time()
            image = self.sd.decode_to_images(job_latents)[0]
            decode_s = time.time() - t0
            out = (
                job.output
                or f"{job.index:04d}_s{job.seed}_{job.width}x{job.height}.png"
            )
            out_path = os.path.join(args.out_dir, out)
            os.makedirs(os.path.dirname(out_path) or args.out_dir, exist_ok=True)
            image.save(out_path)
            report.append(
                {
                    "index": job.index,
                    "output": out_path,
                    "prompt": job.prompt,
                    "seed": job.seed,
                    "size": [job.width, job.height],
                    "steps": job.steps,
                    "guidance": job.guidance,
                    "loras": list(job.lora_sig),
                    "sample_s": round(sample_s, 2),
                    "decode_s": round(decode_s, 2),
                }
            )
            print(
                f"[job {job.index + 1}/{len(jobs)}] {out} "
                f"sample {sample_s:.1f}s decode {decode_s:.1f}s "
                f"| {gpu_mem()}",
                flush=True,
            )
        print(
            f"[phase] DECODE done in {time.time() - t_dec:.1f}s | {gpu_mem()}",
            flush=True,
        )
        del pending

        total = time.time() - t_start
        print(
            f"[done] {len(jobs)} images in {total:.1f}s "
            f"({60 * len(jobs) / max(total, 1e-9):.2f} img/min) | "
            f"prefix cache: {self.prefix_store.hits} hits/"
            f"{self.prefix_store.misses} misses, "
            f"{self.prefix_store.bytes_used / 2**30:.2f} GiB held | {gpu_mem()}",
            flush=True,
        )
        with open(os.path.join(args.out_dir, "report.json"), "w") as f:
            json.dump({"jobs": report}, f, indent=2, ensure_ascii=False)


def resolve_model_path(model: str) -> str:
    """Repo id -> local snapshot dir (downloading if needed); a directory
    passes through. Loading from the snapshot dir keeps the ComfyUI
    candidate logic out of the way and works offline once cached."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    # full snapshot: the loader reads transformer/text_encoder/vae/processor
    # from this one directory
    return snapshot_download(model)


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "no cuda"
    return (
        f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
        f"(peak {torch.cuda.max_memory_allocated() / 2**30:.2f})"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch inference for Qwen-Image 2.1 / 2.1-Turbo (mxfp8, 16 GB cards)"
    )
    p.add_argument("--jobs", help="JSON/JSONL job file")
    p.add_argument("--out-dir", default="out_qwen21")
    p.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    p.add_argument("--qtype", default="mxfp8_static",
                   help="DiT quantization: mxfp8_static (Blackwell) | convrot8 | bf16")
    p.add_argument("--te-qtype", default="convrot8")
    p.add_argument("--no-mxfp8-attn", action="store_true",
                   help="disable the fused mxfp8 attention / quantized KV cache")
    p.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefix-cache-reuse", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="keep finished prefix KV caches (LRU) so repeat jobs skip prefill")
    p.add_argument("--prefix-cache-budget-gb", type=float, default=4.0)
    p.add_argument("--sample-batch-size", type=int, default=1,
                   help=">1 pads prompts into real batches (SDPA attention, bf16 cache)")
    p.add_argument("--lora-mode", choices=["hook", "merge"], default="hook")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the DiT blocks (use with --lora-mode merge or no LoRA)")
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--control-max-pixels", type=int, default=1024 * 1024)
    p.add_argument("--prompt-cache-dir", default="")
    p.add_argument("--ignore-repo-sigmas", action="store_true",
                   help="ignore the repo's shipped sample_sigmas (A/B against the "
                        "linspace+mu-shift schedule)")
    # single-prompt convenience (no jobs file)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None,
                   help="default: 8 for *turbo* models, else 28")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-per-prompt", type=int, default=1)
    p.add_argument("--lora", action="append", default=[],
                   help="'path[:strength]' applied to every --prompt job")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.no_mxfp8_attn:
        os.environ["AITK_MXFP8_ATTN"] = "0"

    default_steps = args.steps
    if default_steps is None:
        default_steps = 8 if "turbo" in args.model.lower() else 28

    if args.jobs:
        raw_jobs = load_jobs(args.jobs)
    else:
        loras = []
        for spec in args.lora:
            path, _, strength = spec.partition(":")
            loras.append({"path": path, "strength": float(strength or 1.0)})
        raw_jobs = [
            {
                "prompt": prompt,
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
                "num_images": args.num_per_prompt,
                "loras": loras,
            }
            for prompt in args.prompt
        ]
    if not raw_jobs:
        print("no jobs: pass --jobs or --prompt", file=sys.stderr)
        sys.exit(1)

    jobs = expand_jobs(
        raw_jobs,
        cli_defaults={
            "width": args.width,
            "height": args.height,
            "steps": default_steps,
            "seed": args.seed,
        },
    )

    if args.compile and args.lora_mode == "hook":
        print(
            "[warn] --compile with hook-mode LoRAs: forward hooks inside "
            "compiled blocks cause graph breaks. Prefer --lora-mode merge.",
            flush=True,
        )

    engine = BatchEngine(args)
    engine.load()
    engine.run(jobs)


if __name__ == "__main__":
    main()
