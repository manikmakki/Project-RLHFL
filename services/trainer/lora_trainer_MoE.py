"""
Project RLHFL - LoRA Trainer (QLoRA, Intel Arc XPU)

Trains LoRA adapters for Gemma-4-26B-A4B (MoE, 128 experts, top-8 active).
Supports SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).

Device strategy:
  4-bit NF4 quantization (via Unsloth or bitsandbytes directly) reduces the
  48 GiB BF16 model to ~13 GiB, fitting the full Gemma-4 26B-A4B on the
  Arc Pro B60's 22 GiB VRAM with ~9 GiB headroom for activations.

  Unsloth path (preferred): patches SFTTrainer/DPOTrainer with chunked CE and
  RoPE kernels for extra speed.  Requires upstream PR #6106 (Intel XPU device
  detection) to be merged, OR the Dockerfile patch that backports it.

  bitsandbytes path (fallback): standard transformers BitsAndBytesConfig + PEFT
  get_peft_model.  Works when Unsloth cannot detect the XPU accelerator.
"""

import gc
import logging
import os
import json
import time as _time
import types
import torch

# ── Unsloth (optional) ───────────────────────────────────────────────────────
# unsloth_zoo.device_type.get_device_type() raises NotImplementedError for XPU
# until upstream PR #6106 is merged (or the Dockerfile patch is applied).
# We catch all exceptions so a missing / incompatible Unsloth never crashes the
# service on startup.
_UNSLOTH_AVAILABLE = False
_FastModel = None
_unsloth_err = None
try:
    from unsloth import FastModel as _FastModel
    _UNSLOTH_AVAILABLE = True
except Exception as _e:
    _unsloth_err = _e

from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
# DPOTrainer/DPOConfig are imported lazily inside _train_dpo() to avoid triggering
# trl → mergekit → pydantic-core which fails to build on Python 3.14 (PyO3 cap).
# ── dill / Python 3.14 compatibility ────────────────────────────────────────
# Python 3.14 changed pickle.save_dict (pickle.py line ~1064) to call
#   self._batch_setitems(obj.items(), obj)
# with THREE positional args.  dill.Pickler (and datasets._dill.Pickler which
# inherits it) subclass the C _pickle.Pickler whose _batch_setitems is NOT
# exposed as a Python attribute — so our old patch tried to grab it, hit
# AttributeError, and was silently swallowed by the except clause.
#
# Fix: use pickle._Pickler (the pure-Python fallback class).  Python 3.14
# already updated _Pickler._batch_setitems to accept (self, items, obj=None).
# Inject that pure-Python method onto dill.Pickler and datasets._dill.Pickler
# so the 3-arg call is accepted on both.  Must happen before any datasets use.
import sys as _sys
if _sys.version_info >= (3, 14):
    try:
        import pickle as _std_pickle
        # _Pickler = pure-Python fallback; its _batch_setitems is already updated
        _py_bsi = _std_pickle._Pickler._batch_setitems
        import dill._dill as _dill_mod
        _dill_mod.Pickler._batch_setitems = _py_bsi
        # Also patch datasets' own Pickler subclass directly for full coverage
        import datasets.utils._dill as _ds_dill
        _ds_dill.Pickler._batch_setitems = _py_bsi
        del _py_bsi, _std_pickle, _dill_mod, _ds_dill
    except Exception as _patch_err:
        import warnings as _w
        _w.warn(f"Python 3.14 dill compat patch failed — training will crash on "
                f"Dataset.from_dict(): {_patch_err}")
        del _patch_err, _w

from datasets import Dataset
import datasets as _datasets
_datasets.disable_caching()  # skip cache I/O (no benefit re-running transforms)

# ── transformers XPU detection patch ─────────────────────────────────────────
# transformers.training_args._setup_devices() calls is_torch_xpu_available() to
# decide whether to place the Trainer on XPU.  Older builds gate this on IPEX
# being installed (is_ipex_available()).  We use native torch.xpu, so IPEX is
# not present and the Trainer falls back to CPU even when XPU is live.
# Patch the utility function to use torch.xpu directly before TrainingArguments
# is ever instantiated.
try:
    import transformers.utils as _tu
    def _native_xpu_available(check_device: bool = False) -> bool:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return False
        if check_device:
            return torch.xpu.device_count() > 0
        return True
    _tu.is_torch_xpu_available = _native_xpu_available
    # training_args imports the symbol at module load time so patch there too
    import transformers.training_args as _ta
    _ta.is_torch_xpu_available = _native_xpu_available
    del _tu, _ta, _native_xpu_available
except Exception as _xpu_patch_err:
    import warnings as _w
    _w.warn(f"XPU detection patch failed — Trainer may use CPU: {_xpu_patch_err}")
    del _w, _xpu_patch_err

from shared.config import SystemConfig

# SFTConfig (TRL >=0.15) extends TrainingArguments with dataset_text_field etc.
try:
    from trl import SFTConfig
    _HAS_SFT_CONFIG = True
except ImportError:
    _HAS_SFT_CONFIG = False

# prepare_model_for_kbit_training: required for gradient checkpointing with 4-bit
try:
    from peft import prepare_model_for_kbit_training
    _HAS_KBIT_PREP = True
except ImportError:
    _HAS_KBIT_PREP = False

logger = logging.getLogger(__name__)

if not _UNSLOTH_AVAILABLE:
    logger.warning(
        "Unsloth not available (%s). Using bitsandbytes + standard PEFT path. "
        "Unsloth Intel XPU support: upstream PR #6106.",
        _unsloth_err,
    )
else:
    logger.info("Unsloth loaded — using FastModel path with patched kernels.")

_XPU_HARD_FAIL = False  # True after DEVICE_LOST; xe context is dead for this session


class _TimingCallback(TrainerCallback):
    """Benchmark the first N steps; set too_slow if the projected run exceeds budget.

    WARMUP_STEPS=10 (up from 3) skips the SYCL JIT compilation spike that
    inflates the first few step times on a cold XPU context.
    """

    WARMUP_STEPS = 10
    MEASURE_STEPS = 5

    def __init__(self, max_hours: float):
        self.max_hours = max_hours
        self.too_slow = False
        self.projected_hours = 0.0
        self._t0: float = 0.0
        self._times: list = []

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = _time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        self._times.append(_time.monotonic() - self._t0)

        measured = self._times[self.WARMUP_STEPS:]
        if len(measured) < self.MEASURE_STEPS:
            return

        avg_sec = sum(measured[: self.MEASURE_STEPS]) / self.MEASURE_STEPS
        self.projected_hours = avg_sec * state.max_steps / 3600
        logger.info(
            f"Timing benchmark ({self.MEASURE_STEPS} steps after {self.WARMUP_STEPS} warmup): "
            f"{avg_sec:.1f}s/step → {self.projected_hours:.1f}h projected "
            f"({state.max_steps} total optimizer steps)"
        )
        if self.projected_hours > self.max_hours:
            logger.warning(
                f"Projected {self.projected_hours:.1f}h > {self.max_hours}h budget — "
                f"stopping training early"
            )
            self.too_slow = True
            control.should_training_stop = True



class _NF4Experts3D:
    """NF4 store for a fused 3D MoE expert weight [num_experts, out, in].

    Drop-in replacement for Gemma4TextExperts.gate_up_proj / .down_proj.
    Gemma4's forward accesses experts by index::

        nn.functional.linear(x, self.gate_up_proj[expert_idx])

    so __getitem__ returning a dequantized 2D BF16 slice is a transparent
    drop-in — the forward method runs unchanged.

    This is a plain Python object, NOT an nn.Parameter or nn.Module, so it is
    invisible to model._apply() and model.to() — accelerate, Unsloth, and the
    Trainer never try to move or train it.  We dispatch it explicitly to XPU
    after model dispatch via the .to(device) method.
    """

    def __init__(self, weight_3d: torch.Tensor, quant_type: str = "nf4",
                 blocksize: int = 64):
        from bitsandbytes.functional import quantize_4bit
        self.shape = tuple(weight_3d.shape)
        self.dtype = torch.bfloat16
        self.num_experts = weight_3d.shape[0]
        self._packed: list = []   # uint8 tensors, one per expert (moved to device)
        self._qstate: list = []   # QuantState objects, always kept on CPU
        w3 = weight_3d.to(torch.bfloat16)
        for e in range(self.num_experts):
            w = w3[e].contiguous()  # 2D slice [out, in]
            qw, qs = quantize_4bit(
                w, quant_type=quant_type, blocksize=blocksize,
                compress_statistics=False,
                quant_storage=torch.uint8,
            )
            self._packed.append(qw)
            self._qstate.append(qs)  # stays on CPU — dequantize_4bit works cross-device

    def to(self, device):
        # QuantState.to() mutates in-place and returns None in bnb 0.49, so we
        # never call it. But leaving the QuantState's absmax/code on CPU while
        # qw is on XPU forces bnb to dequantize on CPU and transfer the small
        # result back — fine for one call, but with ~100+ experts hit per
        # layer (untrained router spreads load near-uniformly across all 128)
        # x 30 layers x 2 (gradient-checkpoint recompute), the CPU-roundtrip
        # overhead dominates: 0% GPU util, 200%+ CPU util, minutes per step.
        # Move just the tensor attributes — not the QuantState object itself —
        # so dequantize_4bit runs entirely on-device, no per-call CPU bounce.
        self._packed = [p.to(device) for p in self._packed]
        for qs in self._qstate:
            qs.absmax = qs.absmax.to(device)
            if getattr(qs, "code", None) is not None:
                qs.code = qs.code.to(device)
        return self

    def __getitem__(self, e: int) -> torch.Tensor:
        """Dequantize expert e to BF16. After .to(device), qw + qs absmax/code
        all live on device, so this runs entirely on-device, no CPU bounce."""
        from bitsandbytes.functional import dequantize_4bit
        e = int(e)
        result = dequantize_4bit(self._packed[e], self._qstate[e]).to(self.dtype)
        # Ensure result is on the same device as the packed tensor (bnb may return CPU)
        target = self._packed[e].device
        return result.to(target) if result.device != target else result


def _nf4_moe_forward(self, hidden_states, top_k_index, top_k_weights):
    """Per-expert dequant forward — replacement for grouped_mm_experts_forward.

    transformers (integrations/moe.py: grouped_mm_experts_forward) calls
    `self.gate_up_proj.transpose(-2, -1)` on the *whole* 3D weight directly
    (no per-expert indexing) so it can hand it to a grouped/batched mm kernel.
    Our _NF4Experts3D store only supports per-expert __getitem__ dequant, not
    arbitrary tensor ops on the full [128, out, in] stack — materializing all
    128 experts in BF16 to satisfy .transpose() would defeat the point of NF4
    storage (~1.4 GiB transient per layer).

    moe.py itself documents the equivalent naive loop as a commented-out
    reference "eager mm implementation" of the same Experts.forward contract —
    this is that loop, reading weights through __getitem__ so only the
    experts actually routed-to in this batch (expert_hit) are dequantized,
    one at a time.
    """
    # _NF4Experts3D.__getitem__ always dequantizes to bf16 (its .dtype). The
    # incoming hidden_states may be a different working dtype (e.g. fp32, if
    # the non-quantized layers weren't loaded in bf16) — same situation
    # bnb.nn.Linear4bit handles internally by casting input to compute dtype
    # for the matmul, then casting the result back. We do that explicitly
    # since this loop bypasses Linear4bit entirely.
    compute_dtype = self.gate_up_proj.dtype
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = int(expert_idx[0])
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx].to(compute_dtype)
        gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        routing_weight = top_k_weights[token_idx, top_k_pos, None].to(compute_dtype)
        current_hidden_states = current_hidden_states * routing_weight
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _swap_experts_to_nf4(model: torch.nn.Module) -> int:
    """Quantize Gemma4TextExperts 3D weight tensors to NF4 in-place.

    Pops gate_up_proj / down_proj from each expert module's _parameters dict
    so model._apply() / model.to() never sees them again, then replaces them
    with _NF4Experts3D plain-attribute stores.  Frees the BF16 tensors
    layer-by-layer to avoid an RSS spike.  Also patches the module's forward
    to the per-expert dequant loop (_nf4_moe_forward) since the stock
    grouped_mm_experts_forward requires the full 3D weight as a real tensor.

    Returns the number of tensors swapped (expected: 2 × num_layers = 60).
    """
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Gemma4TextExperts":
            continue
        swapped_any = False
        for attr in ("gate_up_proj", "down_proj"):
            if attr not in mod._parameters:
                continue
            w_param = mod._parameters.pop(attr)     # unregister from nn.Module
            nf4_store = _NF4Experts3D(w_param.data)
            object.__setattr__(mod, attr, nf4_store) # bypass nn.Module setattr guard
            del w_param
            n += 1
            swapped_any = True
        if swapped_any:
            mod.forward = types.MethodType(_nf4_moe_forward, mod)
        gc.collect()    # free BF16 shards layer-by-layer to limit peak RSS
    if n:
        try:
            import psutil as _p
            rss = _p.Process().memory_info().rss / 1024**3
            logger.info(f"[NF4] Swapped {n} fused 3D expert tensors to NF4; RSS {rss:.1f} GiB")
        except Exception:
            logger.info(f"[NF4] Swapped {n} fused 3D expert tensors to NF4")
    return n


class LoRATrainer:
    """Train LoRA adapters for Gemma-4-26B-A4B (MoE) via QLoRA."""

    _LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention (GQA + sliding window)
        # gate_proj / up_proj / down_proj are EXCLUDED: the MoE expert blocks
        # (layers.*.experts) are BF16 and CPU-pinned for VRAM reasons — adding
        # LoRA adapters there would run all expert computation on CPU.
        # Attention-only LoRA is sufficient for persona shaping.
    ]

    # Gemma-4's tokenizer_config.json ships without a chat_template.
    _GEMMA4_CHAT_TEMPLATE = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}\n"
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|turn>user\n' + message['content'] | trim + '<turn|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|turn>model\n' + message['content'] | trim + '<turn|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
    )

    def __init__(self, config: SystemConfig, base_model_path: str):
        self.config = config
        self.base_model_path = base_model_path

    def _apply_chat_template(self, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<eos>"
        if tokenizer.chat_template is None:
            tokenizer.chat_template = self._GEMMA4_CHAT_TEMPLATE
            logger.info("Set Gemma-4 chat template (not present in tokenizer_config.json)")

    def _load_and_prepare(self):
        """Load the model in 4-bit NF4 and apply QLoRA.

        Tries the Unsloth path first (faster kernels); falls back to the standard
        bitsandbytes + PEFT path if Unsloth is unavailable.

        Returns (model, tokenizer).
        """
        if _UNSLOTH_AVAILABLE:
            return self._load_and_prepare_unsloth()
        return self._load_and_prepare_bnb()

    def _load_model_to_xpu(self):
        """CPU-load, NF4-quantize (attention + experts), then dispatch to XPU.

        Phase 1: Load on CPU with BitsAndBytesConfig.  bnb skips quantization
                 on CPU by default (CUDA-only kernel); we detect and fix this.
        Phase 2: Swap Gemma4TextExperts 3D tensors to _NF4Experts3D, freeing
                 ~42.5 GiB of BF16.  With only ~5B non-expert params left, the
                 total NF4 model is ~13 GiB — well inside 22.7 GiB VRAM.
        Phase 3: Dispatch standard Parameters/buffers to XPU via _apply();
                 move _NF4Experts3D stores explicitly (plain attrs, not Parameters).

        Returns (model, tokenizer, xpu_ok).
        """
        from transformers import BitsAndBytesConfig

        _xpu_ok = (hasattr(torch, "xpu") and torch.xpu.is_available()
                   and torch.xpu.device_count() > 0)
        if _xpu_ok:
            logger.info(
                f"[Load] XPU detected ({torch.xpu.device_count()} device(s), "
                f"{torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GiB)"
            )
        else:
            logger.warning("[Load] XPU not available — model will stay on CPU (slow training)")

        # ── Phase 1: load on CPU, bnb quantizes attention linears ────────────
        logger.info(f"[Load] Phase 1: loading {self.base_model_path} on CPU with NF4 config")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map={"": "cpu"},
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )

        # bnb skips NF4 quantization for non-CUDA targets — detect & fix
        try:
            import psutil as _psutil
            _rss_gb = _psutil.Process().memory_info().rss / 1024**3
        except Exception:
            _psutil = None
            _rss_gb = -1.0

        _sample_mod = next(
            (m for _, m in model.named_modules()
             if getattr(m, 'weight', None) is not None
             and getattr(m.weight, 'ndim', 0) >= 2),
            None,
        )
        _w_dtype = str(_sample_mod.weight.dtype) if _sample_mod else "?"
        _has_qs = (
            _sample_mod is not None
            and getattr(getattr(_sample_mod, 'weight', None), 'quant_state', None) is not None
        )
        logger.info(
            f"[Load] CPU load done — RSS: {_rss_gb:.1f} GiB; "
            f"first linear dtype={_w_dtype}, quant_state={'OK' if _has_qs else 'MISSING'}"
        )

        if not _has_qs or _w_dtype != "torch.uint8":
            logger.info("[Load] Attention weights not yet NF4 — applying manual CPU pre-quantization")
            try:
                from bitsandbytes.functional import quantize_4bit as _q4
                from bitsandbytes.nn import Params4bit as _P4
                _n_q = 0
                for _mn, _mod in model.named_modules():
                    _w = getattr(_mod, 'weight', None)
                    if not isinstance(_w, _P4):
                        continue
                    if getattr(_w, 'bnb_quantized', False):
                        continue
                    if _w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        continue
                    _qt = getattr(_mod, 'bnb_4bit_quant_type', 'nf4')
                    _cs = getattr(_mod, 'bnb_4bit_use_double_quant', True)
                    _wq, _qs = _q4(
                        _w.data.contiguous().to(torch.float16),
                        quant_type=_qt, compress_statistics=_cs,
                        quant_storage=torch.uint8,
                    )
                    _mod.weight = _P4(
                        data=_wq, requires_grad=False, quant_state=_qs,
                        bnb_quantized=True, blocksize=64, quant_type=_qt,
                        compress_statistics=_cs, quant_storage=torch.uint8,
                    )
                    _n_q += 1
                try:
                    _rss2 = _psutil.Process().memory_info().rss / 1024**3
                except Exception:
                    _rss2 = -1.0
                logger.info(f"[Load] NF4 pre-quantization: {_n_q} attention layers; RSS {_rss2:.1f} GiB")
            except Exception as _qe:
                logger.error(f"[Load] NF4 pre-quantization FAILED: {_qe} — attention may stay BF16")

        # Strip vision components (text-only training; they are BF16 and large)
        for _va in ("vision_tower", "vision_model", "image_encoder",
                    "multi_modal_projector", "img_projection", "vision_projection",
                    "visual", "siglip_vision_model"):
            _vmod = getattr(model, _va, None)
            if _vmod is not None:
                setattr(model, _va, None)
                del _vmod
                logger.info(f"[Load] Stripped vision component: {_va}")
        gc.collect()

        # ── Phase 2: quantize fused 3D expert tensors to NF4 ─────────────────
        # bitsandbytes never quantizes these (they are nn.Parameter, not nn.Linear).
        # After this call, ~42.5 GiB BF16 → ~10.6 GiB NF4 as plain attrs.
        logger.info("[Load] Phase 2: quantizing Gemma4TextExperts 3D tensors to NF4 ...")
        _n_experts = _swap_experts_to_nf4(model)
        if _n_experts == 0:
            logger.warning(
                "[Load] No Gemma4TextExperts modules found — expert weights remain BF16. "
                "XPU dispatch may OOM (42.5 GiB BF16 experts + ~5B attention = ~52 GiB)."
            )

        # ── Phase 3: dispatch to XPU ──────────────────────────────────────────
        if _xpu_ok:
            # Quick probe: does bnb keep Params4bit as uint8 on XPU, or expand to BF16?
            _p4_probe = next(
                (m.weight for _, m in model.named_modules()
                 if getattr(getattr(m, 'weight', None), 'bnb_quantized', False)),
                None,
            )
            if _p4_probe is not None:
                _cpu_b = _p4_probe.nbytes
                try:
                    _xpu_probe = _p4_probe.to(torch.device("xpu:0"))
                    torch.xpu.synchronize()
                    _ratio = _xpu_probe.nbytes / max(_cpu_b, 1)
                    _probe_dtype = str(_xpu_probe.dtype)
                    del _xpu_probe
                except Exception as _pe:
                    logger.warning(f"[Load] 4-bit XPU probe failed: {_pe}")
                    _ratio, _probe_dtype = 1.0, str(_p4_probe.dtype)
                torch.xpu.empty_cache()
                logger.info(
                    f"[Load] 4-bit XPU probe: {_p4_probe.dtype} → {_probe_dtype}; "
                    f"ratio {_ratio:.1f}× ({'4-bit kept' if _ratio < 2.5 else 'dequantized to BF16'})"
                )
                if _ratio > 2.5:
                    logger.warning(
                        "[Load] bnb expands Params4bit to BF16 on XPU (no native 4-bit kernel). "
                        "Attention layers will be BF16 (~10 GiB) instead of NF4 (~2 GiB). "
                        "With NF4 experts (~10.6 GiB) + BF16 attention (~10 GiB) ≈ 20.6 GiB — "
                        "may fit but close to the 22.7 GiB limit. Proceeding."
                    )

            _total_vram = 22.5
            try:
                _total_vram = torch.xpu.get_device_properties(0).total_memory / 1024**3
            except Exception:
                pass
            _pre = torch.xpu.memory_allocated(0) / 1024**3
            logger.info(
                f"[Load] Phase 3: XPU dispatch — before: {_pre:.2f}/{_total_vram:.1f} GiB"
            )

            _move_count = [0]
            def _convert_with_periodic_sync(t):
                result = t.to(torch.device("xpu:0"))
                _move_count[0] += 1
                if _move_count[0] % 50 == 0:
                    torch.xpu.synchronize()
                    logger.info(
                        f"  [dispatch] {_move_count[0]} tensors; "
                        f"{torch.xpu.memory_allocated(0) / 1024**3:.2f} GiB on xpu:0"
                    )
                return result

            model._apply(_convert_with_periodic_sync)
            torch.xpu.synchronize()

            # Move NF4 expert stores to XPU (_apply skips them — plain attrs, not Parameters)
            _expert_stores = 0
            for mod in model.modules():
                if type(mod).__name__ == "Gemma4TextExperts":
                    for attr in ("gate_up_proj", "down_proj"):
                        store = getattr(mod, attr, None)
                        if isinstance(store, _NF4Experts3D):
                            store.to(torch.device("xpu:0"))
                            _expert_stores += 1
            torch.xpu.synchronize()

            logger.info(
                f"[Load] Dispatch complete: {_move_count[0]} param tensors + "
                f"{_expert_stores} NF4 expert stores; "
                f"{torch.xpu.memory_allocated(0) / 1024**3:.1f} GiB allocated, "
                f"{torch.xpu.memory_reserved(0) / 1024**3:.1f} GiB reserved"
            )
            # Prevent accelerate.prepare_model() from calling model.to(device).
            model.hf_device_map = {"": "xpu:0"}

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self._apply_chat_template(tokenizer)
        return model, tokenizer, _xpu_ok

    def _load_and_prepare_unsloth(self):
        """Unsloth FastModel path: shared CPU→XPU loader + Unsloth patched PEFT."""
        logger.info(f"[Unsloth] Loading {self.base_model_path}")
        model, tokenizer, _xpu_ok = self._load_model_to_xpu()

        model = self._unwrap_clippable_linear(model)
        if _HAS_KBIT_PREP:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        model = _FastModel.get_peft_model(
            model,
            r=self.config.training.lora_rank,
            target_modules=self._LORA_TARGET_MODULES,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=self.config.training.max_seq_length,
        )
        if _xpu_ok and not getattr(model, "hf_device_map", None):
            model.hf_device_map = {"": "xpu:0"}
        self._unfreeze_moe_router(model)
        model.print_trainable_parameters()
        return model, tokenizer

    def _load_and_prepare_bnb(self):
        """bitsandbytes BitsAndBytesConfig path: shared CPU→XPU loader + standard PEFT."""
        logger.info(f"[bnb] Loading {self.base_model_path}")
        model, tokenizer, _xpu_ok = self._load_model_to_xpu()

        model = self._unwrap_clippable_linear(model)
        if _HAS_KBIT_PREP:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            r=self.config.training.lora_rank,
            lora_alpha=self.config.training.lora_alpha,
            target_modules=self._LORA_TARGET_MODULES,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if _xpu_ok and not getattr(model, "hf_device_map", None):
            model.hf_device_map = {"": "xpu:0"}
        self._unfreeze_moe_router(model)
        model.print_trainable_parameters()
        return model, tokenizer

    def _unwrap_clippable_linear(self, model):
        """Replace Gemma4ClippableLinear wrappers with their inner linear module.

        PEFT only recognises standard torch.nn.Linear (and its bitsandbytes
        Linear4bit subclass). Gemma4 wraps projections in ClippableLinear for
        weight-clipping during full fine-tuning. LoRA freezes all base weights so
        the clipping never fires — unwrapping is numerically identical and lets
        PEFT inject LoRA adapters normally.
        """
        replaced = 0
        for parent in model.modules():
            for child_name, child in list(parent.named_children()):
                if type(child).__name__ == "Gemma4ClippableLinear":
                    if hasattr(child, "linear"):
                        setattr(parent, child_name, child.linear)
                        replaced += 1
        if replaced:
            logger.info(f"Unwrapped {replaced} Gemma4ClippableLinear → inner linear for PEFT")
        return model

    def _unfreeze_moe_router(self, model):
        """Unfreeze MoE router (top-k expert selection layer — not a linear proj).

        Skips quantized (non-float) parameters: bitsandbytes stores 4-bit weights
        as uint8 which cannot have requires_grad=True.
        """
        _FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
        router_count = 0
        for name, param in model.named_parameters():
            if set(name.split(".")[:-1]) & {"router", "gate"}:
                if param.dtype in _FLOAT_DTYPES:
                    param.requires_grad = True
                    router_count += 1
        if router_count:
            logger.info(f"Unfroze {router_count} MoE router parameters for training")

    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        val_dataset: List[Dict[str, Any]],
        checkpoint_dir: str,
    ) -> Tuple[str, Dict[str, float]]:
        """Train a LoRA adapter on the provided dataset.

        Returns:
            Tuple of (adapter_path, metrics_dict) where metrics_dict contains
            train_loss and eval_loss.
        """
        global _XPU_HARD_FAIL
        logger.info(f"Starting LoRA training with {len(train_dataset)} samples")

        if _XPU_HARD_FAIL:
            raise RuntimeError(
                "XPU context was lost in a previous attempt this session "
                "(DEVICE_LOST — xe hang timeout). Restart the container to re-enable XPU."
            )

        is_dpo_mode = self.config.training.enable_dpo
        max_hours = self.config.training.max_training_hours
        model = None

        try:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            model, tokenizer = self._load_and_prepare()

            timing_cb = _TimingCallback(max_hours) if max_hours > 0 else None
            extra_callbacks = [timing_cb] if timing_cb else []

            if is_dpo_mode:
                adapter_path, metrics = self._train_dpo(
                    model, tokenizer, train_dataset, val_dataset, checkpoint_path
                )
            else:
                adapter_path, metrics = self._train_sft(
                    model, tokenizer, train_dataset, val_dataset, checkpoint_path,
                    extra_callbacks=extra_callbacks,
                )

            if timing_cb and timing_cb.too_slow:
                logger.warning(
                    f"Training stopped early: projected {timing_cb.projected_hours:.1f}h "
                    f"exceeded {max_hours}h budget — adapter may be undertrained"
                )

            self._save_metadata(checkpoint_path, metrics, train_dataset,
                                val_dataset, is_dpo_mode)
            return adapter_path, metrics

        except RuntimeError as exc:
            if "DEVICE_LOST" in str(exc).upper():
                _XPU_HARD_FAIL = True
                logger.warning(
                    f"XPU context lost (DEVICE_LOST — xe GPU hang timeout). "
                    f"XPU disabled for this session. Restart container to retry. "
                    f"Error: {exc}"
                )
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            if model is not None:
                try:
                    torch.xpu.synchronize(0)
                except Exception:
                    pass
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            gc.collect()
            gc.collect()
            try:
                torch.xpu.empty_cache()
            except (AttributeError, RuntimeError):
                pass
            try:
                free, total = torch.xpu.mem_get_info(0)
                logger.info(
                    f"XPU memory after cleanup: "
                    f"{free // 1024**3}GiB free / {total // 1024**3}GiB total"
                )
            except Exception:
                pass

    def _save_metadata(self, checkpoint_path, metrics, train_dataset,
                       val_dataset, is_dpo_mode):
        metadata = {
            "train_loss": metrics.get("train_loss", 0.0),
            "eval_loss": metrics.get("eval_loss", 0.0),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": self.config.training.num_epochs,
            "mode": "dpo" if is_dpo_mode else "sft",
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{checkpoint_path}/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Training complete. Loss: {metrics.get('train_loss', 0):.4f} "
            f"(eval: {metrics.get('eval_loss', 0):.4f})"
        )

    def _train_sft(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path,
                   extra_callbacks=None):
        """SFT training via TRL SFTTrainer (Unsloth-patched when available)."""
        logger.info("Using SFT (Supervised Fine-Tuning) mode")

        train_ds = self._prepare_dataset(train_dataset, tokenizer)
        val_ds = self._prepare_dataset(val_dataset, tokenizer)

        args_kwargs = dict(
            output_dir=str(checkpoint_path),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            logging_steps=10,
            logging_dir=f"{checkpoint_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            # bf16 trainer-level autocast needs transformers' XPU detection (requires IPEX).
            # bitsandbytes bnb_4bit_compute_dtype=bfloat16 handles bf16 computation directly.
            bf16=False,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
        if _HAS_SFT_CONFIG:
            training_args = SFTConfig(**args_kwargs, dataset_text_field=None, packing=False)
        else:
            training_args = TrainingArguments(**args_kwargs)

        class _DataCollator:
            """Pads batch, masks prompt tokens from labels, and injects Gemma 4 extras."""
            def __init__(self, tok, max_length):
                self.tokenizer = tok
                self.max_length = max_length

            def __call__(self, features):
                batch = self.tokenizer.pad(
                    features, padding=True, max_length=self.max_length, return_tensors="pt"
                )
                labels = batch["input_ids"].clone()
                for i, f in enumerate(features):
                    p_len = f.get("prompt_len", 0)
                    if p_len > 0:
                        labels[i, :p_len] = -100
                return {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": labels,
                    # Gemma 4 is multimodal: create_causal_mask_mapping requires
                    # mm_token_type_ids (text=0, image=1) when model.training=True.
                    # Text-only batches are all zeros.
                    "mm_token_type_ids": torch.zeros_like(batch["input_ids"]),
                }

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=_DataCollator(tokenizer, self.config.training.max_seq_length),
            processing_class=tokenizer,  # prevents AutoProcessor from loading Gemma4Processor (needs PIL)
            callbacks=extra_callbacks or [],
        )

        try:
            result = trainer.train()
        except Exception:
            del trainer
            gc.collect()
            raise

        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        eval_metrics = trainer.evaluate()
        return adapter_path, {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }

    def _train_dpo(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path):
        """DPO training via TRL DPOTrainer (Unsloth-patched when available)."""
        from trl import DPOTrainer, DPOConfig  # lazy: avoids mergekit → pydantic-core on Py3.14

        logger.info("Using DPO (Direct Preference Optimization) mode")

        train_ds = Dataset.from_list(train_dataset)
        val_ds = Dataset.from_list(val_dataset)
        logger.info(f"DPO datasets: {len(train_ds)} train, {len(val_ds)} val")

        dpo_config = DPOConfig(
            output_dir=str(checkpoint_path),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            logging_steps=10,
            logging_dir=f"{checkpoint_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=False,  # same reasoning as SFT: bnb compute dtype handles bf16
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            beta=self.config.training.dpo_beta,
            max_length=self.config.training.max_seq_length,
            max_prompt_length=self.config.training.max_seq_length // 2,
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )
        logger.info(f"DPOTrainer created with beta={self.config.training.dpo_beta}")

        try:
            result = trainer.train()
        except Exception:
            del trainer
            gc.collect()
            raise

        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        eval_metrics = trainer.evaluate()
        return adapter_path, {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }

    def _prepare_dataset(self, dataset: List[Dict[str, Any]], tokenizer) -> Dataset:
        """Tokenize SFT dataset using Gemma-4 chat template.

        Produces input_ids, attention_mask, and prompt_len (used by _DataCollator
        to mask prompt tokens from the loss so only the completion is trained on).
        """

        def format_example(example):
            prompt = example["prompt"]
            completion = example["completion"]

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.training.max_seq_length // 2,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_tokens["input_ids"])

            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.training.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "prompt_len": prompt_len,
            }

        formatted = [format_example(ex) for ex in dataset]
        return Dataset.from_dict({
            "input_ids": [ex["input_ids"] for ex in formatted],
            "attention_mask": [ex["attention_mask"] for ex in formatted],
            "prompt_len": [ex["prompt_len"] for ex in formatted],
        })

    def merge_adapter(self, adapter_path: str, output_path: str) -> str:
        """Merge LoRA adapter with base model and save state dict."""
        logger.info(f"Merging adapter {adapter_path} with base model")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="cpu",
                torch_dtype="auto",
            )
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            os.makedirs(output_path, exist_ok=True)
            model.config.save_pretrained(output_path)

            state_dict_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(model.state_dict(), state_dict_path)
            logger.info(f"Merged model saved to {state_dict_path}")

            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            tokenizer.save_pretrained(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}", exc_info=True)
            raise
        finally:
            gc.collect()
