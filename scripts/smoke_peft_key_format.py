"""
Smoke test: what do PEFT adapter_model.safetensors keys actually look like
for our Gemma4 module nesting + LoRA target modules?

Confirmed module nesting (transformers/models/gemma4/modeling_gemma4.py):
    Gemma4ForConditionalGeneration.model        -> Gemma4Model
    Gemma4Model.language_model                  -> Gemma4TextModel
    Gemma4TextModel.layers[i].self_attn.q_proj  -> nn.Linear

No GPU / base model weights needed — this only tests PEFT's key-naming
convention against a dummy module with the same dotted-path structure as
the real model, for the same target_modules our trainer uses.

Run inside the training container:
  docker compose run --rm trainer python /app/scripts/smoke_peft_key_format.py
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors import safe_open


class _Attn(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _DecoderLayer(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.self_attn = _Attn(hidden)


class _TextModel(nn.Module):
    """Mirrors Gemma4TextModel: has .layers (the attribute PEFT target-module
    matching walks through)."""
    def __init__(self, n_layers=2, hidden=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(hidden) for _ in range(n_layers)])


class _Gemma4Model(nn.Module):
    """Mirrors Gemma4Model: has .language_model (the extra indirection the
    GGUF converter regex didn't originally account for)."""
    def __init__(self, n_layers=2, hidden=16):
        super().__init__()
        self.language_model = _TextModel(n_layers, hidden)


class _DummyGemma4ForCausalLM(nn.Module):
    """Mirrors Gemma4ForConditionalGeneration: has .model."""
    def __init__(self, n_layers=2, hidden=16):
        super().__init__()
        self.model = _Gemma4Model(n_layers, hidden)
        # PEFT's get_peft_model wants a model with a forward(); not exercised
        # here since we only call save_pretrained(), but keep it real-ish.
        self.lm_head = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        h = x
        for layer in self.model.language_model.layers:
            h = layer.self_attn.q_proj(h)
        return self.lm_head(h)


dummy = _DummyGemma4ForCausalLM()

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
peft_model = get_peft_model(dummy, lora_config)
peft_model.print_trainable_parameters()

with tempfile.TemporaryDirectory() as tmpdir:
    peft_model.save_pretrained(tmpdir)
    st_path = Path(tmpdir) / "adapter_model.safetensors"
    print(f"\nadapter_model.safetensors exists: {st_path.exists()}")

    with safe_open(str(st_path), framework="pt") as f:
        keys = list(f.keys())

print(f"\n{len(keys)} keys in adapter_model.safetensors:")
for k in sorted(keys):
    print(f"  {k}")

# --- Check against the GGUF converter's regex ---
sys.path.insert(0, "/app/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("convert_lora_to_gguf", "/app/scripts/convert_lora_to_gguf.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

print("\n_hf_to_gguf_name() mapping results:")
n_mapped = 0
for k in sorted(keys):
    mapped = conv._hf_to_gguf_name(k)
    status = "MAPPED" if mapped else "SKIPPED"
    if mapped:
        n_mapped += 1
    print(f"  {status:8s} {k}  ->  {mapped}")

print(f"\n{n_mapped}/{len(keys)} keys mapped.")
if n_mapped == 0:
    print("FAIL: convert_lora_to_gguf.py's regex does not match real PEFT key format")
    sys.exit(1)
else:
    print("PASS: at least some keys map correctly")
    sys.exit(0)
