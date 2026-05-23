#!/usr/bin/env python3
"""
Convert a HuggingFace PEFT LoRA adapter (safetensors) to GGUF LoRA format for llama.cpp.

Usage:
    python convert_lora_to_gguf.py \\
        --adapter-dir /checkpoints/checkpoint_20241201 \\
        --outfile /checkpoints/checkpoint_20241201/adapter.gguf

Requirements:
    pip install gguf>=0.10.0 safetensors torch

Architecture support:
    - Gemma / Gemma-2 / Gemma-4 (attention + dense FFN + MoE expert layers)
"""

import sys
import json
import re
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor name mapping: HuggingFace PEFT → llama.cpp GGUF
# ---------------------------------------------------------------------------

# Matches: base_model.model.model.layers.{i}.{module}.lora_{A|B}.weight
_PEFT_PATTERN = re.compile(
    r"^base_model\.model\."       # PEFT prefix
    r"model\.layers\.(\d+)\."    # layer index
    r"(.+?)\."                   # module path (e.g. self_attn.q_proj)
    r"lora_(A|B)\.weight$"       # LoRA matrix direction
)

# HuggingFace module name → llama.cpp tensor base name
_ATTN_AND_DENSE_MAP = {
    "self_attn.q_proj":   "attn_q",
    "self_attn.k_proj":   "attn_k",
    "self_attn.v_proj":   "attn_v",
    "self_attn.o_proj":   "attn_output",
    "mlp.gate_proj":      "ffn_gate",
    "mlp.up_proj":        "ffn_up",
    "mlp.down_proj":      "ffn_down",
}

# MoE expert: mlp.experts.{e}.{gate|up|down}_proj
_MOE_EXPERT_PATTERN = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
_MOE_PROJ_MAP = {
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
}


def _hf_to_gguf_name(hf_key: str) -> Optional[str]:
    m = _PEFT_PATTERN.match(hf_key)
    if not m:
        return None
    layer_idx = m.group(1)
    module = m.group(2)
    ab = m.group(3)  # "A" → lora_a, "B" → lora_b

    if module in _ATTN_AND_DENSE_MAP:
        gguf_base = f"blk.{layer_idx}.{_ATTN_AND_DENSE_MAP[module]}"
    else:
        me = _MOE_EXPERT_PATTERN.match(module)
        if me:
            exp_idx = me.group(1)
            proj = _MOE_PROJ_MAP[me.group(2)]
            gguf_base = f"blk.{layer_idx}.{proj}_exp{exp_idx}"
        else:
            return None

    suffix = "lora_a" if ab == "A" else "lora_b"
    return f"{gguf_base}.{suffix}"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(adapter_dir: Path, outfile: Path) -> bool:
    try:
        from gguf import GGUFWriter
    except ImportError:
        logger.error("gguf package not found: pip install gguf>=0.10.0")
        return False

    try:
        from safetensors.torch import load_file as st_load
    except ImportError:
        logger.error("safetensors not found: pip install safetensors")
        return False

    # --- adapter_config.json ---
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        logger.error(f"adapter_config.json not found in {adapter_dir}")
        return False
    with open(config_file) as f:
        adapter_cfg = json.load(f)

    rank = int(adapter_cfg.get("r", adapter_cfg.get("lora_r", 8)))
    alpha = float(adapter_cfg.get("lora_alpha", rank))

    # --- adapter_model.safetensors ---
    st_file = adapter_dir / "adapter_model.safetensors"
    if not st_file.exists():
        logger.error(f"adapter_model.safetensors not found in {adapter_dir}")
        return False

    raw_tensors = st_load(str(st_file), device="cpu")

    # --- Map names and convert to F32 numpy ---
    mapped: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for hf_key, tensor in raw_tensors.items():
        gguf_name = _hf_to_gguf_name(hf_key)
        if gguf_name:
            mapped[gguf_name] = tensor.float().numpy()
        else:
            skipped.append(hf_key)

    if skipped:
        logger.debug(f"Skipped {len(skipped)} unmapped tensors (first 5): {skipped[:5]}")

    if not mapped:
        logger.error("No tensors mapped — verify adapter_model.safetensors key format")
        return False

    logger.info(f"Mapped {len(mapped)} tensors (rank={rank}, alpha={alpha})")

    # --- Write GGUF ---
    outfile.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(outfile), "gemma4")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_uint32("adapter.lora.r", rank)
    writer.add_float32("adapter.lora.alpha", alpha)

    for name, arr in mapped.items():
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / (1024 * 1024)
    logger.info(f"Written: {outfile} ({size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA adapter to GGUF for llama.cpp")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing adapter_model.safetensors and adapter_config.json")
    parser.add_argument("--outfile", help="Output GGUF path (default: adapter_dir/adapter.gguf)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    adapter_dir = Path(args.adapter_dir).resolve()
    outfile = Path(args.outfile).resolve() if args.outfile else (adapter_dir / "adapter.gguf")

    success = convert(adapter_dir, outfile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
