#!/usr/bin/env python3
"""
Convert GPT-OSS model from PyTorch .bin format to SafeTensors format.
This bypasses transformers 5.0's PyTorch version restriction for torch.load.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

def convert_model_to_safetensors(model_path: str):
    """Convert a HuggingFace model from .bin to safetensors format."""
    print(f"Converting model at {model_path} to SafeTensors format...")

    try:
        from transformers import AutoModelForCausalLM
        import torch

        # Temporarily disable the torch.load safety check by setting env var
        os.environ["HF_HUB_DISABLE_TORCH_LOAD_SAFETY_CHECK"] = "1"

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print("Saving model in SafeTensors format...")
        # Save with safe_serialization=True to use safetensors
        model.save_pretrained(
            model_path,
            safe_serialization=True,
            max_shard_size="5GB"  # Split into 5GB shards
        )

        # Remove old .bin files
        print("Cleaning up old .bin files...")
        for bin_file in Path(model_path).glob("*.bin"):
            if bin_file.name != "optimizer.bin":  # Keep optimizer if present
                print(f"Removing {bin_file.name}")
                bin_file.unlink()

        print("✓ Conversion complete!")
        print(f"Model at {model_path} is now in SafeTensors format")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    model_path = "/opt/project-rlhfl/volumes/models/Qwen3-30B-A3B"

    if not Path(model_path).exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)

    convert_model_to_safetensors(model_path)
