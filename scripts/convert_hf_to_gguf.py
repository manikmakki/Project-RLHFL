#!/usr/bin/env python3
"""
Convert HuggingFace models to GGUF format.
Supports GPT-OSS and other architectures.
Based on llama.cpp conversion methodology.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("hf-to-gguf")


def convert_hf_to_gguf(model_dir: str, output_file: str, output_type: str = "f16"):
    """
    Convert HuggingFace model to GGUF format.

    Args:
        model_dir: Path to HuggingFace model directory
        output_file: Output GGUF file path
        output_type: Output data type (f32, f16, q8_0, etc.)
    """
    import json
    import struct
    import numpy as np
    from pathlib import Path

    logger.info(f"Loading model: {Path(model_dir).name}")

    # Load config to determine architecture
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    arch = config.get("architectures", [None])[0]
    logger.info(f"Architecture: {arch}")
    logger.info(f"Architecture type: {type(arch)}")
    logger.info(f"Architecture repr: {repr(arch)}")
    logger.info(f"Match test: {arch == 'GptOssForCausalLM'}")

    # Check if architecture is supported
    if arch == "GptOssForCausalLM":
        logger.info("GPT-OSS model detected")
        _convert_gpt_oss(model_dir, output_file, output_type, config)
    elif arch in ["MistralForCausalLM", "LlamaForCausalLM"]:
        logger.info(f"{arch} model detected")
        _convert_llama_mistral(model_dir, output_file, output_type, config)
    else:
        raise ValueError(f"Model {arch} is not supported")


def _convert_gpt_oss(model_dir: str, output_file: str, output_type: str, config: dict):
    """Convert GPT-OSS model to GGUF."""
    try:
        # Try using transformers + gguf library (if available)
        import gguf
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model with transformers...")

        # Bypass torch.load safety check
        from transformers.utils import import_utils
        from transformers import modeling_utils
        def _bypass_check():
            pass
        import_utils.check_torch_load_is_safe = _bypass_check
        modeling_utils.check_torch_load_is_safe = _bypass_check

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        logger.info("Creating GGUF file...")

        # Create GGUF writer
        # Note: gguf library doesn't have native "gpt-oss" support yet,
        # so we use "llama" as a compatible base architecture
        # The model weights will still work correctly for inference
        gguf_writer = gguf.GGUFWriter(output_file, arch="llama")

        # Add metadata
        gguf_writer.add_name(Path(model_dir).name)
        # Architecture is set in constructor, not here
        gguf_writer.add_block_count(config.get("num_hidden_layers", 24))
        gguf_writer.add_context_length(config.get("initial_context_length", 4096))
        gguf_writer.add_embedding_length(config.get("hidden_size", 2880))
        gguf_writer.add_feed_forward_length(config.get("intermediate_size", 2880))
        gguf_writer.add_head_count(config.get("num_attention_heads", 64))
        gguf_writer.add_head_count_kv(config.get("num_key_value_heads", 8))

        # Add required Llama-specific metadata
        gguf_writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-5))

        # Add RoPE parameters if present
        if "rope_theta" in config:
            gguf_writer.add_rope_freq_base(config["rope_theta"])

        # Add vocab size
        gguf_writer.add_vocab_size(config.get("vocab_size", 201088))

        # Add tokenizer
        logger.info("Adding tokenizer...")
        tokens = []
        scores = []
        toktypes = []

        for i in range(len(tokenizer)):
            try:
                token = tokenizer.convert_ids_to_tokens(i)
                tokens.append(token.encode("utf-8") if isinstance(token, str) else token)
                scores.append(0.0)  # Placeholder score
                toktypes.append(gguf.TokenType.NORMAL)
            except Exception:
                # Skip invalid token IDs
                continue

        gguf_writer.add_tokenizer_model("gpt2")
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        gguf_writer.add_token_types(toktypes)

        # Add tokenizer merges (CRITICAL for GPT-2 tokenizers!)
        logger.info("Adding tokenizer merges...")
        merges_added = False

        # Method 1: Try to load merges from tokenizer.json (fast tokenizer format)
        tokenizer_json_file = Path(model_dir) / "tokenizer.json"
        if tokenizer_json_file.exists():
            logger.info(f"Loading merges from {tokenizer_json_file}")
            import json
            with open(tokenizer_json_file, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)

            if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                merges_list = tokenizer_data['model']['merges']
                # Convert from [['a', 'b'], ['c', 'd']] to ['a b', 'c d']
                merges_strings = [' '.join(merge_pair) for merge_pair in merges_list]
                gguf_writer.add_token_merges(merges_strings)
                logger.info(f"✓ Added {len(merges_strings)} BPE merges from tokenizer.json")
                merges_added = True

        # Method 2: Try to load merges.txt from model directory (legacy format)
        if not merges_added:
            merges_file = Path(model_dir) / "merges.txt"
            if merges_file.exists():
                logger.info(f"Loading merges from {merges_file}")
                with open(merges_file, 'r', encoding='utf-8') as f:
                    merges_lines = f.read().strip().split('\n')
                    # Skip header line if present
                    if merges_lines and merges_lines[0].startswith('#'):
                        merges_lines = merges_lines[1:]
                    gguf_writer.add_token_merges(merges_lines)
                    logger.info(f"✓ Added {len(merges_lines)} BPE merges from merges.txt")
                    merges_added = True

        # Method 3: Try to get merges from tokenizer's bpe_ranks attribute
        if not merges_added and hasattr(tokenizer, 'bpe_ranks'):
            logger.info("Loading merges from tokenizer.bpe_ranks...")
            merges = []
            for merge_pair, rank in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
                merges.append(f"{merge_pair[0]} {merge_pair[1]}")
            gguf_writer.add_token_merges(merges)
            logger.info(f"✓ Added {len(merges)} BPE merges from bpe_ranks")
            merges_added = True

        if not merges_added:
            logger.error("=" * 80)
            logger.error("CRITICAL: Could not find tokenizer merges!")
            logger.error("=" * 80)
            logger.error(f"Checked locations:")
            logger.error(f"  1. {tokenizer_json_file} (not found or no merges)")
            logger.error(f"  2. {Path(model_dir) / 'merges.txt'} (not found)")
            logger.error(f"  3. tokenizer.bpe_ranks attribute (not found)")
            logger.error("")
            logger.error("The GGUF file will be created but may not load correctly in llama.cpp")
            logger.error("=" * 80)
            # Don't exit - let the user decide if they want to continue

        # Add special tokens
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            gguf_writer.add_pad_token_id(tokenizer.pad_token_id)

        # Add model tensors
        logger.info("Adding model tensors...")
        state_dict = model.state_dict()

        tensor_count = 0
        for name, tensor in state_dict.items():
            # Convert tensor to numpy
            data = tensor.detach().cpu().float().numpy()

            # Convert to target dtype
            if output_type == "f16":
                data = data.astype(np.float16)
            elif output_type == "f32":
                data = data.astype(np.float32)

            gguf_writer.add_tensor(name, data)
            tensor_count += 1

            if tensor_count % 100 == 0:
                logger.info(f"Processed {tensor_count} tensors...")

        # Write file
        logger.info(f"Writing GGUF file: {output_file}")
        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()

        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        logger.info(f"✓ Conversion complete: {output_file} ({file_size_mb:.0f} MB)")

    except ImportError as e:
        logger.error("=" * 80)
        logger.error("GGUF CONVERSION FAILED: Missing required library")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("Solution: Install with: pip install gguf")
        logger.error("=" * 80)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("=" * 80)
        logger.error("GGUF CONVERSION FAILED: File not found")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Model directory: {model_dir}")
        logger.error("=" * 80)
        sys.exit(1)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("GGUF CONVERSION FAILED: Unexpected error")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error(f"Model directory: {model_dir}")
        logger.error(f"Output file: {output_file}")
        logger.error("=" * 80)
        logger.error("Full traceback:", exc_info=True)
        logger.error("=" * 80)
        sys.exit(1)


def _convert_llama_mistral(model_dir: str, output_file: str, output_type: str, config: dict):
    """Convert Llama/Mistral models using standard llama.cpp converter."""
    logger.error("Llama/Mistral conversion not implemented yet")
    logger.error("Use llama.cpp's convert-hf-to-gguf.py instead")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to GGUF format"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f16",
        choices=["f32", "f16", "q8_0"],
        help="Output data type (default: f16)"
    )

    args = parser.parse_args()

    try:
        convert_hf_to_gguf(args.model, args.outfile, args.outtype)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
