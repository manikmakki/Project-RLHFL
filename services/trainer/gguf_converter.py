"""
GGUF Conversion Pipeline

Converts HuggingFace PEFT LoRA adapters to quantized GGUF format
for use with llama.cpp inference.

Pipeline: HF PEFT LoRA -> merged HF model -> FP16 GGUF -> Q4_K_M GGUF
"""

import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CONVERT_SCRIPT_PATH = "/app/scripts/convert_hf_to_gguf.py"
QUANTIZE_BINARY_PATH = "/app/bin/llama-quantize"


class GGUFConversionError(Exception):
    """Raised when GGUF conversion fails at any stage."""
    pass


class GGUFConverter:
    """Converts HuggingFace models to quantized GGUF format."""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def convert_adapter_to_gguf(
        self,
        adapter_path: str,
        merged_model_dir: str,
        output_gguf_path: str,
        quantization_type: str = "Q4_K_M",
        lora_trainer=None,
    ) -> str:
        """
        Full pipeline: adapter -> merged HF model -> FP16 GGUF -> quantized GGUF.

        Args:
            adapter_path: Path to the PEFT LoRA adapter directory.
            merged_model_dir: Temporary directory for the merged HF model.
            output_gguf_path: Final output path for the quantized GGUF.
            quantization_type: GGUF quantization type (default Q4_K_M).
            lora_trainer: LoRATrainer instance (has merge_adapter method).

        Returns:
            Path to the final quantized GGUF file.

        Raises:
            GGUFConversionError: If any stage fails.
        """
        fp16_gguf_path = output_gguf_path.replace(".gguf", "-fp16.gguf")

        try:
            # Stage 1: Merge adapter with base model
            logger.info(f"Stage 1/3: Merging adapter {adapter_path} with base model")
            self._merge_adapter(adapter_path, merged_model_dir, lora_trainer)

            # Stage 2: Convert merged HF model to FP16 GGUF
            logger.info("Stage 2/3: Converting merged model to FP16 GGUF")
            self._convert_hf_to_gguf(merged_model_dir, fp16_gguf_path)

            # Stage 3: Quantize FP16 GGUF to target quantization
            logger.info(f"Stage 3/3: Quantizing to {quantization_type}")
            self._quantize_gguf(fp16_gguf_path, output_gguf_path, quantization_type)

            logger.info(f"GGUF conversion complete: {output_gguf_path}")
            return output_gguf_path

        except Exception as e:
            logger.error(f"GGUF conversion pipeline failed: {e}", exc_info=True)
            self._cleanup_file(output_gguf_path)
            raise GGUFConversionError(f"Conversion failed: {e}") from e

        finally:
            # Always clean up intermediate files (FP16 GGUF ~14GB, merged dir ~14GB)
            self._cleanup_file(fp16_gguf_path)
            self._cleanup_dir(merged_model_dir)

    def _merge_adapter(self, adapter_path: str, output_dir: str, lora_trainer) -> None:
        """Stage 1: Merge LoRA adapter into base model."""
        if lora_trainer is None:
            raise GGUFConversionError("LoRATrainer instance required for merging")
        try:
            lora_trainer.merge_adapter(adapter_path, output_dir)
        except Exception as e:
            raise GGUFConversionError(f"Adapter merge failed: {e}") from e

    def _convert_hf_to_gguf(self, model_dir: str, output_path: str) -> None:
        """Stage 2: Convert HF model to FP16 GGUF using convert script."""
        if not Path(CONVERT_SCRIPT_PATH).exists():
            raise GGUFConversionError(
                f"Conversion script not found: {CONVERT_SCRIPT_PATH}"
            )

        cmd = [
            "python3", CONVERT_SCRIPT_PATH,
            model_dir,
            "--outfile", output_path,
            "--outtype", "f16",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            raise GGUFConversionError(
                f"HF-to-GGUF conversion failed (rc={result.returncode}): "
                f"{result.stderr[-1000:]}"
            )
        logger.info(f"FP16 GGUF created: {output_path}")

    def _quantize_gguf(
        self, input_path: str, output_path: str, quant_type: str
    ) -> None:
        """Stage 3: Quantize GGUF to target type using llama-quantize."""
        if not Path(QUANTIZE_BINARY_PATH).exists():
            raise GGUFConversionError(
                f"Quantize binary not found: {QUANTIZE_BINARY_PATH}"
            )

        cmd = [QUANTIZE_BINARY_PATH, input_path, output_path, quant_type]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            raise GGUFConversionError(
                f"Quantization failed (rc={result.returncode}): "
                f"{result.stderr[-1000:]}"
            )

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Quantized GGUF created: {output_path} ({size_mb:.0f} MB)")

    @staticmethod
    def _cleanup_file(path: str) -> None:
        """Safely remove a file if it exists."""
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
                logger.debug(f"Cleaned up: {path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {path}: {e}")

    @staticmethod
    def _cleanup_dir(path: str) -> None:
        """Safely remove a directory tree if it exists."""
        try:
            p = Path(path)
            if p.exists():
                shutil.rmtree(p)
                logger.debug(f"Cleaned up directory: {path}")
        except Exception as e:
            logger.warning(f"Failed to clean up directory {path}: {e}")
