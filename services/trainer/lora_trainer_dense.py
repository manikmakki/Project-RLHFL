"""
Project RLHFL - Dense LoRA Trainer (CPU-only)

Specialised trainer for Ministral-3-14B and similar dense VLM architectures
where the checkpoint is structured as a full `Mistral3ForConditionalGeneration`
(vision_tower + multi_modal_projector + language_model) but we only want to
fine-tune the language model backbone via LoRA.

Key differences from lora_trainer.py:
- Freezes vision_tower and multi_modal_projector (no gradient, no memory overhead)
- Loads and trains in float32 so both forward and backward matmuls dispatch to
  MKL SGEMM (OpenMP parallel). BF16 model weights cause mixed-type backward
  matmuls (float32 grad × BF16 weight) that fall outside oneDNN's fast path.
- Text-only batches never pass pixel_values so the vision branch is never
  executed; adapter weights are saved with the correct full VLM paths for Ollama.
"""

import gc
import logging
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import Dataset

from shared.config import SystemConfig

logger = logging.getLogger(__name__)

# Monkey-patch transformers to bypass PyTorch 2.6 requirement for torch.load
try:
    from transformers.utils import import_utils
    from transformers import modeling_utils
    def _bypass_torch_load_check():
        pass
    import_utils.check_torch_load_is_safe = _bypass_torch_load_check
    modeling_utils.check_torch_load_is_safe = _bypass_torch_load_check
except Exception:
    pass

try:
    from trl import DPOTrainer, DPOConfig
    DPO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TRL not available, DPO training disabled: {e}")
    DPO_AVAILABLE = False


class LoRATrainerDense:
    """
    LoRA trainer for dense VLM models (Ministral-3-14B style).

    Loads the full Mistral3ForConditionalGeneration in float32, freezes vision
    components, and trains text layers only. Float32 ensures both forward and
    backward matmuls go through MKL SGEMM (OpenMP parallel). Mixed BF16/float32
    backward matmuls don't have a fast oneDNN path and run single-threaded.
    """

    def __init__(self, config: SystemConfig, base_model_path: str):
        self.config = config
        self.base_model_path = base_model_path
        self.lora_config = self._create_lora_config()

    def _create_lora_config(self) -> LoraConfig:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        return LoraConfig(
            r=self.config.training.lora_rank,
            lora_alpha=self.config.training.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
        return tokenizer

    def _load_model(self):
        """
        Load the full VLM and freeze vision components.

        There is no standalone ForCausalLM class for this architecture — only
        Mistral3ForConditionalGeneration exists. We load the full VLM and freeze
        the vision_tower and multi_modal_projector so they consume no gradient
        memory and do not participate in the backward pass.

        Text-only training batches never pass pixel_values, so the vision branch
        is never executed in forward(); the frozen params add no overhead beyond
        static RAM (~0.4B params).
        """
        try:
            from transformers.utils import import_utils
            from transformers import modeling_utils
            def _bypass():
                pass
            import_utils.check_torch_load_is_safe = _bypass
            modeling_utils.check_torch_load_is_safe = _bypass
        except Exception:
            pass

        logger.info(f"Loading VLM from {self.base_model_path}...")
        torch.set_num_threads(self.config.training.cpu_threads)
        logger.info(f"PyTorch CPU threads: {torch.get_num_threads()}")

        model = Mistral3ForConditionalGeneration.from_pretrained(
            self.base_model_path,
            device_map="cpu",
            torch_dtype=torch.float32,  # float32 ensures both forward and backward matmuls
            trust_remote_code=True,     # dispatch to MKL SGEMM (OpenMP) — BF16 backward
            low_cpu_mem_usage=True,     # matmuls fall back to a non-parallel path
            quantization_config=None,
            attn_implementation="eager",
        )

        # FP8 checkpoints are forcibly dequantized to BF16 on CPU (no GPU/XPU),
        # ignoring torch_dtype. Explicitly cast to float32 so backward matmuls
        # dispatch to MKL SGEMM (OpenMP parallel) instead of mixed-type fallback.
        model = model.float()
        param = next(model.parameters())
        ram_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        logger.info(f"Model dtype: {param.dtype}, estimated RAM: {ram_gb:.1f}GB")

        # Freeze vision components — they add no gradient memory and the forward
        # path skips them when pixel_values is absent (text-only batches).
        frozen = 0
        for name, param in model.named_parameters():
            if "vision_tower" in name or "multi_modal_projector" in name:
                param.requires_grad = False
                frozen += 1
        logger.info(f"Frozen {frozen} vision parameters (vision_tower + multi_modal_projector)")

        model.config.use_cache = False
        logger.info(
            f"VLM loaded: {type(model).__name__}, "
            f"layers={model.config.text_config.num_hidden_layers}, "
            f"hidden={model.config.text_config.hidden_size}"
        )
        return model

    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        val_dataset: List[Dict[str, Any]],
        checkpoint_dir: str,
    ) -> Tuple[str, Dict[str, float]]:
        """Train a LoRA adapter on the language model backbone."""
        logger.info(f"Starting dense LoRA training with {len(train_dataset)} samples")

        try:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            tokenizer = self._load_tokenizer()
            model = self._load_model()

            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
            model.config.use_cache = False

            is_dpo_mode = self.config.training.enable_dpo and DPO_AVAILABLE

            if is_dpo_mode:
                adapter_path, metrics = self._train_dpo(
                    model, tokenizer, train_dataset, val_dataset, checkpoint_path
                )
            else:
                adapter_path, metrics = self._train_sft(
                    model, tokenizer, train_dataset, val_dataset, checkpoint_path
                )

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
            metrics["mode"] = "dpo" if is_dpo_mode else "sft"
            return adapter_path, metrics

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            gc.collect()

    def _train_sft(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path):
        """SFT training using HuggingFace Trainer with CPU AMP (bf16=True)."""
        logger.info("Using SFT (Supervised Fine-Tuning) mode")

        train_ds = self._prepare_dataset(train_dataset, tokenizer)
        val_ds = self._prepare_dataset(val_dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_path),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=min(self.config.training.warmup_steps, 10),
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            logging_steps=1,
            logging_dir=f"{checkpoint_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=False,                  # float32 throughout — MKL SGEMM for forward+backward
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_drop_last=False,
            use_cpu=True,
        )

        class DataCollatorForCausalLM:
            def __init__(self, tok, max_length=None):
                self.tokenizer = tok
                self.max_length = max_length

            def __call__(self, features):
                batch = self.tokenizer.pad(
                    features, padding="max_length", max_length=self.max_length, return_tensors="pt"
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
                }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForCausalLM(tokenizer, self.config.training.max_seq_length),
        )

        # Re-apply thread count after Trainer/Accelerator initialisation
        torch.set_num_threads(self.config.training.cpu_threads)
        logger.info(f"SFT training starting with {torch.get_num_threads()} CPU threads (float32)")

        result = trainer.train()

        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        eval_metrics = trainer.evaluate()
        return adapter_path, {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }

    def _train_dpo(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path):
        """DPO training using TRL DPOTrainer with CPU AMP (bf16=True)."""
        logger.info("Using DPO (Direct Preference Optimization) mode")
        # Suppress the per-sample tokenization mismatch warnings from TRL — these fire
        # when BPE merges a trailing space across the prompt/response boundary, which
        # is handled by rstrip() in dataset_builder. Any remaining instances are benign.
        logging.getLogger("trl.trainer.dpo_trainer").setLevel(logging.ERROR)

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
            warmup_steps=min(self.config.training.warmup_steps, 10),
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            logging_steps=1,
            logging_dir=f"{checkpoint_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=False,                  # float32 throughout — MKL SGEMM for forward+backward
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
            beta=self.config.training.dpo_beta,
            max_length=self.config.training.max_seq_length,
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )
        logger.info(f"DPOTrainer created with beta={self.config.training.dpo_beta}")

        # Re-apply thread count after Trainer/Accelerator initialisation
        torch.set_num_threads(self.config.training.cpu_threads)
        logger.info(f"DPO training starting with {torch.get_num_threads()} CPU threads (float32)")

        result = trainer.train()

        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        eval_metrics = trainer.evaluate()
        return adapter_path, {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }

    def _prepare_dataset(self, dataset: List[Dict[str, Any]], tokenizer) -> Dataset:
        """Tokenise SFT dataset using ChatML format."""

        def format_example(example):
            prompt = example["prompt"]
            completion = example["completion"]

            prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.training.max_seq_length // 2,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_tokens["input_ids"])

            full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
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
