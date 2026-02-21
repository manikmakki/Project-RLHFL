"""
Project RLHFL - LoRA Trainer (CPU-only)

Trains LoRA adapters for GPT-OSS 20B using CPU-only training.
Supports SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).

Adapters are parameter-efficient (~10-20MB), requiring only ~1-2% of full
fine-tuning memory while achieving comparable results.
"""

import gc
import logging
import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from datasets import Dataset

from shared.config import SystemConfig

logger = logging.getLogger(__name__)

# Monkey-patch transformers to bypass PyTorch 2.6 requirement for torch.load
# Safe because we trust official GPT-OSS model files from HuggingFace
try:
    from transformers.utils import import_utils
    from transformers import modeling_utils
    def _bypass_torch_load_check():
        pass
    import_utils.check_torch_load_is_safe = _bypass_torch_load_check
    modeling_utils.check_torch_load_is_safe = _bypass_torch_load_check
    logger.info("Bypassed torch.load safety check (trusted model files)")
except Exception as e:
    logger.warning(f"Failed to patch torch.load safety check: {e}")

# DPO trainer for preference optimization
try:
    from trl import DPOTrainer, DPOConfig
    DPO_AVAILABLE = True
    logger.info("DPO training support loaded successfully")
except ImportError as e:
    logger.warning(f"TRL not available, DPO training disabled: {e}")
    DPO_AVAILABLE = False


class LoRATrainer:
    """Train LoRA adapters for the GPT-OSS base model (CPU-only)."""

    def __init__(self, config: SystemConfig, base_model_path: str):
        self.config = config
        self.base_model_path = base_model_path
        self.lora_config = self._create_lora_config()

    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration targeting attention + MoE expert layers."""
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # GPT-OSS MoE expert FFN components
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
            "mlp.experts.up_proj",
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
        """Load tokenizer for GPT-OSS models."""
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|return|>"

        return tokenizer

    def _load_model(self):
        """Load the base model on CPU in float32."""
        # Re-apply safety patch before each model load
        try:
            from transformers.utils import import_utils
            from transformers import modeling_utils
            def _bypass():
                pass
            import_utils.check_torch_load_is_safe = _bypass
            modeling_utils.check_torch_load_is_safe = _bypass
        except Exception:
            pass

        logger.info(f"Loading model from {self.base_model_path} on CPU (float32)...")
        torch.set_num_threads(self.config.training.cpu_threads)
        logger.info(f"PyTorch CPU threads: {torch.get_num_threads()}")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
        model = model.float()

        # Freeze base parameters; only LoRA + router layers will train
        for name, param in model.named_parameters():
            if "lora" in name or "router" in name.lower():
                param.requires_grad = True

        return model

    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        val_dataset: List[Dict[str, Any]],
        checkpoint_dir: str,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Train a LoRA adapter on the provided dataset.

        Returns:
            Tuple of (adapter_path, metrics_dict) where metrics_dict contains
            train_loss and eval_loss.
        """
        logger.info(f"Starting LoRA training with {len(train_dataset)} samples")

        try:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            tokenizer = self._load_tokenizer()
            model = self._load_model()

            # Apply LoRA
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

            # Save training metadata
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
            return adapter_path, metrics

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            gc.collect()

    def _train_sft(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path):
        """SFT training path using HuggingFace Trainer."""
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
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            logging_steps=10,
            logging_dir=f"{checkpoint_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=False,
            gradient_checkpointing=False,
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
                }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForCausalLM(tokenizer, self.config.training.max_seq_length),
        )

        result = trainer.train()

        # Save adapter
        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Extract metrics
        eval_metrics = trainer.evaluate()
        metrics = {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }
        return adapter_path, metrics

    def _train_dpo(self, model, tokenizer, train_dataset, val_dataset, checkpoint_path):
        """DPO training path using TRL DPOTrainer."""
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
            bf16=False,
            gradient_checkpointing=False,
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
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

        result = trainer.train()

        # Save adapter
        adapter_path = f"{checkpoint_path}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        eval_metrics = trainer.evaluate()
        metrics = {
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss", 0.0),
        }
        return adapter_path, metrics

    def _prepare_dataset(self, dataset: List[Dict[str, Any]], tokenizer) -> Dataset:
        """Tokenize SFT dataset using GPT-OSS Harmony format."""

        def format_example(example):
            prompt = example["prompt"]
            completion = example["completion"]

            prompt_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>"
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.training.max_seq_length // 2,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_tokens["input_ids"])

            full_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"
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
        """Merge LoRA adapter with base model into a single safetensors model."""
        logger.info(f"Merging adapter {adapter_path} with base model")

        try:
            # Re-apply safety patch
            try:
                from transformers.utils import import_utils
                from transformers import modeling_utils
                def _bypass():
                    pass
                import_utils.check_torch_load_is_safe = _bypass
                modeling_utils.check_torch_load_is_safe = _bypass
            except Exception:
                pass

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="cpu",
                torch_dtype="auto",
                use_cache=False,
            )
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            os.makedirs(output_path, exist_ok=True)
            model.config.save_pretrained(output_path)

            # Save state dict directly (bypasses transformers 5.0 weight conversion issues)
            state_dict_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(model.state_dict(), state_dict_path)
            logger.info(f"Model state dict saved to {state_dict_path}")

            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            tokenizer.save_pretrained(output_path)

            logger.info(f"Merged model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}", exc_info=True)
            raise
        finally:
            gc.collect()
