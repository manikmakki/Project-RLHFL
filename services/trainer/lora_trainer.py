"""
Project RLHFL - LoRA Trainer

This module handles the actual training of LoRA (Low-Rank Adaptation) adapters
for the GPT-OSS base language model. It uses QLoRA (Quantized LoRA) with 4-bit
quantization to enable training on GPUs with 14GB+ VRAM.

The trainer:
- Loads the GPT-OSS base model with 4-bit quantization
- Applies LoRA configuration to attention and MLP projection layers
- Trains on weighted datasets with supervision using Harmony format
- Saves lightweight adapter weights (~10-20MB)
- Supports merging adapters back into the base model

LoRA adapters are parameter-efficient, requiring only ~1-2% of full
fine-tuning memory while achieving comparable results.
"""

import logging
import os
import shutil
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
import json

# Set PyTorch CUDA memory management to avoid fragmentation
# expandable_segments reduces fragmentation when loading large models
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from shared.config import SystemConfig

# Initialize logger BEFORE trying to import DPO components
logger = logging.getLogger(__name__)

# DPO trainer for preference optimization
# Note: TRL 0.7.4 is incompatible with transformers 5.0.0
# Requires TRL >= 0.12.0 for transformers 5.0+ support
try:
    from trl import DPOTrainer, DPOConfig
    DPO_AVAILABLE = True
    logger.info("DPO training support loaded successfully")
except ImportError as e:
    logger.warning(f"TRL library not available or incompatible, DPO training will not be available: {e}")
    DPO_AVAILABLE = False

# Monkey patch bitsandbytes to ignore _is_hf_initialized parameter
# This fixes compatibility issues between transformers 5.0, accelerate, and bitsandbytes
try:
    import bitsandbytes as bnb
    from functools import wraps

    # Patch Int8Params
    if hasattr(bnb.nn, 'Int8Params'):
        _original_int8_new = bnb.nn.Int8Params.__new__

        @wraps(_original_int8_new)
        def _patched_int8_new(cls, *args, **kwargs):
            # Remove the problematic parameter
            kwargs.pop('_is_hf_initialized', None)
            return _original_int8_new(cls, *args, **kwargs)

        bnb.nn.Int8Params.__new__ = staticmethod(_patched_int8_new)
        logger.info("Applied Int8Params compatibility patch")

    # Patch Params4bit if it exists
    if hasattr(bnb.nn, 'Params4bit'):
        _original_4bit_new = bnb.nn.Params4bit.__new__

        @wraps(_original_4bit_new)
        def _patched_4bit_new(cls, *args, **kwargs):
            # Remove the problematic parameter
            kwargs.pop('_is_hf_initialized', None)
            return _original_4bit_new(cls, *args, **kwargs)

        bnb.nn.Params4bit.__new__ = staticmethod(_patched_4bit_new)
        logger.info("Applied Params4bit compatibility patch")
except Exception as e:
    logger.warning(f"Failed to apply bitsandbytes patch: {e}")


def log_gpu_memory(stage: str):
    """Log GPU memory usage at key training stages."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"[Memory] {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        logger.warning(f"[Memory] {stage}: CUDA not available")


class LoRATrainer:
    """Train LoRA adapters for the base model."""
    
    def __init__(self, config: SystemConfig, base_model_path: str):
        self.config = config
        self.base_model_path = base_model_path

        # Create default LoRA config (trains all layers)
        self.lora_config = self._create_lora_config()

    def _create_lora_config(self, layers_to_train: Optional[List[int]] = None) -> LoraConfig:
        """Create LoRA configuration, optionally restricted to specific layers.

        Args:
            layers_to_train: Optional list of layer indices to train. None means all layers.

        Returns:
            LoraConfig object with optional layer restriction.
        """
        config_params = {
            "r": self.config.training.lora_rank,
            "lora_alpha": self.config.training.lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

        # Add layers_to_transform if specified (for sequential training)
        if layers_to_train is not None:
            config_params["layers_to_transform"] = layers_to_train
            logger.info(f"LoRA will ONLY train layers: {layers_to_train}")
        else:
            logger.info("LoRA will train ALL layers (standard mode)")

        return LoraConfig(**config_params)
    
    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        val_dataset: List[Dict[str, Any]],
        checkpoint_dir: str
    ) -> str:
        """
        Train a LoRA adapter on the provided dataset.
        
        Returns:
            str: Path to the trained adapter
        """
        logger.info(f"Starting LoRA training with {len(train_dataset)} samples")

        try:
            import shutil
            from pathlib import Path

            # Create checkpoint directory
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Load tokenizer with fallback logic
            logger.info(f"Loading tokenizer from {self.base_model_path}")
            tokenizer = None
            tokenizer_errors = []
            tokenizer_json_backup = None

            # Try 1: Bypass corrupted tokenizer.json by temporarily renaming it
            try:

                tokenizer_json_path = Path(self.base_model_path) / "tokenizer.json"
                if tokenizer_json_path.exists():
                    tokenizer_json_backup = Path(self.base_model_path) / "tokenizer.json.backup"
                    logger.warning(f"Temporarily moving potentially corrupted tokenizer.json to {tokenizer_json_backup}")
                    shutil.move(str(tokenizer_json_path), str(tokenizer_json_backup))

                # Load without tokenizer.json (forces legacy tokenizer)
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained(
                    self.base_model_path,
                    use_fast=False
                )
                logger.info("Tokenizer loaded successfully (GPT2Tokenizer without tokenizer.json)")

            except Exception as e1:
                tokenizer_errors.append(f"GPT2Tokenizer without tokenizer.json: {e1}")

                # Try 2: Restore tokenizer.json and try slow tokenizer
                if tokenizer_json_backup and tokenizer_json_backup.exists():
                    try:
                        shutil.move(str(tokenizer_json_backup), str(tokenizer_json_path))
                        logger.info("Restored tokenizer.json")
                    except:
                        pass

                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.base_model_path,
                        use_fast=False
                    )
                    logger.info("Tokenizer loaded successfully (slow tokenizer)")
                except Exception as e2:
                    tokenizer_errors.append(f"Slow tokenizer from {self.base_model_path}: {e2}")

                    # Try 3: Use GPT-2 tokenizer directly as generic fallback
                    try:
                        logger.warning("Using generic GPT2Tokenizer as fallback")
                        from transformers import GPT2Tokenizer
                        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                        logger.info("Tokenizer loaded (generic GPT-2)")
                    except Exception as e3:
                        tokenizer_errors.append(f"Generic GPT-2 fallback: {e3}")

            # Restore tokenizer.json if we backed it up and haven't already
            if tokenizer_json_backup and tokenizer_json_backup.exists():
                try:
                    tokenizer_json_path = Path(self.base_model_path) / "tokenizer.json"
                    if not tokenizer_json_path.exists():
                        shutil.move(str(tokenizer_json_backup), str(tokenizer_json_path))
                        logger.info("Restored tokenizer.json after successful load")
                except Exception as e:
                    logger.warning(f"Failed to restore tokenizer.json: {e}")

            if tokenizer is None:
                error_msg = "Failed to load tokenizer after all attempts:\n" + "\n".join(tokenizer_errors)
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Add GPT-OSS Harmony special tokens if not present
            harmony_tokens = [
                "<|startoftext|>", "<|endoftext|>", "<|return|>", "<|constrain|>",
                "<|channel|>", "<|start|>", "<|end|>", "<|message|>", "<|call|>",
                "<|endofprompt|>"
            ]

            # Check if special tokens are missing and add them
            existing_tokens = set(tokenizer.get_vocab().keys())
            missing_tokens = [t for t in harmony_tokens if t not in existing_tokens]

            if missing_tokens:
                logger.warning(f"Adding {len(missing_tokens)} missing Harmony tokens to tokenizer")
                tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})

            # Ensure tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"

            # Ensure EOS token is set (should be <|return|> for GPT-OSS Harmony)
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<|return|>"

            logger.info(f"Tokenizer configured: vocab_size={len(tokenizer)}, pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")

            # Check if CPU training mode is enabled
            if self.config.training.enable_cpu_training:
                # CPU TRAINING MODE: Train on CPU with full precision (plenty of RAM available)
                logger.info("=" * 80)
                logger.info("CPU TRAINING MODE ENABLED")
                logger.info(f"Using {self.config.training.cpu_threads} CPU threads (leaving rest for inference)")
                logger.info("Training will be slower but allows GPU for inference")
                logger.info("=" * 80)

                # Set CPU thread count to avoid overwhelming system
                torch.set_num_threads(self.config.training.cpu_threads)
                logger.info(f"PyTorch CPU threads set to: {torch.get_num_threads()}")

                # Load model on CPU without quantization (256GB RAM is plenty)
                logger.info(f"Loading base model from {self.base_model_path} on CPU (no quantization)")
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    device_map="cpu",  # Force CPU placement
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_cache=False,
                    torch_dtype=torch.float32  # Full precision on CPU
                )
                logger.info(f"Model loaded on CPU - RAM usage will be ~40-50GB")

                # For CPU training, we don't need prepare_model_for_kbit_training
                # Just freeze non-LoRA parameters
                for param in model.parameters():
                    param.requires_grad = False
                logger.info("Model parameters frozen for LoRA training")

            else:
                # GPU TRAINING MODE: 8-bit quantization with CPU offload
                logger.info(f"Loading base model from {self.base_model_path} with 8-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_cache=False,
                    torch_dtype=torch.bfloat16
                )
                log_gpu_memory("After model load")

                # Prepare model for k-bit training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

            # Apply LoRA
            logger.info("Applying LoRA configuration")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
            log_gpu_memory("After LoRA application")

            # Ensure use_cache is disabled for training
            model.config.use_cache = False
            
            # Check training mode: DPO or SFT
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
            """Format a single example for training using GPT-OSS Harmony format."""
            prompt = example['prompt']
            completion = example['completion']

            # Use Harmony format for GPT-OSS training
            # Tokenize prompt alone to get prompt length
            prompt_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>"
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.training.max_seq_length // 2,
                padding=False,
                return_tensors=None
            )
            prompt_len = len(prompt_tokens["input_ids"])

            # Full example (prompt + completion with harmony format)
            full_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.training.max_seq_length,
                padding="max_length",
                return_tensors=None
            )

            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "prompt_len": prompt_len
            }

        # Create HuggingFace dataset
        formatted_data = [format_example(ex) for ex in dataset]

        ds = Dataset.from_dict({
            "input_ids": [ex["input_ids"] for ex in formatted_data],
            "attention_mask": [ex["attention_mask"] for ex in formatted_data],
            "prompt_len": [ex["prompt_len"] for ex in formatted_data]
        })
        
        return ds
    
    def merge_adapter(
        self,
        adapter_path: str,
        output_path: str
    ) -> str:
        """
        Merge LoRA adapter with base model.

        This creates a single merged model that can be converted to GGUF.
        GPU memory is explicitly freed after saving so the API container
        has VRAM available to load the resulting GGUF.
        """
        logger.info(f"Merging adapter {adapter_path} with base model")

        model = None
        tokenizer = None
        try:
            # Load base model on CPU for merge (avoids meta-tensor errors
            # that occur with device_map="auto" when GPU memory is tight)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="cpu",
                torch_dtype="auto",
                use_cache=False
            )

            # Load adapter
            model = PeftModel.from_pretrained(model, adapter_path)

            # Merge
            model = model.merge_and_unload()

            # Save merged model
            os.makedirs(output_path, exist_ok=True)
            model.save_pretrained(output_path)

            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            tokenizer.save_pretrained(output_path)

            logger.info(f"Merged model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}", exc_info=True)
            raise
        finally:
            # Explicitly free GPU memory so the API container can load the new GGUF
            self._unload_models(model, tokenizer)

    @staticmethod
    def _unload_models(*objects):
        """Force-free GPU VRAM by deleting model objects and clearing CUDA cache."""
        import gc
        for obj in objects:
            if obj is not None:
                del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Trainer GPU memory freed (torch.cuda.empty_cache)")
