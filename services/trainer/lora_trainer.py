"""
Project RLHFL - LoRA Trainer

This module handles the actual training of LoRA (Low-Rank Adaptation) adapters
for the base language model. It uses QLoRA (Quantized LoRA) with 4-bit
quantization to enable training on consumer GPUs with 16GB VRAM.

The trainer:
- Loads the base model with 4-bit quantization
- Applies LoRA configuration to attention layers
- Trains on weighted datasets with supervision
- Saves lightweight adapter weights (~10-20MB)
- Supports merging adapters back into the base model

LoRA adapters are parameter-efficient, requiring only ~1-2% of full
fine-tuning memory while achieving comparable results.
"""

import logging
import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from shared.config import SystemConfig

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Train LoRA adapters for the base model."""
    
    def __init__(self, config: SystemConfig, base_model_path: str):
        self.config = config
        self.base_model_path = base_model_path
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
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
            # Create checkpoint directory
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                use_fast=False  # Use slow tokenizer to avoid tokenizer.json parsing issues
            )
            
            # Ensure tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model with 4-bit quantization (QLoRA)
            logger.info(f"Loading base model from {self.base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False  # Disable KV cache for training to avoid attention mask mismatches
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Enable gradient checkpointing to save memory
            # Note: Disabled due to attention mask dimension issues with KV cache
            # model.gradient_checkpointing_enable()
            
            # Apply LoRA
            logger.info("Applying LoRA configuration")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
            
            # Ensure use_cache is disabled for training
            model.config.use_cache = False
            
            # Prepare datasets
            train_ds = self._prepare_dataset(train_dataset, tokenizer)
            val_ds = self._prepare_dataset(val_dataset, tokenizer)
            
            # Training arguments
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
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                fp16=True,
                fp16_opt_level="O2",
                report_to="none",
                remove_unused_columns=False,
                dataloader_drop_last=False,
            )
            
            # Custom data collator for causal LM that pads and masks prompt tokens
            class DataCollatorForCausalLM:
                def __init__(self, tokenizer, max_length=None):
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __call__(self, features):
                    # features is a list of dicts with keys: input_ids, attention_mask, prompt_len
                    batch = self.tokenizer.pad(
                        features,
                        padding=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    # Build labels and mask prompt portion with -100
                    labels = input_ids.clone()
                    prompt_lens = [f.get("prompt_len", 0) for f in features]
                    for i, p_len in enumerate(prompt_lens):
                        if p_len > 0:
                            labels[i, :p_len] = -100

                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }

            data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, max_length=self.config.training.max_seq_length)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save adapter
            adapter_path = f"{checkpoint_path}/adapter"
            logger.info(f"Saving adapter to {adapter_path}")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            
            # Save training metadata
            metadata = {
                "train_loss": train_result.training_loss,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "epochs": self.config.training.num_epochs,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"{checkpoint_path}/training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")
            
            return adapter_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _prepare_dataset(
        self,
        dataset: List[Dict[str, Any]],
        tokenizer
    ) -> Dataset:
        """Prepare dataset for training."""
        
        def format_example(example):
            """Format a single example for training and record prompt length."""
            prompt = example['prompt']
            completion = example['completion']

            # Tokenize prompt alone to get prompt length (including INST markers)
            prompt_text = f"[INST] {prompt} [/INST]"
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.training.max_seq_length // 2,
                padding=False,
                return_tensors=None
            )
            prompt_len = len(prompt_tokens["input_ids"])

            # Full example (prompt + completion)
            full_text = f"[INST] {prompt} [/INST] {completion}"
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
