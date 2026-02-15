import gc
import logging
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

from shared.config import SystemConfig
from shared.models import CheckpointMetadata

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model checkpoints and determine if they should be deployed."""
    
    def __init__(self, config: SystemConfig, base_model_path: str, checkpoints_path: str):
        self.config = config
        self.base_model_path = base_model_path
        self.checkpoints_path = checkpoints_path
        self.current_checkpoint_file = Path(checkpoints_path) / "current_checkpoint.json"
    
    def evaluate_checkpoint(
        self,
        adapter_path: str,
        val_dataset: List[Dict[str, Any]],
        checkpoint_id: str
    ) -> tuple[bool, CheckpointMetadata]:
        """
        Evaluate a checkpoint and decide if it should be deployed.
        
        Returns:
            tuple[bool, CheckpointMetadata]: (should_deploy, metadata)
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_id}")
        
        try:
            # Calculate metrics
            metrics = self._calculate_metrics(adapter_path, val_dataset)
            
            # Load current checkpoint metrics for comparison
            current_metrics = self._load_current_metrics()
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                adapter_path=adapter_path,
                timestamp=datetime.now(),
                metrics=metrics,
                parent_checkpoint=self._get_current_checkpoint_id(),
                training_samples=len(val_dataset),
                can_rollback=True,
                deployed=False
            )
            
            # Save checkpoint metadata
            self._save_checkpoint_metadata(metadata)
            
            # Decide if we should deploy
            should_deploy = self._should_deploy(metrics, current_metrics)
            
            if should_deploy:
                logger.info(f"Checkpoint {checkpoint_id} passed validation, deploying")
                metadata.deployed = True
                self._save_checkpoint_metadata(metadata)
                self._set_current_checkpoint(checkpoint_id, metadata)
            else:
                logger.warning(
                    f"Checkpoint {checkpoint_id} failed validation, "
                    f"keeping current model"
                )
            
            return should_deploy, metadata
            
        except Exception as e:
            logger.error(f"Failed to evaluate checkpoint: {e}", exc_info=True)
            # Return safe defaults with empty metrics dict
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                adapter_path=adapter_path,
                timestamp=datetime.now(),
                metrics={},  # Empty metrics dict - only floats allowed
                training_samples=len(val_dataset),
                can_rollback=True,
                deployed=False
            )
            return False, metadata
    
    def _calculate_metrics(
        self,
        adapter_path: str,
        val_dataset: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate validation metrics for the model."""
        logger.info("Calculating validation metrics...")
        
        model = None
        tokenizer = None
        try:
            # Load model with adapter
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )

            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()

            # Calculate perplexity on validation set
            perplexities = []

            # Sample subset for evaluation (to save time)
            eval_samples = val_dataset[:min(len(val_dataset), 20)]

            for sample in eval_samples:
                prompt = sample['prompt']
                completion = sample['completion']
                # Use GPT-OSS Harmony format (not Llama format)
                text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"

                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    perplexity = np.exp(loss)
                    perplexities.append(perplexity)

            avg_perplexity = np.mean(perplexities)

            # Calculate sentiment alignment (how well responses match expected sentiment)
            sentiment_scores = []
            for sample in eval_samples:
                if 'sentiment' in sample and sample['sentiment'] != 0:
                    # Positive sentiment samples should have low perplexity (model is confident)
                    if sample['sentiment'] > 0:
                        score = 1.0 / (1.0 + perplexities[len(sentiment_scores)])
                    else:
                        score = 0.5  # Neutral for negative examples
                    sentiment_scores.append(score)

            sentiment_alignment = np.mean(sentiment_scores) if sentiment_scores else 0.5

            metrics = {
                "perplexity": float(avg_perplexity),
                "sentiment_alignment": float(sentiment_alignment)
            }

            logger.info(f"Metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {
                "perplexity": 999.0,
                "sentiment_alignment": 0.0
            }
        finally:
            # Free GPU vRAM used by evaluation model
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Evaluator GPU memory freed")
    
    def _should_deploy(
        self,
        new_metrics: Dict[str, float],
        current_metrics: Optional[Dict[str, float]]
    ) -> bool:
        """
        Decide if new checkpoint should be deployed.
        
        Criteria:
        - Perplexity improved OR stayed within 5%
        - Sentiment alignment improved OR stayed within threshold
        - No catastrophic quality drop
        """
        
        # If no current checkpoint, deploy the first one
        if current_metrics is None:
            logger.info("No current checkpoint, deploying first checkpoint")
            return True
        
        # Check if new metrics are valid
        if "error" in new_metrics:
            logger.warning("New metrics contain errors, not deploying")
            return False
        
        # Check perplexity
        new_perplexity = new_metrics.get("perplexity", 999.0)
        current_perplexity = current_metrics.get("perplexity", 999.0)
        
        perplexity_ok = new_perplexity <= current_perplexity * 1.05  # Allow 5% degradation
        
        logger.info(
            f"Perplexity check: {new_perplexity:.2f} vs {current_perplexity:.2f} "
            f"({'PASS' if perplexity_ok else 'FAIL'})"
        )
        
        # Check sentiment alignment
        new_alignment = new_metrics.get("sentiment_alignment", 0.0)
        current_alignment = current_metrics.get("sentiment_alignment", 0.0)
        
        alignment_ok = new_alignment >= current_alignment - 0.05  # Allow small degradation
        
        logger.info(
            f"Sentiment alignment check: {new_alignment:.2f} vs {current_alignment:.2f} "
            f"({'PASS' if alignment_ok else 'FAIL'})"
        )
        
        # Deploy if both checks pass
        should_deploy = perplexity_ok and alignment_ok
        
        return should_deploy
    
    def _save_checkpoint_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata to disk."""
        metadata_path = Path(self.checkpoints_path) / f"{metadata.checkpoint_id}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
    
    def _load_current_metrics(self) -> Optional[Dict[str, float]]:
        """Load metrics from the current deployed checkpoint."""
        try:
            if not self.current_checkpoint_file.exists():
                return None
            
            with open(self.current_checkpoint_file, 'r') as f:
                current = json.load(f)
            
            return current.get("metrics")
            
        except Exception as e:
            logger.error(f"Failed to load current metrics: {e}")
            return None
    
    def _get_current_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the current deployed checkpoint."""
        try:
            if not self.current_checkpoint_file.exists():
                return None
            
            with open(self.current_checkpoint_file, 'r') as f:
                current = json.load(f)
            
            return current.get("checkpoint_id")
            
        except Exception:
            return None
    
    def _set_current_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata):
        """Set a checkpoint as the current deployed checkpoint."""
        current_data = {
            "checkpoint_id": checkpoint_id,
            "adapter_path": metadata.adapter_path,
            "gguf_model_path": metadata.gguf_model_path,
            "metrics": metadata.metrics,
            "deployed_at": datetime.now().isoformat()
        }

        with open(self.current_checkpoint_file, 'w') as f:
            json.dump(current_data, f, indent=2)

        logger.info(f"Set current checkpoint to: {checkpoint_id}")
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []
        
        for metadata_file in Path(self.checkpoints_path).glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Convert timestamp string back to datetime
                if isinstance(data.get('timestamp'), str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                checkpoint = CheckpointMetadata(**data)
                checkpoints.append(checkpoint)
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint metadata from {metadata_file}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a previous checkpoint and notify the API to reload."""
        try:
            metadata_file = Path(self.checkpoints_path) / f"{checkpoint_id}_metadata.json"

            if not metadata_file.exists():
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False

            with open(metadata_file, 'r') as f:
                data = json.load(f)

            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])

            metadata = CheckpointMetadata(**data)

            # Verify GGUF file exists for this checkpoint
            if metadata.gguf_model_path and not Path(metadata.gguf_model_path).exists():
                logger.error(
                    f"GGUF file for checkpoint {checkpoint_id} not found: "
                    f"{metadata.gguf_model_path}"
                )
                return False

            # Set as current
            self._set_current_checkpoint(checkpoint_id, metadata)

            # Notify API to reload (best-effort)
            if metadata.gguf_model_path:
                try:
                    import requests
                    requests.post(
                        "http://api:8000/v1/model/reload",
                        json={
                            "gguf_model_path": metadata.gguf_model_path,
                            "checkpoint_id": checkpoint_id,
                            "metrics": metadata.metrics,
                        },
                        timeout=30,
                    )
                    logger.info("API notified of rollback")
                except Exception as e:
                    logger.warning(f"Failed to notify API during rollback: {e}")

            logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint: {e}")
            return False
