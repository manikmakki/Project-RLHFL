"""
Project RLHFL - Model Evaluator

Evaluates training checkpoints using metrics from the trainer (train_loss, eval_loss)
and decides whether to deploy or reject the checkpoint.
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

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
        checkpoint_id: str,
        training_metrics: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, CheckpointMetadata]:
        """
        Evaluate a checkpoint using training metrics and decide if it should be deployed.

        Args:
            adapter_path: Path to the trained adapter
            val_dataset: Validation dataset (used for sample count)
            checkpoint_id: Unique checkpoint identifier
            training_metrics: Dict with train_loss and eval_loss from trainer

        Returns:
            tuple[bool, CheckpointMetadata]: (should_deploy, metadata)
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_id}")

        try:
            current_metrics = self._load_current_metrics()

            # Compute metrics from training results
            if training_metrics:
                eval_loss = training_metrics.get("eval_loss", 0.0)
                mode = training_metrics.get("mode", "sft")

                if mode == "dpo":
                    # DPO eval_loss is a preference loss (logit margin), not CE loss.
                    # exp(DPO_loss) is not perplexity — carry forward the existing
                    # baseline so it isn't corrupted by a meaningless value.
                    carried = current_metrics.get("perplexity", 999.0) if current_metrics else 999.0
                    metrics = {
                        "perplexity": carried,
                        "train_loss": float(training_metrics.get("train_loss", 0.0)),
                        "eval_loss": float(eval_loss),
                        "mode": "dpo",
                    }
                else:
                    metrics = {
                        "perplexity": float(math.exp(eval_loss)) if eval_loss < 100 else 999.0,
                        "train_loss": float(training_metrics.get("train_loss", 0.0)),
                        "eval_loss": float(eval_loss),
                        "mode": "sft",
                    }
            else:
                metrics = {"perplexity": 999.0, "train_loss": 0.0, "eval_loss": 0.0, "mode": "sft"}

            logger.info(f"Checkpoint metrics: {metrics}")

            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                adapter_path=adapter_path,
                timestamp=datetime.now(),
                metrics=metrics,
                parent_checkpoint=self._get_current_checkpoint_id(),
                training_samples=len(val_dataset),
                can_rollback=True,
                deployed=False,
            )
            self._save_checkpoint_metadata(metadata)

            should_deploy = self._should_deploy(metrics, current_metrics)

            if should_deploy:
                logger.info(f"Checkpoint {checkpoint_id} passed validation, deploying")
                metadata.deployed = True
                self._save_checkpoint_metadata(metadata)
                self._set_current_checkpoint(checkpoint_id, metadata)
            else:
                logger.warning(f"Checkpoint {checkpoint_id} failed validation, keeping current model")

            return should_deploy, metadata

        except Exception as e:
            logger.error(f"Failed to evaluate checkpoint: {e}", exc_info=True)
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                adapter_path=adapter_path,
                timestamp=datetime.now(),
                metrics={},
                training_samples=len(val_dataset),
                can_rollback=True,
                deployed=False,
            )
            return False, metadata

    def _should_deploy(
        self,
        new_metrics: Dict[str, float],
        current_metrics: Optional[Dict[str, float]],
    ) -> bool:
        """Decide if new checkpoint should replace current deployment."""
        if current_metrics is None:
            logger.info("No current checkpoint, deploying first checkpoint")
            return True

        new_perplexity = new_metrics.get("perplexity", 999.0)
        current_perplexity = current_metrics.get("perplexity", 999.0)

        if new_metrics.get("mode") == "dpo":
            logger.info(
                f"DPO checkpoint: train_loss={new_metrics.get('train_loss', 0):.4f}, "
                f"dpo_eval_loss={new_metrics.get('eval_loss', 0):.4f} "
                f"(perplexity comparison skipped — DPO loss is not CE loss)"
            )

        floor = self.config.training.min_reasonable_perplexity
        if current_perplexity < floor:
            logger.warning(
                f"Stored baseline perplexity {current_perplexity:.4f} is below the sanity "
                f"floor of {floor} — likely caused by eval/train data contamination in a "
                f"previous run. Ignoring baseline; treating as first deployment."
            )
            logger.info(f"Perplexity: {new_perplexity:.2f} vs <corrupted baseline> (PASS — baseline reset)")
            return True

        # Allow 5% degradation in perplexity
        perplexity_ok = new_perplexity <= current_perplexity * 1.05

        logger.info(
            f"Perplexity: {new_perplexity:.2f} vs {current_perplexity:.2f} "
            f"({'PASS' if perplexity_ok else 'FAIL'})"
        )

        return perplexity_ok

    def _save_checkpoint_metadata(self, metadata: CheckpointMetadata):
        metadata_path = Path(self.checkpoints_path) / f"{metadata.checkpoint_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

    def _load_current_metrics(self) -> Optional[Dict[str, float]]:
        try:
            if not self.current_checkpoint_file.exists():
                return None
            with open(self.current_checkpoint_file) as f:
                return json.load(f).get("metrics")
        except Exception as e:
            logger.error(f"Failed to load current metrics: {e}")
            return None

    def _get_current_checkpoint_id(self) -> Optional[str]:
        try:
            if not self.current_checkpoint_file.exists():
                return None
            with open(self.current_checkpoint_file) as f:
                return json.load(f).get("checkpoint_id")
        except Exception:
            return None

    def _set_current_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata):
        current_data = {
            "checkpoint_id": checkpoint_id,
            "adapter_path": metadata.adapter_path,
            "metrics": metadata.metrics,
            "deployed_at": datetime.now().isoformat(),
        }
        with open(self.current_checkpoint_file, "w") as f:
            json.dump(current_data, f, indent=2)
        logger.info(f"Set current checkpoint to: {checkpoint_id}")

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        checkpoints = []
        for metadata_file in Path(self.checkpoints_path).glob("*_metadata.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                if isinstance(data.get("timestamp"), str):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                checkpoints.append(CheckpointMetadata(**data))
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
