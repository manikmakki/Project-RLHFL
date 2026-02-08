"""
Project RLHFL - Training Worker

This module orchestrates the automatic training pipeline. It runs as a
background service that:

1. Periodically checks if training should be triggered based on:
   - Number of new interactions
   - Time since last interaction
   - Time since last training
   - User-initiated requests

2. When triggered, executes the full training pipeline:
   - Pulls interactions from memory
   - Builds training dataset with weighting
   - Trains LoRA adapter using QLoRA
   - Validates checkpoint
   - Deploys if validation passes

3. Maintains checkpoints and enables rollback if needed

The worker runs continuously with hourly checks, ensuring the model
continuously improves based on user interactions without manual intervention.
"""

import gc
import logging
import time
import sys
import torch
import requests
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import load_config, settings
from shared.models import Interaction
from api.memory_manager import MemoryManager
from trainer.training_scheduler import TrainingScheduler
from trainer.dataset_builder import DatasetBuilder
from trainer.lora_trainer import LoRATrainer
from trainer.model_evaluator import ModelEvaluator
from trainer.gguf_converter import GGUFConverter, GGUFConversionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerWorker:
    """Main training worker that orchestrates the training pipeline."""
    
    def __init__(self):
        logger.info("Initializing training worker...")
        
        # Load configuration
        self.config = load_config()
        
        # Initialize components
        self.memory_manager = MemoryManager(self.config)
        self.scheduler = TrainingScheduler(self.config)
        self.dataset_builder = DatasetBuilder(self.config)
        self.lora_trainer = LoRATrainer(self.config, settings.base_model_path)
        self.model_evaluator = ModelEvaluator(
            self.config,
            settings.base_model_path,
            settings.checkpoints_path
        )
        
        logger.info("Training worker initialized successfully")
    
    def check_and_train(self):
        """Check if training should be triggered and execute if needed."""
        try:
            logger.info("Checking training conditions...")
            
            # Get training statistics
            stats = self.memory_manager.get_training_stats()
            
            # Log status
            status_message = self.scheduler.get_status_message(stats)
            logger.info(f"Training status:\n{status_message}")
            
            # Check if training should be triggered
            should_train, reason = self.scheduler.should_trigger_training(stats)
            
            if should_train:
                logger.info(f"Training triggered: {reason}")
                self._execute_training()
            else:
                logger.info("No training trigger conditions met")
                
        except Exception as e:
            logger.error(f"Error in training check: {e}", exc_info=True)
    
    def _request_api_reload_previous(self):
        """Tell the API to reload its previous model (after training didn't produce a new deployment)."""
        try:
            checkpoint_file = Path("/data/current_checkpoint.json")
            if checkpoint_file.exists():
                import json
                data = json.loads(checkpoint_file.read_text())
                gguf_path = data.get("gguf_model_path")
                cp_id = data.get("checkpoint_id")
                if gguf_path and Path(gguf_path).exists():
                    logger.info(f"Reloading previous model: {gguf_path}")
                    self._notify_api_reload(gguf_path, cp_id, {})
                    return
            # Fallback: reload base GGUF model from config
            base_model = settings.model_path
            if Path(base_model).exists():
                logger.info(f"Reloading base model: {base_model}")
                self._notify_api_reload(base_model, None, {})
            else:
                logger.warning(f"No model found to reload after training (tried {base_model})")
        except Exception as e:
            logger.warning(f"Failed to reload previous model (poller will recover): {e}")

    def _request_api_unload(self):
        """Ask the API to unload its model so the trainer gets full GPU vRAM."""
        try:
            response = requests.post(
                "http://api:8000/v1/model/unload",
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info("API model unloaded, full GPU vRAM available for training")
                    return True
                else:
                    logger.warning(f"API unload returned failure: {data.get('message')}")
            else:
                logger.warning(f"API unload returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not reach API for model unload: {e}")
        return False

    def _execute_training(self):
        """Execute the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Step 0: Free GPU vRAM by unloading the API's llama-cpp model
            logger.info("Step 0: Requesting API model unload to free GPU vRAM...")
            self._request_api_unload()

            # Step 1: Prepare dataset
            logger.info("Step 1: Preparing training dataset...")
            
            last_training = self.memory_manager._load_training_state()
            interactions = self.memory_manager.get_interactions_since(last_training)
            golden_examples = self.memory_manager.get_golden_examples()
            
            if not interactions and not golden_examples:
                logger.warning("No training data available, skipping training")
                return
            
            dataset = self.dataset_builder.build_training_dataset(
                interactions,
                golden_examples
            )
            
            if not dataset:
                logger.warning("Empty dataset after preparation, skipping training")
                return
            
            # Step 2: Split into train/validation
            logger.info("Step 2: Creating train/validation split...")
            
            train_data, val_data = self.dataset_builder.create_validation_split(
                dataset,
                self.config.training.validation_split
            )
            
            # Format for training
            train_formatted = self.dataset_builder.format_for_training(train_data)
            val_formatted = self.dataset_builder.format_for_training(val_data)
            
            # Step 3: Train LoRA adapter
            logger.info("Step 3: Training LoRA adapter...")
            
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir = f"{settings.checkpoints_path}/{checkpoint_id}"
            
            adapter_path = self.lora_trainer.train(
                train_formatted,
                val_formatted,
                checkpoint_dir
            )
            
            logger.info(f"Training complete. Adapter saved to: {adapter_path}")
            
            # Step 4: Evaluate checkpoint
            logger.info("Step 4: Evaluating checkpoint...")
            
            should_deploy, metadata = self.model_evaluator.evaluate_checkpoint(
                adapter_path,
                val_formatted,
                checkpoint_id
            )
            
            # Step 5: Deploy if validation passed
            if should_deploy:
                logger.info("Step 5: Converting to GGUF and deploying...")
                try:
                    gguf_path = self._convert_and_deploy(
                        adapter_path, checkpoint_id, checkpoint_dir, metadata
                    )
                    logger.info(f"Checkpoint {checkpoint_id} deployed as GGUF: {gguf_path}")
                    logger.info(f"Metrics: {metadata.metrics}")
                except Exception as e:
                    logger.error(
                        f"GGUF conversion/deploy failed for {checkpoint_id}: {e}",
                        exc_info=True,
                    )
                    logger.info("Keeping current model; adapter saved for retry")
                    self._request_api_reload_previous()
            else:
                logger.warning("Checkpoint failed validation, not deployed")
                logger.info(f"Metrics: {metadata.metrics}")
                self._request_api_reload_previous()
            
            # Step 6: Mark training complete
            self.memory_manager.mark_training_complete()
            
            # Step 7: Cleanup old memories
            logger.info("Step 6: Cleaning up old memories...")
            self.memory_manager.cleanup_old_memories()
            
            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"TRAINING PIPELINE COMPLETE (took {elapsed_time:.1f}s)")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE FAILED")
            logger.info("=" * 80)
            # Ensure the API model is reloaded even if training crashed
            self._request_api_reload_previous()
    
    def _convert_and_deploy(self, adapter_path, checkpoint_id, checkpoint_dir, metadata):
        """
        Convert adapter to GGUF and notify the API to reload.

        1. Merge adapter + base model → HF model
        2. Convert HF → FP16 GGUF
        3. Quantize to Q4_K_M
        4. Update checkpoint metadata with GGUF path
        5. Free all trainer GPU memory
        6. Notify API to reload
        7. Clean up old GGUF files
        """
        converter = GGUFConverter(base_model_path=settings.base_model_path)

        merged_dir = f"{checkpoint_dir}/merged_model"
        gguf_filename = f"model-{checkpoint_id}.Q4_K_M.gguf"
        gguf_path = f"/models/{gguf_filename}"

        # Stages 1-3: merge, convert, quantize
        converter.convert_adapter_to_gguf(
            adapter_path=adapter_path,
            merged_model_dir=merged_dir,
            output_gguf_path=gguf_path,
            quantization_type="Q4_K_M",
            lora_trainer=self.lora_trainer,
        )

        # Verify output
        gguf_file = Path(gguf_path)
        if not gguf_file.exists() or gguf_file.stat().st_size < 100_000_000:
            raise GGUFConversionError(
                f"GGUF output missing or too small: {gguf_path}"
            )

        # Update checkpoint metadata with GGUF path
        metadata.gguf_model_path = gguf_path
        self.model_evaluator._save_checkpoint_metadata(metadata)
        self.model_evaluator._set_current_checkpoint(checkpoint_id, metadata)

        # Free all trainer GPU memory before API attempts reload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Trainer GPU memory fully cleared before API reload")

        # Notify API to reload (best-effort; poller is the safety net)
        self._notify_api_reload(gguf_path, checkpoint_id, metadata.metrics)

        # Clean up old GGUF files
        max_old = self.config.model.max_old_gguf_files
        self._cleanup_old_gguf_files(keep_count=max_old, current_gguf=gguf_path)

        return gguf_path

    def _notify_api_reload(self, gguf_path, checkpoint_id, metrics):
        """Send HTTP POST to the API service to trigger model reload."""
        try:
            response = requests.post(
                "http://api:8000/v1/model/reload",
                json={
                    "gguf_model_path": gguf_path,
                    "checkpoint_id": checkpoint_id,
                    "metrics": metrics,
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info("API notified successfully, model reload confirmed")
                else:
                    logger.warning(f"API reload returned failure: {data.get('message')}")
            else:
                logger.warning(
                    f"API reload notification returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to notify API of new model (poller will pick it up): {e}"
            )

    def _cleanup_old_gguf_files(self, keep_count=2, current_gguf=""):
        """Remove old fine-tuned GGUF files, keeping base model + N most recent."""
        models_dir = Path("/models")

        # Find all fine-tuned GGUF files (pattern: model-checkpoint_*.gguf)
        gguf_files = sorted(
            list(models_dir.glob("model-checkpoint_*.gguf")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        # Always keep the base model and the current deployment
        files_to_keep = {current_gguf}
        for f in gguf_files[:keep_count]:
            files_to_keep.add(str(f))

        for f in gguf_files:
            if str(f) not in files_to_keep:
                try:
                    f.unlink()
                    logger.info(f"Cleaned up old GGUF: {f}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {f}: {e}")

    def check_user_requested_training(self):
        """Fast-poll check: only runs training if a user explicitly requested it."""
        try:
            stats = self.memory_manager.get_training_stats()
            if stats.user_requested_training:
                logger.info("User-requested training detected, starting immediately")
                self._execute_training()
        except Exception as e:
            logger.error(f"Error in user-request check: {e}", exc_info=True)

    def run(self):
        """Run the training worker with scheduled checks."""
        logger.info("Starting training worker scheduler...")

        # Create scheduler
        scheduler = BlockingScheduler()

        # Schedule training checks every hour
        scheduler.add_job(
            self.check_and_train,
            trigger=IntervalTrigger(hours=1),
            id='training_check',
            name='Check training conditions',
            replace_existing=True
        )

        # Fast poll (every 30s) for user-requested training only
        scheduler.add_job(
            self.check_user_requested_training,
            trigger=IntervalTrigger(seconds=30),
            id='user_request_check',
            name='Check for user-requested training',
            replace_existing=True
        )

        # Run an immediate check on startup
        logger.info("Running initial training check...")
        self.check_and_train()

        logger.info("Training worker running. Hourly checks + 30s user-request poll.")

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Training worker stopped")


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("LOCAL LLM TRAINING SERVICE")
    logger.info("=" * 80)
    
    # Wait a bit for API service to start
    logger.info("Waiting 30 seconds for API service to initialize...")
    time.sleep(30)
    
    # Create and run worker
    worker = TrainerWorker()
    worker.run()


if __name__ == "__main__":
    main()
