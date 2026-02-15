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

            # Check for pending training request
            request = self.memory_manager.get_training_request()

            if request:
                reason = request["reason"]
                requested_mode = request.get("training_mode", "auto")

                # Manual requests bypass schedule
                if reason == "manual":
                    logger.info("Manual training requested, bypassing schedule")
                    self._execute_training(requested_mode=requested_mode)
                    return

                # Automatic triggers check schedule
                if self._is_within_schedule():
                    logger.info(f"Training triggered: {reason}")
                    self._execute_training(requested_mode=requested_mode)
                else:
                    logger.info(
                        f"Training requested ({reason}) but outside schedule window. "
                        f"Waiting for {self.config.training.schedule_time} "
                        f"{self.config.training.schedule_timezone}"
                    )
                return

            # No pending request, check if conditions are met
            stats = self.memory_manager.get_training_stats()

            # Log status
            status_message = self.scheduler.get_status_message(stats)
            logger.info(f"Training status:\n{status_message}")

            # Check if training should be triggered
            should_train, reason, mode = self.scheduler.should_trigger_training(stats)

            if should_train:
                # Create request for scheduled execution
                logger.info(f"Training conditions met: {reason} (mode: {mode})")
                self.memory_manager.request_training(reason=reason, training_mode=mode)
                logger.info("Training request created for next schedule window")
            else:
                logger.info("No training trigger conditions met")

        except Exception as e:
            logger.error(f"Error in training check: {e}", exc_info=True)

    def _is_within_schedule(self) -> bool:
        """
        Check if current time is within configured training schedule.

        Returns:
            True if schedule disabled OR current time matches schedule window
        """
        if not self.config.training.schedule_enabled:
            return True  # Schedule disabled, always allow

        try:
            import pytz
            from datetime import datetime

            # Get configured timezone
            tz = pytz.timezone(self.config.training.schedule_timezone)
            now = datetime.now(tz)

            # Parse schedule time (e.g., "01:00")
            schedule_hour, schedule_minute = map(int, self.config.training.schedule_time.split(':'))

            # Calculate window
            window_minutes = self.config.training.schedule_window_minutes

            # Check if current time is within window of scheduled time
            scheduled_time = now.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
            time_diff_minutes = abs((now - scheduled_time).total_seconds() / 60)

            # Handle day boundary (e.g., schedule at 01:00, current time 23:50)
            if time_diff_minutes > 12 * 60:  # More than 12 hours diff means crossed midnight
                time_diff_minutes = 24 * 60 - time_diff_minutes

            within_window = time_diff_minutes <= window_minutes

            if within_window:
                logger.info(
                    f"Within training schedule window "
                    f"({time_diff_minutes:.1f} min from {self.config.training.schedule_time})"
                )
            else:
                logger.debug(
                    f"Outside training schedule window "
                    f"({time_diff_minutes:.1f} min from {self.config.training.schedule_time})"
                )

            return within_window

        except Exception as e:
            logger.error(f"Error checking schedule (allowing training): {e}")
            return True  # On error, allow training
    
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

    def _execute_training(self, requested_mode: str = "auto"):
        """
        Execute the complete training pipeline.

        Args:
            requested_mode: Training mode ("auto", "sft", or "dpo")
                          "auto" = select based on data sentiment distribution
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()
        selected_mode = "sft"  # Default fallback

        try:
            # Step 0: Free GPU vRAM (only needed for GPU training)
            if self.config.training.enable_cpu_training:
                logger.info("Step 0: SKIPPED - CPU training mode enabled, GPU model stays loaded for inference")
                logger.info("Training will proceed on CPU while GPU serves requests")
            else:
                logger.info("Step 0: Requesting API model unload to free GPU vRAM...")
                self._request_api_unload()

            # Step 1: Collect training data
            logger.info("Step 1: Collecting training data...")

            # Get golden examples first to reserve space in top_n
            golden_examples = self.memory_manager.get_golden_examples()

            # Adjust top_n to include golden examples in total count
            top_n = self.config.memory.top_n_weighted_interactions
            top_n_adjusted = max(0, top_n - len(golden_examples))
            logger.info(
                f"Top-N interactions: {top_n} total "
                f"({top_n_adjusted} weighted + {len(golden_examples)} golden)"
            )

            interactions = self.memory_manager.get_top_interactions_by_weight(top_n_adjusted)

            if not interactions and not golden_examples:
                logger.warning("No training data available, skipping training")
                return

            # Step 1.5: Select training mode
            logger.info("Step 1.5: Selecting training mode...")

            if requested_mode == "auto":
                # Auto-select based on sentiment distribution
                stats = self.memory_manager.get_training_stats()
                selected_mode = self.scheduler._select_mode_from_stats(stats)
                logger.info(f"Auto-selected mode: {selected_mode}")
            elif requested_mode in ["sft", "dpo"]:
                selected_mode = requested_mode
                logger.info(f"Using requested mode: {selected_mode}")
            else:
                logger.warning(f"Unknown mode '{requested_mode}', defaulting to SFT")
                selected_mode = "sft"

            # Temporarily override config for this training run
            original_enable_dpo = self.config.training.enable_dpo
            self.config.training.enable_dpo = (selected_mode == "dpo")

            # Step 2: Generate synthetic rejections for DPO (if DPO mode selected)
            rejections = {}
            if self.config.training.enable_dpo:
                logger.info("Step 1.5: Generating synthetic rejections for DPO...")

                from trainer.rejection_generator import RejectionGenerator

                # Initialize generator (uses Ollama API)
                generator = RejectionGenerator(ollama_url="http://api:8000")

                # Collect all prompts that need rejections
                all_prompts = []

                # Positive and neutral interactions
                positive = [i for i in interactions if i.sentiment > self.config.sentiment.positive_threshold]
                neutral = [
                    i for i in interactions
                    if abs(i.sentiment) <= max(
                        self.config.sentiment.positive_threshold,
                        abs(self.config.sentiment.negative_threshold)
                    )
                ]

                for interaction in positive + neutral:
                    all_prompts.append(interaction.user_message)

                # Golden examples (critical for preventing catastrophic forgetting)
                for example in golden_examples:
                    all_prompts.append(example.user_message)

                logger.info(f"Generating rejections for {len(all_prompts)} prompts...")

                # Batch generate rejections using currently loaded model
                rejections = generator.generate_rejections(
                    all_prompts,
                    batch_size=self.config.training.dpo_batch_size,
                    timeout=self.config.training.dpo_rejection_timeout
                )

                logger.info(
                    f"Rejection generation complete: {len(rejections)}/{len(all_prompts)} "
                    f"({len(rejections)/len(all_prompts)*100:.1f}% success rate)"
                )

            # Step 2: Build dataset
            logger.info("Step 2: Building dataset...")

            if self.config.training.enable_dpo and rejections:
                logger.info("Using DPO dataset builder (preference pairs)")
                dataset = self.dataset_builder.build_dpo_dataset(
                    interactions,
                    golden_examples,
                    rejections
                )
            else:
                logger.info("Using SFT dataset builder (supervised fine-tuning)")
                dataset = self.dataset_builder.build_training_dataset(
                    interactions,
                    golden_examples
                )

            if not dataset:
                logger.warning("Empty dataset after preparation, skipping training")
                return

            # Step 3: Split into train/validation
            logger.info("Step 3: Creating train/validation split...")

            train_data, val_data = self.dataset_builder.create_validation_split(
                dataset,
                self.config.training.validation_split
            )

            # Format for training (DPO or SFT)
            if self.config.training.enable_dpo:
                logger.info("Formatting dataset for DPO training...")
                train_formatted = self.dataset_builder.format_for_dpo_training(train_data)
                val_formatted = self.dataset_builder.format_for_dpo_training(val_data)
            else:
                logger.info("Formatting dataset for SFT training...")
                train_formatted = self.dataset_builder.format_for_training(train_data)
                val_formatted = self.dataset_builder.format_for_training(val_data)
            
            # Step 4: Train LoRA adapter (DPO or SFT mode)
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir = f"{settings.checkpoints_path}/{checkpoint_id}"

            # Determine training mode
            training_mode = "DPO" if self.config.training.enable_dpo else "SFT"

            # Check if sequential training is enabled (memory-efficient mode)
            if self.config.training.enable_sequential_training:
                logger.info(f"Step 4: Training LoRA adapter ({training_mode}, SEQUENTIAL MODE)...")
                logger.info(f"Sequential training enabled: {self.config.training.layers_per_pass} layers per pass")
                adapter_path = self.lora_trainer.train_sequential(
                    train_formatted,
                    val_formatted,
                    checkpoint_dir
                )
            else:
                logger.info(f"Step 4: Training LoRA adapter ({training_mode}, STANDARD MODE)...")
                adapter_path = self.lora_trainer.train(
                    train_formatted,
                    val_formatted,
                    checkpoint_dir
                )

            logger.info(f"Training complete. Adapter saved to: {adapter_path}")

            # Step 5: Evaluate checkpoint
            logger.info("Step 5: Evaluating checkpoint...")

            should_deploy, metadata = self.model_evaluator.evaluate_checkpoint(
                adapter_path,
                val_formatted,
                checkpoint_id
            )

            # Step 6: Deploy if validation passed
            if should_deploy:
                logger.info("Step 6: Converting to GGUF and deploying...")
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

                # Cleanup failed checkpoint directory
                logger.info(f"Cleaning up failed checkpoint: {checkpoint_id}")
                self._cleanup_checkpoint_dir(checkpoint_dir)

            # Step 7: Mark training complete
            self.memory_manager.mark_training_complete(training_mode=selected_mode)

            # Step 8: Cleanup old memories
            logger.info("Step 8: Cleaning up old memories...")
            self.memory_manager.cleanup_old_memories()

            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"TRAINING PIPELINE COMPLETE (mode: {selected_mode}, took {elapsed_time:.1f}s)")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE FAILED")
            logger.info("=" * 80)

            # Cleanup failed checkpoint if it was created
            if 'checkpoint_dir' in locals():
                logger.info(f"Cleaning up failed checkpoint directory: {checkpoint_dir}")
                self._cleanup_checkpoint_dir(checkpoint_dir)

            # Ensure the API model is reloaded even if training crashed
            self._request_api_reload_previous()

        finally:
            # Restore original config
            if 'original_enable_dpo' in locals():
                self.config.training.enable_dpo = original_enable_dpo
    
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

    def _cleanup_checkpoint_dir(self, checkpoint_dir: str):
        """
        Remove a failed checkpoint directory, metadata, and any associated files.

        Args:
            checkpoint_dir: Path to the checkpoint directory to remove
        """
        import shutil

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_id = checkpoint_path.name

        # Clean up checkpoint directory
        if checkpoint_path.exists():
            try:
                shutil.rmtree(checkpoint_path)
                logger.info(f"Cleaned up checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up checkpoint directory {checkpoint_dir}: {e}")

        # Clean up metadata file (saved in checkpoints_path root)
        metadata_file = Path(settings.checkpoints_path) / f"{checkpoint_id}_metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
                logger.info(f"Cleaned up checkpoint metadata: {metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up metadata {metadata_file}: {e}")

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
