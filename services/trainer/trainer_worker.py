"""
Project RLHFL - Training Worker

Orchestrates the automatic training pipeline:
1. Checks training triggers (sentiment thresholds, schedule, manual)
2. Builds datasets from stored interactions
3. Trains LoRA adapters (CPU-only, SFT or DPO)
4. Evaluates checkpoints and deploys via Ollama
5. Manages checkpoint storage
"""

import gc
import logging
import os
import shutil
import time
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Must be set before torch (and oneDNN/MKLDNN) are imported.
# oneDNN uses OMP for its thread pool — torch.set_num_threads() does NOT control it.
# Falls back to docker-compose env vars if already set there.
def _configure_blas_threads():
    try:
        import yaml
        config_path = os.environ.get('CONFIG_PATH', '/config/system_config.yaml')
        with open(config_path) as _f:
            _cfg = yaml.safe_load(_f)
        n = str(_cfg.get('training', {}).get('cpu_threads', 32))
    except Exception:
        n = '32'
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, n)

_configure_blas_threads()

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import load_config, settings
from shared.models import Interaction
from api.memory_manager import MemoryManager
from trainer.training_scheduler import TrainingScheduler
from trainer.dataset_builder import DatasetBuilder
from trainer.lora_trainer_dense import LoRATrainerDense
from trainer.model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerWorker:
    """Main training worker that orchestrates the training pipeline."""

    def __init__(self):
        logger.info("Initializing training worker...")
        self.config = load_config()
        self.memory_manager = MemoryManager(self.config)
        self.scheduler = TrainingScheduler(self.config)
        self.dataset_builder = DatasetBuilder(self.config)
        self.lora_trainer = LoRATrainerDense(self.config, settings.base_model_path)
        self.model_evaluator = ModelEvaluator(
            self.config, settings.base_model_path, settings.checkpoints_path
        )
        logger.info("Training worker initialized successfully")

    def check_and_train(self):
        """Check if training should be triggered and execute if needed."""
        try:
            logger.info("Checking training conditions...")
            request = self.memory_manager.get_training_request()

            if request:
                reason = request["reason"]
                requested_mode = request.get("training_mode", "auto")

                if reason == "manual":
                    logger.info("Manual training requested, bypassing schedule")
                    self._execute_training(requested_mode=requested_mode)
                    return

                if self._is_within_schedule():
                    logger.info(f"Training triggered: {reason}")
                    self._execute_training(requested_mode=requested_mode)
                else: # TODO: Fix logic here, training will fire without respect to actual schedule.
                    logger.info(
                        f"Training requested ({reason}) but outside schedule window. "
                        f"Waiting for {self.config.training.schedule_time} "
                        f"{self.config.training.schedule_timezone}"
                    )
                return

            stats = self.memory_manager.get_training_stats()
            status_message = self.scheduler.get_status_message(stats)
            logger.info(f"Training status:\n{status_message}")

            should_train, reason, mode = self.scheduler.should_trigger_training(stats)
            if should_train:
                logger.info(f"Training conditions met: {reason} (mode: {mode})")
                self.memory_manager.request_training(reason=reason, training_mode=mode)
            else:
                logger.info("No training trigger conditions met")

        except Exception as e:
            logger.error(f"Error in training check: {e}", exc_info=True)

    def _is_within_schedule(self) -> bool:
        """Check if current time is within configured training schedule."""
        if not self.config.training.schedule_enabled:
            return True

        try:
            import pytz

            tz = pytz.timezone(self.config.training.schedule_timezone)
            now = datetime.now(tz)
            schedule_hour, schedule_minute = map(int, self.config.training.schedule_time.split(':'))
            window_minutes = self.config.training.schedule_window_minutes
            scheduled_time = now.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
            time_diff_minutes = abs((now - scheduled_time).total_seconds() / 60)

            if time_diff_minutes > 12 * 60:
                time_diff_minutes = 24 * 60 - time_diff_minutes

            within_window = time_diff_minutes <= window_minutes
            if within_window:
                logger.info(f"Within training schedule window ({time_diff_minutes:.1f} min from {self.config.training.schedule_time})")
            return within_window

        except Exception as e:
            logger.error(f"Error checking schedule (allowing training): {e}")
            return True

    def _cleanup_dangling_checkpoints(self):
        """Remove checkpoint dirs from previous runs that were killed before saving an adapter."""
        checkpoints_dir = Path(settings.checkpoints_path)
        for d in checkpoints_dir.iterdir():
            if not (d.is_dir() and d.name.startswith("checkpoint_")):
                continue
            adapter_dir = d / "adapter"
            if not adapter_dir.exists():
                logger.warning(f"Found dangling checkpoint (no adapter saved): {d.name}, removing...")
                try:
                    shutil.rmtree(d)
                    meta = checkpoints_dir / f"{d.name}_metadata.json"
                    if meta.exists():
                        meta.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove dangling checkpoint {d.name}: {e}")

    def _execute_training(self, requested_mode: str = "auto"):
        """Execute the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        self._cleanup_dangling_checkpoints()

        start_time = time.time()
        selected_mode = "sft"

        try:
            # Step 1: Collect training data
            logger.info("Step 1: Collecting training data...")
            golden_examples = self.memory_manager.get_golden_examples()
            top_n = self.config.memory.top_n_weighted_interactions
            top_n_adjusted = max(0, top_n - len(golden_examples))
            interactions = self.memory_manager.get_top_interactions_by_weight(top_n_adjusted)

            if not interactions and not golden_examples:
                logger.warning("No training data available, skipping training")
                return

            # Step 2: Select training mode
            logger.info("Step 2: Selecting training mode...")
            if requested_mode == "auto":
                stats = self.memory_manager.get_training_stats()
                selected_mode = self.scheduler._select_mode_from_stats(stats)
            elif requested_mode in ["sft", "dpo"]:
                selected_mode = requested_mode
            else:
                selected_mode = "sft"
            logger.info(f"Selected mode: {selected_mode}")

            original_enable_dpo = self.config.training.enable_dpo
            self.config.training.enable_dpo = (selected_mode == "dpo")

            # Step 3: Generate rejections for DPO
            rejections = {}
            rejections_cache = Path(settings.checkpoints_path) / "rejections_cache.json"
            if self.config.training.enable_dpo:
                logger.info("Step 3: Generating synthetic rejections for DPO...")

                # Invalidate cache if it's stale (older than 6h, from a previous crashed run)
                if rejections_cache.exists():
                    cache_age_hours = (time.time() - rejections_cache.stat().st_mtime) / 3600
                    if cache_age_hours > 6:
                        logger.warning(
                            f"Rejection cache is {cache_age_hours:.1f}h old (likely from a crashed run), discarding"
                        )
                        rejections_cache.unlink(missing_ok=True)

                # Try loading cached rejections first
                if rejections_cache.exists():
                    try:
                        with open(rejections_cache) as f:
                            rejections = json.load(f)
                        logger.info(f"Loaded {len(rejections)} cached rejections")
                    except Exception as e:
                        logger.warning(f"Failed to load rejection cache: {e}")
                        rejections = {}

                if not rejections:
                    from trainer.rejection_generator import RejectionGenerator

                    generator = RejectionGenerator(
                        ollama_url="http://llm-ollama:11434",
                        model_name=self.config.training.ollama_model_name,
                    )
                    all_prompts = []

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
                    for example in golden_examples:
                        all_prompts.append(example.user_message)

                    logger.info(f"Generating rejections for {len(all_prompts)} prompts...")
                    rejections = generator.generate_rejections(
                        all_prompts,
                        batch_size=self.config.training.dpo_batch_size,
                        timeout=self.config.training.dpo_rejection_timeout
                    )
                    logger.info(f"Rejections: {len(rejections)}/{len(all_prompts)} success")

                    # Cache rejections for retry on failure
                    if rejections:
                        try:
                            with open(rejections_cache, "w") as f:
                                json.dump(rejections, f)
                            logger.info(f"Cached {len(rejections)} rejections to {rejections_cache}")
                        except Exception as e:
                            logger.warning(f"Failed to cache rejections: {e}")

                if not rejections:
                    logger.warning("All rejections failed, falling back to SFT mode")
                    self.config.training.enable_dpo = False

            # Step 4: Build dataset
            logger.info("Step 4: Building dataset...")
            if self.config.training.enable_dpo and rejections:
                dataset = self.dataset_builder.build_dpo_dataset(interactions, golden_examples, rejections)
            else:
                dataset = self.dataset_builder.build_training_dataset(interactions, golden_examples)

            if not dataset:
                logger.warning("Empty dataset, skipping training")
                return

            # Step 5: Train/validation split
            logger.info("Step 5: Creating train/validation split...")
            train_data, val_data = self.dataset_builder.create_validation_split(
                dataset, self.config.training.validation_split
            )

            if self.config.training.enable_dpo:
                train_formatted = self.dataset_builder.format_for_dpo_training(train_data)
                val_formatted = self.dataset_builder.format_for_dpo_training(val_data)
            else:
                train_formatted = self.dataset_builder.format_for_training(train_data)
                val_formatted = self.dataset_builder.format_for_training(val_data)

            # Step 6: Train LoRA adapter
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir = f"{settings.checkpoints_path}/{checkpoint_id}"
            training_mode = "DPO" if self.config.training.enable_dpo else "SFT"
            logger.info(f"Step 6: Training LoRA adapter ({training_mode})...")

            adapter_path, train_metrics = self.lora_trainer.train(
                train_formatted, val_formatted, checkpoint_dir
            )
            logger.info(f"Training complete. Adapter: {adapter_path}")

            # Step 7: Evaluate checkpoint
            logger.info("Step 7: Evaluating checkpoint...")
            should_deploy, metadata = self.model_evaluator.evaluate_checkpoint(
                adapter_path, val_formatted, checkpoint_id,
                training_metrics=train_metrics
            )

            # Step 8: Deploy if validation passed
            if should_deploy and self.config.training.enable_ollama_deployment:
                logger.info("Step 8: Merging and deploying to Ollama...")
                try:
                    self._merge_and_deploy(adapter_path, checkpoint_id, checkpoint_dir, metadata)
                except Exception as e:
                    logger.error(f"Deployment failed: {e}", exc_info=True)
                    logger.info("Keeping current model; adapter saved for retry")
                    self._notify_api_reload()
            elif should_deploy:
                logger.info(f"Step 8: Adapter saved at {adapter_path} (Ollama deployment disabled)")
            else:
                logger.warning("Checkpoint failed validation, not deployed")
                self._notify_api_reload()
                self._cleanup_checkpoint_dir(checkpoint_dir)

            # Step 9: Cleanup
            self.memory_manager.mark_training_complete(training_mode=selected_mode)
            self.memory_manager.cleanup_old_memories()
            self._cleanup_old_checkpoints()
            if rejections_cache.exists():
                rejections_cache.unlink()
                logger.info("Cleared rejections cache after successful training")

            # Step 8: Cleanup old memories
            logger.info("Step 8: Cleaning up old memories...")
            self.memory_manager.cleanup_old_memories()
            self._cleanup_old_checkpoints()

            elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"TRAINING PIPELINE COMPLETE (mode: {selected_mode}, took {elapsed:.1f}s)")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            if 'checkpoint_dir' in locals():
                self._cleanup_checkpoint_dir(checkpoint_dir)
            self._notify_api_reload()

        finally:
            if 'original_enable_dpo' in locals():
                self.config.training.enable_dpo = original_enable_dpo
            gc.collect()

    def _merge_and_deploy(self, adapter_path, checkpoint_id, checkpoint_dir, metadata):
        """Deploy LoRA adapter to Ollama via adapter layering (no merge needed)."""
        from shared.ollama_deployer import OllamaDeployer

        # Deploy adapter directly to Ollama (layers adapter on base model)
        deployer = OllamaDeployer(ollama_base_url=self.config.llm_proxy.base_url)
        model_name = self.config.training.ollama_model_name
        base_model = self.config.model.ollama_base_model

        success = deployer.deploy_model(
            adapter_path=adapter_path,
            model_name=model_name,
            base_model_name=base_model,
            checkpoint_id=checkpoint_id,
        )

        if not success:
            raise RuntimeError("Ollama deployment failed")

        # Update metadata and config
        metadata.deployed = True
        self.model_evaluator._save_checkpoint_metadata(metadata)
        self.model_evaluator._set_current_checkpoint(checkpoint_id, metadata)
        deployer.update_config_pointer(model_name)

        # Notify API to resume inference
        self._notify_api_reload()

        # Rotate old checkpoints
        deployer.rotate_checkpoints(keep_count=self.config.training.max_checkpoint_count)

        logger.info(f"Deployed {model_name} from checkpoint {checkpoint_id}")

    def _notify_api_reload(self):
        """Notify API to reload model from Ollama."""
        try:
            response = requests.post(
                "http://api:8000/v1/model/reload",
                json={"checkpoint_id": None},
                timeout=60,
            )
            if response.status_code == 200:
                logger.info("API notified to reload model")
            else:
                logger.warning(f"API reload returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to notify API: {e}")

    def _cleanup_checkpoint_dir(self, checkpoint_dir: str):
        """Remove a failed checkpoint directory and its metadata."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_id = checkpoint_path.name

        if checkpoint_path.exists():
            # Preserve logs before deletion
            logs_dir = checkpoint_path / "logs"
            if logs_dir.exists():
                persistent_logs = Path(settings.checkpoints_path) / "logs" / checkpoint_id
                persistent_logs.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copytree(logs_dir, persistent_logs, dirs_exist_ok=True)
                    logger.info(f"Preserved training logs to {persistent_logs}")
                except Exception as e:
                    logger.warning(f"Failed to preserve logs: {e}")

            try:
                shutil.rmtree(checkpoint_path)
                logger.info(f"Cleaned up checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {checkpoint_dir}: {e}")

        metadata_file = Path(settings.checkpoints_path) / f"{checkpoint_id}_metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
            except Exception:
                pass

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint directories, preserving logs."""
        max_count = self.config.training.max_checkpoint_count
        checkpoints_dir = Path(settings.checkpoints_path)

        checkpoint_dirs = sorted(
            [d for d in checkpoints_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Read current checkpoint to always preserve it
        current_id = None
        current_file = checkpoints_dir / "current_checkpoint.json"
        if current_file.exists():
            try:
                current_id = json.loads(current_file.read_text()).get("checkpoint_id")
            except Exception:
                pass

        to_delete = []
        kept = 0
        for d in checkpoint_dirs:
            if d.name == current_id:
                continue  # Always keep current
            if kept < max_count:
                kept += 1
                continue
            to_delete.append(d)

        for d in to_delete:
            logger.info(f"Removing old checkpoint: {d.name}")
            # Preserve logs
            logs_dir = d / "logs"
            if logs_dir.exists():
                persistent_logs = checkpoints_dir / "logs" / d.name
                persistent_logs.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copytree(logs_dir, persistent_logs, dirs_exist_ok=True)
                except Exception:
                    pass
            shutil.rmtree(d, ignore_errors=True)

            # Clean up metadata
            meta = checkpoints_dir / f"{d.name}_metadata.json"
            if meta.exists():
                meta.unlink(missing_ok=True)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old checkpoints")

    def check_user_requested_training(self):
        """Fast-poll: run training if user explicitly requested it."""
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
        scheduler = BlockingScheduler()

        scheduler.add_job(
            self.check_and_train,
            trigger=IntervalTrigger(hours=1),
            id='training_check',
            name='Check training conditions',
            replace_existing=True,
        )
        scheduler.add_job(
            self.check_user_requested_training,
            trigger=IntervalTrigger(seconds=30),
            id='user_request_check',
            name='Check for user-requested training',
            replace_existing=True,
        )

        logger.info("Running initial training check...")
        self.check_and_train()
        logger.info("Training worker running. Hourly checks + 30s user-request poll.")

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Training worker stopped")


def main():
    logger.info("=" * 80)
    logger.info("LOCAL LLM TRAINING SERVICE (CPU-only)")
    logger.info("=" * 80)

    logger.info("Waiting 30 seconds for API service to initialize...")
    time.sleep(30)

    worker = TrainerWorker()
    worker.run()


if __name__ == "__main__":
    main()
