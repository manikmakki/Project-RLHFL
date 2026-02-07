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

import logging
import time
import sys
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
    
    def _execute_training(self):
        """Execute the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
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
                logger.info("Step 5: Deploying checkpoint...")
                
                # Optionally merge adapter with base model
                # This would create a merged model that can be converted to GGUF
                # For now, we just track the adapter path
                
                logger.info(f"Checkpoint {checkpoint_id} deployed successfully")
                logger.info(f"Metrics: {metadata.metrics}")
            else:
                logger.warning("Checkpoint failed validation, not deployed")
                logger.info(f"Metrics: {metadata.metrics}")
            
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
        
        # Run an immediate check on startup
        logger.info("Running initial training check...")
        self.check_and_train()
        
        logger.info("Training worker running. Checks scheduled every hour.")
        
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
