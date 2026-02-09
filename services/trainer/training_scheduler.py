import logging
from datetime import datetime
from shared.config import SystemConfig
from shared.models import TrainingStats

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Determine when training should be triggered based on multiple criteria."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.min_interactions = config.training.min_interactions_threshold
        self.inactivity_hours = config.training.inactivity_threshold_hours
        self.max_days_between = config.training.max_days_between_training
    
    def should_trigger_training(self, stats: TrainingStats) -> tuple[bool, str]:
        """
        Determine if training should be triggered.
        
        Returns:
            tuple[bool, str]: (should_train, reason)
        """
        
        # Trigger 1: User explicitly requested training
        if stats.user_requested_training:
            logger.info("Training trigger: User request")
            return True, "user_request"
        
        # # Trigger 2: Minimum interaction threshold met
        # if stats.new_interactions_since_last_training >= self.min_interactions:
        #     logger.info(
        #         f"Training trigger: Interaction threshold "
        #         f"({stats.new_interactions_since_last_training} >= {self.min_interactions})"
        #     )
        #     return True, "interaction_threshold"
        
        # Trigger 3: Inactivity period with some new data
        # (consolidate learnings during quiet periods)
        if stats.hours_since_last_interaction >= self.inactivity_hours:
            if stats.new_interactions_since_last_training >= 10:
                logger.info(
                    f"Training trigger: Inactivity period "
                    f"({stats.hours_since_last_interaction:.1f}h >= {self.inactivity_hours}h) "
                    f"with {stats.new_interactions_since_last_training} new interactions"
                )
                return True, "inactivity_consolidation"
        
        # # Trigger 4: Maximum time elapsed (prevent staleness)
        # if stats.days_since_last_training >= self.max_days_between:
        #     if stats.new_interactions_since_last_training > 0:
        #         logger.info(
        #             f"Training trigger: Maximum time elapsed "
        #             f"({stats.days_since_last_training:.1f} days >= {self.max_days_between} days)"
        #         )
        #         return True, "time_threshold"
        
        # No trigger conditions met
        return False, "none"
    
    def get_status_message(self, stats: TrainingStats) -> str:
        """Get a human-readable status message."""
        messages = []
        
        messages.append(f"Total interactions: {stats.total_interactions}")
        messages.append(
            f"New interactions since last training: {stats.new_interactions_since_last_training}"
        )
        messages.append(
            f"Hours since last interaction: {stats.hours_since_last_interaction:.1f}"
        )
        messages.append(
            f"Days since last training: {stats.days_since_last_training:.1f}"
        )
        
        # Check each trigger condition
        if stats.new_interactions_since_last_training >= self.min_interactions:
            messages.append(
                f"✓ Interaction threshold met "
                f"({stats.new_interactions_since_last_training}/{self.min_interactions})"
            )
        else:
            messages.append(
                f"  Interaction threshold: "
                f"{stats.new_interactions_since_last_training}/{self.min_interactions}"
            )
        
        if stats.hours_since_last_interaction >= self.inactivity_hours:
            messages.append(
                f"✓ Inactivity threshold met "
                f"({stats.hours_since_last_interaction:.1f}h/{self.inactivity_hours}h)"
            )
        
        if stats.days_since_last_training >= self.max_days_between:
            messages.append(
                f"✓ Time threshold met "
                f"({stats.days_since_last_training:.1f}/{self.max_days_between} days)"
            )
        
        return "\n".join(messages)
