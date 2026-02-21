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
    
    def should_trigger_training(self, stats: TrainingStats, pattern_analysis=None) -> tuple[bool, str, str]:
        """
        Determine if training should be triggered and which mode to use.

        Uses sentiment-based triggers to automatically select between SFT and DPO:
        - Negative sentiment accumulation → DPO (correct bad behavior)
        - Positive/golden accumulation → SFT (reinforce good behavior)
        - Inactivity consolidation → auto-select based on data

        Args:
            stats: Training statistics
            pattern_analysis: Optional PatternAnalysis from Ego Slow (psyche module).
                If provided, its recommendations are logged but do not override
                the existing trigger logic. Future versions may use this to
                adjust thresholds dynamically.

        Returns:
            tuple[bool, str, str]: (should_train, reason, mode)
            mode = "sft" | "dpo" | "auto"
        """
        # Log Ego Slow recommendations if available
        if pattern_analysis and pattern_analysis.recommendation_notes:
            logger.info(
                f"Ego Slow recommendations: {pattern_analysis.recommendation_notes}"
            )

        # Priority 1: User manual request (mode selected based on data)
        if stats.user_requested_training:
            mode = self._select_mode_from_stats(stats)
            logger.info(f"Training trigger: User request (mode: {mode})")
            return True, "user_request", mode

        # Priority 2: Negative sentiment + refusal accumulation → DPO to correct behavior
        # Refusals are high-value DPO signals (model refusing safe requests)
        dpo_threshold = self.config.training.dpo_trigger_threshold
        dpo_signals = stats.new_negative_since_dpo + stats.new_refusals_since_dpo
        if dpo_signals >= dpo_threshold:
            logger.info(
                f"Training trigger: {dpo_signals} DPO signals "
                f"({stats.new_negative_since_dpo} negative + {stats.new_refusals_since_dpo} refusals "
                f">= {dpo_threshold}) → DPO mode"
            )
            return True, f"{dpo_signals}_dpo_signals", "dpo"

        # Priority 3: Positive/golden accumulation → SFT to reinforce behavior
        new_positive_total = stats.new_positive_count + stats.new_golden_count
        sft_threshold = self.config.training.sft_trigger_threshold
        if new_positive_total >= sft_threshold:
            logger.info(
                f"Training trigger: {new_positive_total} new positive/golden examples "
                f"(>= {sft_threshold}) → SFT mode"
            )
            return True, f"{new_positive_total}_positive_examples", "sft"

        # Priority 4: Inactivity consolidation (train on whatever we have)
        if stats.hours_since_last_interaction >= self.inactivity_hours:
            if stats.new_interactions_since_last_training >= 10:
                mode = self._select_mode_from_stats(stats)
                logger.info(
                    f"Training trigger: Inactivity period "
                    f"({stats.hours_since_last_interaction:.1f}h >= {self.inactivity_hours}h) "
                    f"with {stats.new_interactions_since_last_training} new interactions "
                    f"(mode: {mode})"
                )
                return True, "inactivity_consolidation", mode

        # No trigger conditions met
        return False, "none", "none"

    def _select_mode_from_stats(self, stats: TrainingStats) -> str:
        """
        Select training mode based on sentiment distribution.

        Prefers DPO if sufficient negatives exist, otherwise SFT.

        Returns:
            "sft" or "dpo"
        """
        # DPO for correcting bad behavior (negatives + refusals)
        dpo_mode_threshold = self.config.training.dpo_mode_threshold
        dpo_signals = stats.new_negative_since_dpo + stats.new_refusals_since_dpo
        if dpo_signals >= dpo_mode_threshold:
            logger.info(
                f"Selecting DPO mode: {dpo_signals} signals to train out "
                f"({stats.new_negative_since_dpo} negative + {stats.new_refusals_since_dpo} refusals "
                f">= {dpo_mode_threshold})"
            )
            return "dpo"

        # SFT for reinforcing good behavior
        logger.info(f"Selecting SFT mode: reinforcing positive patterns")
        return "sft"
    
    def get_status_message(self, stats: TrainingStats) -> str:
        """Get a human-readable status message including sentiment breakdown."""
        messages = []

        messages.append(f"Total interactions: {stats.total_interactions}")
        messages.append(
            f"New interactions since last training: {stats.new_interactions_since_last_training}"
        )
        messages.append(
            f"  └─ Positive: {stats.new_positive_count}, "
            f"Negative: {stats.new_negative_count}, "
            f"Neutral: {stats.new_neutral_count}, "
            f"Golden: {stats.new_golden_count}"
        )
        messages.append(
            f"Hours since last interaction: {stats.hours_since_last_interaction:.1f}"
        )
        messages.append(
            f"Days since last training: {stats.days_since_last_training:.1f}"
        )

        # DPO-specific stats
        if stats.last_dpo_training_timestamp:
            messages.append(
                f"Days since last DPO training: {stats.days_since_dpo_training:.1f}"
            )
            messages.append(
                f"New negatives since DPO: {stats.new_negative_since_dpo}"
            )

        # Check sentiment-based trigger conditions
        messages.append("\nTrigger Status:")

        # DPO trigger (negatives + refusals)
        dpo_threshold = self.config.training.dpo_trigger_threshold
        dpo_signals = stats.new_negative_since_dpo + stats.new_refusals_since_dpo
        if dpo_signals >= dpo_threshold:
            messages.append(
                f"✓ DPO threshold met: {dpo_signals}/{dpo_threshold} "
                f"({stats.new_negative_since_dpo} negative + {stats.new_refusals_since_dpo} refusals)"
            )
        else:
            messages.append(
                f"  DPO threshold: {dpo_signals}/{dpo_threshold} "
                f"({stats.new_negative_since_dpo} negative + {stats.new_refusals_since_dpo} refusals)"
            )

        # SFT trigger
        new_positive_total = stats.new_positive_count + stats.new_golden_count
        sft_threshold = self.config.training.sft_trigger_threshold
        if new_positive_total >= sft_threshold:
            messages.append(
                f"✓ SFT threshold met: {new_positive_total}/{sft_threshold} positive/golden"
            )
        else:
            messages.append(
                f"  SFT threshold: {new_positive_total}/{sft_threshold} positive/golden"
            )

        # Inactivity trigger
        if stats.hours_since_last_interaction >= self.inactivity_hours:
            messages.append(
                f"✓ Inactivity threshold met: "
                f"{stats.hours_since_last_interaction:.1f}h/{self.inactivity_hours}h"
            )

        return "\n".join(messages)
