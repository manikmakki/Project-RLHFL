import logging
from datetime import datetime
from shared.config import SystemConfig
from shared.models import TrainingStats

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Determine when training should be triggered and which mode to use."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def should_trigger_training(self, stats: TrainingStats) -> tuple[bool, str, str]:
        """
        Determine if training should be triggered and which mode to use.

        Trigger conditions (checked in priority order):
          1. Manual user request — always fires.
          2. Normal threshold — ≥ min_interactions_threshold new evaluated interactions
             since the last training run.
          3. Timeout — ≥ max_days_between_training days since last training AND
             at least 1 new evaluated interaction (avoids re-training on zero new data).

        Mode selection (applied to all automatic triggers):
          - DPO if new_negative_since_dpo >= dpo_trigger_threshold (corrects bad behaviour
            that has accumulated since the last DPO pass).
          - SFT otherwise (always the default path).

        Returns:
            (should_train, reason, mode)  where mode is "sft" | "dpo" | "auto"
        """
        min_threshold = self.config.training.min_interactions_threshold
        max_days = self.config.training.max_days_between_training

        # Priority 1: manual request
        if stats.user_requested_training:
            mode = self._select_mode(stats)
            logger.info(f"Training trigger: manual request (mode: {mode})")
            return True, "user_request", mode

        # Priority 2: normal threshold
        new = stats.new_interactions_since_last_training
        if new >= min_threshold:
            mode = self._select_mode(stats)
            logger.info(
                f"Training trigger: {new} new evaluated interactions "
                f">= threshold {min_threshold} (mode: {mode})"
            )
            return True, f"{new}_new_interactions", mode

        # Priority 3: timeout — only if there is at least one new signal
        if stats.days_since_last_training >= max_days:
            if new >= 1:
                mode = self._select_mode(stats)
                logger.info(
                    f"Training trigger: {stats.days_since_last_training:.1f} days since last "
                    f"training >= {max_days}-day timeout, {new} new interaction(s) (mode: {mode})"
                )
                return True, f"{max_days}d_timeout", mode
            else:
                logger.info(
                    f"Timeout ({stats.days_since_last_training:.1f}d >= {max_days}d) but "
                    f"0 new evaluated interactions — skipping training"
                )

        return False, "none", "none"

    def _select_mode(self, stats: TrainingStats) -> str:
        """
        Choose SFT or DPO based on how many new negative signals have accumulated
        since the last DPO training run. SFT is always the default.
        """
        dpo_threshold = self.config.training.dpo_trigger_threshold
        neg = stats.new_negative_since_dpo
        if neg >= dpo_threshold:
            logger.info(
                f"Mode: DPO — {neg} new negative signals since last DPO "
                f">= threshold {dpo_threshold}"
            )
            return "dpo"
        logger.info(
            f"Mode: SFT — {neg} new negative signals since last DPO "
            f"< threshold {dpo_threshold}"
        )
        return "sft"

    def get_status_message(self, stats: TrainingStats) -> str:
        """Human-readable summary of current trigger state."""
        min_threshold = self.config.training.min_interactions_threshold
        max_days = self.config.training.max_days_between_training
        dpo_threshold = self.config.training.dpo_trigger_threshold

        new = stats.new_interactions_since_last_training
        neg = stats.new_negative_since_dpo

        lines = [
            f"Total evaluated: {stats.total_interactions}",
            f"New since last training: {new}/{min_threshold} threshold"
            + (" ✓" if new >= min_threshold else ""),
            f"  └─ Positive: {stats.new_positive_count}, "
            f"Negative: {stats.new_negative_count}, "
            f"Neutral: {stats.new_neutral_count}, "
            f"Golden: {stats.new_golden_count}",
            f"Days since last training: {stats.days_since_last_training:.1f}/{max_days} max"
            + (" ✓ (timeout)" if stats.days_since_last_training >= max_days and new >= 1 else ""),
            "",
            "Mode selection:",
            f"  Negatives since last DPO: {neg}/{dpo_threshold} threshold"
            + (" → DPO" if neg >= dpo_threshold else " → SFT (default)"),
        ]
        return "\n".join(lines)
