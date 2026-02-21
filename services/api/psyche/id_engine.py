"""
Psyche Module - Id Engine (Reward-Driven Engine)

Drives the agent toward immediate success by computing real-time reward
signals from the conversation. Wraps the existing SentimentAnalyzer to
track per-session reward trajectories.

The Id Engine:
1. Analyzes recent user messages for sentiment trend
2. Computes session-level reward trajectory (improving/declining/stable)
3. Generates system prompt fragments about recent performance
"""

import asyncio
import logging
from collections import deque
from typing import Optional, List, Dict, Any

from shared.config import SystemConfig
from shared.models import ChatMessage, MessageRole
from api.psyche.models import RewardContext

logger = logging.getLogger(__name__)


class IdEngine:
    """
    Reward-driven engine that produces real-time reward signals.

    Tracks per-session sentiment trajectory using a sliding window.
    Calls the existing SentimentAnalyzer (via Ollama) synchronously
    for each request (user accepted 1-5s latency).
    """

    def __init__(self, config: SystemConfig, sentiment_analyzer):
        """
        Args:
            config: System configuration
            sentiment_analyzer: Existing SentimentAnalyzer instance
        """
        self.config = config
        self.psyche_config = config.psyche
        self.analyzer = sentiment_analyzer
        self._enabled = config.psyche.id_enabled

        # Per-conversation session reward buffers
        # conversation_id -> deque of recent sentiment scores
        self._session_rewards: Dict[str, deque] = {}
        self._window_size = config.psyche.id_session_window

        logger.info(
            f"IdEngine initialized: enabled={self._enabled}, "
            f"session_window={self._window_size}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.id_enabled

    async def compute_reward_context(
        self,
        conversation_id: str,
        messages: List[ChatMessage],
    ) -> RewardContext:
        """
        Analyze the conversation to produce a real-time reward signal.

        Runs sentiment analysis on the most recent user messages in the
        conversation to detect the reward trajectory.

        Args:
            conversation_id: Conversation identifier
            messages: Full conversation messages

        Returns:
            RewardContext with sentiment, trend, and temperature adjustment
        """
        if not self.enabled:
            return RewardContext()

        # Extract recent user messages for sentiment analysis
        user_messages = [
            msg for msg in messages
            if msg.role == MessageRole.USER
        ]

        if not user_messages:
            return RewardContext()

        # Analyze the most recent user message
        latest_user = user_messages[-1]
        user_text = latest_user.get_text_content()
        if not user_text:
            return RewardContext()

        # Get the previous assistant response (for sentiment context)
        prev_assistant_text = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                prev_assistant_text = msg.get_text_content() or ""
                break

        # Run sentiment analysis (calls Ollama synchronously -- accepted latency)
        try:
            sentiment = await asyncio.to_thread(
                self.analyzer.analyze,
                user_message=user_text,
                assistant_response=prev_assistant_text,
                conversation_continuing=True,
            )
        except Exception as e:
            logger.error(f"Id Engine sentiment analysis failed: {e}")
            return RewardContext()

        # Update session buffer
        if conversation_id not in self._session_rewards:
            self._session_rewards[conversation_id] = deque(maxlen=self._window_size)
        self._session_rewards[conversation_id].append(sentiment)

        # Compute trend
        buffer = self._session_rewards[conversation_id]
        trend = self._compute_trend(buffer)
        session_mean = sum(buffer) / len(buffer) if buffer else 0.0

        # Suggest temperature adjustment based on trajectory
        temp_adjustment = 0.0
        if trend == "declining":
            temp_adjustment = -0.1  # More conservative when declining
        elif trend == "improving" and session_mean > 0.5:
            temp_adjustment = 0.05  # Slightly more creative when doing well

        reward_ctx = RewardContext(
            current_sentiment=round(sentiment, 3),
            trend=trend,
            session_reward_mean=round(session_mean, 3),
            suggested_temperature_adjustment=round(temp_adjustment, 3),
        )

        logger.debug(
            f"Id Engine reward context: sentiment={sentiment:.2f}, "
            f"trend={trend}, mean={session_mean:.2f}"
        )

        return reward_ctx

    async def record_outcome(
        self,
        conversation_id: str,
        response_text: str,
    ):
        """
        Called post-response to record the outcome.

        Currently a no-op since sentiment is computed from user messages
        (which arrive on the NEXT turn). This hook exists for future
        extension (e.g., self-evaluation of response quality).
        """
        pass

    def _compute_trend(self, buffer: deque) -> str:
        """
        Compute sentiment trend from the session buffer.

        Uses simple linear regression direction over the window.
        """
        if len(buffer) < 2:
            return "stable"

        values = list(buffer)
        n = len(values)

        # Simple linear trend: compare first half mean to second half mean
        mid = n // 2
        first_half = values[:mid] if mid > 0 else values[:1]
        second_half = values[mid:]

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        diff = second_mean - first_mean

        if diff > 0.15:
            return "improving"
        elif diff < -0.15:
            return "declining"
        return "stable"

    def cleanup_session(self, conversation_id: str):
        """Remove session buffer for a conversation (on cleanup/timeout)."""
        self._session_rewards.pop(conversation_id, None)

    def is_healthy(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "enabled": self.enabled,
            "active_sessions": len(self._session_rewards),
            "session_window": self._window_size,
        }
