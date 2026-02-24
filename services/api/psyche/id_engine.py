"""
Psyche Module - Id Engine (Reward-Driven Engine)

Drives the agent toward immediate success by computing real-time reward
signals from the conversation. Wraps the existing SentimentAnalyzer to
track per-session reward trajectories.

In Psyche v2, also generates a natural-language "desire signal" via the
Id model, which is injected into the Ego's system prompt to guide response
generation toward user satisfaction.
"""

import asyncio
import logging
from collections import deque
from typing import Optional, List, Dict, Any

from shared.config import SystemConfig
from shared.models import ChatMessage, MessageRole
from api.psyche.models import RewardContext

logger = logging.getLogger(__name__)

# Desire signal prompt for the Id model
DESIRE_SYSTEM_PROMPT = """You analyze user messages to identify what the user truly wants from this interaction. Focus on:
- Emotional tone and urgency
- Implicit expectations (directness, detail level, creativity)
- Whether they want information, action, empathy, or validation

Respond in 1-2 concise sentences describing the user's core desire and how the assistant should respond to maximize satisfaction. Be specific and actionable."""


class IdEngine:
    """
    Reward-driven engine that produces real-time reward signals and desire framing.

    Tracks per-session sentiment trajectory using a sliding window.
    Optionally calls a small Id model to generate natural-language desire signals.
    """

    def __init__(self, config: SystemConfig, sentiment_analyzer, llm_proxy=None):
        self.config = config
        self.psyche_config = config.psyche
        self.analyzer = sentiment_analyzer
        self.llm_proxy = llm_proxy
        self._enabled = config.psyche.id_enabled

        self._session_rewards: Dict[str, deque] = {}
        self._window_size = config.psyche.id_session_window

        logger.info(
            f"IdEngine initialized: enabled={self._enabled}, "
            f"session_window={self._window_size}, "
            f"has_llm_proxy={'yes' if llm_proxy else 'no'}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.id_enabled

    async def compute_reward_context(
        self,
        conversation_id: str,
        messages: List[ChatMessage],
        id_temperature: float = 0.7,
    ) -> RewardContext:
        """
        Analyze the conversation to produce a reward signal and desire framing.

        Args:
            conversation_id: Conversation identifier
            messages: Full conversation messages
            id_temperature: Temperature for Id model (controlled by Superego)

        Returns:
            RewardContext with sentiment, trend, and desire_signal
        """
        if not self.enabled:
            return RewardContext()

        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        if not user_messages:
            return RewardContext()

        latest_user = user_messages[-1]
        user_text = latest_user.get_text_content()
        if not user_text:
            return RewardContext()

        prev_assistant_text = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                prev_assistant_text = msg.get_text_content() or ""
                break

        # Run sentiment analysis
        try:
            sentiment = await asyncio.to_thread(
                self.analyzer.analyze,
                user_message=user_text,
                assistant_response=prev_assistant_text,
                conversation_continuing=True,
            )
        except Exception as e:
            logger.error(f"Id Engine sentiment analysis failed: {e}")
            sentiment = 0.0

        # Update session buffer
        if conversation_id not in self._session_rewards:
            self._session_rewards[conversation_id] = deque(maxlen=self._window_size)
        self._session_rewards[conversation_id].append(sentiment)

        buffer = self._session_rewards[conversation_id]
        trend = self._compute_trend(buffer)
        session_mean = sum(buffer) / len(buffer) if buffer else 0.0

        temp_adjustment = 0.0
        if trend == "declining":
            temp_adjustment = -0.1
        elif trend == "improving" and session_mean > 0.5:
            temp_adjustment = 0.05

        # Generate desire signal via Id model (if available)
        desire_signal = await self._compute_desire_signal(
            user_text, prev_assistant_text, sentiment, trend, id_temperature
        )

        reward_ctx = RewardContext(
            current_sentiment=round(sentiment, 3),
            trend=trend,
            session_reward_mean=round(session_mean, 3),
            suggested_temperature_adjustment=round(temp_adjustment, 3),
            desire_signal=desire_signal,
        )

        logger.debug(
            f"Id Engine: sentiment={sentiment:.2f}, trend={trend}, "
            f"desire={'yes' if desire_signal else 'heuristic'}"
        )

        return reward_ctx

    async def _compute_desire_signal(
        self,
        user_text: str,
        prev_assistant_text: str,
        sentiment: float,
        trend: str,
        temperature: float,
    ) -> Optional[str]:
        """
        Generate a natural-language desire signal via the Id model.

        Falls back to heuristic if no Id model is configured.
        """
        id_model = self.psyche_config.id_model

        # If we have an Id model and LLM proxy, use the model
        if id_model and self.llm_proxy:
            try:
                context = f"User message: {user_text[:500]}"
                if prev_assistant_text:
                    context += f"\n\nPrevious assistant response (first 300 chars): {prev_assistant_text[:300]}"
                context += f"\n\nUser sentiment score: {sentiment:.2f} (trend: {trend})"

                messages = [
                    ChatMessage(role=MessageRole.SYSTEM, content=DESIRE_SYSTEM_PROMPT),
                    ChatMessage(role=MessageRole.USER, content=context),
                ]

                response = await self.llm_proxy.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=80,
                    model_override=id_model,
                )

                desire = (response.get("content") or "").strip()
                if desire:
                    return desire

            except Exception as e:
                logger.warning(f"Id model desire signal failed, using heuristic: {e}")

        # Heuristic fallback
        return self._heuristic_desire(sentiment, trend)

    def _heuristic_desire(self, sentiment: float, trend: str) -> str:
        """Generate a simple heuristic desire signal from sentiment data."""
        if trend == "declining":
            return (
                "User satisfaction is declining. Prioritize directness and precision. "
                "Avoid filler, hedging, or unnecessary caveats. Address their request head-on."
            )
        elif trend == "improving" and sentiment > 0.5:
            return (
                "User is engaged and satisfied. Maintain current approach — "
                "be thorough and responsive while keeping momentum."
            )
        elif sentiment < -0.2:
            return (
                "User appears frustrated. Be concise, accurate, and solution-focused. "
                "Skip preamble and get straight to the answer."
            )
        else:
            return (
                "User sentiment is neutral. Provide a clear, well-structured response "
                "that directly addresses their request."
            )

    async def record_outcome(self, conversation_id: str, response_text: str):
        """Called post-response. Currently a no-op (sentiment comes from next user turn)."""
        pass

    def _compute_trend(self, buffer: deque) -> str:
        if len(buffer) < 2:
            return "stable"
        values = list(buffer)
        mid = len(values) // 2
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
        self._session_rewards.pop(conversation_id, None)

    def is_healthy(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active_sessions": len(self._session_rewards),
            "session_window": self._window_size,
            "id_model": self.psyche_config.id_model or "(heuristic)",
        }
