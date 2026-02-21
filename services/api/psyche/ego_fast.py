"""
Psyche Module - Ego Fast (Per-Request Reasoning Layer)

Lightweight per-request reasoning that leverages the thinking model's native
chain-of-thought. Does NOT add outer multi-call loops.

Two functions:
1. enrich_system_prompt(): Inject think-plan-act-reflect template + reward context
2. parse_response(): Extract self-critique, reflection, and confidence from output
"""

import re
import logging
from typing import Optional, List, Dict, Any

from shared.config import SystemConfig
from shared.models import ChatMessage, MessageRole
from api.psyche.models import RewardContext, EgoParseResult

logger = logging.getLogger(__name__)


# Reasoning template injected into system prompt to encourage structured thinking
REASONING_TEMPLATE = (
    "When approaching complex tasks, use structured reasoning: "
    "(1) Think - analyze the request and identify key requirements, "
    "(2) Plan - outline your approach before acting, "
    "(3) Act - execute your plan, "
    "(4) Reflect - briefly assess whether your response fully addresses the request. "
    "If you notice gaps in your response, address them."
)

# Reward context templates
REWARD_IMPROVING = (
    "Your recent responses have been well-received. Continue the current approach."
)
REWARD_DECLINING = (
    "Recent responses have not fully met expectations. Be more careful and precise. "
    "Pay close attention to what the user is actually asking for."
)
REWARD_STABLE = ""  # No injection when stable

# Regex patterns for self-critique detection
REFLECTION_PATTERNS = [
    re.compile(r"(?:on reflection|upon reflection|thinking about it|actually|wait)", re.IGNORECASE),
    re.compile(r"(?:I should have|I could have|a better approach|let me reconsider)", re.IGNORECASE),
    re.compile(r"(?:correction|to clarify|more precisely|to be more accurate)", re.IGNORECASE),
]

CONFIDENCE_HIGH_PATTERNS = [
    re.compile(r"(?:I'm confident|definitely|certainly|clearly|this is correct)", re.IGNORECASE),
]

CONFIDENCE_LOW_PATTERNS = [
    re.compile(r"(?:I'm not sure|I think|perhaps|possibly|might be|unclear)", re.IGNORECASE),
    re.compile(r"(?:I'm uncertain|I don't know if|it's hard to say)", re.IGNORECASE),
]


class EgoFast:
    """
    Per-request reasoning layer.

    Enriches system prompts with structured reasoning templates and
    reward context from the Id Engine. Parses responses for self-critique
    and reflection markers.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.psyche_config = config.psyche
        self._enabled = config.psyche.ego_fast_enabled

        logger.info(
            f"EgoFast initialized: enabled={self._enabled}, "
            f"reasoning_prompt={self.psyche_config.ego_reasoning_prompt}, "
            f"parse_reflection={self.psyche_config.ego_parse_reflection}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.ego_fast_enabled

    async def enrich_system_prompt(
        self,
        messages: List[ChatMessage],
        reward_context: Optional[RewardContext] = None,
    ) -> List[ChatMessage]:
        """
        Enrich the system prompt with reasoning template and reward context.

        If a system message exists, appends to it. If none exists, prepends one.
        Returns a new list (does not mutate input).

        Args:
            messages: Conversation messages
            reward_context: Optional reward signals from the Id Engine
        """
        if not self.enabled:
            return messages

        fragments = []

        # Add reasoning template
        if self.psyche_config.ego_reasoning_prompt:
            fragments.append(REASONING_TEMPLATE)

        # Add reward context if significant
        if reward_context:
            threshold = self.psyche_config.id_reward_prompt_threshold
            if reward_context.trend == "improving" and reward_context.session_reward_mean > threshold:
                fragments.append(REWARD_IMPROVING)
            elif reward_context.trend == "declining" and reward_context.session_reward_mean < -threshold:
                fragments.append(REWARD_DECLINING)

        if not fragments:
            return messages

        injection = " ".join(fragments)
        messages = list(messages)

        # Find existing system message
        system_idx = next(
            (i for i, msg in enumerate(messages) if msg.role == MessageRole.SYSTEM),
            None,
        )

        if system_idx is not None:
            existing_content = messages[system_idx].get_text_content() or ""
            messages[system_idx] = ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"{existing_content}\n\n{injection}",
            )
        else:
            messages.insert(0, ChatMessage(
                role=MessageRole.SYSTEM,
                content=injection,
            ))

        return messages

    async def parse_response(self, response_dict: Dict[str, Any]) -> Optional[EgoParseResult]:
        """
        Parse the model's response for self-critique, reflection, and confidence.

        Uses regex heuristics -- no additional LLM call. Extracts metadata
        that can be stored alongside the interaction for training analysis.

        Args:
            response_dict: Raw response from LLMProxy.generate()

        Returns:
            EgoParseResult with extracted metadata, or None if parsing disabled
        """
        if not self.enabled or not self.psyche_config.ego_parse_reflection:
            return None

        content = response_dict.get("content") or ""
        thinking = response_dict.get("thinking")

        # Combine thinking + content for analysis
        full_text = f"{thinking or ''} {content}".strip()
        if not full_text:
            return None

        # Detect reflection
        has_reflection = False
        reflection_text = None
        for pattern in REFLECTION_PATTERNS:
            match = pattern.search(full_text)
            if match:
                has_reflection = True
                # Extract surrounding context (up to 200 chars after match)
                start = max(0, match.start() - 50)
                end = min(len(full_text), match.end() + 200)
                reflection_text = full_text[start:end].strip()
                break

        # Estimate confidence
        confidence = 0.5  # neutral default
        for pattern in CONFIDENCE_HIGH_PATTERNS:
            if pattern.search(full_text):
                confidence = min(confidence + 0.2, 1.0)
        for pattern in CONFIDENCE_LOW_PATTERNS:
            if pattern.search(full_text):
                confidence = max(confidence - 0.2, 0.0)

        return EgoParseResult(
            thinking=thinking,
            reflection=reflection_text if has_reflection else None,
            confidence_estimate=round(confidence, 2),
        )

    def is_healthy(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "enabled": self.enabled,
            "reasoning_prompt": self.psyche_config.ego_reasoning_prompt,
            "parse_reflection": self.psyche_config.ego_parse_reflection,
        }
