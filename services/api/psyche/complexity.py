"""
Psyche Module - Complexity Classifier

Heuristic-based prompt complexity classifier that determines the refinement
turn budget and quality threshold. No model needed — uses message length,
keyword detection, question count, and tool presence.
"""

import re
import logging
from typing import Tuple, List, Optional, Dict, Any

from shared.config import SystemConfig

logger = logging.getLogger(__name__)

# Keywords that suggest higher complexity
COMPLEX_KEYWORDS = re.compile(
    r"\b(explain|analyze|compare|contrast|evaluate|discuss|describe in detail|"
    r"walk me through|break down|pros and cons|tradeoffs?|trade-offs?|"
    r"step by step|in depth|comprehensive|thorough)\b",
    re.IGNORECASE,
)

CODE_KEYWORDS = re.compile(
    r"\b(implement|write (?:a |the )?(?:code|function|class|script|program)|"
    r"refactor|debug|fix (?:the |this )?(?:bug|error|issue)|create (?:a )?(?:module|api|endpoint)|"
    r"code review|pull request)\b",
    re.IGNORECASE,
)

# Simple factual patterns (likely don't need refinement)
SIMPLE_PATTERNS = re.compile(
    r"^(?:what is|who is|when was|where is|how many|how much|define |what's |"
    r"translate |convert |calculate )",
    re.IGNORECASE,
)


class ComplexityClassifier:
    """Heuristic complexity classifier for refinement turn budget."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.bypass_chars = config.psyche.complexity_bypass_chars
        self.bypass_tools = config.psyche.refinement_bypass_tools
        self.max_turns = config.psyche.max_refinement_turns

    def classify(
        self,
        user_message_text: str,
        has_tools: bool = False,
    ) -> Tuple[int, float, str]:
        """
        Classify prompt complexity and return refinement parameters.

        Args:
            user_message_text: The user's message text
            has_tools: Whether the request includes tool definitions

        Returns:
            (turn_budget, quality_threshold, complexity_class)
            - turn_budget: 1 = bypass, 3 = moderate, 5 = complex
            - quality_threshold: score needed to accept response
            - complexity_class: "simple", "moderate", or "complex"
        """
        text = user_message_text.strip()
        text_len = len(text)

        # Bypass: tool calls (need exact structured output)
        if has_tools and self.bypass_tools:
            logger.debug("Complexity: simple (tools present, bypass)")
            return (1, 0.0, "simple")

        # Bypass: very short messages
        if text_len < self.bypass_chars:
            logger.debug(f"Complexity: simple (len={text_len} < {self.bypass_chars})")
            return (1, 0.0, "simple")

        # Bypass: simple factual questions
        if SIMPLE_PATTERNS.match(text) and text_len < 100:
            logger.debug("Complexity: simple (factual pattern)")
            return (1, 0.0, "simple")

        # Count question marks as a complexity signal
        question_count = text.count("?")

        # Check for complexity keywords
        has_complex = bool(COMPLEX_KEYWORDS.search(text))
        has_code = bool(CODE_KEYWORDS.search(text))

        # Scoring
        if has_complex or question_count >= 3 or text_len > 500:
            budget = min(5, self.max_turns)
            threshold = 0.8
            complexity = "complex"
        elif has_code or question_count >= 2 or text_len > 200:
            budget = min(3, self.max_turns)
            threshold = 0.75 if has_code else 0.7
            complexity = "moderate"
        else:
            budget = min(3, self.max_turns)
            threshold = 0.7
            complexity = "moderate"

        logger.debug(
            f"Complexity: {complexity} (len={text_len}, questions={question_count}, "
            f"complex_kw={has_complex}, code_kw={has_code}) → budget={budget}, threshold={threshold}"
        )

        return (budget, threshold, complexity)
