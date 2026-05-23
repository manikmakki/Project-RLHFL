import json
import logging
from datetime import datetime
from typing import Optional

import httpx

from shared.config import SystemConfig

logger = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are a rigorous, impartial training data evaluator. Your sole purpose is to assess \
conversation quality for LLM fine-tuning. Apply a high standard — most conversations \
should score 5 or below. Reserve scores above 7 for genuinely excellent responses. \
Reserve is_golden for explicit instructions, preferences, or corrections that should \
permanently shape model behavior. Do not reward fluency alone.

The conversation is a 3-turn sliding window in this format:
  User: <initial message>
  TamAGI: <assistant response>
  User: <follow-up message>

The follow-up message is the strongest signal you have. A natural, engaged, on-topic \
follow-up indicates the assistant response was helpful. A confused, frustrated, or \
topic-abandoning follow-up indicates a poor response. Weight sentiment and quality \
heavily toward this implicit feedback.

Parse the conversation below and return ONLY valid JSON matching the schema. \
Do not include explanation or markdown — raw JSON only.

Conversation:
{content}

Return exactly this JSON structure:
{{
  "user_message": "<text of the first User turn only>",
  "assistant_response": "<text of the TamAGI turn only>",
  "user_followup": "<text of the second User turn only>",
  "sentiment": <float -1.0 to 1.0; derive primarily from the tone and content of the follow-up>,
  "quality": <integer 0-10; 10=exceptionally insightful, 5=adequate, 0=harmful or wrong; factor in whether the follow-up shows the response landed well>,
  "topics": ["<topic1>", "<topic2>"],
  "is_golden": <true ONLY if any turn contains an explicit behavioral instruction, preference, or meaningful correction the model should permanently remember>,
  "is_noise": <true if the exchange is trivial, too short, or too ambiguous to train on>,
  "reasoning": "<one sentence justification, referencing the follow-up where relevant>"
}}"""


def _compute_weight(sentiment: float, quality: int, is_golden: bool) -> float:
    base = 1.0 + abs(sentiment) * 0.5
    quality_factor = (quality / 10.0) * 1.5 + 0.5
    golden_boost = 3.0 if is_golden else 1.0
    return base * quality_factor * golden_boost


class LLMEvaluator:
    """Evaluates conversation quality using the llama.cpp server with structured JSON output."""

    def __init__(self, config: SystemConfig):
        self.base_url = config.llama_cpp.base_url.rstrip("/")
        self.timeout = config.llama_cpp.read_timeout_seconds

    def evaluate(self, content: str) -> Optional[dict]:
        """
        Evaluate a single conversation narrative from data.content.

        Returns a dict of rlhfl_* fields ready to be written to meta, or None on failure.
        """
        prompt = _EVAL_PROMPT.format(content=content)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]

            parsed = json.loads(raw)
            return self._build_evaluation(parsed)

        except json.JSONDecodeError as e:
            logger.warning(f"LLM evaluator returned invalid JSON: {e} | raw={raw[:200]!r}")
            return None
        except Exception as e:
            logger.warning(f"LLM evaluator call failed: {e}")
            return None

    def _build_evaluation(self, parsed: dict) -> Optional[dict]:
        """Validate the parsed LLM output and build the meta.rlhfl_* update dict."""
        user_message = parsed.get("user_message", "").strip()
        assistant_response = parsed.get("assistant_response", "").strip()
        user_followup = parsed.get("user_followup", "").strip()

        if not user_message or not assistant_response:
            logger.warning("Evaluator output missing user_message or assistant_response, skipping")
            return None

        try:
            sentiment = float(parsed["sentiment"])
            quality = int(parsed["quality"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Evaluator output has invalid sentiment/quality: {e}")
            return None

        sentiment = max(-1.0, min(1.0, sentiment))
        quality = max(0, min(10, quality))

        is_golden = bool(parsed.get("is_golden", False))
        is_noise = bool(parsed.get("is_noise", False))
        topics = [str(t) for t in parsed.get("topics", []) if t]
        reasoning = str(parsed.get("reasoning", ""))[:500]

        weight = _compute_weight(sentiment, quality, is_golden)

        return {
            "rlhfl_evaluated_at": datetime.utcnow().isoformat(),
            "rlhfl_sentiment": round(sentiment, 4),
            "rlhfl_quality": quality,
            "rlhfl_weight": round(weight, 4),
            "rlhfl_is_golden": is_golden,
            "rlhfl_is_noise": is_noise,
            "rlhfl_topics": topics,
            "rlhfl_user_message": user_message,
            "rlhfl_assistant_response": assistant_response,
            "rlhfl_user_followup": user_followup,
            "rlhfl_reasoning": reasoning,
        }
