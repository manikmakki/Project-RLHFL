import json
import logging
import re
from datetime import datetime
from typing import Optional

import httpx

from shared.config import SystemConfig

logger = logging.getLogger(__name__)

# Matches chat template tokens from common formats:
#   ChatML:  <|im_start|>, <|im_end|>, <|endoftext|>
#   Llama3:  <|eot_id|>, <|start_header_id|>, <|end_header_id|>
#   Llama2:  [INST], [/INST]
#   Generic: <s>, </s>
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>|\[/?INST\]|</?s>")

_EVAL_PROMPT = """\
You are a rigorous, impartial training data evaluator. Your sole purpose is to assess \
conversation quality for LLM fine-tuning. Apply a high standard — most conversations \
should score 5 or below. Reserve scores above 7 for genuinely excellent responses. \
Reserve is_golden for explicit instructions, preferences, or corrections that should \
permanently shape model behavior. Do not reward fluency alone.

The conversation may appear in one of two formats:

Format A — 3-turn sliding window (preferred):
  User: <initial message>
  TamAGI: <assistant response>
  User: <follow-up message>

Format B — narrative summary (legacy):
  Conversation about: <topic summary>. Latest exchange: User asked '<user message>', \
TamAGI responded about <assistant response summary>.

For Format A, the follow-up is the strongest quality signal: a natural, engaged \
follow-up indicates a good response; a confused or topic-abandoning follow-up \
indicates a poor one. Weight sentiment and quality heavily toward this implicit feedback.
For Format B, extract user_message and assistant_response from the narrative. \
Leave user_followup empty.

For is_noise, use a strict structural definition — mark true ONLY if:
- There is no recognisable user message or assistant response after parsing
- The content is a raw tool call, system log, or error trace with no conversational exchange
Reserve quality scores of 0-2 for content that is harmful, wrong, or useless as training signal.
Casual, emotional, or brief conversations are NOT noise — they receive low quality scores instead.

Return ONLY valid JSON. No explanation, no markdown, no preamble.
Keep user_message, assistant_response, and user_followup to at most 300 characters each.

Conversation:
{content}

Return exactly this JSON structure:
{{
  "user_message": "<first 300 chars of the user's message>",
  "assistant_response": "<first 300 chars of TamAGI's response>",
  "user_followup": "<first 300 chars of the follow-up, or empty string>",
  "sentiment": <float -1.0 to 1.0; use follow-up tone as primary signal when available>,
  "quality": <integer 0-10; 10=exceptionally insightful, 5=adequate, 0=harmful or wrong>,
  "topics": ["<topic1>", "<topic2>"],
  "is_golden": <true ONLY if any turn contains an explicit behavioral instruction, preference, or meaningful correction>,
  "is_noise": <true ONLY if no recognisable conversation can be parsed; false for all real exchanges regardless of quality>,
  "reasoning": "<one sentence justification>"
}}"""


def _compute_weight(sentiment: float, quality: int, is_golden: bool) -> float:
    base = 1.0 + abs(sentiment) * 0.5
    quality_factor = (quality / 10.0) * 1.5 + 0.5
    golden_boost = 3.0 if is_golden else 1.0
    return base * quality_factor * golden_boost


class LLMEvaluator:
    """Evaluates conversation quality using the llama.cpp server.

    Pipeline:
      _sanitize()  — strip chat template tokens (prevents model from treating them as delimiters)
      _precheck()  — pure-Python structural filter (empty, too long, repetitive)
      _evaluate()  — single LLM call for full quality/sentiment/golden/noise analysis
    """

    def __init__(self, config: SystemConfig):
        self.base_url = config.llama_cpp.base_url.rstrip("/")
        self.timeout = config.llama_cpp.read_timeout_seconds

    # A real 3-turn conversation won't exceed ~6000 chars. Beyond that it's almost
    # certainly a looping or malformed response.
    _MAX_CONTENT_CHARS = 6000
    # Repetition window: if the same 300-char chunk appears twice in the first 3000
    # chars, the content is looping.
    _REPEAT_WINDOW = 300

    @staticmethod
    def _sanitize(content: str) -> str:
        """Strip chat template tokens and normalise artifacts from JSON storage."""
        cleaned = _SPECIAL_TOKEN_RE.sub("", content)
        # Doubled quotes appear when TamAGI stored quoted words inside a JSON string
        # (e.g. ""better"" → "better"). They look like JSON empty-string literals to
        # the evaluator model and can cause it to stall generating output.
        cleaned = re.sub(r'"{2,}', '"', cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _precheck(self, content: str) -> Optional[str]:
        """Zero-cost Python structural filter. Returns a reason string if noise, else None."""
        if not content.strip():
            return "empty content"
        if len(content) > self._MAX_CONTENT_CHARS:
            return f"content length {len(content)} chars suggests looping/malformed response"
        scan = content[:3000]
        seen: set[str] = set()
        for i in range(0, len(scan) - self._REPEAT_WINDOW, self._REPEAT_WINDOW // 2):
            chunk = scan[i : i + self._REPEAT_WINDOW]
            if chunk in seen:
                return "repetitive content pattern detected"
            seen.add(chunk)
        return None

    def evaluate(self, content: str) -> Optional[dict]:
        """Evaluate a single conversation. Returns rlhfl_* meta dict or None on failure."""
        content = self._sanitize(content)

        precheck_reason = self._precheck(content)
        if precheck_reason:
            logger.info(f"Pre-filter: {precheck_reason!r}")
            return self._noise_only_result(precheck_reason)

        prompt = _EVAL_PROMPT.format(content=content)
        raw = ""
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0)) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        # No response_format: llama.cpp grammar mode buffers ALL output until JSON
                        # closes; if max_tokens fires mid-JSON it returns empty string. Free-form
                        # output + _extract_json() is more robust.
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 2048,
                        # Disable KV cache reuse — a previous runaway request can leave the cache
                        # in a state that causes subsequent requests to return 0 tokens.
                        "cache_prompt": False,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                resp_json = response.json()
                raw = resp_json["choices"][0]["message"]["content"] or ""
                finish_reason = resp_json["choices"][0].get("finish_reason", "unknown")

            if finish_reason == "length":
                logger.warning(f"LLM hit max_tokens=2048 cap, response may be truncated ({len(raw)} chars)")
            logger.debug(f"LLM response ({len(raw)} chars, finish={finish_reason}): {raw[:300]!r}")

            parsed = json.loads(self._extract_json(raw))
            return self._build_evaluation(parsed)

        except json.JSONDecodeError as e:
            logger.warning(f"LLM evaluator returned invalid JSON: {e} | raw={raw[:500]!r}")
            return None
        except Exception as e:
            logger.warning(f"LLM evaluator call failed: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first complete {...} block out of free-form model output."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _noise_only_result(reason: str) -> dict:
        """Minimal rlhfl_* record written when the pre-filter fires."""
        return {
            "rlhfl_evaluated_at": datetime.utcnow().isoformat(),
            "rlhfl_sentiment": 0.0,
            "rlhfl_quality": 0,
            "rlhfl_weight": 0.0,
            "rlhfl_is_golden": False,
            "rlhfl_is_noise": True,
            "rlhfl_topics": [],
            "rlhfl_user_message": "",
            "rlhfl_assistant_response": "",
            "rlhfl_user_followup": "",
            "rlhfl_reasoning": f"[pre-filter] {reason}",
        }

    def _build_evaluation(self, parsed: dict) -> Optional[dict]:
        """Validate LLM output and build the meta.rlhfl_* update dict."""
        user_message = parsed.get("user_message", "").strip()[:400]
        assistant_response = parsed.get("assistant_response", "").strip()[:400]
        user_followup = parsed.get("user_followup", "").strip()[:400]

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
