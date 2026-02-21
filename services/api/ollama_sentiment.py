"""
Ollama-based Sentiment Analyzer

Uses a lightweight LLM via Ollama API to classify sentiment,
replacing the heavyweight RoBERTa ML model.
"""

import logging
import re
import httpx

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = (
    "Rate the sentiment of this user message on a scale from -1.0 (very negative/frustrated) "
    "to 1.0 (very positive/happy). Consider tone, word choice, and context. "
    "Reply with ONLY a single number between -1.0 and 1.0, nothing else.\n\n"
    "Message: {text}"
)


class OllamaSentimentAnalyzer:
    """Analyze sentiment via Ollama API using a lightweight model."""

    def __init__(self, ollama_url: str = "http://llm-ollama:11434", model: str = "qwen2.5:0.5b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=10.0)

    def analyze(
        self,
        user_message: str,
        assistant_response: str = "",
        conversation_continuing: bool = True,
    ) -> float:
        """
        Analyze sentiment of a user message via Ollama.

        Returns:
            float: Sentiment score from -1.0 to 1.0. Falls back to 0.0 on error.
        """
        try:
            prompt = SENTIMENT_PROMPT.format(text=user_message[:500])  # Truncate long messages

            response = self._client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10,
                    },
                },
                timeout=5.0,
            )

            if response.status_code != 200:
                logger.warning(f"Ollama sentiment returned {response.status_code}")
                return 0.0

            text = response.json().get("response", "").strip()

            # Parse float from response
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                score = float(match.group())
                score = max(-1.0, min(1.0, score))
                return score

            logger.warning(f"Could not parse sentiment from Ollama response: {text!r}")
            return 0.0

        except httpx.TimeoutException:
            logger.debug("Ollama sentiment timed out, using fallback")
            return 0.0
        except Exception as e:
            logger.warning(f"Ollama sentiment error: {e}")
            return 0.0

    def close(self):
        self._client.close()
