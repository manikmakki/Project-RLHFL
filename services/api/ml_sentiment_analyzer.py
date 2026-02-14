"""
ML-based sentiment analyzer using transformer models for nuanced feedback detection.

This module provides a machine learning-based alternative to regex-based sentiment analysis,
using pre-trained models to detect implicit sentiment in user feedback.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading ML dependencies unless needed
_transformers_available = False
_tokenizer = None
_model = None
_device = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _transformers_available = True
except ImportError:
    logger.warning("transformers or torch not available. ML sentiment analysis will be disabled.")


class MLSentimentAnalyzer:
    """
    ML-based sentiment analyzer using RoBERTa model.

    Uses cardiffnlp/twitter-roberta-base-sentiment-latest for nuanced sentiment detection.
    Runs on CPU to avoid conflicts with GPU-based inference/training.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the ML sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier. Defaults to RoBERTa sentiment model.
        """
        global _tokenizer, _model, _device

        if not _transformers_available:
            raise RuntimeError(
                "transformers and torch are required for ML sentiment analysis. "
                "Install with: pip install transformers torch"
            )

        # Lazy initialization - only load model once
        if _model is None:
            logger.info(f"Loading ML sentiment model: {model_name}")
            try:
                _tokenizer = AutoTokenizer.from_pretrained(model_name)
                _model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # Force CPU to avoid GPU conflicts with inference/training
                _device = torch.device("cpu")
                _model.to(_device)
                _model.eval()  # Set to evaluation mode

                logger.info(f"ML sentiment model loaded successfully on {_device}")
            except Exception as e:
                logger.error(f"Failed to load ML sentiment model: {e}")
                raise

        self.tokenizer = _tokenizer
        self.model = _model
        self.device = _device
        self.model_name = model_name

    def analyze(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
        conversation_continuing: bool = True
    ) -> float:
        """
        Analyze sentiment of user message in context of previous assistant response.

        Args:
            user_message: The user's message to analyze
            assistant_response: Previous assistant response (provides context)
            conversation_continuing: Whether the conversation is continuing

        Returns:
            Sentiment score from -1.0 (very negative) to +1.0 (very positive)
        """
        try:
            # Build context-aware input
            # Include previous assistant response to understand user's reaction
            if assistant_response and len(assistant_response) > 0:
                # Take first 200 chars of previous response for context
                context = assistant_response[:200]
                if len(assistant_response) > 200:
                    context += "..."
                text = f"Previous: {context} User reply: {user_message}"
            else:
                text = user_message

            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Extract sentiment scores
            # Model outputs: [negative, neutral, positive]
            neg, neu, pos = scores[0].tolist()

            # Map to -1.0 to +1.0 scale
            # Simple approach: positive - negative
            sentiment = pos - neg

            # Apply continuation baseline for truly neutral messages
            # This maintains the heuristic that continuing a conversation is slightly positive
            if abs(sentiment) < 0.1 and conversation_continuing:
                sentiment = 0.1

            logger.debug(
                f"ML sentiment analysis: neg={neg:.3f}, neu={neu:.3f}, pos={pos:.3f} -> {sentiment:.3f}"
            )

            return sentiment

        except Exception as e:
            logger.error(f"ML sentiment analysis failed: {e}")
            # Don't raise - allow fallback to regex-based analysis
            return 0.0

    def analyze_batch(
        self,
        messages: list[tuple[str, Optional[str]]],
        conversation_continuing: bool = True
    ) -> list[float]:
        """
        Analyze sentiment for multiple messages in batch (more efficient).

        Args:
            messages: List of (user_message, assistant_response) tuples
            conversation_continuing: Whether conversations are continuing

        Returns:
            List of sentiment scores
        """
        try:
            # Build context-aware texts
            texts = []
            for user_msg, asst_resp in messages:
                if asst_resp and len(asst_resp) > 0:
                    context = asst_resp[:200]
                    if len(asst_resp) > 200:
                        context += "..."
                    texts.append(f"Previous: {context} User reply: {user_msg}")
                else:
                    texts.append(user_msg)

            # Batch tokenization
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Process results
            sentiments = []
            for score_tensor in scores:
                neg, neu, pos = score_tensor.tolist()
                sentiment = pos - neg

                # Apply continuation baseline
                if abs(sentiment) < 0.1 and conversation_continuing:
                    sentiment = 0.1

                sentiments.append(sentiment)

            logger.debug(f"ML batch sentiment analysis: processed {len(messages)} messages")
            return sentiments

        except Exception as e:
            logger.error(f"ML batch sentiment analysis failed: {e}")
            # Return neutral scores as fallback
            return [0.0] * len(messages)

    @staticmethod
    def is_available() -> bool:
        """Check if ML sentiment analysis is available."""
        return _transformers_available

    def __repr__(self) -> str:
        return f"MLSentimentAnalyzer(model={self.model_name}, device={self.device})"
