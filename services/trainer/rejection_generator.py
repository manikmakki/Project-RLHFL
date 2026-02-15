"""
Generate synthetic rejected responses for DPO training using Ollama API.

This module generates lower-quality responses by querying the currently loaded
model (base GGUF or latest checkpoint) with degraded sampling parameters.
Rejections are generated in batches for efficiency and are NOT stored in the
vector database.
"""

import logging
import requests
from typing import List, Dict, Optional
from time import sleep

logger = logging.getLogger(__name__)


class RejectionGenerator:
    """Generates synthetic rejected responses via Ollama API."""

    def __init__(self, ollama_url: str = "http://api:8000"):
        """
        Initialize rejection generator.

        Args:
            ollama_url: Base URL for Ollama API (defaults to API service)
        """
        self.ollama_url = ollama_url
        self.model_name = None  # Will be fetched from current loaded model

    def generate_rejections(
        self,
        prompts: List[str],
        batch_size: int = 8,
        timeout: int = 300
    ) -> Dict[str, str]:
        """
        Generate rejected responses for a list of prompts.

        Uses currently loaded model with degraded sampling parameters to
        produce lower-quality responses that serve as negative examples
        for DPO training.

        Args:
            prompts: List of user prompts to generate rejections for
            batch_size: Number of prompts to process in parallel (currently sequential)
            timeout: Request timeout in seconds per prompt

        Returns:
            Dictionary mapping prompts to rejected responses
        """
        if not prompts:
            logger.warning("No prompts provided for rejection generation")
            return {}

        logger.info(f"Generating {len(prompts)} rejection samples...")
        logger.info("Using currently loaded model with degraded sampling parameters")

        # Get currently loaded model name
        self._detect_model()

        rejections = {}
        failed_count = 0

        # Process prompts (currently sequential for stability)
        for idx, prompt in enumerate(prompts, 1):
            try:
                rejection = self._generate_single(prompt, timeout)
                if rejection:
                    rejections[prompt] = rejection
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate rejection for prompt {idx}/{len(prompts)}")

                # Progress logging
                if idx % 10 == 0 or idx == len(prompts):
                    logger.info(f"Generated {idx}/{len(prompts)} rejections ({failed_count} failed)")

            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating rejection {idx}/{len(prompts)}: {e}")
                continue

        success_rate = (len(rejections) / len(prompts)) * 100 if prompts else 0
        logger.info(
            f"Rejection generation complete: {len(rejections)}/{len(prompts)} successful "
            f"({success_rate:.1f}%)"
        )

        return rejections

    def _detect_model(self) -> None:
        """Detect currently loaded model name from API."""
        try:
            # Try to get model info from API status endpoint
            response = requests.get(
                f"{self.ollama_url}/health",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # Model name might be in different fields depending on API version
                self.model_name = data.get("model") or "current"
                logger.info(f"Detected model for rejection generation: {self.model_name}")
            else:
                logger.warning(f"Could not detect model name, using 'current'")
                self.model_name = "current"

        except Exception as e:
            logger.warning(f"Error detecting model name: {e}, using 'current'")
            self.model_name = "current"

    def _generate_single(self, prompt: str, timeout: int) -> Optional[str]:
        """
        Generate a single rejected response using degraded sampling.

        Args:
            prompt: User prompt
            timeout: Request timeout in seconds

        Returns:
            Generated rejection text, or None if generation failed
        """
        try:
            # Call chat completion endpoint with degraded parameters
            response = requests.post(
                f"{self.ollama_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 1.5,      # High randomness (worse quality)
                    "top_p": 0.6,           # Reduced vocabulary diversity
                    "top_k": 20,            # Limit token choices
                    "frequency_penalty": 0.0,  # Allow repetition
                    "presence_penalty": 0.0,   # Allow repetition
                    "max_tokens": 200,      # Shorter responses than normal
                    "stream": False
                },
                timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                rejection = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if rejection and len(rejection.strip()) > 10:
                    return rejection.strip()
                else:
                    logger.warning(f"Generated rejection too short: {len(rejection)} chars")
                    return None
            else:
                logger.error(
                    f"Rejection generation failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Rejection generation timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error in rejection generation: {e}")
            return None
