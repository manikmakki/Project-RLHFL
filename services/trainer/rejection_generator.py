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
from shared.config import load_config

logger = logging.getLogger(__name__)

class RejectionGenerator:
    """Generates synthetic rejected responses via Ollama API."""

    def __init__(self, ollama_url: str = "http://llm-api:8000", model_name: Optional[str] = None):
        model_name = model_name or load_config().model.model_id
        """
        Initialize rejection generator.

        Args:
            ollama_url: Base URL for Ollama API (defaults to API service)
            model_name: Model name for generation requests
        """
        self.ollama_url = ollama_url
        self.model_name = model_name

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
        logger.info(f"Using model '{self.model_name}' with degraded sampling parameters")

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
                    "temperature": 1.0,      # Higher randomness (worse quality)
                    "top_p": 0.7,           # Reduced vocabulary diversity
                    "max_tokens": 512,      # Enough for thinking + content
                    "stream": False
                },
                timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                thinking = message.get("thinking", "")

                # Handle content as array of blocks
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                elif not isinstance(content, str):
                    content = str(content) if content else ""

                rejection = content.strip()

                # If content is empty but thinking is present, the model
                # spent all its tokens on chain-of-thought — use thinking
                if not rejection and thinking:
                    rejection = thinking.strip()
                    logger.debug("Using thinking field as rejection (content was empty)")

                if len(rejection) > 10:
                    return rejection
                else:
                    logger.warning(f"Generated rejection too short ({len(rejection)} chars)")
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
