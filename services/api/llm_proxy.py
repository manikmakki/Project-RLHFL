"""
HTTP Proxy client for forwarding LLM requests to external services.

Supports both Ollama and OpenAI-compatible endpoints, with automatic endpoint
detection, request/response format conversion, streaming support, and error handling.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx

from shared.config import SystemConfig
from shared.models import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class LLMProxy:
    """
    HTTP proxy client for external LLM services.

    Forwards chat completion requests to external LLM endpoints (Ollama, OpenAI)
    while maintaining the same interface as LLMEngine for drop-in replacement.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize LLM proxy client.

        Args:
            config: System configuration containing llm_proxy settings
        """
        self.config = config
        self.base_url = config.llm_proxy.base_url
        self.model_name = config.llm_proxy.external_model_name
        self.api_key = config.llm_proxy.api_key
        self.endpoint_type = self._detect_endpoint_type()

        # Initialize HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=config.llm_proxy.connect_timeout_seconds,
                read=config.llm_proxy.read_timeout_seconds,
                write=config.llm_proxy.connect_timeout_seconds,
                pool=None  # No timeout for connection pool
            ),
            limits=httpx.Limits(
                max_connections=config.llm_proxy.max_connections,
                max_keepalive_connections=config.llm_proxy.max_keepalive_connections
            )
        )

        logger.info(
            f"LLM Proxy initialized: endpoint={self.endpoint_type}, "
            f"base_url={self.base_url}, model={self.model_name}"
        )

    def _detect_endpoint_type(self) -> str:
        """Detect endpoint type from URL or config."""
        if self.config.llm_proxy.endpoint_type != "auto":
            return self.config.llm_proxy.endpoint_type

        # Auto-detect from URL
        url_lower = self.base_url.lower()
        if "/api/chat" in url_lower or "/api/generate" in url_lower or "ollama" in url_lower:
            return "ollama"
        elif "/v1/chat/completions" in url_lower or "openai.com" in url_lower:
            return "openai"
        else:
            # Default to Ollama for local endpoints
            return "ollama"

    def _build_request_ollama(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build Ollama API request format."""
        ollama_messages = []
        for msg in messages:
            message_dict = {"role": msg.role.value}
            if isinstance(msg.content, list):
                # Extract text and base64 images from content blocks.
                # Ollama expects content as a string and images as a separate list.
                text_parts = []
                images = []
                for part in msg.content:
                    if not isinstance(part, dict):
                        text_parts.append(str(part))
                        continue
                    part_type = part.get("type", "")
                    if part_type == "text":
                        text_parts.append(part.get("text", ""))
                    elif part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # Strip data URI prefix if present (data:image/...;base64,<data>)
                        if url.startswith("data:") and ";base64," in url:
                            images.append(url.split(";base64,", 1)[1])
                        elif url:
                            images.append(url)
                message_dict["content"] = "\n".join(text_parts)
                if images:
                    message_dict["images"] = images
            else:
                message_dict["content"] = msg.get_text_content() or ""
            ollama_messages.append(message_dict)

        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }
        if top_k is not None:
            options["top_k"] = top_k

        request_payload = {
            "model": model_override or self.model_name,
            "messages": ollama_messages,
            "stream": stream,
            "options": options,
        }

        if tools:
            request_payload["tools"] = tools

        return request_payload

    def _build_request_openai(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build OpenAI API request format."""
        openai_messages = []
        for msg in messages:
            # Preserve multipart content arrays (images, files, etc.) as-is;
            # OpenAI's API natively supports List[content_block] for content.
            content = msg.content if isinstance(msg.content, list) else (msg.get_text_content() or "")
            message_dict = {
                "role": msg.role.value,
                "content": content,
            }
            if msg.tool_calls:
                message_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                message_dict["name"] = msg.name
            openai_messages.append(message_dict)

        request_payload = {
            "model": model_override or self.model_name,
            "messages": openai_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if tools:
            request_payload["tools"] = tools
            if tool_choice:
                request_payload["tool_choice"] = tool_choice

        return request_payload

    def _build_endpoint_url(self) -> str:
        """Build the full endpoint URL based on type."""
        if self.endpoint_type == "ollama":
            # Ollama uses /api/chat endpoint
            if not self.base_url.endswith("/api/chat"):
                return f"{self.base_url.rstrip('/')}/api/chat"
            return self.base_url
        else:  # openai
            # OpenAI uses /v1/chat/completions
            if not self.base_url.endswith("/v1/chat/completions"):
                return f"{self.base_url.rstrip('/')}/v1/chat/completions"
            return self.base_url

    async def _call_with_retry(self, request_payload: Dict[str, Any]) -> httpx.Response:
        """Call external LLM with retry on transient errors."""
        endpoint_url = self._build_endpoint_url()
        last_error = None

        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.endpoint_type == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.config.llm_proxy.max_retries + 1):
            try:
                response = await self.client.post(
                    endpoint_url,
                    json=request_payload,
                    headers=headers
                )

                # Check for HTTP errors
                response.raise_for_status()
                return response

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as e:
                last_error = e
                if attempt < self.config.llm_proxy.max_retries:
                    delay = self.config.llm_proxy.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"External LLM request failed (attempt {attempt + 1}/"
                        f"{self.config.llm_proxy.max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    break

            except httpx.HTTPStatusError as e:
                # Don't retry HTTP errors (4xx, 5xx from external LLM)
                logger.error(f"External LLM returned error {e.response.status_code}: {e.response.text}")
                raise

        # All retries failed
        if last_error:
            raise last_error

    def _parse_response_ollama(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama response format."""
        if "message" not in response_data:
            logger.error(f"Unexpected Ollama response format: {response_data}")
            return {
                "content": "",
                "tool_calls": None,
                "thinking": None,
                "finish_reason": "error"
            }

        message = response_data["message"]

        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
            "thinking": message.get("thinking"),  # Extended reasoning field
            "finish_reason": response_data.get("done_reason", "stop")
        }

    def _parse_response_openai(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI response format."""
        if "choices" not in response_data or not response_data["choices"]:
            logger.error(f"Unexpected OpenAI response format: {response_data}")
            return {
                "content": "",
                "tool_calls": None,
                "thinking": None,
                "finish_reason": "error"
            }

        choice = response_data["choices"][0]
        message = choice.get("message", {})

        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
            "thinking": message.get("reasoning") or message.get("thoughts"),  # Some providers use these fields
            "finish_reason": choice.get("finish_reason", "stop")
        }

    async def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate non-streaming completion from external LLM.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling (Ollama only, None = Ollama default)
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            tool_choice: Optional tool choice strategy
            model_override: Use a different model than the default

        Returns:
            Dict with content, tool_calls, thinking, and finish_reason
        """
        if self.endpoint_type == "ollama":
            request_payload = self._build_request_ollama(
                messages, temperature, top_p, top_k, max_tokens,
                tools, tool_choice, stream=False, model_override=model_override,
            )
        else:
            request_payload = self._build_request_openai(
                messages, temperature, top_p, top_k, max_tokens,
                tools, tool_choice, stream=False, model_override=model_override,
            )

        # Log raw request payload for debugging
        logger.debug(f"LLM request payload: {json.dumps(request_payload, default=str)[:2000]}")

        # Call external LLM with retry
        response = await self._call_with_retry(request_payload)
        response_data = response.json()

        # Log raw response for debugging
        logger.debug(f"LLM raw response: {json.dumps(response_data, default=str)[:2000]}")

        # Parse response based on endpoint type
        if self.endpoint_type == "ollama":
            return self._parse_response_ollama(response_data)
        else:  # openai
            return self._parse_response_openai(response_data)

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming completion from external LLM.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling (Ollama only, None = Ollama default)
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            tool_choice: Optional tool choice strategy
            model_override: Use a different model than the default

        Yields:
            Chunks in OpenAI streaming format
        """
        if self.endpoint_type == "ollama":
            request_payload = self._build_request_ollama(
                messages, temperature, top_p, top_k, max_tokens,
                tools, tool_choice, stream=True, model_override=model_override,
            )
        else:
            request_payload = self._build_request_openai(
                messages, temperature, top_p, top_k, max_tokens,
                tools, tool_choice, stream=True, model_override=model_override,
            )

        endpoint_url = self._build_endpoint_url()

        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.endpoint_type == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Stream from external LLM
        async with self.client.stream("POST", endpoint_url, json=request_payload, headers=headers) as response:
            response.raise_for_status()

            if self.endpoint_type == "ollama":
                # Ollama streams NDJSON (one JSON per line)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)

                        # Convert Ollama chunk to OpenAI format
                        if "message" in chunk:
                            message = chunk["message"]
                            content = message.get("content", "")

                            # Yield chunk in OpenAI format
                            yield {
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content} if content else {},
                                    "finish_reason": chunk.get("done_reason") if chunk.get("done") else None
                                }]
                            }

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Ollama streaming chunk: {e}")
                        continue

            else:  # openai
                # OpenAI streams SSE format (data: {...})
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Strip "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        # Yield chunk as-is (already in OpenAI format)
                        yield chunk

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse OpenAI streaming chunk: {e}")
                        continue

    async def unload_model(self):
        """
        Unload model from Ollama to free GPU vRAM.

        Sends a request with keep_alive: 0 to tell Ollama to unload the model.
        Only works with Ollama endpoints.
        """
        if self.endpoint_type != "ollama":
            logger.warning(f"Model unload not supported for {self.endpoint_type} endpoints")
            return

        try:
            endpoint_url = f"{self.base_url.rstrip('/')}/api/generate"
            request_payload = {
                "model": self.model_name,
                "prompt": "",  # Empty prompt
                "keep_alive": 0  # Unload immediately
            }

            response = await self.client.post(endpoint_url, json=request_payload, timeout=10.0)
            response.raise_for_status()

            logger.info(f"Ollama model {self.model_name} unloaded (keep_alive: 0)")

        except Exception as e:
            logger.error(f"Failed to unload Ollama model: {e}")
            raise

    async def load_model(self):
        """
        Load/reload model in Ollama.

        Sends a request with keep_alive: "5m" to tell Ollama to load the model
        and keep it in memory for 5 minutes.
        Only works with Ollama endpoints.
        """
        if self.endpoint_type != "ollama":
            logger.warning(f"Model load not supported for {self.endpoint_type} endpoints")
            return

        try:
            endpoint_url = f"{self.base_url.rstrip('/')}/api/generate"
            request_payload = {
                "model": self.model_name,
                "prompt": "",  # Empty prompt to trigger load
                "keep_alive": "5m"  # Keep loaded for 5 minutes
            }

            response = await self.client.post(endpoint_url, json=request_payload, timeout=30.0)
            response.raise_for_status()

            logger.info(f"Ollama model {self.model_name} loaded (keep_alive: 5m)")

        except Exception as e:
            logger.error(f"Failed to load Ollama model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if proxy is ready (always True for proxy)."""
        return True

    async def close(self):
        """Close HTTP client connection pool."""
        await self.client.aclose()
        logger.info("LLM Proxy closed")

    def __repr__(self) -> str:
        return f"LLMProxy(endpoint={self.endpoint_type}, base_url={self.base_url}, model={self.model_name})"
