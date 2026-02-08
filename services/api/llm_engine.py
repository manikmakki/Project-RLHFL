import logging
import json
import uuid
import gc
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_cpp import Llama
from shared.models import ChatMessage, MessageRole
from shared.config import SystemConfig
import jinja2


logger = logging.getLogger(__name__)

# Mistral Instruct v0.3 chat template with native tool calling support.
# This replaces chatml-function-calling which uses the wrong token format for Mistral.
MISTRAL_V3_CHAT_TEMPLATE = (
    "{%- if messages[0][\"role\"] == \"system\" %}"
    "{%- set system_message = messages[0][\"content\"] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{%- if not tools is defined %}"
    "{%- set tools = none %}"
    "{%- endif %}"
    "{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}"
    "{{- bos_token }}"
    "{%- for message in loop_messages %}"
    "{%- if message[\"role\"] == \"user\" %}"
    "{%- if tools is not none and (message == user_messages[-1]) %}"
    "{{- \"[AVAILABLE_TOOLS] [\" }}"
    "{%- for tool in tools %}"
    "{%- set tool = tool.function %}"
    "{{- '{\"type\": \"function\", \"function\": {' }}"
    "{%- for key, val in tool.items() if key != \"return\" %}"
    "{%- if val is string %}"
    "{{- '\"' + key + '\": \"' + val + '\"' }}"
    "{%- else %}"
    "{{- '\"' + key + '\": ' + val|tojson }}"
    "{%- endif %}"
    "{%- if not loop.last %}"
    "{{- \", \" }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{{- \"}}\" }}"
    "{%- if not loop.last %}"
    "{{- \", \" }}"
    "{%- else %}"
    "{{- \"]\" }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{{- \"[/AVAILABLE_TOOLS]\" }}"
    "{%- endif %}"
    "{%- if loop.last and system_message is defined %}"
    "{{- \"[INST] \" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}"
    "{%- else %}"
    "{{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}"
    "{%- endif %}"
    "{%- elif message.tool_calls is defined and message.tool_calls is not none %}"
    "{{- \"[TOOL_CALLS] [\" }}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- set out = tool_call.function|tojson %}"
    "{{- out[:-1] }}"
    "{%- if not tool_call.id is defined or tool_call.id|length != 9 %}"
    "{{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}"
    "{%- endif %}"
    "{{- ', \"id\": \"' + tool_call.id + '\"}' }}"
    "{%- if not loop.last %}"
    "{{- \", \" }}"
    "{%- else %}"
    "{{- \"]\" + eos_token }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- elif message[\"role\"] == \"assistant\" %}"
    "{{- \" \" + message[\"content\"]|trim + eos_token}}"
    "{%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}"
    "{%- if message.content is defined and message.content.content is defined %}"
    "{%- set content = message.content.content %}"
    "{%- else %}"
    "{%- set content = message.content %}"
    "{%- endif %}"
    "{{- '[TOOL_RESULTS] {\"content\": ' + content|string + \", \" }}"
    "{%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}"
    "{{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}"
    "{%- endif %}"
    "{{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}"
    "{%- elif message[\"role\"] == \"system\" %}"
    "{%- endif %}"
    "{%- endfor %}"
)


def _raise_template_exception(message):
    """Helper for Jinja2 template raise_exception calls."""
    raise jinja2.exceptions.TemplateError(message)


class LLMEngine:
    """LLM inference engine using llama.cpp."""

    def __init__(self, config: SystemConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        self.current_adapter_path: Optional[str] = None
        self.current_checkpoint_id: Optional[str] = None
        self.llm: Optional[Llama] = None
        self._model_lock = threading.RLock()
        self._is_reloading = False

        # Serialize inference calls — only one create_completion at a time.
        # Concurrent calls would double GPU KV cache allocation and OOM.
        self._inference_lock = threading.Lock()

        # Initialize Jinja2 environment with Mistral v0.3 chat template
        env = jinja2.Environment()
        env.filters['tojson'] = lambda val: json.dumps(val, ensure_ascii=False)
        env.globals['raise_exception'] = _raise_template_exception
        self._chat_template = env.from_string(MISTRAL_V3_CHAT_TEMPLATE)

        self._load_model()

    def _load_model(self):
        """Load the GGUF model with llama.cpp."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.model.context_length,
                n_gpu_layers=self.config.model.n_gpu_layers,
                n_batch=self.config.model.n_batch,
                n_threads=self.config.model.n_threads,
                kv_cache_type=self.config.model.kv_cache_type,
                use_mmap=self.config.model.use_mmap,
                use_mlock=self.config.model.use_mlock,
                verbose=True  # Temporarily enable to check GPU offloading
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def reload_model(self, new_model_path: str, checkpoint_id: str = None) -> bool:
        """
        Reload the model from a new GGUF file with automatic rollback on failure.

        Strategy: unload current model first to free GPU memory (two 7B models
        cannot coexist), then load the new one. If the new load fails, attempt
        to restore the previous model.

        During the brief reload window (~5-15s), self.llm is None and requests
        will get RuntimeError("Model not loaded") → HTTP 503.

        Args:
            new_model_path: Absolute path to the new GGUF file.
            checkpoint_id: Optional checkpoint identifier for tracking.

        Returns:
            True if reload succeeded, False if failed (previous model restored).
        """
        new_path = Path(new_model_path)

        if not new_path.exists():
            logger.error(f"Reload failed: GGUF file not found: {new_model_path}")
            return False

        if new_path.stat().st_size < 100_000_000:
            logger.error(
                f"Reload failed: GGUF file suspiciously small "
                f"({new_path.stat().st_size} bytes): {new_model_path}"
            )
            return False

        if new_model_path == self.model_path and self.llm is not None:
            logger.info("Reload skipped: same model path already loaded")
            return True

        logger.info(f"Model reload initiated: {self.model_path} -> {new_model_path}")

        with self._model_lock:
            self._is_reloading = True
            old_model_path = self.model_path
            old_checkpoint_id = self.current_checkpoint_id

            try:
                # Step 1: Unload current model to free GPU memory
                logger.info("Unloading current model to free GPU memory...")
                if self.llm is not None:
                    del self.llm
                    self.llm = None
                    gc.collect()

                # Step 2: Load new model
                logger.info(f"Loading new model from {new_model_path}")
                self.model_path = new_model_path
                self._load_model()

                # Step 3: Update tracking
                self.current_checkpoint_id = checkpoint_id
                self._is_reloading = False

                logger.info(
                    f"Model reload successful: now serving {new_model_path} "
                    f"(checkpoint: {checkpoint_id})"
                )
                return True

            except Exception as e:
                logger.error(f"Model reload FAILED: {e}", exc_info=True)

                # Recovery: try to reload the previous model
                logger.info(f"Attempting recovery: reloading previous model {old_model_path}")
                try:
                    self.model_path = old_model_path
                    self._load_model()
                    self.current_checkpoint_id = old_checkpoint_id
                    logger.info("Recovery successful: previous model restored")
                except Exception as recovery_err:
                    logger.critical(
                        f"CRITICAL: Recovery failed, no model available: {recovery_err}"
                    )
                    self.llm = None

                self._is_reloading = False
                return False

    def _convert_messages_to_dict(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert ChatMessage Pydantic objects to dict format for template rendering.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of message dicts compatible with Mistral v0.3 chat template
        """
        result = []
        for msg in messages:
            msg_dict = {
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content is not None else ""
            }

            # Add optional name field
            if msg.name:
                msg_dict["name"] = msg.name

            # Add tool_calls if present (for assistant messages)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    tc.dict() if hasattr(tc, 'dict') else tc
                    for tc in msg.tool_calls
                ]

            # Add tool_call_id if present (for tool messages)
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id

            result.append(msg_dict)
            logger.debug(f"Converted message: {msg_dict}")

        return result

    def _render_chat_prompt(
        self,
        messages_dict: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Render messages into a prompt string using the Mistral v0.3 chat template.

        Args:
            messages_dict: List of message dicts
            tools: Optional list of tool definitions (OpenAI format)

        Returns:
            Rendered prompt string ready for create_completion
        """
        render_kwargs = {
            "messages": messages_dict,
            "bos_token": "",  # Let llama.cpp handle BOS token
            "eos_token": "</s>",
        }
        if tools:
            render_kwargs["tools"] = tools

        rendered = self._chat_template.render(**render_kwargs)
        logger.debug(f"Rendered prompt ({len(rendered)} chars): ...{rendered[-200:]}")
        return rendered

    def _parse_tool_calls(self, text: str, tool_names: Optional[List[str]] = None):
        """
        Parse tool calls from model output.

        Handles two cases:
        1. Explicit [TOOL_CALLS] prefix (text token) followed by JSON array
        2. Output starts with a JSON array whose "name" fields match known
           tool names — [TOOL_CALLS] is a special/control token in Mistral's
           tokenizer so create_completion may omit it from text output.

        Args:
            text: Raw model output text
            tool_names: List of known tool function names (for heuristic detection)

        Returns:
            tuple: (content_text_or_None, tool_calls_list_or_None)
        """
        if not text:
            return None, None

        tool_json = None
        content = None

        if "[TOOL_CALLS]" in text:
            # Case 1: explicit [TOOL_CALLS] text token present
            parts = text.split("[TOOL_CALLS]", 1)
            content = parts[0].strip() or None
            tool_json = parts[1].strip()
            logger.debug("Detected tool calls via [TOOL_CALLS] prefix")
        else:
            # Case 2: [TOOL_CALLS] is a special token not in text output.
            # Check if output starts with a JSON array that looks like tool calls.
            stripped = text.strip()
            if stripped.startswith("["):
                # Find the end of the JSON array (first unbalanced ']')
                bracket_depth = 0
                end_idx = None
                for i, ch in enumerate(stripped):
                    if ch == "[":
                        bracket_depth += 1
                    elif ch == "]":
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            end_idx = i + 1
                            break

                if end_idx:
                    candidate = stripped[:end_idx]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list) and len(parsed) > 0 and "name" in parsed[0]:
                            # Verify the name matches a known tool
                            if not tool_names or parsed[0]["name"] in tool_names:
                                tool_json = candidate
                                content = None
                                logger.debug("Detected tool calls via JSON array heuristic (special token omitted)")
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass

        if tool_json is None:
            return text.strip(), None

        # Clean up EOS tokens and trailing whitespace
        tool_json = tool_json.replace("</s>", "").strip()

        try:
            raw_calls = json.loads(tool_json)
            if not isinstance(raw_calls, list):
                raw_calls = [raw_calls]

            tool_calls = []
            for tc in raw_calls:
                # Use model-generated ID (9 chars) or generate one
                call_id = tc.get("id", uuid.uuid4().hex[:9])
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {}))
                    }
                })

            logger.debug(f"Parsed {len(tool_calls)} tool call(s)")
            return content, tool_calls

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse tool calls: {e}, raw: {tool_json!r}")
            return text.strip(), None

    def generate(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.

        Uses the Mistral v0.3 chat template for prompt rendering and
        create_completion for inference, with post-processing to parse
        [TOOL_CALLS] from the raw output.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            tools: List of tool/function definitions (OpenAI format)
            tool_choice: Tool choice control ("auto", "none", or specific function)

        Returns:
            Dict with 'content', 'tool_calls', and 'finish_reason'
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        # Convert messages to dict format
        messages_dict = self._convert_messages_to_dict(messages)

        # Use provided parameters or fall back to config defaults
        temp = temperature if temperature is not None else self.config.model.temperature
        top_p_val = top_p if top_p is not None else self.config.model.top_p
        max_tok = max_tokens if max_tokens is not None else self.config.model.max_tokens

        # Only include tools if tool_choice isn't "none"
        effective_tools = tools if tool_choice != "none" else None

        try:
            # Render prompt using Mistral v0.3 template
            prompt = self._render_chat_prompt(messages_dict, tools=effective_tools)

            # Guard: count prompt tokens to prevent context overflow crash
            prompt_tokens = len(self.llm.tokenize(prompt.encode('utf-8')))
            context_limit = self.config.model.context_length
            available = context_limit - prompt_tokens
            if available <= 0:
                logger.warning(
                    f"Prompt ({prompt_tokens} tokens) exceeds context "
                    f"({context_limit}). Rejecting request."
                )
                return {
                    "content": "I'm sorry, but this conversation has become too long for me to process. Please start a new conversation or shorten your message.",
                    "tool_calls": None,
                    "finish_reason": "stop"
                }
            # Clamp max_tokens so prompt + generation fits in context
            effective_max_tok = min(max_tok, available)
            if effective_max_tok < max_tok:
                logger.info(
                    f"Clamped max_tokens {max_tok} -> {effective_max_tok} "
                    f"(prompt={prompt_tokens}, ctx={context_limit})"
                )

            logger.debug(
                f"Calling create_completion (tools={bool(effective_tools)}, "
                f"temp={temp}, max_tokens={effective_max_tok}, "
                f"prompt_tokens={prompt_tokens})"
            )

            with self._inference_lock:
                response = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=effective_max_tok,
                    temperature=temp,
                    top_p=top_p_val,
                    stop=["</s>"]
                )

            raw_text = response['choices'][0]['text']
            logger.debug(f"Raw completion text: {raw_text!r}")

            # Parse tool calls from output if tools were provided
            if effective_tools:
                tool_names = [t["function"]["name"] for t in effective_tools if "function" in t]
                content, tool_calls = self._parse_tool_calls(raw_text, tool_names=tool_names)
            else:
                content = raw_text.strip() if raw_text else None
                tool_calls = None

            finish_reason = "tool_calls" if tool_calls else "stop"

            result = {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason
            }

            logger.debug(f"Returning result: content={result['content']!r}, tool_calls={result['tool_calls']}, finish_reason={result['finish_reason']}")

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None
    ):
        """
        Generate a streaming response from the model.

        Uses create_completion with stream=True and converts output to
        chat completion chunk format. Tool calling is not supported in
        streaming mode (main.py falls back to non-streaming when tools
        are present).

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            tools: List of tool/function definitions (OpenAI format)
            tool_choice: Tool choice control

        Yields:
            Response chunks in chat completion chunk format
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        # Convert messages to dict format
        messages_dict = self._convert_messages_to_dict(messages)

        temp = temperature if temperature is not None else self.config.model.temperature
        top_p_val = top_p if top_p is not None else self.config.model.top_p
        max_tok = max_tokens if max_tokens is not None else self.config.model.max_tokens

        try:
            # Render prompt using Mistral v0.3 template (no tools for streaming)
            prompt = self._render_chat_prompt(messages_dict, tools=None)

            # Guard: count prompt tokens to prevent context overflow crash
            prompt_tokens = len(self.llm.tokenize(prompt.encode('utf-8')))
            context_limit = self.config.model.context_length
            available = context_limit - prompt_tokens
            if available <= 0:
                logger.warning(
                    f"Streaming: Prompt ({prompt_tokens} tokens) exceeds "
                    f"context ({context_limit}). Returning error chunk."
                )
                yield {
                    'choices': [{
                        'index': 0,
                        'delta': {'content': "I'm sorry, but this conversation has become too long for me to process. Please start a new conversation or shorten your message."},
                        'finish_reason': 'stop'
                    }]
                }
                return
            effective_max_tok = min(max_tok, available)
            if effective_max_tok < max_tok:
                logger.info(
                    f"Streaming: Clamped max_tokens {max_tok} -> "
                    f"{effective_max_tok} (prompt={prompt_tokens}, ctx={context_limit})"
                )

            logger.debug(
                f"Starting streaming completion (temp={temp}, "
                f"max_tokens={effective_max_tok}, prompt_tokens={prompt_tokens})"
            )

            with self._inference_lock:
                stream = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=effective_max_tok,
                    temperature=temp,
                    top_p=top_p_val,
                    stop=["</s>"],
                    stream=True
                )

                for chunk in stream:
                    # Convert create_completion chunk format to chat completion chunk format
                    text = chunk['choices'][0].get('text', '')
                    finish_reason = chunk['choices'][0].get('finish_reason')

                    chat_chunk = {
                        'choices': [{
                            'index': 0,
                            'delta': {'content': text} if text else {},
                            'finish_reason': finish_reason
                        }]
                    }
                    yield chat_chunk

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def unload_model(self) -> bool:
        """
        Unload the current model to free GPU vRAM.

        Used before training starts so the trainer gets full GPU access.
        The model can be reloaded afterwards via reload_model().

        Returns:
            True if model was unloaded, False if already unloaded or error.
        """
        with self._model_lock:
            if self.llm is None:
                logger.info("Unload requested but no model is loaded")
                return True

            logger.info("Unloading model to free GPU vRAM for training...")
            try:
                del self.llm
                self.llm = None
                gc.collect()
                logger.info("Model unloaded, GPU vRAM freed")
                return True
            except Exception as e:
                logger.error(f"Error unloading model: {e}", exc_info=True)
                return False

    def is_loaded(self) -> bool:
        """Check if model is loaded and not mid-reload."""
        return self.llm is not None and not self._is_reloading

    def is_reloading(self) -> bool:
        """Check if model is currently being reloaded."""
        return self._is_reloading

    def get_current_adapter(self) -> Optional[str]:
        """Get the current adapter path."""
        return self.current_adapter_path
