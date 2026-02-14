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

# GPT-OSS Harmony chat template with native tool calling support.
# Reference: https://huggingface.co/openai/gpt-oss-20b
GPT_OSS_HARMONY_CHAT_TEMPLATE = (
    "{%- if not tools is defined %}"
    "{%- set tools = none %}"
    "{%- endif %}"
    # System message with tool definitions
    "{%- if messages[0][\"role\"] == \"system\" %}"
    "{{- \"<|start|>system<|message|>\" }}"
    "{{- messages[0][\"content\"] }}"
    "{{- \"\\nReasoning: medium\\n\" }}"
    "{%- if tools is not none %}"
    "{{- \"\\n# Tools\\n\\nnamespace functions {\\n\" }}"
    "{%- for tool in tools %}"
    "{%- set func = tool.function %}"
    "{{- \"  type \" + func.name + \" = (_: {\" }}"
    "{%- if func.parameters and func.parameters.properties %}"
    "{%- for param_name, param_spec in func.parameters.properties.items() %}"
    "{{- \"\\n    \" + param_name }}"
    "{%- if param_name not in (func.parameters.required or []) %}"
    "{{- \"?\" }}"
    "{%- endif %}"
    "{{- \": \" }}"
    "{%- if param_spec.type == \"string\" and param_spec.get(\"enum\") %}"
    "{{- '\"' + '\" | \"'.join(param_spec.enum) + '\"' }}"
    "{%- elif param_spec.type == \"string\" %}"
    "{{- \"string\" }}"
    "{%- elif param_spec.type == \"number\" or param_spec.type == \"integer\" %}"
    "{{- \"number\" }}"
    "{%- elif param_spec.type == \"boolean\" %}"
    "{{- \"boolean\" }}"
    "{%- else %}"
    "{{- \"any\" }}"
    "{%- endif %}"
    "{{- \",\" if not loop.last else \"\" }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{- \"\\n  }) => any;\\n\" }}"
    "{%- endfor %}"
    "{{- \"}\\n\" }}"
    "{%- endif %}"
    "{{- \"<|end|>\" }}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{{- \"<|start|>system<|message|>You are a helpful AI assistant.\\nReasoning: medium\" }}"
    "{%- if tools is not none %}"
    "{{- \"\\n\\n# Tools\\n\\nnamespace functions {\\n\" }}"
    "{%- for tool in tools %}"
    "{%- set func = tool.function %}"
    "{{- \"  type \" + func.name + \" = (_: {\" }}"
    "{%- if func.parameters and func.parameters.properties %}"
    "{%- for param_name, param_spec in func.parameters.properties.items() %}"
    "{{- \"\\n    \" + param_name }}"
    "{%- if param_name not in (func.parameters.required or []) %}"
    "{{- \"?\" }}"
    "{%- endif %}"
    "{{- \": \" }}"
    "{%- if param_spec.type == \"string\" and param_spec.get(\"enum\") %}"
    "{{- '\"' + '\" | \"'.join(param_spec.enum) + '\"' }}"
    "{%- elif param_spec.type == \"string\" %}"
    "{{- \"string\" }}"
    "{%- elif param_spec.type == \"number\" or param_spec.type == \"integer\" %}"
    "{{- \"number\" }}"
    "{%- elif param_spec.type == \"boolean\" %}"
    "{{- \"boolean\" }}"
    "{%- else %}"
    "{{- \"any\" }}"
    "{%- endif %}"
    "{{- \",\" if not loop.last else \"\" }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{- \"\\n  }) => any;\\n\" }}"
    "{%- endfor %}"
    "{{- \"}\\n\" }}"
    "{%- endif %}"
    "{{- \"<|end|>\" }}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    # Message loop
    "{%- for message in loop_messages %}"
    "{%- if message[\"role\"] == \"user\" %}"
    "{{- \"<|start|>user<|message|>\" + message[\"content\"] + \"<|end|>\" }}"
    "{%- elif message[\"role\"] == \"assistant\" %}"
    "{%- if message.tool_calls is defined and message.tool_calls is not none %}"
    # Tool call response with analysis and commentary channels
    "{%- for tool_call in message.tool_calls %}"
    "{{- \"<|start|>assistant<|channel|>analysis<|message|>Calling function \" + tool_call.function.name + \"<|end|>\" }}"
    "{{- \"<|start|>assistant to=functions.\" + tool_call.function.name + \"<|channel|>commentary json<|message|>\" }}"
    "{{- tool_call.function.arguments }}"
    "{{- \"<|call|>\" }}"
    "{%- endfor %}"
    "{%- else %}"
    # Regular assistant message on final channel
    "{{- \"<|start|>assistant<|channel|>final<|message|>\" + message[\"content\"] + \"<|return|>\" }}"
    "{%- endif %}"
    "{%- elif message[\"role\"] == \"tool\" %}"
    # Tool result message
    "{{- \"<|start|>functions.\" + message.get(\"name\", \"unknown\") + \" to=assistant<|channel|>commentary<|message|>\" }}"
    "{%- if message.content is defined and message.content.content is defined %}"
    "{{- message.content.content }}"
    "{%- else %}"
    "{{- message.content }}"
    "{%- endif %}"
    "{{- \"<|end|>\" }}"
    "{%- endif %}"
    "{%- endfor %}"
    # Add generation prompt for assistant response
    "{%- if add_generation_prompt %}"
    "{{- \"<|start|>assistant<|channel|>final<|message|>\" }}"
    "{%- endif %}"
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

        # Load chat template from model directory or use fallback
        self._chat_template = self._load_chat_template()

        self._load_model()

    def _load_chat_template(self):
        """
        Load chat template from model directory or use GPT-OSS Harmony fallback.

        Priority:
        1. Load from model's HF directory (chat_template.jinja)
        2. Fall back to built-in Harmony template
        """
        from datetime import datetime

        env = jinja2.Environment()
        env.filters['tojson'] = lambda val: json.dumps(val, ensure_ascii=False)
        env.globals['raise_exception'] = _raise_template_exception
        env.globals['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)

        # Try to load template from model directory
        model_template_path = Path(self.config.model.base_model_hf_path) / "chat_template.jinja"
        if model_template_path.exists():
            try:
                with open(model_template_path, 'r') as f:
                    template_str = f.read()
                logger.info(f"Loaded chat template from {model_template_path}")
                return env.from_string(template_str)
            except Exception as e:
                logger.warning(f"Failed to load chat template from {model_template_path}: {e}")

        # Fall back to GPT-OSS Harmony template
        logger.info("Using built-in GPT-OSS Harmony chat template")
        return env.from_string(GPT_OSS_HARMONY_CHAT_TEMPLATE)

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

        Strategy: unload current model first to free GPU memory (two 20B models
        cannot coexist on 24GB VRAM), then load the new one. If the new load fails,
        attempt to restore the previous model.

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
            List of message dicts compatible with GPT-OSS Harmony chat template
        """
        result = []
        for msg in messages:
            # Extract text from content — handles string, array-of-parts, or None
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content = "\n".join(
                    part.get("text", "") for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ) or ""
            elif msg.content is not None:
                content = str(msg.content)
            else:
                content = ""

            msg_dict = {
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": content
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
        Render messages into a prompt string using the GPT-OSS Harmony chat template.

        Args:
            messages_dict: List of message dicts
            tools: Optional list of tool definitions (OpenAI format)

        Returns:
            Rendered prompt string ready for create_completion
        """
        render_kwargs = {
            "messages": messages_dict,
            "add_generation_prompt": True,
        }
        if tools:
            render_kwargs["tools"] = tools

        rendered = self._chat_template.render(**render_kwargs)
        logger.debug(f"Rendered prompt ({len(rendered)} chars): ...{rendered[-200:]}")
        return rendered

    def _parse_tool_calls(self, text: str, tool_names: Optional[List[str]] = None):
        """
        Parse tool calls from model output.

        Supports two formats:
        1. Harmony format (native gpt-oss): Uses special tokens like <|call|>
        2. Markdown JSON fallback: Extracts JSON from markdown code blocks

        Args:
            text: Raw model output text
            tool_names: List of known tool function names (for validation)

        Returns:
            tuple: (content_text_or_None, tool_calls_list_or_None)
        """
        if not text:
            return None, None

        # Debug: log raw model output
        logger.debug(f"=== RAW MODEL OUTPUT ({len(text)} chars) ===")
        logger.debug(text[:500] + ("..." if len(text) > 500 else ""))
        logger.debug("=== END RAW OUTPUT ===")

        tool_calls = []
        content = None
        import re

        # Method 1: Try Harmony format first (native gpt-oss-20b)
        # Supports both official format (with <|call|>) and jinx format (without <|call|>)
        if "to=functions." in text:
            # Try official format first: to=functions.FUNC...json<|message|>JSON<|call|>
            pattern_official = r'to=functions\.(\w+).*?json<\|message\|>(.*?)<\|call\|>'
            matches = re.findall(pattern_official, text, re.DOTALL)

            # Try jinx format: <|channel|>commentary to=functions.FUNC <|constrain|>json<|message|>JSON
            if not matches:
                pattern_jinx = r'to=functions\.(\w+).*?<\|constrain\|>json<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)'
                matches = re.findall(pattern_jinx, text, re.DOTALL)

            for func_name, json_args in matches:
                if tool_names and func_name not in tool_names:
                    logger.warning(f"Unknown tool called: {func_name}")
                    continue

                try:
                    args = json.loads(json_args.strip())
                    call_id = uuid.uuid4().hex[:9]
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args)
                        }
                    })
                    logger.debug(f"Parsed tool call (Harmony format): {func_name}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse Harmony tool call: {e}, raw: {json_args!r}")

        # Method 2: Fallback to markdown JSON extraction (for fine-tuned models)
        if not tool_calls and tool_names:
            # Look for JSON in markdown code blocks
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            json_matches = re.findall(json_pattern, text, re.DOTALL)

            for json_block in json_matches:
                try:
                    parsed = json.loads(json_block.strip())

                    # Try to infer function name from context or parsed structure
                    func_name = None
                    args = parsed

                    # Check if JSON has a "name" field (direct format)
                    if isinstance(parsed, dict) and "name" in parsed:
                        func_name = parsed["name"]
                        args = parsed.get("arguments", {})
                    # Otherwise infer from tool_names if there's only one tool
                    elif len(tool_names) == 1:
                        func_name = tool_names[0]
                    # Try to find tool name mentioned in text before the JSON
                    else:
                        for name in tool_names:
                            if name in text.lower() or name.replace("_", " ") in text.lower():
                                func_name = name
                                break

                    if func_name and func_name in tool_names:
                        call_id = uuid.uuid4().hex[:9]
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else args
                            }
                        })
                        logger.debug(f"Parsed tool call (markdown fallback): {func_name}")
                        break  # Only take first valid JSON block
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Skipping invalid JSON block: {e}")
                    continue

        def clean_template_tokens(text: str) -> str:
            """Remove template tokens that may leak into output."""
            if not text:
                return text

            import re

            # Remove any <|...|> format tokens (catches all special tokens)
            cleaned = re.sub(r'<\|[^|]+\|>', '', text)

            # Remove malformed role+channel combinations that sometimes leak
            # These are template artifacts, not natural language
            template_artifacts = [
                r'\bassistantfinal\b',  # "assistantfinal" without spaces
                r'\buserfinal\b',       # "userfinal" without spaces
                r'\bsystemfinal\b',     # "systemfinal" without spaces
            ]

            for pattern in template_artifacts:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

            # Clean up multiple spaces and trim
            cleaned = " ".join(cleaned.split())
            return cleaned.strip()

        # Extract thinking content from analysis channel (chain-of-thought reasoning)
        thinking = None
        if "<|channel|>analysis<|message|>" in text:
            parts = text.split("<|channel|>analysis<|message|>", 1)
            if len(parts) > 1:
                # Extract thinking up to the next channel, start marker, or end marker
                thinking_raw_original = parts[1]
                logger.debug(f"[THINKING DEBUG] Raw thinking before terminator split (first 300 chars): {thinking_raw_original[:300]!r}")

                # Split on various terminators
                thinking_raw = thinking_raw_original
                for terminator in ["<|channel|>", "<|start|>", "<|end|>", "<|return|>"]:
                    thinking_raw = thinking_raw.split(terminator)[0]

                logger.debug(f"[THINKING DEBUG] After terminator splits (first 300 chars): {thinking_raw[:300]!r}")

                thinking_cleaned = clean_template_tokens(thinking_raw)
                logger.debug(f"[THINKING DEBUG] After clean_template_tokens (first 300 chars): {thinking_cleaned[:300]!r}")
                logger.debug(f"[THINKING DEBUG] Full cleaned thinking length: {len(thinking_cleaned)} chars")

                thinking = thinking_cleaned if thinking_cleaned else None
                thinking_preview = thinking[:100] if thinking else None
                logger.debug(f"Extracted thinking from analysis channel: {thinking_preview!r}...")

        # Extract text content (from channels or raw text)
        # Priority: final channel > raw text (no channels) > empty (if only analysis exists)
        if "<|channel|>final<|message|>" in text:
            # Extract from final channel - this is the actual response
            parts = text.split("<|channel|>final<|message|>", 1)
            if len(parts) > 1:
                # Split on terminators
                content_raw = parts[1]
                for terminator in ["<|return|>", "<|end|>", "<|start|>"]:
                    content_raw = content_raw.split(terminator)[0]

                content_raw = clean_template_tokens(content_raw)
                content = content_raw if content_raw else None
                logger.debug(f"Extracted content from final channel: {content!r}")
        elif "<|channel|>" in text:
            # Has channel markers but no final channel - likely just thinking/analysis
            # Content should be empty since we already extracted thinking
            content = None
            logger.debug("Model output contains only analysis channel, no final content")
        else:
            # Fallback: no channel markers at all, use raw text as content
            cleaned = clean_template_tokens(text)
            content = cleaned if cleaned else None
            logger.debug(f"Extracted content (no channel markers, using raw text): {content!r}")

        if tool_calls:
            logger.debug(f"Parsed {len(tool_calls)} tool call(s)")
            return content, tool_calls, thinking

        # Don't fall back to raw text if we've already determined content should be None
        # (e.g., analysis-only output where thinking was extracted)
        logger.debug(f"Final content to return: {content!r}")
        return content, None, thinking

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

        Uses the GPT-OSS Harmony chat template for prompt rendering and
        create_completion for inference, with post-processing to parse
        tool calls from the harmony-formatted output.

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
            # Render prompt using GPT-OSS Harmony template
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
                    stop=["<|return|>"]  # Only stop on <|return|>, allow <|end|> for multi-channel responses
                )

            raw_text = response['choices'][0]['text']
            logger.debug(f"Raw completion text: {raw_text!r}")

            # Always parse channels and tool calls from output
            # Even without tools, we need to extract content from the final channel
            tool_names = None
            if effective_tools:
                tool_names = [t["function"]["name"] for t in effective_tools if "function" in t]

            content, tool_calls, thinking = self._parse_tool_calls(raw_text, tool_names=tool_names)

            finish_reason = "tool_calls" if tool_calls else "stop"

            result = {
                "content": content,
                "tool_calls": tool_calls,
                "thinking": thinking,
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
            # Render prompt using GPT-OSS Harmony template (no tools for streaming)
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
                    stop=["<|return|>"],  # Only stop on <|return|>, allow <|end|> for multi-channel responses
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
