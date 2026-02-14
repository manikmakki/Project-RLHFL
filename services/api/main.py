"""
Project RLHFL - API Service

This module provides an OpenAI-compatible API for local LLM inference with
automatic memory storage and sentiment analysis. It serves as the main entry
point for chat completions and manages the interaction between the LLM engine,
memory manager, and sentiment analyzer.

The API automatically stores interactions in ChromaDB and infers sentiment
from conversation patterns to enable human-in-the-loop training without
manual intervention.
"""

import logging
import time
import uuid
import json
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
import torch

from shared.config import load_config, settings
from shared.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatMessage,
    MessageRole,
    Usage,
    ModelsResponse,
    ModelInfo,
    HealthStatus,
    ModelReloadRequest,
    ModelReloadResponse,
)
from api.llm_engine import LLMEngine
from api.memory_manager import MemoryManager
from api.sentiment_analyzer import SentimentAnalyzer
from api.prompt_interceptor import PromptInterceptor
from api.admin_ui import admin_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
config = None
llm_engine = None
memory_manager = None
sentiment_analyzer = None
prompt_interceptor = None
recent_api_requests = deque(maxlen=20)  # Rolling buffer of last 20 API requests


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by stripping Ollama-style tags.
    Ollama uses 'model:tag' format where tag defaults to 'latest'.
    This allows clients to request 'jinx-gpt-oss-20b:latest' or just 'jinx-gpt-oss-20b'.
    """
    if ':' in model_name:
        # Strip the tag part (e.g., 'jinx-gpt-oss-20b:latest' -> 'jinx-gpt-oss-20b')
        return model_name.split(':')[0]
    return model_name


def parse_model_response(content: str) -> dict:
    """
    Parse the model's response to extract thinking (analysis) and final content.
    The model outputs special channel tags:
    - <|channel|>analysis<|message|>...text...<|end|> for chain-of-thought reasoning
    - <|channel|>final<|message|>...text...<|end|> for the final user-facing response

    Returns dict with 'thinking' and 'content' fields.
    """
    import re

    result = {
        'thinking': None,
        'content': content  # Default to full content if no parsing needed
    }

    # Extract analysis/thinking section
    analysis_match = re.search(
        r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>)',
        content,
        re.DOTALL
    )

    # Extract final content section
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',
        content,
        re.DOTALL
    )

    if analysis_match:
        result['thinking'] = analysis_match.group(1).strip()

    if final_match:
        result['content'] = final_match.group(1).strip()
    elif analysis_match:
        # If we found analysis but no final section, content should be empty
        result['content'] = ''

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global config, llm_engine, memory_manager, sentiment_analyzer, prompt_interceptor

    # Startup
    logger.info("Starting LLM API service...")

    # Configure thread pool for asyncio.to_thread() to handle blocking operations
    thread_pool_size = int(os.environ.get('ASYNCIO_THREAD_POOL_SIZE', 16))
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=thread_pool_size))
    logger.info(f"Configured asyncio thread pool with {thread_pool_size} workers")

    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded from {settings.config_path}")
    
    # Initialize memory manager
    memory_manager = MemoryManager(config)
    logger.info("Memory manager initialized")

    # Check database size and perform auto-cleanup if needed
    logger.info("Checking database size for auto-cleanup...")
    cleanup_performed = await asyncio.to_thread(memory_manager.auto_cleanup_if_needed)
    if cleanup_performed:
        logger.info("Auto-cleanup completed at startup")

    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(config)
    logger.info("Sentiment analyzer initialized")

    # Initialize prompt interceptor
    rules_path = str(Path(settings.config_path).parent / "prompt_rules.yaml")
    prompt_interceptor = PromptInterceptor(rules_path)
    logger.info("Prompt interceptor initialized")

    # Initialize LLM engine
    llm_engine = LLMEngine(config, settings.model_path)
    logger.info("LLM engine initialized")
    # Wire admin UI dependencies now that managers are initialized
    try:
        from api import admin_ui
        admin_ui.set_dependencies(memory_manager, llm_engine, recent_api_requests, prompt_interceptor)
    except Exception:
        logger.warning("Failed to set admin UI dependencies at startup")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected, running on CPU")
    
    # Start background checkpoint poller
    poll_interval = config.model.checkpoint_poll_interval_seconds
    checkpoint_poller_task = asyncio.create_task(
        _poll_for_new_checkpoints(llm_engine, settings.checkpoints_path, poll_interval)
    )
    logger.info(f"Background checkpoint poller started ({poll_interval}s interval)")

    logger.info("LLM API service ready!")

    yield

    # Shutdown
    checkpoint_poller_task.cancel()
    try:
        await checkpoint_poller_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down LLM API service...")


app = FastAPI(
    title="Local LLM API",
    description="OpenAI-compatible API for local LLM with automatic training",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors (422).
    Provides OpenAI-compatible error format with helpful debugging info.
    """
    errors = exc.errors()
    
    # Build human-readable error details
    error_details = []
    for error in errors:
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        error_type = error.get("type", "unknown")
        error_details.append(f"{loc}: {msg} (type: {error_type})")
    
    logger.warning(f"Request validation error: {'; '.join(error_details)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request body",
                "type": "invalid_request_error",
                "param": errors[0]["loc"][-1] if errors else None,
                "details": error_details
            }
        }
    )


@app.exception_handler(503)
async def service_unavailable_handler(request: Request, exc: HTTPException):
    """Add Retry-After header to all 503 responses."""
    return JSONResponse(
        status_code=503,
        content={"detail": exc.detail},
        headers={"Retry-After": "120"},
    )


# Dependency injection for admin endpoints
def get_memory_manager():
    """Provide memory manager to endpoints."""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager


def get_llm_engine():
    """Provide LLM engine to endpoints."""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not initialized")
    return llm_engine


# Include admin router
from api import admin_ui
app.include_router(admin_router)


# Admin route with dependency injection - monkey patch the routes
for route in app.routes:
    if hasattr(route, 'path') and '/admin' in route.path:
        if hasattr(route, 'dependant'):
            # Add dependencies to admin routes
            original_endpoint = route.endpoint
            
            async def wrapped_endpoint(*args, **kwargs):
                kwargs['memory_manager'] = memory_manager
                kwargs['llm_engine'] = llm_engine
                return await original_endpoint(*args, **kwargs)
            
            route.endpoint = wrapped_endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()

    is_reloading = llm_engine.is_reloading() if llm_engine else False
    if is_reloading:
        status_str = "reloading"
    elif llm_engine and llm_engine.is_loaded():
        status_str = "healthy"
    else:
        status_str = "degraded"

    status = HealthStatus(
        status=status_str,
        model_loaded=llm_engine.is_loaded() if llm_engine else False,
        memory_connected=memory_manager.is_connected() if memory_manager else False,
        gpu_available=gpu_available,
        current_checkpoint=llm_engine.current_checkpoint_id if llm_engine else None
    )

    return status


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    model_id = config.model.model_id if config else "gpt-oss-20b"
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model_id,
                object="model",
                created=int(time.time()),
                owned_by="local"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    Supports both streaming and non-streaming responses.
    """
    if not llm_engine or not llm_engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Use configured model if not specified, or validate the requested model matches
        configured_model_id = config.model.model_id if config else "gpt-oss-20b"
        request_model = request.model or configured_model_id

        # Normalize model name to handle Ollama-style tags (e.g., 'model:latest')
        normalized_model = normalize_model_name(request_model)

        if normalized_model not in [configured_model_id, "local-llm"]:  # Allow legacy "local-llm" for compatibility
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request_model}' not found. Available model: {configured_model_id}"
            )
        
        # Validate n parameter
        if request.n and request.n != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported. Multiple completions are not supported."
            )
        
        logger.debug(f"Request tools: {request.tools}")
        logger.debug(f"Request tool_choice: {request.tool_choice}")

        # Extract conversation ID (or generate one)
        conversation_id = request.user or str(uuid.uuid4())
        
        # Get user message (most recent user message in the conversation)
        user_message = next(
            (msg for msg in reversed(request.messages) if msg.role == MessageRole.USER),
            None
        )

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in messages")

        # Get previous assistant response (n-1) for storage pairing.
        # Storing user prompt + prior LLM response captures user behavior
        # (what the user said in reaction to the last assistant output).
        prev_assistant_message = next(
            (msg for msg in reversed(request.messages) if msg.role == MessageRole.ASSISTANT),
            None
        )
        prev_assistant_text = prev_assistant_message.get_text_content() if prev_assistant_message else None

        # Retrieve relevant context from memory using RAG (if enabled)
        user_message_text = user_message.get_text_content()
        messages_with_context = list(request.messages)

        # Normalize content arrays (OpenAI spec: content can be [{type: "text", text: "..."}])
        messages_with_context = [msg.normalize_content() for msg in messages_with_context]

        # Apply prompt interceptor rules before RAG or LLM processing
        if prompt_interceptor:
            messages_with_context = prompt_interceptor.apply(messages_with_context)
            # Re-extract user_message_text in case it was transformed
            user_message_text = next(
                (msg.get_text_content() for msg in reversed(messages_with_context)
                 if msg.role == MessageRole.USER),
                user_message_text
            )

        if config.memory.rag_enabled and not request.tools:
            # Skip RAG when tools are present — the Harmony template includes
            # tool definitions in the system message, and lengthy RAG context
            # may interfere with proper tool calling behavior.

            # Run embedding generation in thread pool to avoid blocking event loop
            relevant_context = await asyncio.to_thread(
                memory_manager.retrieve_relevant_context,
                query=user_message_text,
                k=config.memory.rag_top_k
            )

            # Inject retrieved context into messages
            if relevant_context:
                # Build context string from retrieved interactions
                context_parts = ["Here are some relevant previous interactions for context:\n"]
                for idx, ctx in enumerate(relevant_context, 1):
                    context_parts.append(
                        f"\n[Context {idx}] (Similarity: {ctx['similarity']:.2f})\n"
                        f"User: {ctx['user_message']}\n"
                        f"Assistant: {ctx['assistant_response']}\n"
                    )
                context_str = "".join(context_parts)

                # Insert context as a system message at the beginning
                context_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=context_str
                )
                messages_with_context.insert(0, context_message)
                logger.info(f"RAG: Injected {len(relevant_context)} context items into prompt")
            else:
                logger.debug("RAG: No relevant context found for this query")
        elif request.tools:
            logger.debug("RAG: Skipped — tools present in request")
        else:
            logger.debug("RAG: Disabled in configuration")

        # Log post-processed request (after interception + RAG) so the admin
        # UI shows exactly what the LLM will receive.
        recent_api_requests.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/v1/chat/completions",
            "model": request_model,
            "stream": request.stream,
            "has_tools": bool(request.tools),
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "messages": [msg.model_dump(exclude_none=True) for msg in messages_with_context],
        })

        # Handle streaming
        # Note: chatml-function-calling does not support streaming with auto tool_choice.
        # When client requests stream=true, always return SSE format (even if we generate non-streaming internally)
        logger.debug(f"stream={request.stream}, has_tools={bool(request.tools)}, tool_choice={request.tool_choice}")

        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    request, conversation_id, user_message_text, request_model, messages_with_context, prev_assistant_text
                ),
                media_type="text/event-stream"
            )

        # Generate non-streaming response with context
        # Run LLM inference in thread pool to avoid blocking event loop
        start_time = time.time()
        response_dict = await asyncio.to_thread(
            llm_engine.generate,
            messages=messages_with_context,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice
        )
        generation_time = time.time() - start_time

        logger.debug(f"Raw LLM response_dict: {response_dict}")

        # Extract response components
        response_text = response_dict.get("content") or ""
        tool_calls = response_dict.get("tool_calls")
        thinking = response_dict.get("thinking")
        finish_reason = response_dict.get("finish_reason", "stop")

        logger.debug(f"Parsed response - finish_reason={finish_reason}, has_content={bool(response_text)}, thinking={bool(thinking)}, tool_calls={tool_calls}")

        # Only analyze sentiment and store if this is a final response (not a tool call)
        # Store user prompt paired with previous LLM response (n-1) to capture
        # user behavior — skip if no prior assistant message in history.
        logger.debug(
            f"Storage gate: finish_reason={finish_reason!r}, "
            f"has_response_text={bool(response_text)}, "
            f"has_prev_assistant={bool(prev_assistant_text)}"
        )
        if finish_reason == "stop" and response_text and prev_assistant_text:
            # Infer sentiment (only uses USER message text, not system prompts or context)
            # Run sentiment analysis in thread pool to avoid blocking
            sentiment = await asyncio.to_thread(
                sentiment_analyzer.analyze,
                user_message=user_message_text,
                assistant_response=prev_assistant_text,
                conversation_continuing=True
            )

            # Calculate weight (only uses USER message text)
            weight = await asyncio.to_thread(
                sentiment_analyzer.calculate_weight,
                user_message=user_message_text,
                sentiment=sentiment,
                conversation_continuing=True
            )

            # Check if golden example (only uses USER message text)
            is_golden = await asyncio.to_thread(
                sentiment_analyzer.is_golden_example,
                user_message=user_message_text,
                sentiment=sentiment,
                weight=weight
            )

            # Store interaction: user prompt + previous assistant response (n-1)
            # Run embedding generation and DB storage in thread pool
            await asyncio.to_thread(
                memory_manager.store_interaction,
                conversation_id=conversation_id,
                user_message=user_message_text,
                assistant_response=prev_assistant_text,
                sentiment=sentiment,
                weight=weight,
                is_golden=is_golden
            )

            logger.info(
                f"Generated response in {generation_time:.2f}s "
                f"(sentiment: {sentiment:.2f}, weight: {weight:.2f}, golden: {is_golden})"
            )
        else:
            logger.info(
                f"Generated tool call response in {generation_time:.2f}s "
                f"(finish_reason: {finish_reason})"
            )

        # Build assistant message with tool_calls if present
        # Per OpenAI spec: content should be None when there are tool calls, but some
        # clients (like Open WebUI) might expect an empty string instead
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_text,  # Keep empty string as-is, don't convert to None
            tool_calls=tool_calls,
            thinking=thinking  # Include thinking if present
        )
        logger.debug(f"Built assistant_message: content={assistant_message.content!r}, thinking={bool(thinking)}, tool_calls={assistant_message.tool_calls}")

        # Build OpenAI-compatible response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason=finish_reason
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_message_text.split()) if user_message_text else 0,  # Rough estimate
                completion_tokens=len(response_text.split()) if response_text else 0,
                total_tokens=len(user_message_text.split()) + len(response_text.split()) if user_message_text and response_text else 0
            )
        )

        # Return in configured response format
        if config.model.response_format == "openclaw":
            # OpenClaw native: return message directly with content as array-of-parts
            msg = assistant_message.model_dump(exclude_none=True)
            if isinstance(msg.get("content"), str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            msg["timestamp"] = int(time.time() * 1000)
            logger.info(f"Returning OpenClaw format: {json.dumps(msg, indent=2)}")
            return JSONResponse(content=msg)

        return JSONResponse(content=completion_response.model_dump(exclude_none=True))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(
    request: ChatCompletionRequest,
    conversation_id: str,
    user_message: str,
    model_id: str,
    messages_with_context: list[ChatMessage],
    prev_assistant_text: str | None = None
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses in OpenAI format with RAG context."""
    try:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        # When tools are present, generate complete response then stream as SSE
        # (llm_engine.generate_stream doesn't support tools properly)
        if request.tools:
            logger.debug("Streaming with tools: using generate() then converting to SSE")
            response_dict = await asyncio.to_thread(
                llm_engine.generate,
                messages=messages_with_context,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice
            )

            full_response = response_dict.get("content") or ""
            tool_calls = response_dict.get("tool_calls")
            finish_reason = response_dict.get("finish_reason", "stop")

            # Send initial chunk with role and empty content
            # Some clients (like Open WebUI) expect content to be present even when empty
            initial_chunk = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': ''},
                    'finish_reason': None
                }]
            }
            logger.debug(f"Sending initial chunk: {json.dumps(initial_chunk)}")
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Send content chunk
            if full_response:
                content_chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model_id,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': full_response},
                        'finish_reason': None
                    }]
                }
                logger.debug(f"Sending content chunk: {json.dumps(content_chunk)}")
                yield f"data: {json.dumps(content_chunk)}\n\n"

            # Send tool_calls chunk if present
            if tool_calls:
                tool_chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model_id,
                    'choices': [{
                        'index': 0,
                        'delta': {'tool_calls': tool_calls},
                        'finish_reason': None
                    }]
                }
                logger.debug(f"Sending tool_calls chunk: {json.dumps(tool_chunk)}")
                yield f"data: {json.dumps(tool_chunk)}\n\n"

            # Send final chunk for tools path
            final_chunk = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': finish_reason
                }]
            }
            logger.debug(f"Sending final chunk: {json.dumps(final_chunk)}")
            yield f"data: {json.dumps(final_chunk)}\n\n"
        else:
            # Normal streaming without tools
            # Send initial chunk with role
            initial_chunk = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': ''},
                    'finish_reason': None
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            full_response = ""
            finish_reason = "stop"
            _STREAM_END = object()

            def get_next_chunk(generator):
                try:
                    return next(generator)
                except StopIteration:
                    return _STREAM_END

            def get_stream_generator():
                return llm_engine.generate_stream(
                    messages=messages_with_context,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=None,
                    tool_choice=None
                )

            stream_gen = await asyncio.to_thread(get_stream_generator)

            while True:
                chunk = await asyncio.to_thread(get_next_chunk, stream_gen)

                if chunk is _STREAM_END:
                    break

                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    delta = choice.get('delta', {})

                    if 'content' in delta and delta['content']:
                        full_response += delta['content']

                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']

                    chunk_data = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model_id,
                        'choices': [{
                            'index': 0,
                            'delta': delta,
                            'finish_reason': choice.get('finish_reason')
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final chunk if not already sent
        if not finish_reason:
            final_chunk = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_id,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            finish_reason = 'stop'

        logger.debug("Sending [DONE] message")
        yield "data: [DONE]\n\n"
        logger.debug("Stream completed successfully")

        # Store user prompt paired with previous LLM response (n-1) to capture
        # user behavior — skip if no prior assistant message in history.
        if finish_reason == "stop" and full_response and prev_assistant_text:
            # Run in thread pool to avoid blocking event loop
            sentiment = await asyncio.to_thread(
                sentiment_analyzer.analyze,
                user_message=user_message,
                assistant_response=prev_assistant_text,
                conversation_continuing=True
            )

            weight = await asyncio.to_thread(
                sentiment_analyzer.calculate_weight,
                user_message=user_message,
                sentiment=sentiment
            )

            is_golden = await asyncio.to_thread(
                sentiment_analyzer.is_golden_example,
                user_message=user_message,
                sentiment=sentiment,
                weight=weight
            )

            await asyncio.to_thread(
                memory_manager.store_interaction,
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_response=prev_assistant_text,
                sentiment=sentiment,
                weight=weight,
                is_golden=is_golden
            )
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}", exc_info=True)
        error_chunk = {
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model_id,
            'error': {'message': str(e), 'type': 'server_error'}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/training/trigger")
async def trigger_training():
    """Manually trigger training (for user-initiated training)."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    # Pre-training cleanup: remove low-weight entries to avoid training on low-value data
    logger.info("Running pre-training cleanup to remove low-weight entries")
    deleted_count = await asyncio.to_thread(memory_manager.cleanup_low_weight_entries)
    logger.info(f"Pre-training cleanup: removed {deleted_count} low-weight entries")

    memory_manager.request_training()

    return {
        "status": "training_requested",
        "message": "Training will begin at the next scheduled check",
        "pre_training_cleanup": {
            "deleted_count": deleted_count
        }
    }


@app.get("/v1/training/stats")
async def get_training_stats():
    """Get current training statistics."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    stats = memory_manager.get_training_stats()
    return stats


@app.post("/v1/model/unload")
async def unload_model():
    """
    Unload the current LLM to free GPU vRAM.
    Called by the trainer before training starts so it gets full GPU access.
    The model should be reloaded via /v1/model/reload after training completes.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="LLM engine not initialized")

    logger.info("Model unload requested (pre-training GPU release)")
    success = await asyncio.to_thread(llm_engine.unload_model)

    if success:
        return {"success": True, "message": "Model unloaded, GPU vRAM freed for training"}
    else:
        return {"success": False, "message": "Failed to unload model (check API logs)"}


@app.post("/v1/model/reload")
async def reload_model(request_body: ModelReloadRequest):
    """
    Reload the LLM with a new GGUF model file.
    Called by the trainer after GGUF conversion, or manually for rollback.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="LLM engine not initialized")

    logger.info(
        f"Model reload requested: checkpoint={request_body.checkpoint_id}, "
        f"path={request_body.gguf_model_path}"
    )

    success = await asyncio.to_thread(
        llm_engine.reload_model,
        new_model_path=request_body.gguf_model_path,
        checkpoint_id=request_body.checkpoint_id,
    )

    if success:
        return ModelReloadResponse(
            success=True,
            message=f"Model reloaded with checkpoint {request_body.checkpoint_id}",
            new_model_path=request_body.gguf_model_path,
            checkpoint_id=request_body.checkpoint_id,
        )
    else:
        return ModelReloadResponse(
            success=False,
            message="Model reload failed; previous model restored (check API logs)",
            checkpoint_id=request_body.checkpoint_id,
        )


# ============================================================================
# Ollama API Compatibility Endpoints
# ============================================================================

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """
    Generate chat completion (Ollama-compatible).
    Converts Ollama request/response format to/from OpenAI format.
    Supports both streaming and non-streaming responses.
    """
    if not llm_engine or not llm_engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse Ollama request
        ollama_req = await request.json()

        # Normalize model name (handle :latest tag)
        model_name = normalize_model_name(ollama_req.get("model", ""))
        configured_model_id = config.model.model_id if config else "gpt-oss-20b"

        if model_name not in [configured_model_id, "local-llm"]:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{ollama_req.get('model')}' not found. Available model: {configured_model_id}:latest"
            )

        # Convert Ollama format to OpenAI format
        # Parse messages - convert to ChatMessage objects
        raw_messages = ollama_req.get("messages", [])
        messages = [
            ChatMessage(
                role=MessageRole(msg["role"]),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls")
            )
            for msg in raw_messages
        ]

        tools = ollama_req.get("tools", None)
        stream = ollama_req.get("stream", True)

        # Extract options (Ollama uses 'options' object for generation params)
        options = ollama_req.get("options", {})
        temperature = options.get("temperature", config.model.temperature if config else 0.7)
        top_p = options.get("top_p", config.model.top_p if config else 0.9)
        max_tokens = options.get("num_predict", config.model.max_tokens if config else 2048)

        # Build OpenAI-compatible request
        openai_request = ChatCompletionRequest(
            model=configured_model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=ollama_req.get("tool_choice", "auto") if tools else None
        )

        # Handle streaming response
        if stream:
            async def ollama_stream_generator():
                """Convert OpenAI SSE format to Ollama NDJSON format."""
                start_time = time.time_ns()
                full_content = ""
                thinking_content = ""
                final_content = ""
                full_tool_calls = []
                current_channel = None  # Track which channel we're in (analysis or final)
                in_thinking = False
                thinking_sent = False

                # Get user message for conversation_id
                user_message_text = next(
                    (msg.get_text_content() for msg in reversed(messages) if msg.role == MessageRole.USER),
                    ""
                )
                conversation_id = ollama_req.get("conversation_id", str(uuid.uuid4()))

                # Normalize content arrays
                messages_normalized = [msg.normalize_content() for msg in messages]

                # Get previous assistant message for memory pairing
                prev_assistant_text = next(
                    (msg.get_text_content() for msg in reversed(messages) if msg.role == MessageRole.ASSISTANT),
                    None
                )

                # Use existing stream_chat_completion function
                async for sse_chunk in stream_chat_completion(
                    openai_request, conversation_id, user_message_text,
                    configured_model_id, messages_normalized, prev_assistant_text
                ):
                    # Parse SSE chunk
                    chunk_text = sse_chunk.decode('utf-8') if isinstance(sse_chunk, bytes) else sse_chunk

                    if chunk_text.startswith("data: "):
                        chunk_text = chunk_text[6:]

                    chunk_text = chunk_text.strip()

                    if chunk_text == "[DONE]":
                        # Send final chunk with done=true
                        final_chunk = {
                            "model": configured_model_id,
                            "created_at": datetime.now().isoformat() + "Z",
                            "message": {
                                "role": "assistant",
                                "content": final_content
                            },
                            "done": True,
                            "done_reason": "stop",
                            "total_duration": time.time_ns() - start_time,
                            "load_duration": 0,
                            "prompt_eval_count": 0,
                            "eval_count": len(final_content.split())
                        }

                        # Add thinking if we collected any
                        if thinking_content:
                            final_chunk["message"]["thinking"] = thinking_content

                        if full_tool_calls:
                            final_chunk["message"]["tool_calls"] = full_tool_calls

                        yield json.dumps(final_chunk) + "\n"
                        break

                    if not chunk_text:
                        continue

                    try:
                        openai_data = json.loads(chunk_text)
                    except json.JSONDecodeError:
                        continue

                    # Extract delta from OpenAI chunk
                    if "choices" in openai_data and len(openai_data["choices"]) > 0:
                        delta = openai_data["choices"][0].get("delta", {})

                        # Handle role
                        if "role" in delta:
                            ollama_chunk = {
                                "model": configured_model_id,
                                "created_at": datetime.now().isoformat() + "Z",
                                "message": {"role": delta["role"]},
                                "done": False
                            }
                            yield json.dumps(ollama_chunk) + "\n"

                        # Handle content - parse channel markers on-the-fly
                        if "content" in delta and delta["content"]:
                            token = delta["content"]
                            full_content += token

                            # Detect channel changes
                            if "<|channel|>analysis<|message|>" in full_content and not in_thinking:
                                in_thinking = True
                                current_channel = "analysis"
                                continue
                            elif "<|channel|>final<|message|>" in full_content:
                                in_thinking = False
                                current_channel = "final"
                                # Send accumulated thinking as a single chunk if we have it
                                if thinking_content and not thinking_sent:
                                    # Clean template tokens from thinking before sending
                                    import re
                                    thinking_cleaned = re.sub(r'<\|[^|]+\|>', '', thinking_content)
                                    # Remove malformed artifacts like "assistantfinal"
                                    thinking_cleaned = re.sub(r'\bassistantfinal\b', '', thinking_cleaned, flags=re.IGNORECASE)
                                    thinking_cleaned = re.sub(r'\buserfinal\b', '', thinking_cleaned, flags=re.IGNORECASE)
                                    thinking_cleaned = re.sub(r'\bsystemfinal\b', '', thinking_cleaned, flags=re.IGNORECASE)
                                    thinking_cleaned = " ".join(thinking_cleaned.split()).strip()

                                    thinking_chunk = {
                                        "model": configured_model_id,
                                        "created_at": datetime.now().isoformat() + "Z",
                                        "message": {"thinking": thinking_cleaned},
                                        "done": False
                                    }
                                    yield json.dumps(thinking_chunk) + "\n"
                                    thinking_sent = True
                                continue

                            # Skip channel markers, template tokens, and end tags
                            if any(marker in token for marker in ["<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|start|>"]):
                                continue

                            # Accumulate content based on current channel
                            if in_thinking or current_channel == "analysis":
                                thinking_content += token
                            elif current_channel == "final":
                                final_content += token
                                # Stream final content in real-time
                                ollama_chunk = {
                                    "model": configured_model_id,
                                    "created_at": datetime.now().isoformat() + "Z",
                                    "message": {"content": token},
                                    "done": False
                                }
                                yield json.dumps(ollama_chunk) + "\n"

                        # Handle tool calls
                        if "tool_calls" in delta:
                            tool_calls = delta["tool_calls"]
                            if tool_calls:
                                for tc in tool_calls:
                                    if tc.get("function"):
                                        full_tool_calls.append(tc)
                                ollama_chunk = {
                                    "model": configured_model_id,
                                    "created_at": datetime.now().isoformat() + "Z",
                                    "message": {"tool_calls": tool_calls},
                                    "done": False
                                }
                                yield json.dumps(ollama_chunk) + "\n"

            return StreamingResponse(
                ollama_stream_generator(),
                media_type="application/x-ndjson"
            )

        # Handle non-streaming response
        else:
            # Get non-streaming response from our chat completions logic
            json_response = await chat_completions(openai_request)

            # Parse the JSON content from the JSONResponse
            import json as json_module
            response_content = json_module.loads(json_response.body.decode('utf-8'))

            # Convert OpenAI response to Ollama format
            choice = response_content["choices"][0]
            message = choice["message"]

            ollama_response = {
                "model": configured_model_id,
                "created_at": datetime.now().isoformat() + "Z",
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content") or ""
                },
                "done": True,
                "done_reason": choice.get("finish_reason") or "stop",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": response_content.get("usage", {}).get("prompt_tokens", 0),
                "eval_count": response_content.get("usage", {}).get("completion_tokens", 0)
            }

            # Add thinking if present in the message
            if message.get("thinking"):
                thinking_full = message["thinking"]
                thinking_preview = thinking_full[:200] if len(thinking_full) > 200 else thinking_full
                logger.debug(f"[OLLAMA THINKING] Preview (first 200 chars): {thinking_preview!r}")
                logger.debug(f"[OLLAMA THINKING] Full length: {len(thinking_full)} chars")
                # Log the end of thinking to see if template tokens are there
                if len(thinking_full) > 200:
                    thinking_end = thinking_full[-200:]
                    logger.debug(f"[OLLAMA THINKING] End (last 200 chars): {thinking_end!r}")
                # Check for template token presence
                has_start_token = "<|start|>" in thinking_full
                has_assistant_final = "assistantfinal" in thinking_full.lower()
                logger.debug(f"[OLLAMA THINKING] Contains <|start|>: {has_start_token}, Contains 'assistantfinal': {has_assistant_final}")
                ollama_response["message"]["thinking"] = thinking_full

            # Add tool calls if present
            if message.get("tool_calls"):
                ollama_response["message"]["tool_calls"] = [
                    {
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }
                    for tc in message["tool_calls"]
                ]

            return JSONResponse(content=ollama_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ollama chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tags")
async def ollama_list_models():
    """
    List available models (Ollama-compatible).
    Returns model information including size, format, and capabilities.
    """
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    model_id = config.model.model_id
    model_path = Path(config.model.base_model_path)

    # Get model file size
    try:
        if model_path.exists():
            model_size = model_path.stat().st_size
            model_modified = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        else:
            model_size = 0
            model_modified = datetime.now().isoformat()
    except Exception as e:
        logger.warning(f"Could not read model file stats: {e}")
        model_size = 0
        model_modified = datetime.now().isoformat()

    # Extract model family from model_id (e.g., "gpt-oss-20b" -> "gpt-oss")
    model_family = "-".join(model_id.split("-")[:-1]) if "-" in model_id else model_id

    # Determine parameter size from model_id (e.g., "20b" from "gpt-oss-20b")
    param_size = model_id.split("-")[-1] if "-" in model_id else "unknown"
    if param_size.endswith("b") or param_size.endswith("B"):
        param_size = param_size.upper()

    # Include :latest tag to match Ollama's convention
    model_name_with_tag = f"{model_id}:latest"

    return {
        "models": [
            {
                "name": model_name_with_tag,
                "model": model_name_with_tag,
                "modified_at": model_modified,
                "size": model_size,
                "digest": "sha256:local",  # Local model, no remote digest
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model_family,
                    "families": [model_family],
                    "parameter_size": param_size,
                    "quantization_level": "Q4_K_M"  # From our GGUF config
                }
            }
        ]
    }


@app.post("/api/show")
async def ollama_show_model(request: Request):
    """
    Show detailed model information (Ollama-compatible).
    Returns model configuration, parameters, capabilities, and template.
    """
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    # Parse request body
    try:
        body = await request.json()
        requested_model = body.get("model", config.model.model_id)
        verbose = body.get("verbose", False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body")

    model_id = config.model.model_id

    # Normalize model name to handle Ollama-style tags (e.g., 'model:latest')
    normalized_model = normalize_model_name(requested_model)

    # Validate requested model matches our model
    if normalized_model not in [model_id, "local-llm"]:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{requested_model}' not found. Available model: {model_id}:latest"
        )

    model_path = Path(config.model.base_model_path)

    # Get model file stats
    try:
        if model_path.exists():
            model_size = model_path.stat().st_size
            model_modified = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        else:
            model_size = 0
            model_modified = datetime.now().isoformat()
    except Exception as e:
        logger.warning(f"Could not read model file stats: {e}")
        model_size = 0
        model_modified = datetime.now().isoformat()

    # Extract model family from model_id
    model_family = "-".join(model_id.split("-")[:-1]) if "-" in model_id else model_id
    param_size = model_id.split("-")[-1] if "-" in model_id else "unknown"
    if param_size.endswith("b") or param_size.endswith("B"):
        param_size = param_size.upper()

    # Build parameters string from config
    parameters_lines = [
        f"temperature {config.model.temperature}",
        f"top_p {config.model.top_p}",
        f"num_ctx {config.model.context_length}",
        f"num_gpu {config.model.n_gpu_layers}",
        f"num_thread {config.model.n_threads}",
    ]

    response = {
        "modelfile": f"# Modelfile for {model_id}\nFROM {model_path}\n",
        "parameters": "\n".join(parameters_lines),
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": model_family,
            "families": [model_family],
            "parameter_size": param_size,
            "quantization_level": "Q4_K_M"
        },
        "model_info": {
            "general.architecture": model_family,
            "general.file_type": "Q4_K_M",
            "general.parameter_count": param_size,
            "general.quantization_version": 2,
            f"{model_family}.context_length": config.model.context_length,
            f"{model_family}.embedding_length": 4096,
            f"{model_family}.block_count": 32,
            f"{model_family}.attention.head_count": 32,
            f"{model_family}.attention.head_count_kv": 32,
        },
        "modified_at": model_modified,
        "capabilities": ["completion", "tools"]  # We support tool calling
    }

    # Add verbose fields if requested
    if verbose:
        response["license"] = "Project RLHFL - Local Model License"
        response["system"] = ""  # No default system prompt

    return response


@app.get("/api/ps")
async def ollama_list_running():
    """
    List currently running (loaded) models (Ollama-compatible).
    Returns information about models loaded in memory.
    """
    if not config or not llm_engine:
        return {"models": []}

    # Check if model is loaded
    if not llm_engine.is_loaded():
        return {"models": []}

    model_id = config.model.model_id
    model_path = Path(config.model.base_model_path)

    # Get model file stats
    try:
        if model_path.exists():
            model_size = model_path.stat().st_size
        else:
            model_size = 0
    except Exception:
        model_size = 0

    # Extract model family and parameter size
    model_family = "-".join(model_id.split("-")[:-1]) if "-" in model_id else model_id
    param_size = model_id.split("-")[-1] if "-" in model_id else "unknown"
    if param_size.endswith("b") or param_size.endswith("B"):
        param_size = param_size.upper()

    # Calculate expiration time (models stay loaded indefinitely in our case)
    # Set to 24 hours from now as a reasonable default
    expires_at = (datetime.now().astimezone()).isoformat()

    # Estimate VRAM usage (rough estimate based on quantization)
    # Q4_K_M uses roughly 4.5 bits per parameter
    try:
        if param_size.endswith("B"):
            param_count_str = param_size[:-1]
            param_count = float(param_count_str) * 1_000_000_000
            # 4.5 bits per parameter for Q4_K_M
            size_vram = int(param_count * 4.5 / 8)
        else:
            size_vram = int(model_size * 0.8)  # Rough estimate
    except Exception:
        size_vram = int(model_size * 0.8)

    # Include :latest tag to match Ollama's convention
    model_name_with_tag = f"{model_id}:latest"

    return {
        "models": [
            {
                "name": model_name_with_tag,
                "model": model_name_with_tag,
                "size": model_size,
                "digest": "sha256:local",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model_family,
                    "families": [model_family],
                    "parameter_size": param_size,
                    "quantization_level": "Q4_K_M"
                },
                "expires_at": expires_at,
                "size_vram": size_vram,
                "context_length": config.model.context_length
            }
        ]
    }


@app.get("/api/version")
async def ollama_version():
    """
    Get version information (Ollama-compatible).
    Returns the API version for compatibility.
    """
    return {
        "version": "0.5.1"  # Ollama API version we're compatible with
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _read_checkpoint_json(checkpoint_file: Path) -> Optional[dict]:
    """Read and parse checkpoint JSON file. Returns None on error."""
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read checkpoint file: {e}")
        return None


async def _poll_for_new_checkpoints(
    engine,
    checkpoints_path: str,
    interval_seconds: int = 30,
):
    """
    Background task that checks current_checkpoint.json for a new GGUF model
    path and triggers reload if needed. Safety net for when HTTP notification fails.
    """
    checkpoint_file = Path(checkpoints_path) / "current_checkpoint.json"

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            if not checkpoint_file.exists():
                continue

            data = await asyncio.to_thread(_read_checkpoint_json, checkpoint_file)
            if data is None:
                continue

            new_checkpoint_id = data.get("checkpoint_id")
            new_gguf_path = data.get("gguf_model_path")

            if not new_gguf_path:
                continue

            if new_checkpoint_id == engine.current_checkpoint_id:
                continue

            if not Path(new_gguf_path).exists():
                logger.debug(f"Poller: GGUF not yet available: {new_gguf_path}")
                continue

            logger.info(
                f"Poller detected new checkpoint: {new_checkpoint_id} "
                f"(current: {engine.current_checkpoint_id})"
            )

            success = await asyncio.to_thread(
                engine.reload_model,
                new_model_path=new_gguf_path,
                checkpoint_id=new_checkpoint_id,
            )

            if success:
                logger.info(f"Poller: Model reloaded to {new_checkpoint_id}")
            else:
                logger.warning(f"Poller: Model reload failed for {new_checkpoint_id}")

        except asyncio.CancelledError:
            logger.info("Checkpoint poller cancelled, shutting down")
            break
        except Exception as e:
            logger.error(f"Checkpoint poller error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)