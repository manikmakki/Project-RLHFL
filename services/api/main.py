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
import httpx

from shared.config import load_config, settings, SystemConfig
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
from api.llm_proxy import LLMProxy
from api.memory_manager import MemoryManager
from api.sentiment_analyzer import SentimentAnalyzer
from api.prompt_interceptor import PromptInterceptor
from api.admin_ui import admin_router
from api.psyche.orchestrator import PsycheOrchestrator
from api.psyche.superego import Superego
from api.psyche.ego_fast import EgoFast
from api.psyche.id_engine import IdEngine

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
config = None
llm_proxy = None  # HTTP proxy to external LLM
memory_manager = None
sentiment_analyzer = None
prompt_interceptor = None
psyche = None  # Psyche orchestrator (Id/Ego/Superego)
background_tasks = set()  # Track background storage tasks for graceful shutdown
recent_api_requests = deque(maxlen=50)  # Rolling buffer of last 50 API requests (includes response data)
training_in_progress = False  # Flag to indicate training is active (model unloaded)
training_start_time = None  # Timestamp when training started


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by stripping Ollama-style tags.
    Ollama uses 'model:tag' format where tag defaults to 'latest'.
    This allows clients to request 'model:tag' or 'model:latest'.
    """
    if ':' in model_name:
        # Strip the tag part (e.g., 'model:latest' -> 'model')
        return model_name.split(':')[0]
    return model_name


def should_store_interaction(model_name: str, config: SystemConfig) -> bool:
    """
    Check if model is whitelisted for sentiment analysis and storage.

    Args:
        model_name: Model name from request (may include :tag)
        config: System configuration

    Returns:
        True if model should be analyzed and stored, False otherwise
    """
    whitelist = config.llm_proxy.sentiment_enabled_models

    # Empty whitelist = analyze all models (default for single-model setups)
    if not whitelist:
        return True

    # Normalize model name (strip :latest, :32b tags)
    normalized = model_name.split(':')[0] if ':' in model_name else model_name

    # Check if in whitelist
    for whitelisted in whitelist:
        whitelisted_normalized = whitelisted.split(':')[0] if ':' in whitelisted else whitelisted
        if normalized == whitelisted_normalized:
            return True

    return False


async def store_interaction_background(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    sentiment_analyzer,
    memory_manager
):
    """
    Run sentiment analysis and storage in background (fire-and-forget).
    Never blocks the response - errors are logged but not propagated.

    Args:
        conversation_id: Unique conversation identifier
        user_message: User's message text
        assistant_response: Assistant's response text
        sentiment_analyzer: SentimentAnalyzer instance
        memory_manager: MemoryManager instance
    """
    try:
        # Run sentiment analysis
        sentiment = await asyncio.to_thread(
            sentiment_analyzer.analyze,
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_continuing=True
        )

        # Calculate weight
        weight = await asyncio.to_thread(
            sentiment_analyzer.calculate_weight,
            user_message=user_message,
            sentiment=sentiment,
            conversation_continuing=True
        )

        # Check if golden example
        is_golden = await asyncio.to_thread(
            sentiment_analyzer.is_golden_example,
            user_message=user_message,
            sentiment=sentiment,
            weight=weight
        )

        # Store interaction
        await asyncio.to_thread(
            memory_manager.store_interaction,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_response=assistant_response,
            sentiment=sentiment,
            weight=weight,
            is_golden=is_golden
        )

        logger.info(
            f"Stored interaction: sentiment={sentiment:.2f}, "
            f"weight={weight:.2f}, golden={is_golden}"
        )

    except Exception as e:
        # Log error but don't propagate (fire-and-forget)
        logger.error(f"Background storage failed: {e}", exc_info=True)


def parse_model_response(content: str) -> dict:
    """
    Parse the model's response to extract thinking and final content.
    Handles Qwen3-style <think>...</think> tags for chain-of-thought reasoning.
    Ollama typically handles this natively, but this is a fallback for raw text.

    Returns dict with 'thinking' and 'content' fields.
    """
    import re

    result = {
        'thinking': None,
        'content': content  # Default to full content if no parsing needed
    }

    # Extract <think>...</think> section (Qwen3 format)
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)

    if think_match:
        result['thinking'] = think_match.group(1).strip()
        # Content is everything after the closing </think> tag
        after_think = content[think_match.end():].strip()
        result['content'] = after_think if after_think else ''

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global config, llm_proxy, memory_manager, sentiment_analyzer, prompt_interceptor, psyche, background_tasks

    # Startup
    logger.info("Starting LLM API service (Proxy Mode)...")

    # Initialize background tasks set for graceful shutdown
    background_tasks = set()

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

    # Initialize LLM proxy (replaces local model loading)
    llm_proxy = LLMProxy(config)
    logger.info(f"LLM Proxy initialized: {llm_proxy}")

    # Initialize Psyche (Id/Ego/Superego) if enabled
    if config.psyche.enabled:
        try:
            superego = Superego(config, llm_proxy=llm_proxy)
            ego_fast = EgoFast(config)
            id_engine = IdEngine(config, sentiment_analyzer, llm_proxy=llm_proxy)
            psyche = PsycheOrchestrator(
                config=config,
                superego=superego,
                ego_fast=ego_fast,
                id_engine=id_engine,
            )
            logger.info(
                f"Psyche orchestrator initialized: "
                f"refinement={'on' if config.psyche.refinement_enabled else 'off'}, "
                f"ego_model={config.psyche.ego_model or '(main)'}, "
                f"id_model={config.psyche.id_model or '(heuristic)'}, "
                f"superego_model={config.psyche.superego_model or '(main)'}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Psyche, continuing without it: {e}")
            psyche = None
    else:
        logger.info("Psyche disabled in config")

    # Wire admin UI dependencies now that managers are initialized
    try:
        from api import admin_ui
        admin_ui.set_dependencies(memory_manager, llm_proxy, recent_api_requests, prompt_interceptor, sentiment_analyzer)
        # Add a test entry to verify the debug logging pipeline works end-to-end
        recent_api_requests.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": "startup-test",
            "model": "none",
            "stream": False,
            "duration_seconds": 0,
            "request": {"messages": [{"role": "system", "content": "Debug pipeline test — this entry confirms admin UI can read from the request log."}]},
            "response": {"content": "Pipeline OK", "finish_reason": "test"},
        })
        logger.info(f"Admin debug pipeline: test entry added, deque has {len(recent_api_requests)} items")
    except Exception as e:
        logger.warning(f"Failed to set admin UI dependencies at startup: {e}")
    
    # Checkpoint polling not needed in proxy mode (external LLM handles model loading)
    # poll_interval = config.model.checkpoint_poll_interval_seconds
    # checkpoint_poller_task = asyncio.create_task(
    #     _poll_for_new_checkpoints(llm_proxy, settings.checkpoints_path, poll_interval)
    # )
    # logger.info(f"Background checkpoint poller started ({poll_interval}s interval)")

    logger.info("LLM API service ready!")

    yield

    # Shutdown
    logger.info("Shutting down LLM API service...")

    # Wait for background tasks to complete (with timeout)
    if background_tasks:
        logger.info(f"Waiting for {len(background_tasks)} background tasks to complete...")
        done, pending = await asyncio.wait(background_tasks, timeout=30.0)
        if pending:
            logger.warning(f"{len(pending)} background tasks did not complete within timeout")
        else:
            logger.info("All background tasks completed")

    # Close LLM proxy connection pool
    if llm_proxy:
        await llm_proxy.close()

    logger.info("Shutdown complete")


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


def get_llm_proxy():
    """Provide LLM engine to endpoints."""
    if llm_proxy is None:
        raise HTTPException(status_code=503, detail="LLM engine not initialized")
    return llm_proxy


# Middleware to return 503 during training
@app.middleware("http")
async def training_mode_middleware(request: Request, call_next):
    """
    Block inference requests with 503 when training is in progress.
    Allows admin endpoints, health checks, and training control endpoints.
    """
    # Paths that are always allowed (admin UI, health, training control)
    allowed_paths = [
        "/health",
        "/admin",
        "/v1/model/unload",
        "/v1/model/reload",
        "/v1/training/trigger",
        "/v1/training/stats",
    ]

    # Check if path is allowed
    is_allowed = any(request.url.path.startswith(path) for path in allowed_paths)

    # If training in progress and path not allowed, return 503
    if training_in_progress and not is_allowed:
        estimated_wait = 300  # 5 minutes (conservative estimate)
        if training_start_time:
            elapsed = time.time() - training_start_time
            # Assume training takes ~5 minutes, decrease estimate as time passes
            estimated_wait = max(60, int(300 - elapsed))

        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable - model training in progress. Retry in ~{estimated_wait}s"
        )

    # Otherwise proceed normally
    response = await call_next(request)
    return response


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
                kwargs['llm_proxy'] = llm_proxy
                return await original_endpoint(*args, **kwargs)
            
            route.endpoint = wrapped_endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # In proxy mode, check proxy readiness instead of local model/GPU
    model_ready = llm_proxy.is_loaded() if llm_proxy else False

    # Status is "training" if training in progress, otherwise "healthy" or "degraded"
    if training_in_progress:
        status_str = "training"
    elif model_ready and memory_manager:
        status_str = "healthy"
    else:
        status_str = "degraded"

    status = HealthStatus(
        status=status_str,
        model_loaded=model_ready and not training_in_progress,
        memory_connected=memory_manager.is_connected() if memory_manager else False,
        gpu_available=False,  # Not relevant in proxy mode
        current_checkpoint=None,  # External LLM manages checkpoints
        training_in_progress=training_in_progress,
        external_model_name=config.llm_proxy.external_model_name if config else None
    )

    response = status.model_dump()
    if psyche:
        response["psyche"] = psyche.is_healthy()

    return response


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    model_id = config.model.model_id if config else "unknown-model"
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
    if not llm_proxy or not llm_proxy.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Use configured model if not specified, or validate the requested model matches
        configured_model_id = config.model.model_id if config else "unknown-model"
        request_model = request.model or configured_model_id

        # Normalize model name to handle Ollama-style tags (e.g., 'model:latest')
        normalized_model = normalize_model_name(request_model)

        if normalized_model not in [configured_model_id, normalize_model_name(configured_model_id), "local-llm"]:
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

        # Extract user message text for sentiment analysis
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

        # --- Psyche processing ---
        # Strip client sampling params when Psyche controls them
        psyche_active = psyche and psyche.enabled
        client_temperature = request.temperature
        client_top_p = request.top_p
        if psyche_active and config.psyche.override_client_params:
            logger.debug(
                f"Psyche: stripping client params (temp={request.temperature}, top_p={request.top_p})"
            )
            request.temperature = None
            request.top_p = None

        # Request snapshot for admin debug log (response added after generation)
        _debug_request = {
            "messages": [msg.model_dump(exclude_none=True) for msg in messages_with_context],
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }

        # Handle streaming
        logger.debug(f"stream={request.stream}, has_tools={bool(request.tools)}, tool_choice={request.tool_choice}")

        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    request, conversation_id, user_message_text, request_model, messages_with_context, prev_assistant_text,
                ),
                media_type="text/event-stream"
            )

        # --- Non-streaming path ---
        start_time = time.time()

        if psyche_active and psyche.refinement_enabled:
            # Psyche v2: single refine() call handles generation + refinement
            try:
                refinement_result = await psyche.refine(
                    messages=messages_with_context,
                    user_message_text=user_message_text,
                    conversation_id=conversation_id,
                    llm_proxy=llm_proxy,
                    has_tools=bool(request.tools),
                )

                if not refinement_result.validation.allowed:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request blocked: {refinement_result.validation.reason}"
                    )

                response_dict = refinement_result.best_response_dict
                response_text = response_dict.get("content") or ""
                tool_calls = response_dict.get("tool_calls")
                thinking = response_dict.get("thinking")
                finish_reason = response_dict.get("finish_reason", "stop")

                # Include deliberation as thinking context when enabled
                if config.psyche.expose_deliberation and refinement_result.deliberation_log:
                    delib_text = refinement_result.deliberation_log
                    thinking = f"{delib_text}\n\n{thinking}" if thinking else delib_text

                logger.info(
                    f"Psyche refine complete: turns={refinement_result.turns_used}/{refinement_result.turns_budget}, "
                    f"score={refinement_result.final_quality_score:.2f}, "
                    f"complexity={refinement_result.complexity_class}, "
                    f"refusal={refinement_result.refusal_detected}"
                )

                # Store refusal as negative training signal for DPO
                if refinement_result.refusal_detected:
                    try:
                        await asyncio.to_thread(
                            memory_manager.store_interaction,
                            conversation_id=conversation_id,
                            user_message=user_message_text,
                            assistant_response=response_text,
                            sentiment=-0.8,
                            weight=2.0,
                            is_refusal=True,
                            psyche_metadata={
                                "refusal_detected": True,
                                "refinement_turns": refinement_result.turns_used,
                                "complexity_class": refinement_result.complexity_class,
                            },
                        )
                        logger.info("Stored refusal interaction as DPO training signal")
                    except Exception as store_err:
                        logger.error(f"Failed to store refusal interaction: {store_err}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Psyche refine failed, falling back to direct generation: {e}", exc_info=True)
                response_dict = await llm_proxy.generate(
                    messages=messages_with_context,
                    temperature=client_temperature,
                    top_p=client_top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice
                )
                response_text = response_dict.get("content") or ""
                tool_calls = response_dict.get("tool_calls")
                thinking = response_dict.get("thinking")
                finish_reason = response_dict.get("finish_reason", "stop")

        elif psyche_active:
            # Psyche v1 legacy: pre_process → generate → post_process
            psyche_pre_result = None
            try:
                psyche_pre_result = await psyche.pre_process(
                    messages=messages_with_context,
                    user_message_text=user_message_text,
                    conversation_id=conversation_id,
                )
                if not psyche_pre_result.validation.allowed:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request blocked: {psyche_pre_result.validation.reason}"
                    )
                messages_with_context = psyche_pre_result.messages
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Psyche pre-process failed, continuing without it: {e}")

            response_dict = await llm_proxy.generate(
                messages=messages_with_context,
                temperature=client_temperature,
                top_p=client_top_p,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice
            )
            response_text = response_dict.get("content") or ""
            tool_calls = response_dict.get("tool_calls")
            thinking = response_dict.get("thinking")
            finish_reason = response_dict.get("finish_reason", "stop")

            if psyche_pre_result:
                try:
                    psyche_post_result = await psyche.post_process(
                        response_dict=response_dict,
                        conversation_id=conversation_id,
                        pre_result=psyche_pre_result,
                        messages=list(request.messages),
                        llm_proxy=llm_proxy,
                    )
                    response_dict = psyche_post_result.response_dict
                    response_text = response_dict.get("content") or ""
                    tool_calls = response_dict.get("tool_calls")
                    thinking = response_dict.get("thinking") or thinking
                    finish_reason = response_dict.get("finish_reason", finish_reason)

                    if psyche_post_result.refusal_detected:
                        try:
                            original_refusal = psyche_post_result.metadata.get("original_refusal", response_text)
                            await asyncio.to_thread(
                                memory_manager.store_interaction,
                                conversation_id=conversation_id,
                                user_message=user_message_text,
                                assistant_response=original_refusal,
                                sentiment=-0.8,
                                weight=2.0,
                                is_refusal=True,
                                psyche_metadata={
                                    "refusal_detected": True,
                                    "retry_attempted": psyche_post_result.retry_attempted,
                                    "retry_succeeded": psyche_post_result.metadata.get("retry_succeeded"),
                                },
                            )
                        except Exception as store_err:
                            logger.error(f"Failed to store refusal interaction: {store_err}")
                except Exception as e:
                    logger.error(f"Psyche post-process failed: {e}", exc_info=True)

        else:
            # No Psyche — direct generation
            response_dict = await llm_proxy.generate(
                messages=messages_with_context,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice
            )
            response_text = response_dict.get("content") or ""
            tool_calls = response_dict.get("tool_calls")
            thinking = response_dict.get("thinking")
            finish_reason = response_dict.get("finish_reason", "stop")

        generation_time = time.time() - start_time
        logger.debug(f"Parsed response - finish_reason={finish_reason}, has_content={bool(response_text)}, thinking={bool(thinking)}, tool_calls={tool_calls}")

        # Log request + response to admin debug buffer
        try:
            _debug_psyche = None
            if psyche_active:
                _ref = locals().get('refinement_result')
                _debug_psyche = {
                    "active": True,
                    "refinement": getattr(psyche, 'refinement_enabled', False),
                    "turns": _ref.turns_used if _ref else None,
                    "quality_score": round(_ref.final_quality_score, 2) if _ref else None,
                    "complexity": _ref.complexity_class if _ref else None,
                }
            recent_api_requests.append({
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/v1/chat/completions",
                "model": request_model,
                "stream": request.stream,
                "duration_seconds": round(generation_time, 2),
                "request": _debug_request,
                "response": {
                    "content": response_text if response_text else None,
                    "tool_calls": tool_calls,
                    "thinking": thinking if thinking else None,
                    "finish_reason": finish_reason,
                },
                "psyche": _debug_psyche,
            })
            logger.debug(f"Admin debug log: appended non-streaming entry, deque now has {len(recent_api_requests)} items")
        except Exception as e:
            logger.error(f"Failed to append to admin debug log: {e}", exc_info=True)

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
    prev_assistant_text: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses in OpenAI format."""
    try:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        psyche_active = psyche and psyche.enabled
        use_refinement = psyche_active and psyche.refinement_enabled

        # ----------------------------------------------------------------
        # When psyche/refinement or tools are active, ALL internal LLM
        # generation is non-streaming. Psyche needs the complete response
        # to evaluate/refine. The final approved response is delivered
        # as SSE chunks to the client.
        #
        # When psyche is disabled, true token-by-token streaming is used.
        # ----------------------------------------------------------------

        if use_refinement or request.tools:
            if use_refinement:
                logger.debug("Streaming with psyche refinement: running refine() then converting to SSE")

                try:
                    refinement_result = await psyche.refine(
                        messages=messages_with_context,
                        user_message_text=user_message,
                        conversation_id=conversation_id,
                        llm_proxy=llm_proxy,
                        has_tools=bool(request.tools),
                    )

                    if not refinement_result.validation.allowed:
                        error_chunk = {
                            'id': completion_id, 'object': 'chat.completion.chunk',
                            'created': created, 'model': model_id,
                            'error': {'message': f"Blocked: {refinement_result.validation.reason}", 'type': 'content_filter'}
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    response_dict = refinement_result.best_response_dict
                    full_response = response_dict.get("content") or ""
                    tool_calls = response_dict.get("tool_calls")
                    thinking = response_dict.get("thinking")
                    finish_reason = response_dict.get("finish_reason", "stop")

                    # Include deliberation as thinking context
                    if config.psyche.expose_deliberation and refinement_result.deliberation_log:
                        delib_text = refinement_result.deliberation_log
                        thinking = f"{delib_text}\n\n{thinking}" if thinking else delib_text

                    logger.info(
                        f"Psyche refine (streaming): turns={refinement_result.turns_used}/{refinement_result.turns_budget}, "
                        f"score={refinement_result.final_quality_score:.2f}, "
                        f"complexity={refinement_result.complexity_class}"
                    )

                    # Store refusal as DPO signal
                    if refinement_result.refusal_detected:
                        try:
                            await asyncio.to_thread(
                                memory_manager.store_interaction,
                                conversation_id=conversation_id,
                                user_message=user_message,
                                assistant_response=full_response,
                                sentiment=-0.8,
                                weight=2.0,
                                is_refusal=True,
                                psyche_metadata={
                                    "refusal_detected": True,
                                    "refinement_turns": refinement_result.turns_used,
                                },
                            )
                        except Exception as store_err:
                            logger.error(f"Failed to store refusal: {store_err}")

                except Exception as e:
                    logger.error(f"Psyche refine (streaming) failed, falling back: {e}", exc_info=True)
                    response_dict = await llm_proxy.generate(
                        messages=messages_with_context,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        tools=request.tools,
                        tool_choice=request.tool_choice
                    )
                    full_response = response_dict.get("content") or ""
                    tool_calls = response_dict.get("tool_calls")
                    thinking = response_dict.get("thinking")
                    finish_reason = response_dict.get("finish_reason", "stop")

            else:
                # Tools present but no refinement — direct generation
                logger.debug("Streaming with tools: generating complete response then converting to SSE")
                response_dict = await llm_proxy.generate(
                    messages=messages_with_context,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice
                )
                full_response = response_dict.get("content") or ""
                tool_calls = response_dict.get("tool_calls")
                thinking = response_dict.get("thinking")
                finish_reason = response_dict.get("finish_reason", "stop")

            # --- Deliver final approved response as SSE chunks ---

            # Initial chunk (role + empty content)
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

            # Content chunk
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
                yield f"data: {json.dumps(content_chunk)}\n\n"

            # Tool calls chunk
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
                yield f"data: {json.dumps(tool_chunk)}\n\n"

            # Final chunk
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
            yield f"data: {json.dumps(final_chunk)}\n\n"

        else:
            # True token-by-token streaming (psyche disabled, no tools)
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
            stream_gen = llm_proxy.generate_stream(
                messages=messages_with_context,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                tools=None,
                tool_choice=None
            )

            async for chunk in stream_gen:
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

        # Log streaming request + response to admin debug buffer
        # (MUST be before final yield — code after last yield may not execute)
        try:
            stream_time = time.time() - created
            _stream_tool_calls = locals().get('tool_calls')
            _stream_psyche = None
            if use_refinement:
                _ref = locals().get('refinement_result')
                if _ref:
                    _stream_psyche = {
                        "active": True,
                        "refinement": True,
                        "turns": _ref.turns_used,
                        "quality_score": round(_ref.final_quality_score, 2),
                        "complexity": _ref.complexity_class,
                    }
            recent_api_requests.append({
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/v1/chat/completions",
                "model": model_id,
                "stream": True,
                "duration_seconds": round(stream_time, 2),
                "request": {
                    "messages": [msg.model_dump(exclude_none=True) for msg in messages_with_context],
                    "tools": [t if isinstance(t, dict) else t.model_dump() if hasattr(t, 'model_dump') else str(t) for t in request.tools] if request.tools else None,
                    "tool_choice": request.tool_choice if isinstance(request.tool_choice, (str, dict, type(None))) else str(request.tool_choice),
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                },
                "response": {
                    "content": full_response if full_response else None,
                    "tool_calls": _stream_tool_calls,
                    "finish_reason": finish_reason,
                },
                "psyche": _stream_psyche,
            })
            logger.debug(f"Admin debug log: appended streaming entry, deque now has {len(recent_api_requests)} items")
        except Exception as e:
            logger.error(f"Failed to append streaming entry to admin debug log: {e}", exc_info=True)

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
    Unload model from Ollama to free GPU vRAM for training.

    Sets training_in_progress flag to true, causing 503 responses for inference.
    Returns immediately after requesting Ollama to unload (keep_alive: 0).
    """
    global training_in_progress, training_start_time

    if not llm_proxy:
        raise HTTPException(status_code=503, detail="Proxy not configured")

    try:
        # Set training flag BEFORE unloading
        training_in_progress = True
        training_start_time = time.time()
        logger.info("Training mode activated - API will return 503 for inference requests")

        # Call Ollama to unload the model (keep_alive: 0)
        await llm_proxy.unload_model()

        return {
            "success": True,
            "message": f"Model {config.llm_proxy.external_model_name} unloaded from Ollama, GPU vRAM freed for training"
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {e}", exc_info=True)
        # Reset flag on failure
        training_in_progress = False
        training_start_time = None
        return {
            "success": False,
            "message": f"Failed to unload model: {str(e)}"
        }


@app.post("/v1/model/reload")
async def reload_model(request_body: ModelReloadRequest):
    """
    Reload model in Ollama after training completes.

    Clears training_in_progress flag to resume normal inference.
    The request_body is ignored in proxy mode (Ollama manages model files).
    """
    global training_in_progress, training_start_time

    if not llm_proxy:
        raise HTTPException(status_code=503, detail="Proxy not configured")

    try:
        # Call Ollama to reload the model (keep_alive: 5m)
        await llm_proxy.load_model()

        # Clear training flag AFTER successful reload
        training_in_progress = False
        elapsed = time.time() - training_start_time if training_start_time else 0
        training_start_time = None

        logger.info(f"Training mode deactivated after {elapsed:.1f}s - API resuming normal inference")

        return ModelReloadResponse(
            success=True,
            message=f"Model {config.llm_proxy.external_model_name} reloaded in Ollama",
            previous_model_path=None,  # Not applicable in proxy mode
            new_model_path=None,  # Not applicable in proxy mode
            checkpoint_id=request_body.checkpoint_id
        )
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        return ModelReloadResponse(
            success=False,
            message=f"Failed to reload model: {str(e)}",
            checkpoint_id=request_body.checkpoint_id
        )


# ============================================================================
# Ollama API Compatibility Endpoints
# ============================================================================

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """
    Transparent proxy to external Ollama instance.
    Forwards requests unchanged and captures messages in background for training.
    Psyche (Id/Ego/Superego) processing is applied identically to /v1/chat/completions.
    """
    if not llm_proxy:
        raise HTTPException(status_code=503, detail="Proxy not configured")

    try:
        # Read raw request body
        request_body = await request.body()
        ollama_req = json.loads(request_body)

        # Extract info for background storage (fire-and-forget)
        messages = ollama_req.get("messages", [])
        model_name = ollama_req.get("model", "")
        conversation_id = ollama_req.get("conversation_id", str(uuid.uuid4()))

        # Build external Ollama endpoint URL
        endpoint_url = f"{llm_proxy.base_url.rstrip('/')}/api/chat"

        # Forward headers (exclude host and content-length as they'll be auto-set)
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ['host', 'content-length']
        }

        # Debug logging for tool calling
        if "tools" in ollama_req:
            logger.info(f"Request with tools - model: {model_name}, num_tools: {len(ollama_req.get('tools', []))}")
            logger.debug(f"Forward headers: {dict(forward_headers)}")
            logger.debug(f"Request body preview: {request_body[:500]}")

        # --- Psyche processing ---
        psyche_active = psyche and psyche.enabled and messages
        use_refinement = psyche_active and psyche.refinement_enabled

        # Strip client sampling params when Psyche controls them
        if psyche_active and config.psyche.override_client_params:
            opts = ollama_req.get("options", {})
            stripped = {k: v for k, v in opts.items() if k not in ("temperature", "top_p", "top_k")}
            if stripped != opts:
                logger.debug(f"Psyche: stripping client Ollama options (had: {opts})")
                ollama_req["options"] = stripped
                request_body = json.dumps(ollama_req).encode("utf-8")

        # Convert messages for Psyche
        chat_messages = None
        user_message_text = ""
        if psyche_active:
            chat_messages = [
                ChatMessage(
                    role=MessageRole(msg.get("role", "user")),
                    content=msg.get("content", ""),
                )
                for msg in messages
            ]
            user_message_text = next(
                (msg.get("content", "") for msg in reversed(messages)
                 if msg.get("role") == "user"),
                ""
            )

        # If refinement is enabled, Psyche handles generation internally via refine()
        # so we skip the transparent proxy entirely and synthesize the Ollama response.
        refinement_result = None
        if use_refinement and chat_messages:
            try:
                refinement_result = await psyche.refine(
                    messages=chat_messages,
                    user_message_text=user_message_text,
                    conversation_id=conversation_id,
                    llm_proxy=llm_proxy,
                    has_tools="tools" in ollama_req,
                )
                if not refinement_result.validation.allowed:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request blocked: {refinement_result.validation.reason}"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Psyche refine failed on /api/chat, falling through to proxy: {e}", exc_info=True)
                refinement_result = None

        # Legacy pre-process fallback (when refinement disabled but psyche enabled)
        psyche_pre_result = None
        if psyche_active and not use_refinement and not refinement_result and chat_messages:
            try:
                psyche_pre_result = await psyche.pre_process(
                    messages=chat_messages,
                    user_message_text=user_message_text,
                    conversation_id=conversation_id,
                )
                if not psyche_pre_result.validation.allowed:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request blocked: {psyche_pre_result.validation.reason}"
                    )
                enriched_messages = psyche_pre_result.messages
                ollama_req["messages"] = [
                    {"role": msg.role.value, "content": msg.get_text_content() or ""}
                    for msg in enriched_messages
                ]
                request_body = json.dumps(ollama_req).encode("utf-8")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Psyche pre-process failed on /api/chat, continuing without it: {e}")

        async def transparent_proxy_stream():
            """
            Stream response from external Ollama.

            When Psyche refinement is active, refine() already ran above and
            produced refinement_result — we synthesize Ollama NDJSON from it.

            When legacy psyche is active (pre/post), buffer, evaluate, deliver.
            When psyche is disabled, bytes pass through transparently.
            """
            accumulated_response = ""
            done = False
            start_time = time.time()

            try:
                if refinement_result is not None:
                    # --- Psyche v2: refinement already completed, synthesize Ollama response ---
                    resp_content = refinement_result.best_response_dict.get("content") or ""
                    accumulated_response = resp_content

                    # Build thinking content from deliberation log
                    thinking_content = ""
                    if config.psyche.expose_deliberation and refinement_result.deliberation_log:
                        thinking_content = refinement_result.deliberation_log
                    orig_thinking = refinement_result.best_response_dict.get("thinking")
                    if orig_thinking:
                        thinking_content = f"{thinking_content}\n\n{orig_thinking}" if thinking_content else orig_thinking

                    # Synthesize Ollama NDJSON response
                    synthetic = {
                        "model": ollama_req.get("model", ""),
                        "created_at": datetime.now().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": resp_content,
                        },
                        "done": True,
                        "done_reason": "stop",
                    }
                    yield json.dumps(synthetic) + "\n"
                    done = True

                    logger.info(
                        f"Psyche refine (/api/chat): turns={refinement_result.turns_used}/{refinement_result.turns_budget}, "
                        f"score={refinement_result.final_quality_score:.2f}, "
                        f"complexity={refinement_result.complexity_class}"
                    )

                    # Store refusal as DPO signal
                    if refinement_result.refusal_detected:
                        try:
                            await asyncio.to_thread(
                                memory_manager.store_interaction,
                                conversation_id=conversation_id,
                                user_message=user_message_text,
                                assistant_response=resp_content,
                                sentiment=-0.8,
                                weight=2.0,
                                is_refusal=True,
                                psyche_metadata={
                                    "refusal_detected": True,
                                    "refinement_turns": refinement_result.turns_used,
                                },
                            )
                        except Exception as store_err:
                            logger.error(f"Failed to store refusal: {store_err}")

                elif psyche_pre_result is not None:
                    # --- Legacy psyche: buffer, post-process, deliver ---
                    logger.debug("Legacy psyche on /api/chat: buffering response for evaluation")
                    buffered_lines = []

                    async with llm_proxy.client.stream(
                        "POST",
                        endpoint_url,
                        content=request_body,
                        headers=forward_headers
                    ) as response:
                        if response.status_code >= 400:
                            error_detail = await response.aread()
                            error_text = error_detail.decode('utf-8') if error_detail else f"HTTP {response.status_code}"
                            logger.error(f"External Ollama error {response.status_code}: {error_text}")
                            yield error_text
                            return

                        async for line in response.aiter_lines():
                            buffered_lines.append(line)
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    if chunk.get("message", {}).get("content"):
                                        accumulated_response += chunk["message"]["content"]
                                    if chunk.get("done"):
                                        done = True
                                except json.JSONDecodeError:
                                    pass

                    response_dict = {"content": accumulated_response, "finish_reason": "stop" if done else None}
                    try:
                        psyche_post = await psyche.post_process(
                            response_dict=response_dict,
                            conversation_id=conversation_id,
                            pre_result=psyche_pre_result,
                            messages=[
                                ChatMessage(role=MessageRole(m.get("role", "user")), content=m.get("content", ""))
                                for m in messages
                            ],
                            llm_proxy=llm_proxy,
                        )

                        retried_content = psyche_post.response_dict.get("content") or ""
                        if psyche_post.retry_attempted and retried_content and retried_content != accumulated_response:
                            synthetic_response = json.dumps({
                                "model": ollama_req.get("model", ""),
                                "created_at": datetime.now().isoformat() + "Z",
                                "message": {"role": "assistant", "content": retried_content},
                                "done": True,
                                "done_reason": "stop",
                            })
                            buffered_lines = [synthetic_response]
                            accumulated_response = retried_content

                        if psyche_post.refusal_detected:
                            try:
                                original_refusal = psyche_post.metadata.get("original_refusal", accumulated_response)
                                await asyncio.to_thread(
                                    memory_manager.store_interaction,
                                    conversation_id=conversation_id,
                                    user_message=user_message_text,
                                    assistant_response=original_refusal,
                                    sentiment=-0.8,
                                    weight=2.0,
                                    is_refusal=True,
                                    psyche_metadata={"refusal_detected": True, "retry_attempted": psyche_post.retry_attempted},
                                )
                            except Exception as store_err:
                                logger.error(f"Failed to store refusal: {store_err}")

                    except Exception as e:
                        logger.error(f"Legacy psyche post-process on /api/chat failed: {e}", exc_info=True)

                    for line in buffered_lines:
                        yield line + "\n"

                else:
                    # --- Psyche disabled: transparent pass-through ---
                    async with llm_proxy.client.stream(
                        "POST",
                        endpoint_url,
                        content=request_body,
                        headers=forward_headers
                    ) as response:
                        if response.status_code >= 400:
                            error_detail = await response.aread()
                            error_text = error_detail.decode('utf-8') if error_detail else f"HTTP {response.status_code}"
                            logger.error(f"External Ollama error {response.status_code}: {error_text}")
                            yield error_text
                            return

                        accumulated_tool_calls = None
                        async for line in response.aiter_lines():
                            yield line + "\n"

                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    if chunk.get("message", {}).get("content"):
                                        accumulated_response += chunk["message"]["content"]
                                    if chunk.get("message", {}).get("tool_calls"):
                                        accumulated_tool_calls = chunk["message"]["tool_calls"]
                                    if chunk.get("done"):
                                        done = True
                                except json.JSONDecodeError:
                                    pass

                # Log to Admin UI for debugging (runs for all branches: refinement, legacy psyche, pass-through)
                try:
                    elapsed_time = time.time() - start_time
                    recent_api_requests.append({
                        "timestamp": datetime.now().isoformat(),
                        "endpoint": "/api/chat",
                        "model": model_name,
                        "duration_seconds": round(elapsed_time, 2),
                        "request": {
                            "messages": [{"role": m.get("role", "?"), "content": m.get("content", "")} for m in messages] if messages else [],
                            "has_tools": "tools" in ollama_req,
                            "tools": ollama_req.get("tools"),
                        },
                        "response": {
                            "content": accumulated_response if accumulated_response else None,
                            "length": len(accumulated_response),
                            "done": done,
                        },
                    })
                    logger.info(f"Admin debug log: appended /api/chat entry, deque now has {len(recent_api_requests)} items")
                except Exception as e:
                    logger.error(f"Failed to append /api/chat entry to admin debug log: {e}", exc_info=True)

                # Background storage (fire-and-forget) after stream completes
                if done and messages:
                    # Check model whitelist
                    should_store = should_store_interaction(model_name, config)

                    if should_store:
                        # Extract current user message (n) - their question/reaction
                        # Filter out system prompts masquerading as user messages (start with ###)
                        user_message = None
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                content = msg.get("content", "").strip()
                                # Skip system prompts (start with ###)
                                if not content.startswith("###"):
                                    user_message = content
                                    break
                                else:
                                    logger.debug(f"Filtered system prompt from storage: {content[:100]}...")

                        # Extract previous assistant message (n-1) - what they're reacting to
                        # This is the RLHF pairing: user's reaction to previous assistant response
                        prev_assistant_message = next(
                            (msg.get("content", "") for msg in reversed(messages)
                             if msg.get("role") == "assistant"),
                            None
                        )

                        # Store ONLY when we have proper N:N-1 pair (user reaction + previous assistant)
                        # This captures the user's sentiment about the previous assistant response
                        if user_message and prev_assistant_message:
                            # Create background task (fire-and-forget)
                            task = asyncio.create_task(
                                store_interaction_background(
                                    conversation_id=conversation_id,
                                    user_message=user_message,
                                    assistant_response=prev_assistant_message,
                                    sentiment_analyzer=sentiment_analyzer,
                                    memory_manager=memory_manager
                                )
                            )
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)
                            logger.debug(
                                f"Stored N:N-1 pair for model '{model_name}' "
                                f"(user_len={len(user_message)}, assistant_len={len(prev_assistant_message)})"
                            )
                        elif user_message and not prev_assistant_message:
                            logger.debug(f"Skipped storage - first turn, no previous assistant (N:N-1 required)")
                        else:
                            logger.debug(f"Skipped storage - no valid user message (likely system prompt)")
                    else:
                        logger.debug(f"Model '{model_name}' not whitelisted - skipping storage")

            except httpx.HTTPStatusError as e:
                # This shouldn't happen anymore since we check status_code above
                # But keep as fallback
                logger.error(f"External Ollama HTTP error {e.response.status_code}: {str(e)}")
                error_chunk = {"error": f"Upstream HTTP error: {e.response.status_code}"}
                yield json.dumps(error_chunk) + "\n"
            except Exception as e:
                logger.error(f"Proxy error: {e}", exc_info=True)
                # Yield error instead of raising to avoid "response already started" errors
                error_chunk = {"error": f"Proxy error: {str(e)}"}
                yield json.dumps(error_chunk) + "\n"

        return StreamingResponse(
            transparent_proxy_stream(),
            media_type="application/x-ndjson"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ollama proxy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def ollama_proxy_catchall(request: Request, path: str):
    """
    Catch-all transparent proxy for all other Ollama API endpoints.
    Forwards requests to external Ollama unchanged.

    Handles: /api/tags, /api/show, /api/ps, /api/version, /api/pull, /api/push, etc.
    Note: /api/chat is handled separately for message capture.
    """
    if not llm_proxy:
        raise HTTPException(status_code=503, detail="Proxy not configured")

    try:
        start_time = time.time()

        # Build external Ollama endpoint URL
        endpoint_url = f"{llm_proxy.base_url.rstrip('/')}/api/{path}"

        # Read request body
        request_body = await request.body()

        # Forward headers (exclude host and content-length)
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ['host', 'content-length']
        }

        # Forward request to external Ollama
        response = await llm_proxy.client.request(
            method=request.method,
            url=endpoint_url,
            content=request_body if request_body else None,
            headers=forward_headers,
            params=request.query_params
        )

        # Log to Admin UI for debugging
        elapsed_time = time.time() - start_time
        recent_api_requests.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": f"/api/{path}",
            "method": request.method,
            "status_code": response.status_code,
            "duration_seconds": round(elapsed_time, 2)
        })

        # Return response with same status code and content type
        return JSONResponse(
            content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in ['content-encoding', 'content-length', 'transfer-encoding']}
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"External Ollama error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Proxy error for /api/{path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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