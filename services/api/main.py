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
from typing import AsyncGenerator
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
    HealthStatus
)
from api.llm_engine import LLMEngine
from api.memory_manager import MemoryManager
from api.sentiment_analyzer import SentimentAnalyzer
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
recent_api_requests = deque(maxlen=20)  # Rolling buffer of last 20 API requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global config, llm_engine, memory_manager, sentiment_analyzer

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
    
    # Initialize LLM engine
    llm_engine = LLMEngine(config, settings.model_path)
    logger.info("LLM engine initialized")
    # Wire admin UI dependencies now that managers are initialized
    try:
        from api import admin_ui
        admin_ui.set_dependencies(memory_manager, llm_engine, recent_api_requests)
    except Exception:
        logger.warning("Failed to set admin UI dependencies at startup")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected, running on CPU")
    
    logger.info("LLM API service ready!")
    
    yield
    
    # Shutdown
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
    
    status = HealthStatus(
        status="healthy" if llm_engine and llm_engine.is_loaded() else "degraded",
        model_loaded=llm_engine.is_loaded() if llm_engine else False,
        memory_connected=memory_manager.is_connected() if memory_manager else False,
        gpu_available=gpu_available,
        current_checkpoint=llm_engine.get_current_adapter() if llm_engine else None
    )
    
    return status


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    model_id = config.model.model_id if config else "mistral-7b-instruct"
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
    # Log the API request for debugging
    recent_api_requests.append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/v1/chat/completions",
        "request": request.model_dump()
    })

    if not llm_engine or not llm_engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Use configured model if not specified, or validate the requested model matches
        configured_model_id = config.model.model_id if config else "mistral-7b-instruct"
        request_model = request.model or configured_model_id
        
        if request_model not in [configured_model_id, "local-llm"]:  # Allow legacy "local-llm" for compatibility
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

        # Retrieve relevant context from memory using RAG (if enabled)
        user_message_text = user_message.get_text_content()
        messages_with_context = list(request.messages)

        if config.memory.rag_enabled and not request.tools:
            # Skip RAG when tools are present — the Mistral template merges
            # the system message into the last [INST] block, and lengthy RAG
            # context between [AVAILABLE_TOOLS] and the user query causes
            # the model to ignore tools and answer directly instead.

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

        # Handle streaming
        # Note: chatml-function-calling does not support streaming with auto tool_choice.
        # Fall back to non-streaming when tools are present with auto/unset tool_choice.
        use_streaming = request.stream
        if use_streaming and request.tools and (not request.tool_choice or request.tool_choice == "auto"):
            logger.info("Falling back to non-streaming: tools with auto tool_choice does not support streaming")
            use_streaming = False

        logger.debug(f"use_streaming={use_streaming}, has_tools={bool(request.tools)}, tool_choice={request.tool_choice}")

        if use_streaming:
            return StreamingResponse(
                stream_chat_completion(
                    request, conversation_id, user_message_text, request_model, messages_with_context
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
        finish_reason = response_dict.get("finish_reason", "stop")

        logger.debug(f"Parsed response - finish_reason={finish_reason}, has_content={bool(response_text)}, tool_calls={tool_calls}")

        # Only analyze sentiment and store if this is a final response (not a tool call)
        if finish_reason == "stop" and response_text:
            # Infer sentiment (only uses USER message text, not system prompts or context)
            # Run sentiment analysis in thread pool to avoid blocking
            sentiment = await asyncio.to_thread(
                sentiment_analyzer.analyze,
                user_message=user_message_text,
                assistant_response=response_text,
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

            # Store interaction in memory (only stores USER message text for analysis)
            # Run embedding generation and DB storage in thread pool
            await asyncio.to_thread(
                memory_manager.store_interaction,
                conversation_id=conversation_id,
                user_message=user_message_text,
                assistant_response=response_text,
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
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_text if response_text else None,
            tool_calls=tool_calls
        )
        logger.debug(f"Built assistant_message: content={assistant_message.content!r}, tool_calls={assistant_message.tool_calls}")

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

        return completion_response
        
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
    messages_with_context: list[ChatMessage]
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses in OpenAI format with RAG context."""
    try:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

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

        # Stream tokens with context
        # Wrap the synchronous generator to avoid blocking event loop
        full_response = ""
        finish_reason = "stop"

        # Sentinel value for end of stream
        _STREAM_END = object()

        # Create a wrapper that safely gets the next chunk without raising StopIteration into a future
        def get_next_chunk(generator):
            try:
                return next(generator)
            except StopIteration:
                return _STREAM_END

        # Create the stream generator
        def get_stream_generator():
            return llm_engine.generate_stream(
                messages=messages_with_context,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice
            )

        # Get the generator in a thread pool
        stream_gen = await asyncio.to_thread(get_stream_generator)

        # Stream chunks (each next() call is wrapped to avoid blocking)
        while True:
            chunk = await asyncio.to_thread(get_next_chunk, stream_gen)

            if chunk is _STREAM_END:
                break

            # Extract delta from chunk
            if 'choices' in chunk and len(chunk['choices']) > 0:
                choice = chunk['choices'][0]
                delta = choice.get('delta', {})

                # Accumulate content if present
                if 'content' in delta and delta['content']:
                    full_response += delta['content']

                # Track finish_reason if present
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']

                # Forward the chunk to client (preserving tool_calls if any)
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

        yield "data: [DONE]\n\n"

        # Only store interaction if this was a final response (not a tool call)
        if finish_reason == "stop" and full_response:
            # Run in thread pool to avoid blocking event loop
            sentiment = await asyncio.to_thread(
                sentiment_analyzer.analyze,
                user_message=user_message,
                assistant_response=full_response,
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
                assistant_response=full_response,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)