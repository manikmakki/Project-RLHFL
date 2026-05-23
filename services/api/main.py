"""
Project RLHFL - API Service

Sidecar service that evaluates TamAGI conversations from Elasticsearch
and orchestrates SFT/DPO training. Exposes a minimal REST API for health
checks, training control, and the admin dashboard.
"""

import logging
import json
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from shared.config import load_config, settings, SystemConfig
from shared.models import HealthStatus
from api.es_memory_manager import ElasticsearchMemoryManager
from api.es_ingester import ElasticsearchIngester
from api.admin_ui import admin_router

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
config: Optional[SystemConfig] = None
memory_manager: Optional[ElasticsearchMemoryManager] = None
es_ingester: Optional[ElasticsearchIngester] = None
recent_api_requests = deque(maxlen=50)
training_in_progress = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global config, memory_manager, es_ingester

    logger.info("Starting RLHFL sidecar service...")

    thread_pool_size = int(os.environ.get('ASYNCIO_THREAD_POOL_SIZE', 16))
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=thread_pool_size))

    config = load_config()
    logger.info(f"Configuration loaded from {settings.config_path}")

    memory_manager = ElasticsearchMemoryManager(config)
    logger.info(f"Elasticsearch memory manager initialized (index: {config.elasticsearch.index})")

    await asyncio.to_thread(memory_manager.auto_cleanup_if_needed)

    if config.elasticsearch.sidecar_mode:
        es_ingester = ElasticsearchIngester(config, memory_manager)
        es_ingester.start()
        logger.info(
            f"ES ingester started (window: "
            f"{config.elasticsearch.evaluation_window.start}–"
            f"{config.elasticsearch.evaluation_window.end} "
            f"{config.elasticsearch.evaluation_window.timezone})"
        )

    try:
        from api import admin_ui
        admin_ui.set_dependencies(memory_manager, recent_api_requests, es_ingester)
    except Exception as e:
        logger.warning(f"Failed to set admin UI dependencies at startup: {e}")

    logger.info("RLHFL sidecar ready.")

    yield

    logger.info("Shutting down RLHFL sidecar...")

    if es_ingester:
        es_ingester.stop()

    logger.info("Shutdown complete")


app = FastAPI(
    title="RLHFL Sidecar",
    description="Elasticsearch-backed LLM evaluation and LoRA training service",
    version="2.0.0",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_details = [
        f"{' -> '.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')} (type: {e.get('type', '')})"
        for e in errors
    ]
    logger.warning(f"Request validation error: {'; '.join(error_details)}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request body",
                "type": "invalid_request_error",
                "param": errors[0]["loc"][-1] if errors else None,
                "details": error_details,
            }
        },
    )


@app.exception_handler(503)
async def service_unavailable_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=503,
        content={"detail": exc.detail},
        headers={"Retry-After": "120"},
    )


from api import admin_ui as _admin_ui_module
app.include_router(admin_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    es_ok = memory_manager.is_connected() if memory_manager else False

    if training_in_progress:
        status_str = "training"
    elif es_ok:
        status_str = "healthy"
    else:
        status_str = "degraded"

    return HealthStatus(
        status=status_str,
        model_loaded=not training_in_progress,
        memory_connected=es_ok,
        gpu_available=False,
        training_in_progress=training_in_progress,
        external_model_name=config.model.model_id if config else None,
    ).model_dump()


@app.post("/v1/training/trigger")
async def trigger_training():
    """Manually trigger training."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    memory_manager.request_training()
    return {"status": "training_requested", "message": "Training will begin at the next scheduled check"}


@app.get("/v1/training/stats")
async def get_training_stats():
    """Get current training statistics."""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager.get_training_stats()


def _read_checkpoint_json(checkpoint_file: Path) -> Optional[dict]:
    """Read and parse a checkpoint JSON file. Returns None on error."""
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read checkpoint file: {e}")
        return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
