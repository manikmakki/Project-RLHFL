"""
Admin UI endpoints for Project RLHFL web interface.
"""

import asyncio
import json
import logging
import psutil
import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from pathlib import Path

from shared.config import settings, load_config

logger = logging.getLogger(__name__)

# Create router
admin_router = APIRouter(prefix="/admin", tags=["admin"])

# These will be set by main.py
memory_manager = None
api_request_log = None
es_ingester = None
def set_dependencies(mm, request_log=None, ingester=None):
    """Set global dependencies from main.py."""
    global memory_manager, api_request_log, es_ingester
    memory_manager = mm
    api_request_log = request_log
    es_ingester = ingester


@admin_router.get("/", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the admin dashboard HTML."""
    html_path = Path(__file__).parent / "static" / "admin.html"
    
    if html_path.exists():
        with open(html_path, 'r') as f:
            return f.read()
    else:
        return """
        <html><body>
        <h1>Admin Dashboard</h1>
        <p>Admin interface file not found. Please ensure static/admin.html exists.</p>
        </body></html>
        """


@admin_router.get("/api/stats")
async def get_admin_stats():
    """Get comprehensive system statistics."""
    try:
        if not memory_manager:
            return {
                "training": {"total_interactions": 0, "new_since_training": 0, "hours_since_interaction": 0, "days_since_training": 0, "last_training": None},
                "evaluation": {"evaluated_count": 0, "unevaluated_count": 0, "golden_count": 0, "noise_count": 0, "avg_quality": 0.0, "sidecar_mode": False},
                "system": {"current_checkpoint": None, "memory_usage_mb": 0, "cpu_percent": 0}
            }

        # Training stats
        training_stats = memory_manager.get_training_stats()
        config = load_config()

        # Evaluation pipeline stats from ES
        eval_stats = {"evaluated_count": 0, "unevaluated_count": 0, "golden_count": 0, "noise_count": 0, "avg_quality": 0.0}
        try:
            agg_result = memory_manager.client.search(
                index=memory_manager.index,
                query={"bool": {"filter": [{"term": {"data.type": "conversation"}}]}},
                aggs={
                    "evaluated": {"filter": {"exists": {"field": "meta.rlhfl_evaluated_at"}}},
                    "unevaluated": {"filter": {"bool": {"must_not": [{"exists": {"field": "meta.rlhfl_evaluated_at"}}]}}},
                    "golden": {"filter": {"term": {"meta.rlhfl_is_golden": True}}},
                    "noise": {"filter": {"term": {"meta.rlhfl_is_noise": True}}},
                    "avg_quality": {"avg": {"field": "meta.rlhfl_quality"}},
                },
                size=0,
            )
            aggs = agg_result["aggregations"]
            eval_stats = {
                "evaluated_count": aggs["evaluated"]["doc_count"],
                "unevaluated_count": aggs["unevaluated"]["doc_count"],
                "golden_count": aggs["golden"]["doc_count"],
                "noise_count": aggs["noise"]["doc_count"],
                "avg_quality": round(aggs["avg_quality"].get("value") or 0.0, 2),
                "sidecar_mode": config.elasticsearch.sidecar_mode,
                "evaluation_window": f"{config.elasticsearch.evaluation_window.start}–{config.elasticsearch.evaluation_window.end} {config.elasticsearch.evaluation_window.timezone}",
                "batch_size": config.elasticsearch.batch_size,
            }
        except Exception as e:
            logger.warning(f"Failed to get evaluation stats: {e}")

        # System resources
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Current checkpoint
        current_checkpoint = None
        try:
            cp_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
            if cp_file.exists():
                current_checkpoint = json.load(open(cp_file)).get("checkpoint_id")
        except Exception:
            pass

        return {
            "training": {
                "total_interactions": training_stats.total_interactions,
                "new_since_training": training_stats.new_interactions_since_last_training,
                "hours_since_interaction": round(training_stats.hours_since_last_interaction, 2),
                "days_since_training": round(training_stats.days_since_last_training, 2),
                "last_training": training_stats.last_training_timestamp.isoformat() if training_stats.last_training_timestamp else None,
                "new_positive": training_stats.new_positive_count,
                "new_negative": training_stats.new_negative_count,
                "new_golden": training_stats.new_golden_count,
            },
            "evaluation": eval_stats,
            "system": {
                "current_checkpoint": current_checkpoint,
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "model_id": config.model.model_id,
                "llama_cpp_url": config.llama_cpp.base_url,
            },
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}", exc_info=True)
        return {
            "error": str(e),
            "training": {"total_interactions": 0, "new_since_training": 0, "hours_since_interaction": 0, "days_since_training": 0, "last_training": None},
            "evaluation": {"evaluated_count": 0, "unevaluated_count": 0, "golden_count": 0, "noise_count": 0, "avg_quality": 0.0},
            "system": {"current_checkpoint": None, "memory_usage_mb": 0, "cpu_percent": 0}
        }


@admin_router.get("/api/memories")
async def get_memories(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    sentiment_filter: Optional[str] = Query(None, regex="^(positive|negative|neutral)$"),
    is_golden: Optional[str] = Query(None),
    hours: Optional[int] = Query(None, ge=1)
):
    """Get paginated list of interactions."""
    try:
        if not memory_manager:
            return {"total": 0, "skip": skip, "limit": limit, "items": []}

        interactions = memory_manager.get_interactions_since(None)

        # Apply sentiment filter
        if sentiment_filter:
            if sentiment_filter == "positive":
                interactions = [i for i in interactions if i.sentiment > 0.3]
            elif sentiment_filter == "negative":
                interactions = [i for i in interactions if i.sentiment < -0.3]
            elif sentiment_filter == "neutral":
                interactions = [i for i in interactions if abs(i.sentiment) <= 0.3]

        # Apply is_golden filter
        if is_golden:
            is_golden_bool = is_golden.lower() == "true"
            interactions = [i for i in interactions if i.metadata.get("rlhfl_is_golden", False) == is_golden_bool]

        # Apply time filter
        if hours:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(hours=hours)
            interactions = [i for i in interactions if i.timestamp >= cutoff]

        # Sort by timestamp descending
        interactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Paginate
        total = len(interactions)
        paginated = interactions[skip:skip + limit]
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "items": [
                {
                    "id": i.id,
                    "timestamp": i.timestamp.isoformat(),
                    "conversation_id": i.conversation_id,
                    "user_message": i.user_message[:200],
                    "assistant_response": i.assistant_response[:200],
                    "sentiment": round(i.sentiment, 3),
                    "weight": round(i.weight, 2),
                    "is_golden": i.metadata.get("rlhfl_is_golden", False),
                    "metadata": {
                        "rlhfl_quality": i.metadata.get("rlhfl_quality"),
                        "rlhfl_topics": i.metadata.get("rlhfl_topics", []),
                        "rlhfl_reasoning": i.metadata.get("rlhfl_reasoning", ""),
                        "rlhfl_is_golden": i.metadata.get("rlhfl_is_golden", False),
                        "rlhfl_is_noise": i.metadata.get("rlhfl_is_noise", False),
                    }
                }
                for i in paginated
            ]
        }
    except Exception as e:
        logger.error(f"Error getting memories: {e}", exc_info=True)
        return {"total": 0, "skip": skip, "limit": limit, "items": [], "error": str(e)}


@admin_router.get("/api/memories/{interaction_id}")
async def get_memory_detail(interaction_id: str):
    """Get full details of a specific interaction."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        result = memory_manager.client.get(index=memory_manager.index, id=interaction_id)
        src = result["_source"]
        meta = src.get("meta", {})

        return {
            "id": interaction_id,
            "conversation_id": meta.get("conversation_id", interaction_id),
            "timestamp": src.get("@timestamp"),
            "user_message": meta.get("rlhfl_user_message", ""),
            "assistant_response": meta.get("rlhfl_assistant_response", ""),
            "sentiment": meta.get("rlhfl_sentiment", 0.0),
            "weight": meta.get("rlhfl_weight", 1.0),
            "is_golden": meta.get("rlhfl_is_golden", False),
            "quality": meta.get("rlhfl_quality"),
            "topics": meta.get("rlhfl_topics", []),
            "reasoning": meta.get("rlhfl_reasoning", ""),
            "evaluated_at": meta.get("rlhfl_evaluated_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.put("/api/memories/{interaction_id}")
async def update_memory(interaction_id: str, updates: dict):
    """Update rlhfl evaluation fields on a specific interaction."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        patch: dict = {}
        if "sentiment" in updates:
            patch["rlhfl_sentiment"] = float(updates["sentiment"])
        if "weight" in updates:
            patch["rlhfl_weight"] = float(updates["weight"])
        if "is_golden" in updates:
            patch["rlhfl_is_golden"] = bool(updates["is_golden"])

        if not patch:
            raise HTTPException(status_code=400, detail="No updatable fields provided")

        ok = memory_manager.update_with_evaluation(interaction_id, patch)
        if not ok:
            raise HTTPException(status_code=500, detail="ES update failed")

        logger.info(f"Updated interaction: {interaction_id}")
        return {"status": "updated", "id": interaction_id, "updates": patch}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/memories/{interaction_id}/toggle-golden")
async def toggle_golden(interaction_id: str):
    """Toggle the golden example status of an interaction."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        result = memory_manager.client.get(index=memory_manager.index, id=interaction_id)
        current = result["_source"].get("meta", {}).get("rlhfl_is_golden", False)
        new_golden = not current

        ok = memory_manager.update_with_evaluation(interaction_id, {"rlhfl_is_golden": new_golden})
        if not ok:
            raise HTTPException(status_code=500, detail="ES update failed")

        logger.info(f"{'Added to' if new_golden else 'Removed from'} golden examples: {interaction_id}")
        return {"status": "toggled", "id": interaction_id, "is_golden": new_golden}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling golden status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/golden/re-evaluate")
async def reevaluate_golden_examples():
    """Queue all golden examples for re-evaluation by clearing their rlhfl_evaluated_at field.
    The ingester will re-evaluate them during the next evaluation window."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        golden_interactions = await asyncio.to_thread(memory_manager.get_golden_examples)

        if not golden_interactions:
            return {"status": "complete", "total_golden": 0, "queued": 0}

        queued = 0
        for interaction in golden_interactions:
            ok = await asyncio.to_thread(
                memory_manager.update_with_evaluation,
                interaction.id,
                {"rlhfl_evaluated_at": None},
            )
            if ok:
                queued += 1

        logger.info(f"Golden re-evaluation: queued {queued}/{len(golden_interactions)} docs")
        return {
            "status": "complete",
            "total_golden": len(golden_interactions),
            "queued": queued,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing golden examples for re-evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.delete("/api/memories/{interaction_id}")
async def delete_memory(interaction_id: str):
    """Mark an interaction as noise so it is excluded from training.
    RLHFL does not own TamAGI's documents; instead of deleting, sets rlhfl_is_noise=true."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        ok = await asyncio.to_thread(
            memory_manager.update_with_evaluation,
            interaction_id,
            {"rlhfl_is_noise": True, "rlhfl_is_golden": False},
        )
        if not ok:
            raise HTTPException(status_code=500, detail="ES update failed")

        logger.info(f"Marked as noise: {interaction_id}")
        return {"status": "excluded", "id": interaction_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking memory as noise: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/api/recent-interactions")
async def get_recent_interactions(
    limit: int = Query(20, ge=1, le=100)
):
    """Get recent interactions with full details for debugging."""
    try:
        if not memory_manager:
            return {"total": 0, "items": []}

        # Get all interactions
        interactions = memory_manager.get_interactions_since(None)

        # Sort by timestamp descending
        interactions.sort(key=lambda x: x.timestamp, reverse=True)

        # Limit results
        recent = interactions[:limit]

        return {
            "total": len(interactions),
            "returned": len(recent),
            "items": [
                {
                    "id": i.id,
                    "timestamp": i.timestamp.isoformat(),
                    "conversation_id": i.conversation_id,
                    "user_message": i.user_message,
                    "assistant_response": i.assistant_response,
                    "sentiment": round(i.sentiment, 3),
                    "weight": round(i.weight, 2),
                    "is_golden": i.metadata.get("rlhfl_is_golden", False)
                }
                for i in recent
            ]
        }
    except Exception as e:
        logger.error(f"Error getting recent interactions: {e}", exc_info=True)
        return {"total": 0, "items": [], "error": str(e)}


@admin_router.get("/api/recent-requests")
async def get_recent_requests():
    """Get recent API requests for debugging."""
    try:
        if api_request_log is None:
            logger.warning("api_request_log is None — set_dependencies may not have been called")
            return {"total": 0, "items": []}

        logger.debug(f"recent-requests endpoint: deque has {len(api_request_log)} items")

        # Convert deque to list (most recent first)
        requests = list(reversed(api_request_log))

        # Use jsonable_encoder to safely serialize Pydantic models, datetimes, etc.
        encoded = jsonable_encoder(requests)

        return {
            "total": len(encoded),
            "items": encoded
        }
    except Exception as e:
        logger.error(f"Error getting recent requests: {e}", exc_info=True)
        return {"total": 0, "items": [], "error": str(e)}


@admin_router.get("/api/checkpoints")
async def get_checkpoints():
    """List all training checkpoints."""
    try:
        checkpoints_path = Path(settings.checkpoints_path)
        checkpoints = []

        if checkpoints_path.exists():
            for metadata_file in checkpoints_path.glob("*_metadata.json"):
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)

                    # Auto-detect GGUF files if not in metadata (for manual conversions)
                    if not data.get("gguf_model_path"):
                        checkpoint_id = data.get("checkpoint_id")
                        if checkpoint_id:
                            checkpoint_dir = checkpoints_path / checkpoint_id
                            gguf_files = list(checkpoint_dir.glob("*.gguf"))
                            if gguf_files:
                                # Prefer Q4_K_M quantized version, then any other GGUF
                                q4_files = [f for f in gguf_files if "Q4_K_M" in f.name or "q4" in f.name.lower()]
                                if q4_files:
                                    data["gguf_model_path"] = str(q4_files[0])
                                else:
                                    data["gguf_model_path"] = str(gguf_files[0])

                    checkpoints.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {metadata_file}: {e}")

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return {"checkpoints": checkpoints}
    except Exception as e:
        logger.error(f"Error getting checkpoints: {e}", exc_info=True)
        return {"checkpoints": [], "error": str(e)}


@admin_router.get("/api/training/history")
async def get_training_history():
    """Get training history with metrics over time for charting."""
    try:
        checkpoints_path = Path(settings.checkpoints_path)
        training_history = []

        if checkpoints_path.exists():
            import json

            # Get all checkpoint directories
            checkpoint_dirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]

            for checkpoint_dir in checkpoint_dirs:
                # Check for sequential training structure (pass_* directories)
                pass_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('pass_')])

                if pass_dirs:
                    # Sequential training - collect data from each pass
                    for pass_dir in pass_dirs:
                        metadata_file = pass_dir / "training_metadata.json"

                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)

                                # Extract pass number and layers from directory name
                                parts = pass_dir.name.split("_")
                                pass_num = int(parts[1])
                                layers = f"{parts[3]}-{parts[4]}"

                                history_entry = {
                                    "timestamp": metadata.get("timestamp"),
                                    "train_loss": metadata.get("train_loss"),
                                    "eval_loss": metadata.get("eval_loss"),
                                    "train_samples": metadata.get("train_samples"),
                                    "val_samples": metadata.get("val_samples"),
                                    "epochs": metadata.get("epochs"),
                                    "checkpoint_name": f"{checkpoint_dir.name} - Pass {pass_num} (Layers {layers})"
                                }

                                # Try to get more detailed metrics from trainer_state.json
                                checkpoint_subdirs = [d for d in pass_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
                                if checkpoint_subdirs:
                                    checkpoint_subdirs.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
                                    trainer_state_file = checkpoint_subdirs[-1] / "trainer_state.json"

                                    if trainer_state_file.exists():
                                        try:
                                            with open(trainer_state_file, 'r') as f:
                                                trainer_state = json.load(f)

                                            log_history = trainer_state.get("log_history", [])
                                            if log_history:
                                                last_log = log_history[-1]
                                                if last_log.get("eval_loss") is not None:
                                                    history_entry["eval_loss"] = last_log.get("eval_loss")
                                                history_entry["eval_runtime"] = last_log.get("eval_runtime")
                                                history_entry["global_step"] = trainer_state.get("global_step")
                                        except Exception as e:
                                            logger.warning(f"Failed to load trainer state from {trainer_state_file}: {e}")

                                training_history.append(history_entry)
                            except Exception as e:
                                logger.warning(f"Failed to load checkpoint metadata {metadata_file}: {e}")
                else:
                    # Standard training - metadata at top level
                    metadata_file = checkpoint_dir / "training_metadata.json"

                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)

                            history_entry = {
                                "timestamp": metadata.get("timestamp"),
                                "train_loss": metadata.get("train_loss"),
                                "eval_loss": metadata.get("eval_loss"),
                                "train_samples": metadata.get("train_samples"),
                                "val_samples": metadata.get("val_samples"),
                                "epochs": metadata.get("epochs"),
                                "checkpoint_name": checkpoint_dir.name
                            }

                            # Try to get more detailed metrics from trainer_state.json
                            checkpoint_subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
                            if checkpoint_subdirs:
                                checkpoint_subdirs.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
                                trainer_state_file = checkpoint_subdirs[-1] / "trainer_state.json"

                                if trainer_state_file.exists():
                                    try:
                                        with open(trainer_state_file, 'r') as f:
                                            trainer_state = json.load(f)

                                        log_history = trainer_state.get("log_history", [])
                                        if log_history:
                                            last_log = log_history[-1]
                                            # Override with trainer_state values if available
                                            if last_log.get("eval_loss") is not None:
                                                history_entry["eval_loss"] = last_log.get("eval_loss")
                                            history_entry["eval_runtime"] = last_log.get("eval_runtime")
                                            history_entry["global_step"] = trainer_state.get("global_step")
                                    except Exception as e:
                                        logger.warning(f"Failed to load trainer state from {trainer_state_file}: {e}")

                            training_history.append(history_entry)
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint metadata {metadata_file}: {e}")

        # Sort by timestamp
        training_history.sort(key=lambda x: x.get('timestamp', ''))

        return {
            "success": True,
            "history": training_history,
            "total_checkpoints": len(training_history)
        }
    except Exception as e:
        logger.error(f"Error getting training history: {e}", exc_info=True)
        return {
            "success": False,
            "history": [],
            "total_checkpoints": 0,
            "error": str(e)
        }


@admin_router.get("/api/database/info")
async def get_database_info():
    """Get ES index size and interaction statistics."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)
        golden_examples = await asyncio.to_thread(memory_manager.get_golden_examples)

        max_size_gb = memory_manager.config.elasticsearch.max_index_size_gb

        return {
            "size_gb": round(size_gb, 3),
            "max_size_gb": max_size_gb,
            "usage_percentage": round((size_gb / max_size_gb) * 100, 1) if max_size_gb > 0 else 0,
            "total_interactions": len(all_interactions),
            "golden_examples": len(golden_examples),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/database/cleanup/age")
async def cleanup_by_age(days: Optional[int] = None):
    """
    Report interactions older than the specified threshold.
    RLHFL does not delete TamAGI's documents; age filtering is applied at query time.

    Args:
        days: Number of days threshold. Uses config default if not provided.
    """
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        age_days = days if days is not None else memory_manager.config.elasticsearch.max_memory_age_days
        await asyncio.to_thread(memory_manager.cleanup_old_memories)

        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)

        return {
            "success": True,
            "message": (
                f"Age-based query filter active (threshold: {age_days} days). "
                "RLHFL does not delete TamAGI documents; old interactions are excluded at query time."
            ),
            "deleted_count": 0,
            "current_size_gb": round(size_gb, 3),
            "remaining_interactions": len(all_interactions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during age-based cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/database/cleanup/weight")
async def cleanup_by_weight(threshold: Optional[float] = None):
    """
    Weight-based filtering is applied at query time via rlhfl_weight sort + rlhfl_is_noise flag.
    RLHFL does not delete TamAGI's documents.
    Use the DELETE /api/memories/{id} endpoint to mark individual interactions as noise.
    """
    size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb) if memory_manager else 0.0
    return {
        "success": True,
        "message": "Weight-based filtering is applied at query time. No documents deleted.",
        "deleted_count": 0,
        "current_size_gb": round(size_gb, 3),
    }


@admin_router.post("/api/database/cleanup/lowest")
async def cleanup_lowest_weight(percentage: Optional[float] = None):
    """
    Weight-based filtering is applied at query time via rlhfl_weight sort + rlhfl_is_noise flag.
    RLHFL does not delete TamAGI's documents.
    Use the DELETE /api/memories/{id} endpoint to mark individual interactions as noise.
    """
    size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb) if memory_manager else 0.0
    return {
        "success": True,
        "message": "Weight-based filtering is applied at query time. No documents deleted.",
        "deleted_count": 0,
        "current_size_gb": round(size_gb, 3),
    }


@admin_router.get("/api/model/status")
async def get_model_status():
    """Get current model and training pipeline status."""
    try:
        # Import main.py's global state for training status
        from api.main import training_in_progress, config

        result = {
            "training_in_progress": training_in_progress,
            "current_checkpoint_id": None,
            "model_name": config.model.model_id if config else None,
        }

        # Read current_checkpoint.json for checkpoint ID, metrics, deployment info
        checkpoint_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    cp_data = json.load(f)
                # Use checkpoint_id from file if not already set
                if not result["current_checkpoint_id"]:
                    result["current_checkpoint_id"] = cp_data.get("checkpoint_id")
                result["adapter_path"] = cp_data.get("adapter_path")
                result["deployed_metrics"] = cp_data.get("metrics")
                result["deployed_at"] = cp_data.get("deployed_at")
            except Exception:
                pass

        return result
    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/training/trigger")
async def trigger_training():
    """Trigger a training run from the admin UI (immediate execution, bypasses schedule)."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        # Request manual training (bypasses schedule, auto-selects mode)
        memory_manager.request_training(reason="manual", training_mode="auto")

        return {
            "success": True,
            "message": "Manual training requested. It will begin immediately (bypasses schedule).",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/evaluation/trigger")
async def trigger_evaluation():
    """Force an immediate evaluation batch, bypassing the off-hours window check."""
    if not es_ingester:
        raise HTTPException(status_code=503, detail="Ingester not initialized")
    try:
        evaluated = await asyncio.get_event_loop().run_in_executor(
            None, es_ingester._run_batch
        )
        return {
            "success": True,
            "evaluated": evaluated,
            "message": f"Evaluation batch complete: {evaluated} document(s) evaluated.",
        }
    except Exception as e:
        logger.error(f"Error triggering evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/api/training/live-status")
async def get_training_live_status():
    """Get real-time training status including CPU usage, current pass, and progress."""
    try:
        import subprocess
        import re

        status = {
            "training_active": False,
            "training_mode": "unknown",
            "cpu_percent": 0,
            "ram_usage": "0B",
            "ram_percent": 0,
            "current_pass": None,
            "total_passes": None,
            "current_layers": None,
            "current_epoch": None,
            "total_epochs": None,
            "last_log_time": None,
            "logs": []
        }

        # Check if trainer container is running and get stats
        try:
            result = subprocess.run(
                ["docker", "stats", "llm-trainer", "--no-stream", "--format", "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|')
                if len(parts) >= 3:
                    status["cpu_percent"] = float(parts[0].rstrip('%'))
                    status["ram_usage"] = parts[1].split('/')[0].strip()
                    status["ram_percent"] = float(parts[2].rstrip('%'))

                    # If CPU > 1000%, training is likely active
                    if status["cpu_percent"] > 1000:
                        status["training_active"] = True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            pass

        # Get recent training logs to extract progress info
        try:
            result = subprocess.run(
                ["docker", "logs", "llm-trainer", "--tail", "100"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                logs = result.stdout + result.stderr

                # Extract training mode
                if "CPU TRAINING MODE ENABLED" in logs:
                    status["training_mode"] = "CPU"
                elif "Loading base model" in logs and "8-bit quantization" in logs:
                    status["training_mode"] = "GPU (8-bit)"

                # Extract current pass and layers
                pass_match = re.search(r'PASS (\d+)/(\d+): Training layers (\d+)-(\d+)', logs)
                if pass_match:
                    status["current_pass"] = int(pass_match.group(1))
                    status["total_passes"] = int(pass_match.group(2))
                    status["current_layers"] = f"{pass_match.group(3)}-{pass_match.group(4)}"

                # Extract epoch info
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', logs)
                if epoch_match:
                    status["current_epoch"] = int(epoch_match.group(1))
                    status["total_epochs"] = int(epoch_match.group(2))

                # Get last few relevant log lines
                log_lines = logs.split('\n')
                relevant_logs = []
                for line in reversed(log_lines[-50:]):
                    if any(keyword in line for keyword in ['Training', 'PASS', 'epoch', 'loss', 'CPU', 'Merging', 'complete']):
                        relevant_logs.append(line.strip())
                        if len(relevant_logs) >= 10:
                            break

                status["logs"] = list(reversed(relevant_logs))

                # Get last log timestamp
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', logs)
                if timestamp_match:
                    status["last_log_time"] = timestamp_match.group(1)

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

        return status

    except Exception as e:
        logger.error(f"Error getting training status: {e}", exc_info=True)
        return {
            "training_active": False,
            "error": str(e)
        }


@admin_router.get("/api/training/progress")
async def get_training_progress():
    """Get training progress from checkpoint metadata files (no docker needed)"""
    try:
        checkpoints_dir = Path(os.environ.get("CHECKPOINTS_PATH", "/checkpoints"))

        # Find the most recent checkpoint directory
        checkpoint_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not checkpoint_dirs:
            return {
                "training_active": False,
                "message": "No training checkpoints found"
            }

        latest_checkpoint = checkpoint_dirs[0]

        # Scan for pass directories
        pass_dirs = sorted(
            [d for d in latest_checkpoint.iterdir() if d.is_dir() and d.name.startswith("pass_")],
            key=lambda x: int(x.name.split("_")[1])
        )

        if not pass_dirs:
            return {
                "training_active": False,
                "message": "No training passes found in checkpoint"
            }

        # Collect data from all passes
        passes_data = []
        total_passes = len(pass_dirs)
        current_pass = None

        for pass_dir in pass_dirs:
            # Read training metadata
            metadata_file = pass_dir / "training_metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Find the latest checkpoint within this pass
            checkpoints = sorted(
                [d for d in pass_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]),
                reverse=True
            )

            trainer_state = None
            if checkpoints:
                trainer_state_file = checkpoints[0] / "trainer_state.json"
                if trainer_state_file.exists():
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)

            # Extract pass number and layers from directory name
            # Format: pass_1_layers_0_4
            parts = pass_dir.name.split("_")
            pass_num = int(parts[1])
            layers = f"{parts[3]}-{parts[4]}"

            pass_info = {
                "pass_number": pass_num,
                "layers": layers,
                "train_loss": metadata.get("train_loss"),
                "train_samples": metadata.get("train_samples"),
                "val_samples": metadata.get("val_samples"),
                "epochs": metadata.get("epochs"),
                "timestamp": metadata.get("timestamp"),
                "completed": True
            }

            if trainer_state:
                pass_info["epoch"] = trainer_state.get("epoch")
                pass_info["global_step"] = trainer_state.get("global_step")
                pass_info["max_steps"] = trainer_state.get("max_steps")

                # Get the latest eval loss from log_history
                log_history = trainer_state.get("log_history", [])
                if log_history:
                    latest_log = log_history[-1]
                    pass_info["eval_loss"] = latest_log.get("eval_loss")
                    pass_info["eval_runtime"] = latest_log.get("eval_runtime")

            passes_data.append(pass_info)

        # Check if training is currently active by looking at timestamps
        if passes_data:
            latest_pass = passes_data[-1]
            latest_timestamp = datetime.fromisoformat(latest_pass["timestamp"].replace("Z", "+00:00"))
            time_since_update = datetime.now(latest_timestamp.tzinfo) - latest_timestamp

            # Consider training active if last update was within 10 minutes
            training_active = time_since_update.total_seconds() < 600

            # If active and not all passes complete, we're on the current pass
            if training_active and len(passes_data) < total_passes:
                current_pass = len(passes_data)
                passes_data[-1]["completed"] = False
        else:
            training_active = False

        return {
            "training_active": training_active,
            "checkpoint_id": latest_checkpoint.name,
            "total_passes": total_passes,
            "current_pass": current_pass or total_passes,
            "passes": passes_data,
            "training_mode": "CPU"  # From config
        }

    except Exception as e:
        logger.error(f"Error getting training progress: {e}", exc_info=True)
        return {
            "training_active": False,
            "error": str(e)
        }


@admin_router.post("/api/model/rollback")
async def rollback_model(checkpoint_id: str = Query(..., description="Checkpoint ID to deploy")):
    """Deploy a specific checkpoint's LoRA adapter to llama-server."""
    try:
        from shared.llama_cpp_deployer import LlamaCppDeployer

        checkpoints_path = Path(settings.checkpoints_path)
        checkpoint_dir = checkpoints_path / checkpoint_id

        if not checkpoint_dir.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_id}")

        adapter_dir = checkpoint_dir / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise HTTPException(status_code=400, detail=f"No adapter found in checkpoint {checkpoint_id}")

        config = load_config()
        deployer = LlamaCppDeployer(config.llama_cpp)

        success = await asyncio.to_thread(
            deployer.deploy_lora,
            adapter_path=str(adapter_dir),
            checkpoint_id=checkpoint_id,
        )

        if not success:
            raise HTTPException(status_code=500, detail="llama.cpp deployment failed. Check logs.")

        # Update current_checkpoint.json
        metadata_file = checkpoints_path / f"{checkpoint_id}_metadata.json"
        metrics = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
                metrics = meta.get("metrics", {})
                meta["deployed"] = True
            with open(metadata_file, "w") as f:
                json.dump(meta, f, indent=2)

        from datetime import datetime
        cp_data = {
            "checkpoint_id": checkpoint_id,
            "adapter_path": str(adapter_dir),
            "metrics": metrics,
            "deployed_at": datetime.now().isoformat(),
        }
        with open(checkpoints_path / "current_checkpoint.json", "w") as f:
            json.dump(cp_data, f, indent=2)

        return {"success": True, "message": f"Deployed checkpoint {checkpoint_id} to llama-server"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying checkpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/model/reload-base")
async def reload_base_model():
    """Unload all LoRA adapters from llama-server, reverting to the base model."""
    try:
        import requests as req
        config = load_config()

        # POST empty adapter list to unload all adapters
        response = req.post(
            f"{config.llama_cpp.base_url.rstrip('/')}/lora",
            json=[],
            timeout=30,
        )
        response.raise_for_status()

        # Clear current_checkpoint.json
        cp_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
        if cp_file.exists():
            cp_file.unlink()

        return {"success": True, "message": "All adapters unloaded — llama-server running base model"}

    except Exception as e:
        logger.error(f"Error reverting to base model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#  llama.cpp Server Management
# ------------------------------------------------------------------ #

@admin_router.get("/api/model/server-status")
async def get_llm_server_status():
    """Query the llama-server health and active LoRA adapters."""
    try:
        from shared.llama_cpp_deployer import LlamaCppDeployer
        config = load_config()
        deployer = LlamaCppDeployer(config.llama_cpp)
        status = await asyncio.to_thread(deployer.get_server_status)
        return {"server_url": config.llama_cpp.base_url, **status}
    except Exception as e:
        logger.error(f"Error querying server status: {e}", exc_info=True)
        return {"server_url": "", "reachable": False, "healthy": False, "loaded_adapters": [], "error": str(e)}


@admin_router.post("/api/model/deploy-lora")
async def deploy_lora_adapter(checkpoint_id: str = Query(..., description="Checkpoint ID to deploy")):
    """Convert and load a checkpoint's LoRA adapter into llama-server via POST /lora."""
    try:
        from shared.llama_cpp_deployer import LlamaCppDeployer

        checkpoints_path = Path(settings.checkpoints_path)
        checkpoint_dir = checkpoints_path / checkpoint_id

        if not checkpoint_dir.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_id}")

        adapter_dir = checkpoint_dir / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise HTTPException(status_code=400, detail=f"No adapter found in checkpoint {checkpoint_id}")

        config = load_config()
        deployer = LlamaCppDeployer(config.llama_cpp)
        logger.info(f"Admin UI: deploying {checkpoint_id} to llama-server")

        success = await asyncio.to_thread(
            deployer.deploy_lora,
            adapter_path=str(adapter_dir),
            checkpoint_id=checkpoint_id,
        )

        if success:
            return {
                "success": True,
                "message": f"Checkpoint {checkpoint_id} loaded into llama-server",
                "checkpoint_id": checkpoint_id,
            }
        raise HTTPException(status_code=500, detail="Deployment failed — check logs for details")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying LoRA adapter: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deployment error: {str(e)}")


