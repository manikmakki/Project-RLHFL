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
from fastapi.responses import HTMLResponse
from pathlib import Path

from shared.config import settings

logger = logging.getLogger(__name__)

# Create router
admin_router = APIRouter(prefix="/admin", tags=["admin"])

# These will be set by main.py
memory_manager = None
llm_engine = None
api_request_log = None
prompt_interceptor = None


def set_dependencies(mm, engine, request_log=None, interceptor=None):
    """Set global dependencies from main.py."""
    global memory_manager, llm_engine, api_request_log, prompt_interceptor
    memory_manager = mm
    llm_engine = engine
    api_request_log = request_log
    prompt_interceptor = interceptor


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
    logger.info(f"admin_ui: memory_manager={memory_manager}, llm_engine={llm_engine}")
    try:
        if not memory_manager or not llm_engine:
            return {
                "training": {"total_interactions": 0, "new_since_training": 0, "hours_since_interaction": 0, "days_since_training": 0, "last_training": None},
                "memory": {"total_interactions": 0, "golden_examples": 0, "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}},
                "system": {"model_loaded": False, "current_checkpoint": None, "memory_usage_mb": 0, "cpu_percent": 0}
            }
        
        # Training stats
        training_stats = memory_manager.get_training_stats()
        
        # Memory stats
        all_interactions = memory_manager.get_interactions_since(None)
        golden_examples = memory_manager.get_golden_examples()
        
        # Sentiment distribution
        positive = sum(1 for i in all_interactions if i.sentiment > 0.3)
        negative = sum(1 for i in all_interactions if i.sentiment < -0.3)
        neutral = len(all_interactions) - positive - negative
        
        # System resources
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "training": {
                "total_interactions": training_stats.total_interactions,
                "new_since_training": training_stats.new_interactions_since_last_training,
                "hours_since_interaction": round(training_stats.hours_since_last_interaction, 2),
                "days_since_training": round(training_stats.days_since_last_training, 2),
                "last_training": training_stats.last_training_timestamp.isoformat() if training_stats.last_training_timestamp else None
            },
            "memory": {
                "total_interactions": len(all_interactions),
                "golden_examples": len(golden_examples),
                "sentiment_distribution": {
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral
                }
            },
            "system": {
                "model_loaded": llm_engine.is_loaded(),
                "is_reloading": llm_engine.is_reloading(),
                "current_checkpoint": llm_engine.current_checkpoint_id,
                "model_path": getattr(llm_engine, 'model_path', None),
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}", exc_info=True)
        return {
            "error": str(e),
            "training": {"total_interactions": 0, "new_since_training": 0, "hours_since_interaction": 0, "days_since_training": 0, "last_training": None},
            "memory": {"total_interactions": 0, "golden_examples": 0, "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}},
            "system": {"model_loaded": False, "current_checkpoint": None, "memory_usage_mb": 0, "cpu_percent": 0}
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
            interactions = [i for i in interactions if i.metadata.get("is_golden", False) == is_golden_bool]

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
                    "user_message": i.user_message[:200],  # Truncate for display
                    "assistant_response": i.assistant_response[:200],
                    "sentiment": round(i.sentiment, 3),
                    "weight": round(i.weight, 2),
                    "is_golden": i.metadata.get("is_golden", False)
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

        # Get from ChromaDB
        results = memory_manager.collection.get(
            ids=[interaction_id],
            include=["metadatas", "documents"]
        )

        if not results['ids']:
            raise HTTPException(status_code=404, detail="Interaction not found")

        metadata = results['metadatas'][0]

        return {
            "id": results['ids'][0],
            "conversation_id": metadata['conversation_id'],
            "timestamp": metadata['timestamp'],
            "user_message": metadata['user_message'],
            "assistant_response": metadata['assistant_response'],
            "sentiment": metadata.get('sentiment', 0.0),
            "weight": metadata.get('weight', 1.0),
            "is_golden": metadata.get('is_golden', False),
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.put("/api/memories/{interaction_id}")
async def update_memory(interaction_id: str, updates: dict):
    """Update a specific interaction's metadata."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        # Get existing data
        results = memory_manager.collection.get(
            ids=[interaction_id],
            include=["metadatas", "documents", "embeddings"]
        )

        if not results['ids']:
            raise HTTPException(status_code=404, detail="Interaction not found")

        metadata = results['metadatas'][0]

        # Update allowed fields
        if 'sentiment' in updates:
            metadata['sentiment'] = float(updates['sentiment'])
        if 'weight' in updates:
            metadata['weight'] = float(updates['weight'])
        if 'is_golden' in updates:
            metadata['is_golden'] = bool(updates['is_golden'])

        # Update in main collection
        memory_manager.collection.update(
            ids=[interaction_id],
            metadatas=[metadata]
        )

        # Handle golden collection
        is_golden = metadata.get('is_golden', False)
        try:
            golden_results = memory_manager.golden_collection.get(ids=[interaction_id])
            exists_in_golden = len(golden_results['ids']) > 0
        except:
            exists_in_golden = False

        if is_golden and not exists_in_golden:
            # Add to golden collection
            memory_manager.golden_collection.add(
                ids=[interaction_id],
                embeddings=[results['embeddings'][0]],
                documents=[results['documents'][0]],
                metadatas=[metadata]
            )
        elif not is_golden and exists_in_golden:
            # Remove from golden collection
            memory_manager.golden_collection.delete(ids=[interaction_id])
        elif is_golden and exists_in_golden:
            # Update in golden collection
            memory_manager.golden_collection.update(
                ids=[interaction_id],
                metadatas=[metadata]
            )

        logger.info(f"Updated interaction: {interaction_id}")
        return {"status": "updated", "id": interaction_id, "metadata": metadata}
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

        # Get existing data
        results = memory_manager.collection.get(
            ids=[interaction_id],
            include=["metadatas", "documents", "embeddings"]
        )

        if not results['ids']:
            raise HTTPException(status_code=404, detail="Interaction not found")

        metadata = results['metadatas'][0]
        current_golden = metadata.get('is_golden', False)
        new_golden = not current_golden
        metadata['is_golden'] = new_golden

        # Update in main collection
        memory_manager.collection.update(
            ids=[interaction_id],
            metadatas=[metadata]
        )

        # Handle golden collection
        if new_golden:
            # Add to golden collection
            memory_manager.golden_collection.add(
                ids=[interaction_id],
                embeddings=[results['embeddings'][0]],
                documents=[results['documents'][0]],
                metadatas=[metadata]
            )
            logger.info(f"Added to golden examples: {interaction_id}")
        else:
            # Remove from golden collection
            try:
                memory_manager.golden_collection.delete(ids=[interaction_id])
                logger.info(f"Removed from golden examples: {interaction_id}")
            except:
                pass

        return {
            "status": "toggled",
            "id": interaction_id,
            "is_golden": new_golden
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling golden status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.delete("/api/memories/{interaction_id}")
async def delete_memory(interaction_id: str):
    """Delete a specific interaction."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        # Delete from ChromaDB
        memory_manager.collection.delete(ids=[interaction_id])

        # Also try to delete from golden examples if present
        try:
            memory_manager.golden_collection.delete(ids=[interaction_id])
        except:
            pass  # May not exist in golden collection

        logger.info(f"Deleted interaction: {interaction_id}")
        return {"status": "deleted", "id": interaction_id}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
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
                    "is_golden": i.metadata.get("is_golden", False)
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
        if not api_request_log:
            return {"total": 0, "items": []}

        # Convert deque to list (most recent first)
        requests = list(reversed(api_request_log))

        return {
            "total": len(requests),
            "items": requests
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
                # Read training_metadata.json
                metadata_file = checkpoint_dir / "training_metadata.json"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        history_entry = {
                            "timestamp": metadata.get("timestamp"),
                            "train_loss": metadata.get("train_loss"),
                            "train_samples": metadata.get("train_samples"),
                            "val_samples": metadata.get("val_samples"),
                            "epochs": metadata.get("epochs"),
                            "checkpoint_name": checkpoint_dir.name
                        }

                        # Find the highest numbered checkpoint subdirectory
                        checkpoint_subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
                        if checkpoint_subdirs:
                            # Sort by checkpoint number and get the last one
                            checkpoint_subdirs.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
                            trainer_state_file = checkpoint_subdirs[-1] / "trainer_state.json"

                            if trainer_state_file.exists():
                                try:
                                    with open(trainer_state_file, 'r') as f:
                                        trainer_state = json.load(f)

                                    log_history = trainer_state.get("log_history", [])
                                    if log_history:
                                        # Get the last entry (most recent)
                                        last_log = log_history[-1]
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
    """Get database size and statistics."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        import asyncio

        # Get size in thread pool to avoid blocking
        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)
        golden_examples = await asyncio.to_thread(memory_manager.get_golden_examples)

        max_size_gb = memory_manager.config.memory.max_db_size_gb
        threshold = memory_manager.config.memory.auto_cleanup_threshold
        threshold_size_gb = max_size_gb * threshold

        return {
            "size_gb": round(size_gb, 3),
            "max_size_gb": max_size_gb,
            "threshold_size_gb": round(threshold_size_gb, 3),
            "usage_percentage": round((size_gb / max_size_gb) * 100, 1) if max_size_gb > 0 else 0,
            "total_interactions": len(all_interactions),
            "golden_examples": len(golden_examples),
            "auto_cleanup_enabled": True,
            "auto_cleanup_percentage": memory_manager.config.memory.auto_cleanup_percentage * 100
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/database/cleanup/age")
async def cleanup_by_age(days: Optional[int] = None):
    """
    Manually trigger age-based cleanup (removes entries older than specified days).

    Args:
        days: Number of days threshold. Uses config default if not provided.
    """
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        import asyncio
        from datetime import datetime, timedelta

        # Use custom days or config default
        age_days = days if days is not None else memory_manager.config.memory.max_memory_age_days

        # Run custom cleanup with specified days
        cutoff = datetime.now() - timedelta(days=age_days)

        def cleanup_with_custom_age():
            try:
                results = memory_manager.collection.get(include=["metadatas"])

                ids_to_delete = []
                for i, metadata in enumerate(results['metadatas']):
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    if timestamp < cutoff and not metadata.get('is_golden', False):
                        ids_to_delete.append(results['ids'][i])

                if ids_to_delete:
                    memory_manager.collection.delete(ids=ids_to_delete)
                    logger.info(f"Cleaned up {len(ids_to_delete)} old memories (older than {age_days} days)")
                    return len(ids_to_delete)
                return 0
            except Exception as e:
                logger.error(f"Failed to cleanup memories: {e}")
                return 0

        # Run cleanup in thread pool to avoid blocking
        deleted_count = await asyncio.to_thread(cleanup_with_custom_age)

        # Get new stats
        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)

        return {
            "success": True,
            "message": f"Age-based cleanup completed (threshold: {age_days} days)",
            "deleted_count": deleted_count,
            "new_size_gb": round(size_gb, 3),
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
    Manually trigger weight-based cleanup (removes entries below weight threshold).

    Args:
        threshold: Optional weight threshold. Uses config default if not provided.
    """
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        import asyncio

        # Run cleanup in thread pool to avoid blocking
        deleted_count = await asyncio.to_thread(
            memory_manager.cleanup_low_weight_entries,
            threshold
        )

        # Get new stats
        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)

        return {
            "success": True,
            "message": f"Weight-based cleanup completed",
            "deleted_count": deleted_count,
            "new_size_gb": round(size_gb, 3),
            "remaining_interactions": len(all_interactions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during weight-based cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/database/cleanup/lowest")
async def cleanup_lowest_weight(percentage: Optional[float] = None):
    """
    Manually trigger cleanup of lowest-weight entries.

    Args:
        percentage: Percentage to remove (0.0-1.0). Uses config default if not provided.
    """
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        import asyncio

        # Run cleanup in thread pool to avoid blocking
        deleted_count = await asyncio.to_thread(
            memory_manager.cleanup_lowest_weight_entries,
            percentage
        )

        # Get new stats
        size_gb = await asyncio.to_thread(memory_manager.get_db_size_gb)
        all_interactions = await asyncio.to_thread(memory_manager.get_interactions_since, None)

        return {
            "success": True,
            "message": f"Lowest-weight cleanup completed",
            "deleted_count": deleted_count,
            "new_size_gb": round(size_gb, 3),
            "remaining_interactions": len(all_interactions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during lowest-weight cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.get("/api/model/status")
async def get_model_status():
    """Get current model and training pipeline status."""
    try:
        result = {
            "model_loaded": False,
            "is_reloading": False,
            "current_checkpoint_id": None,
            "model_path": None,
        }

        if llm_engine:
            result["model_loaded"] = llm_engine.is_loaded()
            result["is_reloading"] = llm_engine.is_reloading()
            result["current_checkpoint_id"] = llm_engine.current_checkpoint_id
            result["model_path"] = getattr(llm_engine, "model_path", None)

        # Read current_checkpoint.json for GGUF and metrics info
        checkpoint_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    cp_data = json.load(f)
                result["deployed_gguf"] = cp_data.get("gguf_model_path")
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
    """Trigger a training run from the admin UI."""
    try:
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not initialized")

        # Pre-training cleanup
        deleted_count = await asyncio.to_thread(memory_manager.cleanup_low_weight_entries)

        memory_manager.request_training()

        return {
            "success": True,
            "message": "Training requested. It will begin at the next scheduled check.",
            "pre_training_cleanup_deleted": deleted_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/model/rollback")
async def rollback_model(checkpoint_id: str = Query(..., description="Checkpoint ID to rollback to")):
    """Rollback to a previous checkpoint and reload the model."""
    try:
        if not llm_engine:
            raise HTTPException(status_code=503, detail="LLM engine not initialized")

        checkpoints_path = Path(settings.checkpoints_path)
        metadata_file = checkpoints_path / f"{checkpoint_id}_metadata.json"

        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_id}")

        with open(metadata_file, "r") as f:
            data = json.load(f)

        gguf_path = data.get("gguf_model_path")
        if not gguf_path:
            raise HTTPException(
                status_code=400,
                detail=f"Checkpoint {checkpoint_id} has no GGUF model (adapter-only checkpoint)"
            )

        if not Path(gguf_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"GGUF file not found on disk: {gguf_path}"
            )

        # Update current_checkpoint.json
        current_cp_file = checkpoints_path / "current_checkpoint.json"
        current_data = {
            "checkpoint_id": checkpoint_id,
            "adapter_path": data.get("adapter_path"),
            "gguf_model_path": gguf_path,
            "metrics": data.get("metrics", {}),
            "deployed_at": datetime.now().isoformat(),
        }
        with open(current_cp_file, "w") as f:
            json.dump(current_data, f, indent=2)

        # Trigger model reload
        success = await asyncio.to_thread(
            llm_engine.reload_model,
            new_model_path=gguf_path,
            checkpoint_id=checkpoint_id,
        )

        if success:
            return {
                "success": True,
                "message": f"Rolled back to checkpoint {checkpoint_id}",
                "checkpoint_id": checkpoint_id,
                "gguf_model_path": gguf_path,
            }
        else:
            return {
                "success": False,
                "message": "Rollback failed; previous model restored. Check API logs.",
                "checkpoint_id": checkpoint_id,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during rollback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/model/reload-base")
async def reload_base_model():
    """Reload the original base model (undo all fine-tuning)."""
    try:
        if not llm_engine:
            raise HTTPException(status_code=503, detail="LLM engine not initialized")

        base_path = settings.model_path
        if not Path(base_path).exists():
            raise HTTPException(status_code=404, detail=f"Base model not found: {base_path}")

        success = await asyncio.to_thread(
            llm_engine.reload_model,
            new_model_path=base_path,
            checkpoint_id=None,
        )

        if success:
            return {
                "success": True,
                "message": "Reverted to base model (no fine-tuning)",
                "model_path": base_path,
            }
        else:
            return {
                "success": False,
                "message": "Failed to reload base model; check API logs",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading base model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#  Prompt Interceptor Rules
# ------------------------------------------------------------------ #

@admin_router.get("/api/prompt-rules")
async def list_prompt_rules():
    """List all prompt interceptor rules."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    return {"rules": prompt_interceptor.get_rules()}


@admin_router.post("/api/prompt-rules")
async def add_prompt_rule(body: dict):
    """Add a new prompt interceptor rule."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    try:
        rule = prompt_interceptor.add_rule(
            pattern=body.get("pattern", ""),
            replacement=body.get("replacement", ""),
            description=body.get("description", ""),
            target_roles=body.get("target_roles", ["system"]),
            enabled=body.get("enabled", True),
            priority=body.get("priority", 50),
        )
        return {"success": True, "rule": rule}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@admin_router.put("/api/prompt-rules/{rule_id}")
async def update_prompt_rule(rule_id: str, body: dict):
    """Update an existing prompt interceptor rule."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    try:
        rule = prompt_interceptor.update_rule(rule_id, body)
        if rule is None:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        return {"success": True, "rule": rule}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@admin_router.delete("/api/prompt-rules/{rule_id}")
async def delete_prompt_rule(rule_id: str):
    """Delete a prompt interceptor rule."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    if not prompt_interceptor.delete_rule(rule_id):
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return {"success": True}


@admin_router.post("/api/prompt-rules/{rule_id}/toggle")
async def toggle_prompt_rule(rule_id: str):
    """Toggle a prompt interceptor rule's enabled state."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    rule = prompt_interceptor.toggle_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return {"success": True, "rule": rule}


@admin_router.post("/api/prompt-rules/test")
async def test_prompt_rules(body: dict):
    """Test all enabled rules against sample text."""
    if not prompt_interceptor:
        raise HTTPException(status_code=503, detail="Prompt interceptor not initialized")
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required")
    return prompt_interceptor.test(text)
