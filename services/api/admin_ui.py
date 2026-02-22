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

from shared.config import settings, load_config

logger = logging.getLogger(__name__)

# Create router
admin_router = APIRouter(prefix="/admin", tags=["admin"])

# These will be set by main.py
memory_manager = None
llm_proxy = None
api_request_log = None
prompt_interceptor = None


def set_dependencies(mm, proxy, request_log=None, interceptor=None):
    """Set global dependencies from main.py."""
    global memory_manager, llm_proxy, api_request_log, prompt_interceptor
    memory_manager = mm
    llm_proxy = proxy
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
    logger.info(f"admin_ui: memory_manager={memory_manager}, llm_proxy={llm_proxy}")
    try:
        if not memory_manager or not llm_proxy:
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
        
        result = {
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
                "model_loaded": llm_proxy.is_loaded(),
                "is_reloading": getattr(llm_proxy, 'is_reloading', lambda: False)(),
                "current_checkpoint": getattr(llm_proxy, 'current_checkpoint_id', None),
                "model_path": getattr(llm_proxy, 'model_path', None),
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
        }

        # Fall back to current_checkpoint.json if llm_proxy doesn't track it
        if not result["system"]["current_checkpoint"]:
            try:
                cp_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
                if cp_file.exists():
                    with open(cp_file, "r") as f:
                        result["system"]["current_checkpoint"] = json.load(f).get("checkpoint_id")
            except Exception:
                pass

        return result
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
        # Import main.py's global state for training status
        from api.main import training_in_progress, config

        result = {
            "model_loaded": False,
            "is_reloading": False,
            "training_in_progress": training_in_progress,
            "current_checkpoint_id": None,
            "model_path": None,
            "external_model_name": None,
        }

        if llm_proxy:
            # Proxy mode
            result["model_loaded"] = llm_proxy.is_loaded() and not training_in_progress
            result["is_reloading"] = training_in_progress  # Training = reloading in proxy mode
            result["current_checkpoint_id"] = getattr(llm_proxy, "current_checkpoint_id", None)
            result["model_path"] = getattr(llm_proxy, "model_path", None)

            # Add external model name for Ollama
            if hasattr(llm_proxy, "model_name"):
                result["external_model_name"] = llm_proxy.model_name
            elif config:
                result["external_model_name"] = config.llm_proxy.external_model_name

        # Read current_checkpoint.json for checkpoint ID, metrics, deployment info
        checkpoint_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    cp_data = json.load(f)
                # Use checkpoint_id from file if not set by llm_proxy
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

        # Pre-training cleanup
        deleted_count = await asyncio.to_thread(memory_manager.cleanup_low_weight_entries)

        # Request manual training (bypasses schedule, auto-selects mode)
        memory_manager.request_training(reason="manual", training_mode="auto")

        return {
            "success": True,
            "message": "Manual training requested. It will begin immediately (bypasses schedule).",
            "pre_training_cleanup_deleted": deleted_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering training: {e}", exc_info=True)
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
    """Deploy a specific checkpoint's LoRA adapter to Ollama."""
    try:
        from shared.ollama_deployer import OllamaDeployer

        checkpoints_path = Path(settings.checkpoints_path)
        checkpoint_dir = checkpoints_path / checkpoint_id

        if not checkpoint_dir.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_id}")

        adapter_dir = checkpoint_dir / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise HTTPException(status_code=400, detail=f"No adapter found in checkpoint {checkpoint_id}")

        config = load_config()
        deployer = OllamaDeployer(ollama_base_url=config.llm_proxy.base_url)
        model_name = config.training.ollama_model_name

        success = await asyncio.to_thread(
            deployer.deploy_model,
            adapter_path=str(adapter_dir),
            model_name=model_name,
            base_model_name=model_name,
            checkpoint_id=checkpoint_id,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Ollama deployment failed. Check logs.")

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

        return {"success": True, "message": f"Deployed checkpoint {checkpoint_id} to Ollama as {model_name}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying checkpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/api/model/reload-base")
async def reload_base_model():
    """Revert to the base model by recreating it without any adapter."""
    try:
        from shared.ollama_deployer import OllamaDeployer
        import requests as req

        config = load_config()
        model_name = config.training.ollama_model_name
        deployer = OllamaDeployer(ollama_base_url=config.llm_proxy.base_url)

        # Unload current model
        deployer._unload_model(model_name)

        # Recreate from base (no adapter)
        response = req.post(
            f"{deployer.ollama_api}/create",
            json={"model": model_name, "from": model_name, "stream": False},
            timeout=120,
        )
        response.raise_for_status()

        # Clear current_checkpoint.json
        cp_file = Path(settings.checkpoints_path) / "current_checkpoint.json"
        if cp_file.exists():
            cp_file.unlink()

        return {"success": True, "message": f"Reverted {model_name} to base model (no adapter)"}

    except Exception as e:
        logger.error(f"Error reverting to base model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#  Ollama Model Deployment
# ------------------------------------------------------------------ #

@admin_router.post("/api/ollama/deploy-checkpoint")
async def deploy_checkpoint_to_ollama(checkpoint_id: str = Query(..., description="Checkpoint ID to deploy")):
    """
    Deploy a specific checkpoint to Ollama.

    This will:
    1. Find the LoRA adapter for the checkpoint
    2. Deploy it to Ollama via adapter layering (replaces current model)
    3. Update config pointer
    4. Next API request will use the deployed model
    """
    try:
        import sys
        sys.path.insert(0, '/app/trainer')
        from shared.ollama_deployer import OllamaDeployer

        checkpoints_path = Path(settings.checkpoints_path)
        checkpoint_dir = checkpoints_path / checkpoint_id

        if not checkpoint_dir.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint directory not found: {checkpoint_id}")

        # Look for adapter directory
        adapter_dir = checkpoint_dir / "adapter"
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise HTTPException(
                status_code=400,
                detail=f"No adapter found in checkpoint {checkpoint_id}. Expected {adapter_file}"
            )

        # Load config
        config = load_config()

        # Deploy to Ollama via adapter layering
        logger.info(f"Admin UI: Deploying checkpoint {checkpoint_id} to Ollama")
        deployer = OllamaDeployer(ollama_base_url=config.llm_proxy.base_url)

        model_name = config.training.ollama_model_name
        success = deployer.deploy_model(
            adapter_path=str(adapter_dir),
            model_name=model_name,
            base_model_name=model_name,
            checkpoint_id=checkpoint_id,
        )

        if success:
            # Update config pointer
            deployer.update_config_pointer(model_name)

            return {
                "success": True,
                "message": f"Checkpoint {checkpoint_id} deployed to Ollama as {model_name}",
                "checkpoint_id": checkpoint_id,
                "model_name": model_name,
                "adapter_path": str(adapter_dir)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Ollama deployment failed. Check trainer logs for details."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying checkpoint to Ollama: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deployment error: {str(e)}")


@admin_router.post("/api/ollama/deploy-base")
async def deploy_base_model_to_ollama():
    """
    Deploy the original base model to Ollama.

    This reverts to the unmodified gpt-oss:20b model.
    """
    try:
        import sys
        import subprocess
        sys.path.insert(0, '/app/trainer')
        from shared.ollama_deployer import OllamaDeployer

        # Load config
        config = load_config()

        logger.info("Admin UI: Reverting to base model in Ollama")
        deployer = OllamaDeployer(ollama_base_url=config.llm_proxy.base_url)

        model_name = config.training.ollama_model_name

        # Pull the original base model from Ollama registry
        # This assumes the base model is available as "gpt-oss:20b" in Ollama
        try:
            result = subprocess.run(
                ["ollama", "pull", "gpt-oss:20b"],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                # Update config to point to base model
                deployer.update_config_pointer("gpt-oss:20b")

                return {
                    "success": True,
                    "message": "Reverted to base gpt-oss:20b model",
                    "model_name": "gpt-oss:20b"
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to pull base model: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="Timeout pulling base model from Ollama")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying base model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deployment error: {str(e)}")


@admin_router.get("/api/ollama/models")
async def list_ollama_models():
    """List all models available in Ollama."""
    try:
        import sys
        sys.path.insert(0, '/app/trainer')
        from shared.ollama_deployer import OllamaDeployer

        # Load config
        config = load_config()

        deployer = OllamaDeployer(ollama_base_url=config.llm_proxy.base_url)
        models = deployer.list_models()

        return {
            "models": models,
            "current_model": config.llm_proxy.external_model_name
        }

    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}", exc_info=True)
        return {"models": [], "error": str(e)}


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
