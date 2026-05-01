import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from shared.models import Interaction, TrainingStats, ChatMessage
from shared.config import SystemConfig, settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory using ChromaDB vector database."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_path = settings.data_path
        
        # Initialize ChromaDB with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=f"{self.data_path}/chromadb",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                chroma_telemetry_impl="chromadb.telemetry.product.posthog.Posthog"
            )
        )
        
        # Get or create collections
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.golden_collection = self.client.get_or_create_collection(
            name="golden_examples",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model (force CPU to avoid CUDA 6.1 compatibility issues)
        logger.info(f"Loading embedding model: {config.memory.embedding_model}")
        self.embedder = SentenceTransformer(
            config.memory.embedding_model,
            device='cpu'  # Force CPU for embeddings (avoids CUDA arch issues)
        )
        logger.info("Embedding model loaded on CPU")
        
        # Track training state
        self.training_state_file = f"{self.data_path}/training_state.json"
        self.user_training_request = False
    
    def store_interaction(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        sentiment: float = 0.0,
        weight: float = 1.0,
        is_golden: bool = False,
        is_refusal: bool = False,
    ) -> str:
        """Store an interaction in the vector database."""
        try:
            timestamp = datetime.now()
            interaction_id = f"{conversation_id}_{timestamp.timestamp()}"

            # Create embedding from user message + response
            text = f"User: {user_message}\nAssistant: {assistant_response}"
            embedding = self.embedder.encode(text).tolist()

            # Store full messages for maximum training fidelity
            # Handle edge case of extremely long messages (ChromaDB metadata limit: ~64KB)
            user_msg_stored = user_message
            assistant_resp_stored = assistant_response
            truncated = False

            if len(user_message) > 60000:
                user_msg_stored = user_message[:60000] + "...[TRUNCATED]"
                truncated = True
            if len(assistant_response) > 60000:
                assistant_resp_stored = assistant_response[:60000] + "...[TRUNCATED]"
                truncated = True

            metadata = {
                "conversation_id": conversation_id,
                "timestamp": timestamp.isoformat(),
                "user_message": user_msg_stored,  # Full message (no 500-char limit)
                "assistant_response": assistant_resp_stored,  # Full response (no 500-char limit)
                "user_message_length": len(user_message),
                "assistant_response_length": len(assistant_response),
                "sentiment": sentiment,
                "weight": weight,
                "is_golden": is_golden,
                "is_refusal": is_refusal,
                "truncated": truncated
            }

            # Store in main collection
            self.collection.add(
                ids=[interaction_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
            
            # Also store in golden collection if marked
            if is_golden:
                self.golden_collection.add(
                    ids=[interaction_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata]
                )
                logger.info(f"Stored golden example: {interaction_id}")
            
            logger.debug(f"Stored interaction: {interaction_id} (sentiment: {sentiment:.2f}, weight: {weight:.2f})")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            raise
    
    def get_interactions_since(
        self,
        timestamp: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Interaction]:
        """Get all interactions since a given timestamp."""
        try:
            results = self.collection.get(
                include=["metadatas", "documents"]
            )
            
            interactions = []
            for i, metadata in enumerate(results['metadatas']):
                interaction_timestamp = datetime.fromisoformat(metadata['timestamp'])
                
                # Filter by timestamp if provided
                if timestamp and interaction_timestamp <= timestamp:
                    continue
                
                interaction = Interaction(
                    id=results['ids'][i],
                    conversation_id=metadata['conversation_id'],
                    timestamp=interaction_timestamp,
                    user_message=metadata['user_message'],
                    assistant_response=metadata['assistant_response'],
                    sentiment=metadata.get('sentiment', 0.0),
                    weight=metadata.get('weight', 1.0),
                    metadata=metadata,
                )
                interactions.append(interaction)
            
            # Sort by timestamp
            interactions.sort(key=lambda x: x.timestamp)
            
            # Apply limit if specified
            if limit:
                interactions = interactions[-limit:]
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}")
            return []
    
    def get_top_interactions_by_weight(self, n: int, exclude_golden: bool = True) -> List[Interaction]:
        """Get top N interactions ranked by weight (from all history, not time-gated)."""
        try:
            results = self.collection.get(
                include=["metadatas", "documents"]
            )

            interactions = []
            for i, metadata in enumerate(results['metadatas']):
                if exclude_golden and metadata.get('is_golden', False):
                    continue

                interaction = Interaction(
                    id=results['ids'][i],
                    conversation_id=metadata['conversation_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    user_message=metadata['user_message'],
                    assistant_response=metadata['assistant_response'],
                    sentiment=metadata.get('sentiment', 0.0),
                    weight=metadata.get('weight', 1.0),
                    metadata=metadata
                )
                interactions.append(interaction)

            # Sort by weight descending, return top N
            interactions.sort(key=lambda x: x.weight, reverse=True)
            return interactions[:n]

        except Exception as e:
            logger.error(f"Failed to retrieve top interactions by weight: {e}")
            return []

    def get_golden_examples(self) -> List[Interaction]:
        """Get all golden examples from the replay buffer."""
        try:
            results = self.golden_collection.get(
                include=["metadatas", "documents"]
            )

            interactions = []
            for i, metadata in enumerate(results['metadatas']):
                interaction = Interaction(
                    id=results['ids'][i],
                    conversation_id=metadata['conversation_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    user_message=metadata['user_message'],
                    assistant_response=metadata['assistant_response'],
                    sentiment=metadata.get('sentiment', 0.0),
                    weight=metadata.get('weight', 1.0),
                    metadata=metadata
                )
                interactions.append(interaction)

            return interactions

        except Exception as e:
            logger.error(f"Failed to retrieve golden examples: {e}")
            return []
    
    def get_training_stats(self) -> TrainingStats:
        """Get statistics about training status including sentiment breakdown."""
        try:
            # Load training state (includes last_training and last_dpo_training)
            training_state = self._load_training_state_full()
            last_training = training_state.get('last_training')
            last_dpo_training = training_state.get('last_dpo_training')

            # Get all interactions
            all_interactions = self.get_interactions_since()

            if not all_interactions:
                return TrainingStats(
                    new_interactions_since_last_training=0,
                    hours_since_last_interaction=0.0,
                    days_since_last_training=999.0,
                    total_interactions=0,
                    last_training_timestamp=None,
                    user_requested_training=self._is_training_requested(),
                    new_positive_count=0,
                    new_negative_count=0,
                    new_neutral_count=0,
                    new_golden_count=0,
                    last_dpo_training_timestamp=None,
                    new_negative_since_dpo=0,
                    days_since_dpo_training=999.0
                )

            # Calculate basic stats
            now = datetime.now()
            latest_interaction = max(all_interactions, key=lambda x: x.timestamp)
            hours_since_last = (now - latest_interaction.timestamp).total_seconds() / 3600

            # Calculate new interactions since last training
            if last_training:
                new_interactions_list = [i for i in all_interactions if i.timestamp > last_training]
                new_interactions = len(new_interactions_list)
                days_since_training = (now - last_training).total_seconds() / 86400
            else:
                new_interactions_list = all_interactions
                new_interactions = len(all_interactions)
                days_since_training = 999.0

            # Calculate sentiment breakdown of NEW interactions
            new_positive = sum(1 for i in new_interactions_list
                             if i.sentiment > self.config.sentiment.positive_threshold)
            new_negative = sum(1 for i in new_interactions_list
                             if i.sentiment < self.config.sentiment.negative_threshold)
            new_neutral = sum(1 for i in new_interactions_list
                            if abs(i.sentiment) <= max(
                                self.config.sentiment.positive_threshold,
                                abs(self.config.sentiment.negative_threshold)
                            ))
            new_golden = sum(1 for i in new_interactions_list
                           if i.metadata.get('is_golden', False))

            # Count refusals in new interactions
            new_refusal_count = sum(1 for i in new_interactions_list
                                    if i.metadata.get('is_refusal', False))

            # Calculate DPO-specific stats (new negatives since last DPO training)
            if last_dpo_training:
                new_negative_since_dpo = sum(1 for i in all_interactions
                                             if i.timestamp > last_dpo_training and
                                             i.sentiment < self.config.sentiment.negative_threshold)
                new_refusals_since_dpo = sum(1 for i in all_interactions
                                              if i.timestamp > last_dpo_training and
                                              i.metadata.get('is_refusal', False))
                days_since_dpo = (now - last_dpo_training).total_seconds() / 86400
            else:
                new_negative_since_dpo = new_negative
                new_refusals_since_dpo = new_refusal_count
                days_since_dpo = 999.0

            return TrainingStats(
                new_interactions_since_last_training=new_interactions,
                hours_since_last_interaction=hours_since_last,
                days_since_last_training=days_since_training,
                total_interactions=len(all_interactions),
                last_training_timestamp=last_training,
                user_requested_training=self._is_training_requested(),
                new_positive_count=new_positive,
                new_negative_count=new_negative,
                new_neutral_count=new_neutral,
                new_golden_count=new_golden,
                last_dpo_training_timestamp=last_dpo_training,
                new_negative_since_dpo=new_negative_since_dpo,
                days_since_dpo_training=days_since_dpo,
                new_refusal_count=new_refusal_count,
                new_refusals_since_dpo=new_refusals_since_dpo
            )

        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return TrainingStats(
                new_interactions_since_last_training=0,
                hours_since_last_interaction=0.0,
                days_since_last_training=999.0,
                total_interactions=0,
                user_requested_training=False,
                new_positive_count=0,
                new_negative_count=0,
                new_neutral_count=0,
                new_golden_count=0,
                last_dpo_training_timestamp=None,
                new_negative_since_dpo=0,
                days_since_dpo_training=999.0
            )
    
    def mark_training_complete(self, training_mode: str = "sft"):
        """
        Mark that training has completed.

        Args:
            training_mode: The mode used for training ("sft" or "dpo")
        """
        now = datetime.now()
        self._save_training_state(now, training_mode)
        self.user_training_request = False
        self.clear_training_request()

    def request_training(self, reason: str = "manual", training_mode: str = "auto"):
        """
        Request training execution. Persists to shared volume so trainer container sees it.

        Args:
            reason: Why training was requested ("manual", "inactivity", "negative_sentiment", etc.)
            training_mode: Force specific mode ("auto", "sft", "dpo")
        """
        self.user_training_request = True

        request_data = {
            "requested_at": datetime.now().isoformat(),
            "reason": reason,
            "training_mode": training_mode,
            "triggered_by": "admin_ui" if reason == "manual" else "scheduler"
        }

        try:
            request_file = Path(self.data_path) / "training_request.json"
            request_file.write_text(json.dumps(request_data, indent=2))
            logger.info(f"Training requested: {reason} (mode: {training_mode})")
        except Exception as e:
            logger.warning(f"Failed to write training request file: {e}")

    def get_training_request(self) -> Optional[dict]:
        """Get pending training request if exists."""
        try:
            request_file = Path(self.data_path) / "training_request.json"
            if request_file.exists():
                return json.loads(request_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to read training request file: {e}")
        return None

    def clear_training_request(self):
        """Clear training request (called after training completes or fails)."""
        try:
            request_file = Path(self.data_path) / "training_request.json"
            if request_file.exists():
                request_file.unlink()
                logger.info("Training request cleared")
        except Exception as e:
            logger.warning(f"Failed to clear training request file: {e}")

    def _is_training_requested(self) -> bool:
        """Check both in-memory flag and shared request file."""
        if self.user_training_request:
            return True
        return self.get_training_request() is not None

    def _write_training_request_file(self):
        """Deprecated: Use request_training() instead."""
        self.request_training(reason="manual", training_mode="auto")

    def _clear_training_request_file(self):
        """Deprecated: Use clear_training_request() instead."""
        self.clear_training_request()
    
    def _load_training_state(self) -> Optional[datetime]:
        """Load the last training timestamp (backward compatibility)."""
        state = self._load_training_state_full()
        return state.get('last_training')

    def _load_training_state_full(self) -> dict:
        """
        Load the full training state including DPO training timestamp.

        Returns dict with:
            - last_training: datetime or None
            - last_dpo_training: datetime or None
            - last_mode: str ("sft" or "dpo")
        """
        try:
            with open(self.training_state_file, 'r') as f:
                state = json.load(f)
                return {
                    'last_training': datetime.fromisoformat(state['last_training']) if 'last_training' in state else None,
                    'last_dpo_training': datetime.fromisoformat(state['last_dpo_training']) if 'last_dpo_training' in state else None,
                    'last_mode': state.get('last_mode', 'sft')
                }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return {
                'last_training': None,
                'last_dpo_training': None,
                'last_mode': 'sft'
            }

    def _save_training_state(self, timestamp: datetime, training_mode: str = "sft"):
        """
        Save the training timestamp.

        Args:
            timestamp: When training completed
            training_mode: The mode used for training ("sft" or "dpo")
        """
        try:
            # Load existing state to preserve DPO timestamp
            existing_state = self._load_training_state_full()

            state = {
                'last_training': timestamp.isoformat(),
                'last_mode': training_mode
            }

            # Update DPO training timestamp if this was a DPO training run
            if training_mode == "dpo":
                state['last_dpo_training'] = timestamp.isoformat()
            else:
                # Preserve existing DPO timestamp
                if existing_state['last_dpo_training']:
                    state['last_dpo_training'] = existing_state['last_dpo_training'].isoformat()

            with open(self.training_state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Training state saved: mode={training_mode}, timestamp={timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
    
    def cleanup_old_memories(self):
        """Remove interactions older than max_memory_age_days."""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.memory.max_memory_age_days)

            results = self.collection.get(include=["metadatas"])

            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if timestamp < cutoff and not metadata.get('is_golden', False):
                    ids_to_delete.append(results['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Cleaned up {len(ids_to_delete)} old memories")

        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}")

    def cleanup_low_weight_entries(self, threshold: Optional[float] = None):
        """
        Remove interactions below a weight threshold (for pre-training cleanup).

        Args:
            threshold: Weight threshold. Uses config.memory.min_weight_threshold if not provided.
        """
        try:
            threshold = threshold if threshold is not None else self.config.memory.min_weight_threshold

            results = self.collection.get(include=["metadatas"])

            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                weight = metadata.get('weight', 1.0)
                is_golden = metadata.get('is_golden', False)

                # Don't delete golden examples
                if not is_golden and weight < threshold:
                    ids_to_delete.append(results['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Cleaned up {len(ids_to_delete)} low-weight entries (threshold: {threshold})")
                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Failed to cleanup low-weight entries: {e}")
            return 0

    def cleanup_lowest_weight_entries(self, percentage: Optional[float] = None):
        """
        Remove the bottom X% of entries by weight (for auto-cleanup when DB is full).

        Args:
            percentage: Percentage to remove (0.0-1.0). Uses config.memory.auto_cleanup_percentage if not provided.

        Returns:
            Number of entries deleted
        """
        try:
            percentage = percentage if percentage is not None else self.config.memory.auto_cleanup_percentage

            results = self.collection.get(include=["metadatas"])

            if not results['ids']:
                return 0

            # Build list of (id, weight, is_golden) tuples
            entries = []
            for i, metadata in enumerate(results['metadatas']):
                entries.append({
                    'id': results['ids'][i],
                    'weight': metadata.get('weight', 1.0),
                    'is_golden': metadata.get('is_golden', False)
                })

            # Filter out golden examples
            non_golden_entries = [e for e in entries if not e['is_golden']]

            if not non_golden_entries:
                logger.info("No non-golden entries to cleanup")
                return 0

            # Sort by weight (ascending)
            non_golden_entries.sort(key=lambda x: x['weight'])

            # Calculate how many to delete
            num_to_delete = int(len(non_golden_entries) * percentage)

            if num_to_delete == 0:
                logger.info("Calculated 0 entries to delete, skipping cleanup")
                return 0

            # Get IDs of lowest weight entries
            ids_to_delete = [e['id'] for e in non_golden_entries[:num_to_delete]]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Cleaned up {len(ids_to_delete)} lowest-weight entries ({percentage*100}% of non-golden)")
                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Failed to cleanup lowest-weight entries: {e}")
            return 0

    def get_db_size_gb(self) -> float:
        """
        Get the current size of the ChromaDB database in GB.

        Returns:
            Database size in gigabytes
        """
        try:
            import os

            db_path = f"{self.data_path}/chromadb"
            total_size = 0

            for dirpath, dirnames, filenames in os.walk(db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)

            size_gb = total_size / (1024 ** 3)  # Convert bytes to GB
            return size_gb

        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return 0.0

    def auto_cleanup_if_needed(self) -> bool:
        """
        Check database size and automatically cleanup if over threshold.

        Returns:
            True if cleanup was performed, False otherwise
        """
        try:
            current_size = self.get_db_size_gb()
            max_size = self.config.memory.max_db_size_gb
            threshold = self.config.memory.auto_cleanup_threshold

            threshold_size = max_size * threshold

            logger.info(f"Database size: {current_size:.2f}GB / {max_size}GB (threshold: {threshold_size:.2f}GB)")

            if current_size >= threshold_size:
                logger.warning(f"Database size ({current_size:.2f}GB) exceeds threshold ({threshold_size:.2f}GB), triggering auto-cleanup")

                # Remove bottom X% by weight
                deleted_count = self.cleanup_lowest_weight_entries()

                new_size = self.get_db_size_gb()
                logger.info(f"Auto-cleanup complete. New size: {new_size:.2f}GB (freed {current_size - new_size:.2f}GB)")

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to perform auto-cleanup: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if ChromaDB is connected."""
        try:
            self.client.heartbeat()
            return True
        except:
            return False
