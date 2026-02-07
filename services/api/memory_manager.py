import logging
import json
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
        is_golden: bool = False
    ) -> str:
        """Store an interaction in the vector database."""
        try:
            timestamp = datetime.now()
            interaction_id = f"{conversation_id}_{timestamp.timestamp()}"
            
            # Create embedding from user message + response
            text = f"User: {user_message}\nAssistant: {assistant_response}"
            embedding = self.embedder.encode(text).tolist()
            
            metadata = {
                "conversation_id": conversation_id,
                "timestamp": timestamp.isoformat(),
                "user_message": user_message[:500],  # Truncate for metadata
                "assistant_response": assistant_response[:500],
                "sentiment": sentiment,
                "weight": weight,
                "is_golden": is_golden
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
                    metadata=metadata
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

    def retrieve_relevant_context(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant interactions from memory using semantic search.

        Args:
            query: The user's message to find relevant context for
            k: Number of results to retrieve
            similarity_threshold: Minimum similarity score (0-1), defaults to config value

        Returns:
            List of relevant context items with user_message, assistant_response, and distance
        """
        try:
            if not query.strip():
                return []

            # Use config threshold if not specified
            threshold = similarity_threshold if similarity_threshold is not None else self.config.memory.similarity_threshold

            # Generate embedding for the query
            query_embedding = self.embedder.encode(query).tolist()

            # Query ChromaDB for similar interactions
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )

            # Parse results
            relevant_context = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # ChromaDB returns distance (lower is more similar)
                    # Convert to similarity score (1 - normalized_distance)
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance  # Cosine distance to similarity

                    # Filter by threshold
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        relevant_context.append({
                            'user_message': metadata.get('user_message', ''),
                            'assistant_response': metadata.get('assistant_response', ''),
                            'similarity': similarity,
                            'timestamp': metadata.get('timestamp', ''),
                            'sentiment': metadata.get('sentiment', 0.0)
                        })

            logger.debug(f"Retrieved {len(relevant_context)} relevant context items for query (threshold: {threshold})")
            return relevant_context

        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
            return []
    
    def get_training_stats(self) -> TrainingStats:
        """Get statistics about training status."""
        try:
            # Load training state
            last_training = self._load_training_state()
            
            # Get all interactions
            all_interactions = self.get_interactions_since()
            
            if not all_interactions:
                return TrainingStats(
                    new_interactions_since_last_training=0,
                    hours_since_last_interaction=0.0,
                    days_since_last_training=999.0,
                    total_interactions=0,
                    last_training_timestamp=None,
                    user_requested_training=self.user_training_request
                )
            
            # Calculate stats
            now = datetime.now()
            latest_interaction = max(all_interactions, key=lambda x: x.timestamp)
            
            hours_since_last = (now - latest_interaction.timestamp).total_seconds() / 3600
            
            if last_training:
                new_interactions = sum(1 for i in all_interactions if i.timestamp > last_training)
                days_since_training = (now - last_training).total_seconds() / 86400
            else:
                new_interactions = len(all_interactions)
                days_since_training = 999.0
            
            return TrainingStats(
                new_interactions_since_last_training=new_interactions,
                hours_since_last_interaction=hours_since_last,
                days_since_last_training=days_since_training,
                total_interactions=len(all_interactions),
                last_training_timestamp=last_training,
                user_requested_training=self.user_training_request
            )
            
        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return TrainingStats(
                new_interactions_since_last_training=0,
                hours_since_last_interaction=0.0,
                days_since_last_training=999.0,
                total_interactions=0,
                user_requested_training=False
            )
    
    def mark_training_complete(self):
        """Mark that training has completed."""
        self._save_training_state(datetime.now())
        self.user_training_request = False
    
    def request_training(self):
        """User explicitly requests training."""
        self.user_training_request = True
        logger.info("User requested training")
    
    def _load_training_state(self) -> Optional[datetime]:
        """Load the last training timestamp."""
        try:
            with open(self.training_state_file, 'r') as f:
                state = json.load(f)
                return datetime.fromisoformat(state['last_training'])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
    def _save_training_state(self, timestamp: datetime):
        """Save the training timestamp."""
        try:
            with open(self.training_state_file, 'w') as f:
                json.dump({'last_training': timestamp.isoformat()}, f)
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
