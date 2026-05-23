import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from elasticsearch import Elasticsearch

from shared.config import SystemConfig
from shared.config import settings as app_settings
from shared.es_client import create_es_client, check_es_connection
from shared.models import Interaction, TrainingStats

logger = logging.getLogger(__name__)

# Fields written back to echo_memory documents under meta.*
_RLHFL_FIELDS = (
    "rlhfl_evaluated_at",
    "rlhfl_sentiment",
    "rlhfl_quality",
    "rlhfl_weight",
    "rlhfl_is_golden",
    "rlhfl_is_noise",
    "rlhfl_topics",
    "rlhfl_user_message",
    "rlhfl_assistant_response",
    "rlhfl_user_followup",
    "rlhfl_reasoning",
)


def _hit_to_interaction(hit: dict) -> Optional[Interaction]:
    """Convert an ES hit to an Interaction. Returns None if required rlhfl fields are absent."""
    src = hit["_source"]
    meta = src.get("meta", {})

    user_msg = meta.get("rlhfl_user_message")
    asst_resp = meta.get("rlhfl_assistant_response")
    if not user_msg or not asst_resp:
        return None

    ts_raw = src.get("@timestamp")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")) if ts_raw else datetime.now()
    except (ValueError, AttributeError):
        ts = datetime.now()

    return Interaction(
        id=hit["_id"],
        conversation_id=meta.get("conversation_id", hit["_id"]),
        timestamp=ts,
        user_message=user_msg,
        assistant_response=asst_resp,
        user_followup=meta.get("rlhfl_user_followup") or None,
        sentiment=float(meta.get("rlhfl_sentiment", 0.0)),
        weight=float(meta.get("rlhfl_weight", 1.0)),
        metadata={k: meta.get(k) for k in _RLHFL_FIELDS},
    )


class ElasticsearchMemoryManager:
    """Manages training data by reading from and annotating TamAGI's ES index."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.es_config = config.elasticsearch
        self.index = self.es_config.index
        self.client: Elasticsearch = create_es_client(self.es_config)

        self.training_state_file = str(Path(app_settings.data_path) / "training_state.json")
        self.user_training_request = False

    # ------------------------------------------------------------------
    # Ingester interface
    # ------------------------------------------------------------------

    def get_unevaluated_interactions(self, limit: int = 25) -> List[dict]:
        """Return up to `limit` raw ES hits that have not been evaluated yet."""
        try:
            result = self.client.search(
                index=self.index,
                query={
                    "bool": {
                        "filter": [{"term": {"data.type": "conversation"}}],
                        "must_not": [{"exists": {"field": "meta.rlhfl_evaluated_at"}}],
                    }
                },
                sort=[{"@timestamp": {"order": "asc"}}],
                size=limit,
            )
            return result["hits"]["hits"]
        except Exception as e:
            logger.error(f"Failed to fetch unevaluated interactions: {e}")
            return []

    def update_with_evaluation(self, doc_id: str, evaluation: dict) -> bool:
        """Write rlhfl evaluation fields back to an existing echo_memory document."""
        try:
            self.client.update(
                index=self.index,
                id=doc_id,
                body={"doc": {"meta": evaluation}},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update doc {doc_id} with evaluation: {e}")
            return False

    # ------------------------------------------------------------------
    # Training pipeline interface (matches MemoryManager signatures)
    # ------------------------------------------------------------------

    def get_golden_examples(self) -> List[Interaction]:
        """Return all interactions marked as golden (no age cutoff — kept forever)."""
        try:
            result = self.client.search(
                index=self.index,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"data.type": "conversation"}},
                            {"term": {"meta.rlhfl_is_golden": True}},
                        ]
                    }
                },
                size=10000,
            )
            interactions = []
            for hit in result["hits"]["hits"]:
                interaction = _hit_to_interaction(hit)
                if interaction:
                    interactions.append(interaction)
            logger.info(f"Retrieved {len(interactions)} golden examples")
            return interactions
        except Exception as e:
            logger.error(f"Failed to retrieve golden examples: {e}")
            return []

    def get_top_interactions_by_weight(
        self, n: int, exclude_golden: bool = True
    ) -> List[Interaction]:
        """Return top-N evaluated, non-noise interactions sorted by weight descending."""
        cutoff = datetime.now() - timedelta(days=self.es_config.max_memory_age_days)

        must_not: list = [{"term": {"meta.rlhfl_is_noise": True}}]
        if exclude_golden:
            must_not.append({"term": {"meta.rlhfl_is_golden": True}})

        try:
            result = self.client.search(
                index=self.index,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"data.type": "conversation"}},
                            {"exists": {"field": "meta.rlhfl_evaluated_at"}},
                            {"range": {"@timestamp": {"gte": cutoff.isoformat()}}},
                        ],
                        "must_not": must_not,
                    }
                },
                sort=[{"meta.rlhfl_weight": {"order": "desc", "unmapped_type": "float"}}],
                size=n,
            )
            interactions = []
            for hit in result["hits"]["hits"]:
                interaction = _hit_to_interaction(hit)
                if interaction:
                    interactions.append(interaction)
            logger.info(f"Retrieved {len(interactions)} top-weighted interactions")
            return interactions
        except Exception as e:
            logger.error(f"Failed to retrieve top interactions by weight: {e}")
            return []

    def get_interactions_since(
        self,
        timestamp: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Interaction]:
        """Return evaluated interactions optionally filtered by a start timestamp."""
        filters: list = [
            {"term": {"data.type": "conversation"}},
            {"exists": {"field": "meta.rlhfl_evaluated_at"}},
        ]
        if timestamp:
            filters.append({"range": {"@timestamp": {"gt": timestamp.isoformat()}}})

        try:
            size = limit if limit else 10000
            result = self.client.search(
                index=self.index,
                query={"bool": {"filter": filters}},
                sort=[{"@timestamp": {"order": "asc"}}],
                size=size,
            )
            interactions = []
            for hit in result["hits"]["hits"]:
                interaction = _hit_to_interaction(hit)
                if interaction:
                    interactions.append(interaction)
            return interactions
        except Exception as e:
            logger.error(f"Failed to retrieve interactions since {timestamp}: {e}")
            return []

    def get_training_stats(self) -> TrainingStats:
        """Compute training trigger statistics via ES aggregations."""
        try:
            training_state = self._load_training_state_full()
            last_training = training_state.get("last_training")
            last_dpo_training = training_state.get("last_dpo_training")

            pos_thresh = 0.3
            neg_thresh = -0.3

            base_filter: list = [
                {"term": {"data.type": "conversation"}},
                {"exists": {"field": "meta.rlhfl_evaluated_at"}},
                {"term": {"meta.rlhfl_is_noise": False}},
            ]

            since_training_filter = (
                [{"range": {"@timestamp": {"gt": last_training.isoformat()}}}]
                if last_training
                else []
            )
            since_dpo_filter = (
                [{"range": {"@timestamp": {"gt": last_dpo_training.isoformat()}}}]
                if last_dpo_training
                else []
            )

            result = self.client.search(
                index=self.index,
                query={"bool": {"filter": base_filter}},
                aggs={
                    "latest_ts": {"max": {"field": "@timestamp"}},
                    "total": {"value_count": {"field": "meta.rlhfl_evaluated_at"}},
                    "since_training": {
                        "filter": {"bool": {"filter": since_training_filter or [{"match_all": {}}]}},
                        "aggs": {
                            "count": {"value_count": {"field": "meta.rlhfl_evaluated_at"}},
                            "positive": {"filter": {"range": {"meta.rlhfl_sentiment": {"gt": pos_thresh}}}},
                            "negative": {"filter": {"range": {"meta.rlhfl_sentiment": {"lt": neg_thresh}}}},
                            "golden": {"filter": {"term": {"meta.rlhfl_is_golden": True}}},
                        },
                    },
                    "neg_since_dpo": {
                        "filter": {
                            "bool": {
                                "filter": (since_dpo_filter or [{"match_all": {}}])
                                + [{"range": {"meta.rlhfl_sentiment": {"lt": neg_thresh}}}]
                            }
                        }
                    },
                },
                size=0,
            )

            aggs = result["aggregations"]
            total = int(aggs["total"]["value"] or 0)

            latest_ts_ms = aggs["latest_ts"].get("value")
            if latest_ts_ms:
                latest_ts = datetime.fromtimestamp(latest_ts_ms / 1000)
                hours_since_last = (datetime.now() - latest_ts).total_seconds() / 3600
            else:
                hours_since_last = 0.0

            since_tr = aggs["since_training"]
            new_count = int(since_tr["count"]["value"] or 0)
            new_positive = int(since_tr["positive"]["doc_count"])
            new_negative = int(since_tr["negative"]["doc_count"])
            new_golden = int(since_tr["golden"]["doc_count"])
            new_neutral = new_count - new_positive - new_negative

            new_neg_since_dpo = int(aggs["neg_since_dpo"]["doc_count"])

            days_since_training = (
                (datetime.now() - last_training).total_seconds() / 86400
                if last_training
                else 0.0
            )
            days_since_dpo = (
                (datetime.now() - last_dpo_training).total_seconds() / 86400
                if last_dpo_training
                else 0.0
            )

            return TrainingStats(
                new_interactions_since_last_training=new_count,
                hours_since_last_interaction=hours_since_last,
                days_since_last_training=days_since_training,
                total_interactions=total,
                last_training_timestamp=last_training,
                user_requested_training=self._is_training_requested(),
                new_positive_count=new_positive,
                new_negative_count=new_negative,
                new_neutral_count=max(0, new_neutral),
                new_golden_count=new_golden,
                last_dpo_training_timestamp=last_dpo_training,
                new_negative_since_dpo=new_neg_since_dpo,
                days_since_dpo_training=days_since_dpo,
                new_refusal_count=0,
                new_refusals_since_dpo=0,
            )

        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return TrainingStats(
                new_interactions_since_last_training=0,
                hours_since_last_interaction=0.0,
                days_since_last_training=0.0,
                total_interactions=0,
                user_requested_training=False,
                new_positive_count=0,
                new_negative_count=0,
                new_neutral_count=0,
                new_golden_count=0,
                last_dpo_training_timestamp=None,
                new_negative_since_dpo=0,
                days_since_dpo_training=0.0,
            )

    # ------------------------------------------------------------------
    # Training state (file-based, unchanged from original)
    # ------------------------------------------------------------------

    def mark_training_complete(self, training_mode: str = "sft"):
        self._save_training_state(datetime.now(), training_mode)
        self.user_training_request = False
        self.clear_training_request()

    def request_training(self, reason: str = "manual", training_mode: str = "auto"):
        self.user_training_request = True
        request_data = {
            "requested_at": datetime.now().isoformat(),
            "reason": reason,
            "training_mode": training_mode,
            "triggered_by": "admin_ui" if reason == "manual" else "scheduler",
        }
        try:
            request_file = Path(app_settings.data_path) / "training_request.json"
            request_file.write_text(json.dumps(request_data, indent=2))
            logger.info(f"Training requested: {reason} (mode: {training_mode})")
        except Exception as e:
            logger.warning(f"Failed to write training request file: {e}")

    def get_training_request(self) -> Optional[dict]:
        try:
            request_file = Path(app_settings.data_path) / "training_request.json"
            if request_file.exists():
                return json.loads(request_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to read training request file: {e}")
        return None

    def clear_training_request(self):
        try:
            request_file = Path(app_settings.data_path) / "training_request.json"
            if request_file.exists():
                request_file.unlink()
                logger.info("Training request cleared")
        except Exception as e:
            logger.warning(f"Failed to clear training request file: {e}")

    def _is_training_requested(self) -> bool:
        if self.user_training_request:
            return True
        return self.get_training_request() is not None

    # ------------------------------------------------------------------
    # Cleanup / size monitoring
    # ------------------------------------------------------------------

    def cleanup_old_memories(self):
        """Log how many old non-golden evaluated docs are beyond the age limit.
        Does not delete — TamAGI owns its index. Age filtering is applied at query time."""
        try:
            cutoff = datetime.now() - timedelta(days=self.es_config.max_memory_age_days)
            result = self.client.count(
                index=self.index,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"data.type": "conversation"}},
                            {"exists": {"field": "meta.rlhfl_evaluated_at"}},
                            {"range": {"@timestamp": {"lt": cutoff.isoformat()}}},
                        ],
                        "must_not": [{"term": {"meta.rlhfl_is_golden": True}}],
                    }
                },
            )
            old_count = result["count"]
            if old_count:
                logger.info(
                    f"{old_count} evaluated docs older than {self.es_config.max_memory_age_days} days "
                    f"are excluded from training queries (not deleted — managed by TamAGI)"
                )
        except Exception as e:
            logger.warning(f"Failed to count old memories: {e}")

    def get_db_size_gb(self) -> float:
        """Return the ES index store size in GB."""
        try:
            stats = self.client.indices.stats(index=self.index, metric="store")
            size_bytes = stats["_all"]["total"]["store"]["size_in_bytes"]
            return size_bytes / (1024 ** 3)
        except Exception as e:
            logger.error(f"Failed to get index size: {e}")
            return 0.0

    def auto_cleanup_if_needed(self) -> bool:
        """Log a warning if the index exceeds the configured size threshold."""
        try:
            current_size = self.get_db_size_gb()
            max_size = self.es_config.max_index_size_gb
            logger.info(f"ES index size: {current_size:.2f} GB / {max_size} GB")
            if current_size >= max_size:
                logger.warning(
                    f"ES index ({current_size:.2f} GB) exceeds max_index_size_gb ({max_size} GB). "
                    f"TamAGI manages index retention — consider adjusting its settings."
                )
            return False
        except Exception as e:
            logger.error(f"Failed to check index size: {e}")
            return False

    def is_connected(self) -> bool:
        return check_es_connection(self.client, self.index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_training_state_full(self) -> dict:
        try:
            with open(self.training_state_file, "r") as f:
                state = json.load(f)
            return {
                "last_training": datetime.fromisoformat(state["last_training"]) if "last_training" in state else None,
                "last_dpo_training": datetime.fromisoformat(state["last_dpo_training"]) if "last_dpo_training" in state else None,
                "last_mode": state.get("last_mode", "sft"),
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return {"last_training": None, "last_dpo_training": None, "last_mode": "sft"}

    def _save_training_state(self, timestamp: datetime, training_mode: str = "sft"):
        try:
            existing = self._load_training_state_full()
            state: dict = {
                "last_training": timestamp.isoformat(),
                "last_mode": training_mode,
            }
            if training_mode == "dpo":
                state["last_dpo_training"] = timestamp.isoformat()
            elif existing["last_dpo_training"]:
                state["last_dpo_training"] = existing["last_dpo_training"].isoformat()

            with open(self.training_state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Training state saved: mode={training_mode}, timestamp={timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
