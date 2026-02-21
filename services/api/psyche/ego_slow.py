"""
Psyche Module - Ego Slow (Background Strategic Planner)

Operates asynchronously to monitor interaction patterns, analyze training
effectiveness, and recommend training strategy adjustments.

Extends (but does not replace) the existing TrainingScheduler by feeding
PatternAnalysis into its decision-making process.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from shared.config import SystemConfig
from shared.models import TrainingStats
from api.psyche.models import PatternAnalysis, TrainingRecommendation

logger = logging.getLogger(__name__)


class EgoSlow:
    """
    Background strategic planner that monitors patterns and recommends
    training strategy adjustments.

    Does NOT replace the TrainingScheduler -- it provides supplementary
    PatternAnalysis that the scheduler can optionally use.
    """

    def __init__(self, config: SystemConfig, memory_manager):
        """
        Args:
            config: System configuration
            memory_manager: MemoryManager instance for querying interactions
        """
        self.config = config
        self.psyche_config = config.psyche
        self.memory = memory_manager
        self._enabled = config.psyche.ego_slow_enabled

        # Cache for pattern analysis (avoid re-computing on every check)
        self._last_analysis: Optional[PatternAnalysis] = None
        self._last_analysis_time: Optional[datetime] = None

        logger.info(
            f"EgoSlow initialized: enabled={self._enabled}, "
            f"analysis_interval={self.psyche_config.ego_pattern_analysis_interval_hours}h"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.ego_slow_enabled

    def analyze_patterns(self) -> PatternAnalysis:
        """
        Query ChromaDB for recent interactions and analyze patterns.

        Returns cached result if analysis was run recently (within
        ego_pattern_analysis_interval_hours).

        Returns:
            PatternAnalysis with trend, failure topics, and recommendations
        """
        if not self.enabled:
            return PatternAnalysis()

        # Check cache freshness
        interval = timedelta(hours=self.psyche_config.ego_pattern_analysis_interval_hours)
        if (
            self._last_analysis
            and self._last_analysis_time
            and datetime.now() - self._last_analysis_time < interval
        ):
            logger.debug("Ego Slow returning cached pattern analysis")
            return self._last_analysis

        try:
            analysis = self._run_analysis()
            self._last_analysis = analysis
            self._last_analysis_time = datetime.now()
            return analysis
        except Exception as e:
            logger.error(f"Ego Slow pattern analysis failed: {e}")
            return self._last_analysis or PatternAnalysis()

    def _run_analysis(self) -> PatternAnalysis:
        """Perform the actual pattern analysis."""
        # Get recent interactions (last 7 days)
        since = datetime.now() - timedelta(days=7)
        interactions = self.memory.get_interactions_since(
            timestamp=since, limit=500
        )

        if not interactions:
            return PatternAnalysis(recommendation_notes=["No recent interactions to analyze"])

        # Compute sentiment trend
        sentiments = [i.sentiment for i in interactions]
        n = len(sentiments)
        if n >= 4:
            mid = n // 2
            first_half_mean = sum(sentiments[:mid]) / mid
            second_half_mean = sum(sentiments[mid:]) / (n - mid)
            sentiment_trend = second_half_mean - first_half_mean
        else:
            sentiment_trend = 0.0

        # Identify common failure topics (interactions with negative sentiment)
        negative_interactions = [i for i in interactions if i.sentiment < -0.3]
        failure_topics = self._extract_topics(negative_interactions)

        # Build recommendations
        notes = []

        if sentiment_trend > 0.2:
            notes.append("Sentiment improving -- consider reducing training frequency")
        elif sentiment_trend < -0.2:
            notes.append("Sentiment declining -- consider prioritizing DPO training")

        if len(negative_interactions) > n * 0.3:
            notes.append(
                f"High negative ratio ({len(negative_interactions)}/{n}) -- "
                "DPO training recommended"
            )

        if failure_topics:
            notes.append(f"Common failure topics: {', '.join(failure_topics[:5])}")

        analysis = PatternAnalysis(
            sentiment_trend=round(sentiment_trend, 3),
            common_failure_topics=failure_topics[:10],
            recommendation_notes=notes,
        )

        logger.info(
            f"Ego Slow analysis: trend={sentiment_trend:.3f}, "
            f"failures={len(negative_interactions)}/{n}, "
            f"recommendations={len(notes)}"
        )

        return analysis

    def _extract_topics(self, interactions: list) -> List[str]:
        """
        Extract common keywords from negative interactions.

        Simple word frequency approach -- no external NLP dependencies.
        """
        if not interactions:
            return []

        # Common stop words to filter out
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their", "what", "which",
            "who", "when", "where", "how", "not", "no", "but", "and",
            "or", "if", "then", "so", "just", "also", "very", "too",
        }

        word_counts: Dict[str, int] = {}
        for interaction in interactions:
            words = interaction.user_message.lower().split()
            for word in words:
                word = word.strip(".,!?;:'\"()[]{}").lower()
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Return top words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if count >= 2]

    def recommend_training_strategy(
        self,
        stats: TrainingStats,
        patterns: Optional[PatternAnalysis] = None,
    ) -> TrainingRecommendation:
        """
        Enhanced training strategy recommendation.

        Factors in pattern analysis alongside raw stats.

        Args:
            stats: Current training statistics
            patterns: Optional pattern analysis (computed if not provided)

        Returns:
            TrainingRecommendation with mode, reason, and urgency
        """
        if not self.enabled:
            return TrainingRecommendation(mode="sft", reason="ego_slow disabled")

        if patterns is None:
            patterns = self.analyze_patterns()

        # If sentiment is declining significantly, prioritize DPO
        if patterns.sentiment_trend < -0.2 and stats.new_negative_since_dpo >= 3:
            return TrainingRecommendation(
                mode="dpo",
                reason=f"Declining sentiment trend ({patterns.sentiment_trend:.2f}) with accumulated negatives",
                urgency="high",
            )

        # If sentiment is improving, reduce urgency
        if patterns.sentiment_trend > 0.2:
            return TrainingRecommendation(
                mode="sft",
                reason="Sentiment improving -- routine reinforcement",
                urgency="low",
            )

        # Default: use stats-based mode selection
        if stats.new_negative_since_dpo >= self.config.training.dpo_mode_threshold:
            return TrainingRecommendation(
                mode="dpo",
                reason=f"{stats.new_negative_since_dpo} negatives since last DPO",
                urgency="medium",
            )

        return TrainingRecommendation(
            mode="sft",
            reason="Default SFT reinforcement",
            urgency="medium",
        )

    def is_healthy(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "enabled": self.enabled,
            "has_cached_analysis": self._last_analysis is not None,
            "last_analysis_time": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            "analysis_interval_hours": self.psyche_config.ego_pattern_analysis_interval_hours,
        }
