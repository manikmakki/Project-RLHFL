"""
Psyche Module - Data Models

Pydantic models for inter-layer communication between Id, Ego, and Superego.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class RewardContext(BaseModel):
    """Real-time reward signal from the Id Engine."""
    current_sentiment: float = 0.0
    trend: Literal["improving", "declining", "stable"] = "stable"
    session_reward_mean: float = 0.0
    suggested_temperature_adjustment: float = 0.0


class EgoParseResult(BaseModel):
    """Extracted reasoning metadata from Ego Fast."""
    thinking: Optional[str] = None
    reflection: Optional[str] = None
    confidence_estimate: Optional[float] = None
    action_plan: Optional[str] = None


class JudgmentResult(BaseModel):
    """Superego discrimination judgment."""
    allowed: bool = True
    reason: Optional[str] = None
    confidence: float = 1.0
    fast_path: bool = False  # True if resolved by pattern match, False if LLM judge was called


class ValidationResult(BaseModel):
    """Combined validation result from Superego."""
    allowed: bool = True
    reason: Optional[str] = None
    judgment: Optional[JudgmentResult] = None
    modified_tool_calls: Optional[List[Dict[str, Any]]] = None


class PatternAnalysis(BaseModel):
    """Strategic pattern analysis from Ego Slow."""
    sentiment_trend: float = 0.0
    common_failure_topics: List[str] = Field(default_factory=list)
    training_effectiveness: Optional[float] = None
    recommendation_notes: List[str] = Field(default_factory=list)


class TrainingRecommendation(BaseModel):
    """Training strategy recommendation from Ego Slow."""
    mode: Literal["sft", "dpo", "skip"] = "sft"
    reason: str = ""
    suggested_lora_rank: Optional[int] = None
    urgency: Literal["low", "medium", "high"] = "medium"


class PreProcessResult(BaseModel):
    """Result of the orchestrator's pre-processing pass."""
    messages: List[Any]  # Enriched ChatMessage list
    reward_context: Optional[RewardContext] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    permissive_framing_applied: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostProcessResult(BaseModel):
    """Result of the orchestrator's post-processing pass."""
    response_dict: Dict[str, Any]
    ego_parse: Optional[EgoParseResult] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    refusal_detected: bool = False
    retry_attempted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
