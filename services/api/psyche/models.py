"""
Psyche Module - Data Models

Pydantic models for inter-layer communication between Id, Ego, and Superego.
Includes both legacy models (PreProcessResult, PostProcessResult) and
Psyche v2 refinement models (RefinementResult, SuperegoProfile, etc.).
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


# ─── Shared / Legacy Models ────────────────────────────────────────────────

class RewardContext(BaseModel):
    """Real-time reward signal from the Id Engine."""
    current_sentiment: float = 0.0
    trend: Literal["improving", "declining", "stable"] = "stable"
    session_reward_mean: float = 0.0
    suggested_temperature_adjustment: float = 0.0
    desire_signal: Optional[str] = None  # Natural-language desire framing from Id model


class EgoParseResult(BaseModel):
    """Extracted reasoning metadata from Ego Fast (legacy)."""
    thinking: Optional[str] = None
    reflection: Optional[str] = None
    confidence_estimate: Optional[float] = None
    action_plan: Optional[str] = None


class JudgmentResult(BaseModel):
    """Superego discrimination judgment (safety layer)."""
    allowed: bool = True
    reason: Optional[str] = None
    confidence: float = 1.0
    fast_path: bool = False


class ValidationResult(BaseModel):
    """Combined validation result from Superego safety check."""
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


# ─── Legacy Orchestrator Results (used when refinement_enabled=False) ──────

class PreProcessResult(BaseModel):
    """Result of the orchestrator's pre-processing pass (legacy)."""
    messages: List[Any]
    reward_context: Optional[RewardContext] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    permissive_framing_applied: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostProcessResult(BaseModel):
    """Result of the orchestrator's post-processing pass (legacy)."""
    response_dict: Dict[str, Any]
    ego_parse: Optional[EgoParseResult] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    refusal_detected: bool = False
    retry_attempted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Psyche v2: Refinement Pipeline Models ─────────────────────────────────

class SuperegoJudgment(BaseModel):
    """Superego's evaluation of a candidate response."""
    quality_score: float = 0.0
    critique: str = ""
    param_adjustments: Dict[str, float] = Field(default_factory=dict)


class RefinementTurn(BaseModel):
    """Record of a single refinement turn in the inner loop."""
    turn_number: int
    quality_score: float
    critique: str
    param_adjustments: Dict[str, float] = Field(default_factory=dict)
    response_preview: str = ""  # First 200 chars of candidate response


class RefinementResult(BaseModel):
    """Complete result of the refinement pipeline."""
    turns_used: int = 1
    turns_budget: int = 1
    final_quality_score: float = 0.0
    efficiency_score: float = 0.0  # quality / turns
    accepted_at_turn: int = 1
    turn_history: List[RefinementTurn] = Field(default_factory=list)
    best_response_dict: Dict[str, Any] = Field(default_factory=dict)
    complexity_class: str = "simple"  # "simple", "moderate", "complex"
    deliberation_log: str = ""  # Human-readable inner loop dialogue
    validation: ValidationResult = Field(default_factory=ValidationResult)
    reward_context: Optional[RewardContext] = None
    refusal_detected: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SuperegoProfile(BaseModel):
    """
    Global persistent personality calibration.

    Evolves slowly across ALL conversations to form a coherent personality.
    Persisted to disk (JSON) so it survives restarts.
    """
    # Baseline sampling params applied to all Ego generations
    baseline_params: Dict[str, float] = Field(
        default_factory=lambda: {"temp": 0.7, "top_p": 0.9, "top_k": 40}
    )

    # Per-prompt-type learned adjustments (e.g. {"code": {"temp": -0.15}})
    type_adjustments: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Adjustment rate — decays over time for stability
    # rate = max(0.02, 0.3 * e^(-total_conversations / 50))
    adjustment_rate: float = 0.3
    total_conversations: int = 0

    # Score tracking for self-introspection (rolling window, last 100)
    recent_scores: List[float] = Field(default_factory=list)
    recent_turn_counts: List[int] = Field(default_factory=list)
    score_trend: Literal["improving", "declining", "stable"] = "stable"

    # Self-calibration bounds (adjusted by introspection)
    quality_threshold: float = 0.7
    id_temperature: float = 0.7  # Superego controls Id's temperature

    # Introspection state
    last_introspection_conversation: int = 0
    introspection_interval: int = 25
