"""
Project RLHFL - Configuration Management

This module provides centralized configuration management using Pydantic
for validation and YAML for storage. It defines all system parameters
including:

- Model settings (paths, context length, generation parameters)
- Training hyperparameters (LoRA rank, learning rate, batch sizes)
- Memory management (embeddings, golden examples, retention)
- Sentiment analysis thresholds and weights

Configuration can be loaded from YAML files or environment variables,
with sensible defaults for all parameters. The configuration is validated
at load time to catch errors early.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    model_id: str = "dolphin3:8b"  # Model identifier for API responses / Ollama tag
    base_model_path: str = "/models/Dolphin3.0-Llama3.1-8B"  # HF safetensors for training
    ollama_base_model: str = "dolphin3:8b"  # Ollama base model tag (for adapter FROM)
    hf_model_id: str = "dphn/Dolphin3.0-Llama3.1-8B"  # HuggingFace model ID (reference)
    context_length: int = 8192
    max_tokens: int = 2048
    n_threads: int = 4
    response_format: str = "openai"  # "openai" or "openclaw"


class TrainingConfig(BaseModel):
    min_interactions_threshold: int = 50
    inactivity_threshold_hours: int = 8
    max_days_between_training: int = 7
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    validation_split: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    max_seq_length: int = 256  # Reduced from 512 for attention mask compatibility

    # CPU training threads (leave room for inference on the same machine)
    cpu_threads: int = 32

    # DPO (Direct Preference Optimization) settings
    enable_dpo: bool = False  # Master DPO toggle (if false, always use SFT)
    dpo_beta: float = 0.1  # DPO temperature parameter (controls strength of preference)
    dpo_batch_size: int = 8  # Batch size for rejection generation
    dpo_rejection_timeout: int = 300  # Timeout per rejection generation (seconds)

    # Sentiment-based training triggers
    sft_trigger_threshold: int = 20  # Trigger SFT when ≥ this many new positive/golden
    dpo_trigger_threshold: int = 5   # Trigger DPO when ≥ this many new negative sentiments
    dpo_mode_threshold: int = 5      # Use DPO mode if ≥ this many negatives exist

    # Training schedule settings
    schedule_enabled: bool = False              # Enable scheduled training (vs immediate)
    schedule_time: str = "01:00"                # Time to run training (24-hour format)
    schedule_timezone: str = "America/New_York"  # Timezone for schedule_time
    schedule_window_minutes: int = 60           # Execute if within N minutes of scheduled time

    # Ollama deployment
    enable_ollama_deployment: bool = False  # Deploy to Ollama after training
    ollama_model_name: str = "dolphin3:8b"  # Ollama model name to update

    # Checkpoint storage management
    max_checkpoint_count: int = 3  # Max checkpoint dirs to keep
    max_gguf_files: int = 2  # Max old GGUF files in Ollama


class MemoryConfig(BaseModel):
    max_memory_age_days: int = 90
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    # Memory management settings
    max_db_size_gb: float = 10.0  # Maximum database size before auto-cleanup
    auto_cleanup_threshold: float = 0.9  # Trigger cleanup at 90% of max size
    auto_cleanup_percentage: float = 0.1  # Remove bottom 10% by weight during auto-cleanup
    min_weight_threshold: float = 0.5  # Remove entries below this weight during pre-training cleanup
    top_n_weighted_interactions: int = 1000  # Total interactions (including golden) to include in training


class SentimentConfig(BaseModel):
    positive_threshold: float = 0.3
    negative_threshold: float = -0.3
    explicit_instruction_boost: float = 3.0
    continuation_baseline: float = 0.1


class LLMProxyConfig(BaseModel):
    """Configuration for HTTP proxy to external LLM services."""
    # External LLM endpoint URL
    base_url: str = "http://llm-ollama:11434"  # Override with LLM_BASE_URL env var

    # Endpoint type: "auto" (detect from URL), "ollama", "openai"
    endpoint_type: str = "auto"

    # Model name to request from external LLM
    external_model_name: str = "qwen2.5:32b"  # Override with LLM_MODEL_NAME env var

    # API key for authenticated endpoints (OpenAI, etc.)
    api_key: Optional[str] = None  # Override with OPENAI_API_KEY env var

    # Lightweight model for sentiment analysis via Ollama
    sentiment_model: str = "qwen2.5:0.5b"

    # Model whitelist for sentiment analysis + storage
    # Empty list = analyze all models (default for single-model setups)
    # Non-empty = only whitelisted models get sentiment + storage
    sentiment_enabled_models: list[str] = []

    # HTTP client settings
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 120  # Long for streaming
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    max_connections: int = 10
    max_keepalive_connections: int = 5


class PsycheConfig(BaseModel):
    """Configuration for the Psyche module (Id / Ego / Superego)."""
    enabled: bool = True

    # Multi-model roles (empty = use main model from llm_proxy.external_model_name)
    id_model: str = ""           # Small reward/desire model (e.g. qwen2.5:1.5b-instruct)
    ego_model: str = ""          # Response generation model (e.g. dolphin3:8b)
    superego_model: str = ""     # Critique/optimization model (e.g. phi4-mini:3.8b)

    # Refinement loop
    refinement_enabled: bool = True
    max_refinement_turns: int = 5
    quality_threshold: float = 0.7       # Accept response at or above this score
    min_quality_for_accept: float = 0.5  # Accept even below threshold if best after max turns
    complexity_bypass_chars: int = 50    # Skip refinement for very short prompts
    refinement_bypass_tools: bool = True  # Skip refinement when tools are present

    # Client parameter control
    override_client_params: bool = True  # Strip client-sent temp/top_p/top_k

    # Id Engine (reward-driven)
    id_enabled: bool = True
    id_session_window: int = 10
    id_reward_prompt_threshold: float = 0.3

    # Ego Fast (per-request reasoning) — legacy, used only when refinement_enabled=False
    ego_fast_enabled: bool = True
    ego_reasoning_prompt: bool = True
    ego_parse_reflection: bool = True

    # Ego Slow (background strategy)
    ego_slow_enabled: bool = True
    ego_pattern_analysis_interval_hours: int = 4

    # Superego (discrimination + optimization)
    superego_enabled: bool = True
    superego_safety_judge: bool = True  # LLM-based safety judge (disable to keep only pattern matching)
    superego_permissive_framing: bool = True
    superego_refusal_retry: bool = True
    superego_max_retries: int = 1
    superego_judge_timeout_seconds: int = 15

    # Superego parameter control
    superego_param_control: bool = True
    superego_temp_range: list = [0.1, 1.2]
    superego_top_p_range: list = [0.3, 1.0]
    superego_top_k_range: list = [5, 100]
    superego_judge_max_tokens: int = 100

    # Superego self-introspection
    superego_introspection_interval: int = 25  # Self-evaluate every N conversations
    superego_profile_path: str = "/data/superego_profile.json"

    # Deliberation exposure
    expose_deliberation: bool = False  # Include inner loop dialogue in response thinking field

    # Safety patterns
    superego_danger_patterns: list[str] = [
        "rm -rf /",
        ":(){ :|:& };:",
        "mkfs.",
        "dd if=/dev/zero",
        "> /dev/sda",
    ]
    superego_safe_patterns: list[str] = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    ]


class SystemConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    llm_proxy: LLMProxyConfig = Field(default_factory=LLMProxyConfig)
    psyche: PsycheConfig = Field(default_factory=PsycheConfig)

    @model_validator(mode='after')
    def _resolve_model_defaults(self) -> 'SystemConfig':
        """
        Auto-derive dependent model names from model.model_id when not explicitly set.

        This makes model.model_id the single source of truth — changing it once
        propagates to llm_proxy, training, and psyche without editing each field.
        """
        mid = self.model.model_id

        # LLM proxy model name
        proxy_default = LLMProxyConfig.model_fields['external_model_name'].default
        if not self.llm_proxy.external_model_name or self.llm_proxy.external_model_name == proxy_default:
            self.llm_proxy.external_model_name = mid

        # Sentiment model
        sentiment_default = LLMProxyConfig.model_fields['sentiment_model'].default
        if not self.llm_proxy.sentiment_model or self.llm_proxy.sentiment_model == sentiment_default:
            self.llm_proxy.sentiment_model = mid

        # Sentiment enabled models whitelist
        if not self.llm_proxy.sentiment_enabled_models:
            self.llm_proxy.sentiment_enabled_models = [mid]

        # Training deployment model name
        training_default = TrainingConfig.model_fields['ollama_model_name'].default
        if not self.training.ollama_model_name or self.training.ollama_model_name == training_default:
            self.training.ollama_model_name = mid

        return self


class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "/models/Dolphin3.0-Llama3.1-8B")
    base_model_path: str = os.getenv("BASE_MODEL_PATH", "/models/Dolphin3.0-Llama3.1-8B")
    config_path: str = os.getenv("CONFIG_PATH", "/config/system_config.yaml")
    data_path: str = os.getenv("DATA_PATH", "/data")
    checkpoints_path: str = os.getenv("CHECKPOINTS_PATH", "/checkpoints")

    # LLM Proxy settings (override config file)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    log_level: str = os.getenv("LOG_LEVEL", "DEBUG")


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load system configuration from YAML file."""
    if config_path is None:
        settings = Settings()
        config_path = settings.config_path
    
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SystemConfig(**config_dict)
    else:
        # Return default configuration
        config = SystemConfig()
        # Save default config for future reference
        save_config(config, config_path)
        return config


def save_config(config: SystemConfig, config_path: str):
    """Save system configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)


# Global settings instance
settings = Settings()
