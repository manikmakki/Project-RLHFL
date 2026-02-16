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
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    model_id: str = "gpt-oss:20b"  # Model identifier for API responses
    base_model_path: str = "/models/gpt-oss-20b-Q4_K_M.gguf"
    base_model_hf_path: str = "/models/gpt-oss-20b-base"
    context_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    n_gpu_layers: int = -1  # Use all GPU layers
    n_batch: int = 512
    n_threads: int = 4
    kv_cache_type: str = "q8_0"
    use_mmap: bool = True
    use_mlock: bool = True
    checkpoint_poll_interval_seconds: int = 30
    max_old_gguf_files: int = 2
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

    # Sequential layer training parameters (for memory-constrained training)
    enable_sequential_training: bool = False  # Toggle sequential mode
    layers_per_pass: int = 5  # How many layers to train per pass
    sequential_merge_checkpoints: bool = True  # Auto-merge between passes

    # CPU training mode (for systems with limited VRAM but abundant RAM/cores)
    enable_cpu_training: bool = False  # Train on CPU instead of GPU
    cpu_threads: int = 32  # Number of CPU threads to use (leave room for inference)

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


class MemoryConfig(BaseModel):
    golden_examples_count: int = 20
    max_memory_age_days: int = 90
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    similarity_threshold: float = 0.7
    # RAG (Retrieval Augmented Generation) settings - DEPRECATED
    # System now operates in transparent proxy mode, focusing on RLHF without RAG
    rag_enabled: bool = False  # Default changed from True (deprecated)
    rag_top_k: int = 5  # Number of similar interactions to retrieve for context (deprecated)
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
    base_url: str = "http://ollama:11434"  # Override with LLM_BASE_URL env var

    # Endpoint type: "auto" (detect from URL), "ollama", "openai"
    endpoint_type: str = "auto"

    # Model name to request from external LLM
    external_model_name: str = "qwen2.5:32b"  # Override with LLM_MODEL_NAME env var

    # API key for authenticated endpoints (OpenAI, etc.)
    api_key: Optional[str] = None  # Override with OPENAI_API_KEY env var

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


class SystemConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    llm_proxy: LLMProxyConfig = Field(default_factory=LLMProxyConfig)


class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "/models/gpt-oss-20b-Q4_K_M.gguf")
    base_model_path: str = os.getenv("BASE_MODEL_PATH", "/models/gpt-oss-20b-base")
    config_path: str = os.getenv("CONFIG_PATH", "/config/system_config.yaml")
    data_path: str = os.getenv("DATA_PATH", "/data")
    checkpoints_path: str = os.getenv("CHECKPOINTS_PATH", "/checkpoints")

    # LLM Proxy settings (override config file)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
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
