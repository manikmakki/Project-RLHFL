"""
Project RLHFL - Configuration Management

Centralized configuration using Pydantic for validation and YAML for storage.
RLHFL is a pure sidecar service: it reads TamAGI's echo_memory ES index,
evaluates conversations with the LLM, and trains LoRA adapters.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    model_id: str = "gemma4:27b"
    base_model_path: str = "/models/gemma-4-26B-A4B"
    context_length: int = 8192
    max_tokens: int = 2048


class TrainingConfig(BaseModel):
    min_interactions_threshold: int = 50
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
    max_seq_length: int = 256

    cpu_threads: int = 32

    enable_dpo: bool = False
    dpo_beta: float = 0.1
    dpo_batch_size: int = 8
    dpo_rejection_timeout: int = 300

    dpo_trigger_threshold: int = 5

    schedule_enabled: bool = False
    schedule_time: str = "01:00"
    schedule_timezone: str = "America/New_York"
    schedule_window_minutes: int = 60

    enable_deployment: bool = False

    max_checkpoint_count: int = 3
    max_gguf_files: int = 2

    # A stored baseline below this value is almost certainly caused by
    # eval/train data contamination and should not be used as the comparison target.
    min_reasonable_perplexity: float = 2.0


class EvaluationWindowConfig(BaseModel):
    """Time window during which the ES ingester is allowed to run evaluations."""
    start: str = "01:00"
    end: str = "05:00"
    timezone: str = "America/New_York"


class ElasticsearchConfig(BaseModel):
    """Elasticsearch connection and sidecar-mode settings."""
    host: str = "http://localhost:9200"
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    index: str = "echo_memory"

    # When True, RLHFL reads from TamAGI's index rather than intercepting chats.
    sidecar_mode: bool = True

    evaluation_window: EvaluationWindowConfig = Field(default_factory=EvaluationWindowConfig)
    batch_size: int = 25

    max_memory_age_days: int = 90
    max_index_size_gb: float = 25.0
    top_n_weighted_interactions: int = 1000


class LlamaCppConfig(BaseModel):
    """Connection settings for the external llama.cpp server."""
    base_url: str = "http://host.docker.internal:8080"
    api_key: Optional[str] = None
    read_timeout_seconds: int = 120
    lora_scale: float = 1.0
    host_checkpoints_path: Optional[str] = None


class SystemConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)


class Settings(BaseSettings):
    base_model_path: str = os.getenv("BASE_MODEL_PATH", "/models/gemma-4-26B-A4B")
    config_path: str = os.getenv("CONFIG_PATH", "/config/system_config.yaml")
    data_path: str = os.getenv("DATA_PATH", "/data")
    checkpoints_path: str = os.getenv("CHECKPOINTS_PATH", "/checkpoints")

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
        config = SystemConfig()
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
