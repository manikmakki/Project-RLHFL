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
    model_id: str = "mistral-7b-instruct"  # Model identifier for API responses
    base_model_path: str = "/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    base_model_hf_path: str = "/models/mistral-7b-instruct-base"
    context_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    n_gpu_layers: int = -1  # Use all GPU layers
    n_batch: int = 512
    n_threads: int = 4


class TrainingConfig(BaseModel):
    min_interactions_threshold: int = 50
    inactivity_threshold_hours: int = 24
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


class MemoryConfig(BaseModel):
    golden_examples_count: int = 20
    max_memory_age_days: int = 90
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    similarity_threshold: float = 0.7
    # RAG (Retrieval Augmented Generation) settings
    rag_enabled: bool = True
    rag_top_k: int = 5  # Number of similar interactions to retrieve for context
    # Memory management settings
    max_db_size_gb: float = 10.0  # Maximum database size before auto-cleanup
    auto_cleanup_threshold: float = 0.9  # Trigger cleanup at 90% of max size
    auto_cleanup_percentage: float = 0.1  # Remove bottom 10% by weight during auto-cleanup
    min_weight_threshold: float = 0.5  # Remove entries below this weight during pre-training cleanup


class SentimentConfig(BaseModel):
    positive_threshold: float = 0.3
    negative_threshold: float = -0.3
    explicit_instruction_boost: float = 3.0
    continuation_baseline: float = 0.1


class SystemConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)


class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    base_model_path: str = os.getenv("BASE_MODEL_PATH", "/models/mistral-7b-instruct-base")
    config_path: str = os.getenv("CONFIG_PATH", "/config/system_config.yaml")
    data_path: str = os.getenv("DATA_PATH", "/data")
    checkpoints_path: str = os.getenv("CHECKPOINTS_PATH", "/checkpoints")
    
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
