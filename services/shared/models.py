"""
Project RLHFL - Data Models

This module defines all Pydantic data models used throughout the system.
These models provide:

- Type safety and validation
- Serialization/deserialization
- OpenAI API compatibility
- Documentation via field descriptions

Key model categories:
- API models: OpenAI-compatible request/response formats
- Memory models: Interaction storage and retrieval
- Training models: Dataset samples and checkpoints
- System models: Health status and metadata
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: Union[str, List, Dict, Any] = Field(
        ...,
        description="Message content - string or array of content blocks"
    )
    name: Optional[str] = Field(
        None,
        description="Optional name of the author of this message"
    )

    def get_text_content(self) -> str:
        """Extract text content for sentiment analysis (only called for USER messages)."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # For arrays of content blocks, extract text
            text_parts = []
            for item in self.content:
                if isinstance(item, dict):
                    if 'text' in item:
                        text_parts.append(str(item['text']))
                    elif 'content' in item:
                        text_parts.append(str(item['content']))
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts) if text_parts else ""
        elif isinstance(self.content, dict):
            # For dict-based content
            if 'text' in self.content:
                return str(self.content['text'])
            elif 'content' in self.content:
                return str(self.content['content'])
            else:
                return str(self.content)
        elif self.content is None:
            return ""
        else:
            return str(self.content)


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible Chat Completion Request.
    Supports both streaming and non-streaming modes.
    """
    model_config = ConfigDict(extra="allow")  # Allow extra fields for forward compatibility
    
    # Required fields
    messages: List[ChatMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far."
    )
    
    # Model selection
    model: Optional[str] = Field(
        None,
        description="ID of the model to use. If not provided, uses the system default."
    )
    
    # Sampling/Generation parameters
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="What sampling temperature to use, between 0 and 2. Higher values make output more random."
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="An alternative to sampling with temperature, called nucleus sampling."
    )
    n: Optional[int] = Field(
        1,
        ge=1,
        description="How many chat completion choices to generate for each input message."
    )
    stream: bool = Field(
        False,
        description="If true, partial message deltas will be sent."
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens."
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="The maximum number of tokens that can be generated in the chat completion."
    )
    presence_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Number between -2.0 and 2.0 that penalizes new tokens based on whether they appear in the text so far."
    )
    frequency_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Number between -2.0 and 2.0 that penalizes new tokens based on their existing frequency."
    )
    logit_bias: Optional[Dict[str, int]] = Field(
        None,
        description="Modify the likelihood of specified tokens appearing in the completion."
    )
    
    # Response format
    response_format: Optional[Dict[str, str]] = Field(
        None,
        description="An object specifying the format that the model must output."
    )
    
    # Reproducibility
    seed: Optional[int] = Field(
        None,
        description="If specified, the system will make a best effort to sample deterministically."
    )
    
    # Function calling / Tools
    tools: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="A list of tools the model may call."
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="Controls which (if any) function is called by the model."
    )
    
    # Logging preferences  
    logprobs: Optional[bool] = Field(
        None,
        description="Include the log probabilities on the logprobs most likely tokens."
    )
    top_logprobs: Optional[int] = Field(
        None,
        ge=0,
        le=20,
        description="An integer between 0 and 20 specifying the number of most likely tokens to return."
    )
    
    # User identification (not standard OpenAI, but useful for tracking)
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing the end-user for tracking and analyzing."
    )


class ChatCompletionDelta(BaseModel):
    """Delta object for streaming responses."""
    role: Optional[MessageRole] = None
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[Any] = None

    class Config:
        use_enum_values = True


class ChatCompletionChoice(BaseModel):
    """Choice in a chat completion response."""
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[ChatCompletionDelta] = None  # For streaming
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class Interaction(BaseModel):
    """Represents a single interaction stored in memory."""
    id: str
    conversation_id: str
    timestamp: datetime
    user_message: str
    assistant_response: str
    sentiment: float = 0.0
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingStats(BaseModel):
    """Statistics about training status."""
    new_interactions_since_last_training: int
    hours_since_last_interaction: float
    days_since_last_training: float
    total_interactions: int
    last_training_timestamp: Optional[datetime] = None
    user_requested_training: bool = False


class TrainingDatasetSample(BaseModel):
    """Single training sample."""
    prompt: str
    response: str
    label: Literal["chosen", "rejected"]
    weight: float = 1.0


class CheckpointMetadata(BaseModel):
    """Metadata for a training checkpoint."""
    checkpoint_id: str
    adapter_path: str
    timestamp: datetime
    metrics: Dict[str, float]
    parent_checkpoint: Optional[str] = None
    training_samples: int
    can_rollback: bool = True
    deployed: bool = False


class HealthStatus(BaseModel):
    """API health status."""
    status: str
    model_loaded: bool
    memory_connected: bool
    gpu_available: bool
    current_checkpoint: Optional[str] = None
