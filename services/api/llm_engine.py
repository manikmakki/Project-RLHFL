import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_cpp import Llama
from shared.models import ChatMessage, MessageRole
from shared.config import SystemConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    """LLM inference engine using llama.cpp."""
    
    def __init__(self, config: SystemConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        self.current_adapter_path: Optional[str] = None
        self.llm: Optional[Llama] = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model with llama.cpp."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.model.context_length,
                n_gpu_layers=self.config.model.n_gpu_layers,
                n_batch=self.config.model.n_batch,
                n_threads=self.config.model.n_threads,
                verbose=True  # Temporarily enable to check GPU offloading
            )
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def reload_with_adapter(self, adapter_path: str):
        """
        Reload model with a LoRA adapter.
        Note: llama.cpp GGUF doesn't directly support LoRA adapters.
        This would require converting the adapted model back to GGUF.
        For simplicity, we'll track the adapter path for the training service.
        """
        self.current_adapter_path = adapter_path
        logger.info(f"Adapter path updated to: {adapter_path}")
        # In production, you'd convert the HF model + adapter to GGUF and reload
    
    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for Mistral-Instruct prompt format."""
        formatted = ""
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                formatted += f"[INST] {msg.content} [/INST]\n"
            elif msg.role == MessageRole.USER:
                formatted += f"[INST] {msg.content} [/INST]\n"
            elif msg.role == MessageRole.ASSISTANT:
                formatted += f"{msg.content}\n"
        
        return formatted.strip()
    
    def generate(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from the model."""
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        prompt = self.format_messages(messages)
        
        # Use provided parameters or fall back to config defaults
        temp = temperature if temperature is not None else self.config.model.temperature
        top_p_val = top_p if top_p is not None else self.config.model.top_p
        max_tok = max_tokens if max_tokens is not None else self.config.model.max_tokens
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tok,
                temperature=temp,
                top_p=top_p_val,
                stop=["[INST]", "</s>"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Generate a streaming response from the model."""
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        prompt = self.format_messages(messages)
        
        temp = temperature if temperature is not None else self.config.model.temperature
        top_p_val = top_p if top_p is not None else self.config.model.top_p
        max_tok = max_tokens if max_tokens is not None else self.config.model.max_tokens
        
        try:
            stream = self.llm(
                prompt,
                max_tokens=max_tok,
                temperature=temp,
                top_p=top_p_val,
                stop=["[INST]", "</s>"],
                echo=False,
                stream=True
            )
            
            for output in stream:
                if 'choices' in output and len(output['choices']) > 0:
                    text = output['choices'][0].get('text', '')
                    if text:
                        yield text
                        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.llm is not None
    
    def get_current_adapter(self) -> Optional[str]:
        """Get the current adapter path."""
        return self.current_adapter_path
