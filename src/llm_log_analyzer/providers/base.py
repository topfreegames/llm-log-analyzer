"""
Base classes and constants for LLM providers.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Model constants
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 8000

class LLMProvider(ABC):
    """Abstract base class for LLM providers - handles API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, backoff_factor: float = 2.0, timeout: Optional[float] = None):
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout  # None means no timeout (default behavior)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_default_chunk_model(self) -> str:
        """Get the default model for chunk analysis."""
        pass
    
    @abstractmethod
    def get_default_aggregation_model(self) -> str:
        """Get the default model for aggregation."""
        pass
    
    @abstractmethod
    def get_default_tokenizer_encoding_name(self) -> str:
        """Get the default tokenizer name."""
        pass

    @abstractmethod
    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make an API request to the provider. Implemented by subclasses."""
        pass
    
    @staticmethod
    def _is_model_not_found(error: Exception) -> bool:
        text = str(error).lower()
        return "404" in text and "not found" in text

    def _fallback_model(self, model: str) -> Optional[str]:
        """The other default model, tried once when a model is retired (404)."""
        for candidate in (self.get_default_chunk_model(), self.get_default_aggregation_model()):
            if candidate != model:
                return candidate
        return None

    def make_request(self, model: str, prompt: str) -> str:
        """Make a request with exponential backoff retry and optional timeout."""
        last_exception = None
        fallback_used = False
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.timeout is not None:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self._make_api_request, model, prompt)
                        try:
                            return future.result(timeout=self.timeout)
                        except FutureTimeoutError:
                            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
                else:
                    return self._make_api_request(model, prompt)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

                fallback = None if fallback_used else self._fallback_model(model)
                if self._is_model_not_found(e) and fallback:
                    self.logger.warning(f"Model {model} not found; retrying with {fallback}")
                    model, fallback_used = fallback, True
                    continue
                
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    self.logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    break
        
        raise Exception(f"All retry attempts failed. Last error: {last_exception}")
