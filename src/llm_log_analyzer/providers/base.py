"""
Base classes and constants for LLM providers.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
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
        # Models that actually answered, and any fallback a provider took (e.g. a retired model).
        self.models_used: List[str] = []
        self.model_fallbacks: List[str] = []
        self._answered_model: Optional[str] = None
    
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
    
    def make_request(self, model: str, prompt: str) -> str:
        """Make a request with exponential backoff retry and optional timeout."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.timeout is not None:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self._make_api_request, model, prompt)
                        try:
                            response = future.result(timeout=self.timeout)
                        except FutureTimeoutError:
                            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
                else:
                    response = self._make_api_request(model, prompt)
                self.models_used.append(self._answered_model or model)
                self._answered_model = None
                return response
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    self.logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    break
        
        raise Exception(f"All retry attempts failed. Last error: {last_exception}")
