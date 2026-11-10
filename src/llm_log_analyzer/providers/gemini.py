"""
Google Gemini LLM provider implementation.
Uses the google-genai SDK (https://github.com/googleapis/python-genai).
"""

from typing import Optional
from llm_log_analyzer.environment import PROVIDER_API_KEYS
from .base import LLMProvider, TEMPERATURE, MAX_OUTPUT_TOKENS
from .types import LLMProviderType

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or PROVIDER_API_KEYS[LLMProviderType.GEMINI.value], **kwargs)
        try:
            from google import genai
            if not self.api_key:
                raise Exception("Gemini API key is required")
            from google.genai import types
            # The -latest aliases exist only on v1beta; google-genai 3.0 defaults to v1, where they 404.
            self._client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(api_version="v1beta"))
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini provider. "
                "Install with: pip install google-genai"
            )

    def get_default_chunk_model(self) -> str:
        # Floating alias -> current Flash tier. Pinned Gemini IDs (e.g. gemini-2.5-*)
        # get retired and start returning 404; the alias tracks the latest GA model.
        return "gemini-flash-latest"

    def get_default_aggregation_model(self) -> str:
        return "gemini-pro-latest"

    def get_default_tokenizer_encoding_name(self) -> str:
        return "cl100k_base"

    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make request to Gemini API."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        if response.candidates and getattr(response.candidates[0], "finish_reason", None) == "SAFETY":
            raise Exception("Response was blocked by safety filters. Try rephrasing the input.")

        text = getattr(response, "text", None) or (response.candidates[0].content.parts[0].text if response.candidates and response.candidates[0].content.parts else None)
        if not text:
            raise Exception("Empty response received from Gemini API")

        response_text = text.strip()
        self.logger.debug(f"Gemini response length: {len(response_text)} chars")
        return response_text
