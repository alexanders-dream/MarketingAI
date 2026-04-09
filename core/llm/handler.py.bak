"""
LLM provider handling for Marketing AI v3
"""
import logging
from typing import Optional, Union, Dict, Any
import requests
import functools

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig

logger = logging.getLogger(__name__)


class LLMProviderHandler:
    """Handles multiple LLM providers with unified interface"""

    SUPPORTED_PROVIDERS = ["GROQ", "OPENAI", "GEMINI", "OLLAMA"]

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def fetch_models(provider: str, endpoint: str, api_key: Optional[str] = None) -> list:
        """
        Fetch available models for a provider

        Args:
            provider: Provider name (GROQ, OPENAI, GEMINI, OLLAMA)
            endpoint: API endpoint
            api_key: API key (not required for Ollama)

        Returns:
            List of available model names
        """
        provider = provider.upper()

        try:
            if provider == "GROQ":
                if not api_key:
                    return [], "API key is required for Groq."
                url = f"{endpoint}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return [model['id'] for model in response.json()['data']], None
                return [], f"Failed to fetch models: {response.text}"

            elif provider == "OPENAI":
                if not api_key:
                    return [], "API key is required for OpenAI."
                url = f"{endpoint}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return [model['id'] for model in response.json()['data']], None
                return [], f"Failed to fetch models: {response.text}"

            elif provider == "GEMINI":
                # Gemini models are predefined
                return [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.0-pro"
                ], None

            elif provider == "OLLAMA":
                url = f"{endpoint}/api/tags"
                response = requests.get(url)
                if response.status_code == 200:
                    return [model['name'] for model in response.json()['models']], None
                return [], f"Failed to connect to Ollama at {endpoint}"

            else:
                logger.warning(f"Unsupported provider: {provider}")
                return [], f"Unsupported provider: {provider}"

        except Exception as e:
            logger.error(f"Error fetching models for {provider}: {str(e)}")
            return [], f"An unexpected error occurred: {str(e)}"

    @staticmethod
    def create_client(provider: str, model: str, api_key: Optional[str] = None,
                     endpoint: Optional[str] = None, temperature: float = AppConfig.DEFAULT_TEMPERATURE,
                     max_tokens: int = AppConfig.DEFAULT_MAX_TOKENS) -> Optional[Union[ChatGroq, ChatOllama, ChatOpenAI, ChatGoogleGenerativeAI]]:
        """
        Create LLM client for specified provider

        Args:
            provider: Provider name
            model: Model name
            api_key: API key
            endpoint: Custom endpoint
            temperature: Temperature setting
            max_tokens: Max tokens

        Returns:
            LLM client instance or None if failed
        """
        provider = provider.upper()

        try:
            if provider == "GROQ":
                if not api_key:
                    raise ValueError("Groq API key is required")
                return ChatGroq(
                    api_key=api_key,
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            elif provider == "OPENAI":
                if not api_key:
                    raise ValueError("OpenAI API key is required")
                return ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    base_url=endpoint or AppConfig.get_endpoint("OPENAI"),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            elif provider == "GEMINI":
                if not api_key:
                    raise ValueError("Gemini API key is required")
                return ChatGoogleGenerativeAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            elif provider == "OLLAMA":
                return ChatOllama(
                    model=model,
                    base_url=endpoint or AppConfig.get_endpoint("OLLAMA"),
                    temperature=temperature,
                    num_predict=max_tokens
                )

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to create {provider} client: {str(e)}")
            return None


class LLMManager:
    """High-level LLM management with caching"""

    def __init__(self):
        self._clients = {}

    def get_client(self, provider: str, model: str, api_key: Optional[str] = None,
                  endpoint: Optional[str] = None, temperature: float = AppConfig.DEFAULT_TEMPERATURE,
                  max_tokens: int = AppConfig.DEFAULT_MAX_TOKENS) -> Optional[Union[ChatGroq, ChatOllama, ChatOpenAI, ChatGoogleGenerativeAI]]:
        """
        Get or create cached LLM client

        Args:
            provider: Provider name
            model: Model name
            api_key: API key
            endpoint: Custom endpoint
            temperature: Temperature setting
            max_tokens: Max tokens

        Returns:
            LLM client instance
        """
        # Create cache key
        cache_key = f"{provider}_{model}_{api_key}_{endpoint}_{temperature}_{max_tokens}"

        if cache_key not in self._clients:
            client = LLMProviderHandler.create_client(
                provider=provider,
                model=model,
                api_key=api_key,
                endpoint=endpoint,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if client:
                self._clients[cache_key] = client

        return self._clients.get(cache_key)

    def clear_cache(self):
        """Clear all cached clients"""
        self._clients.clear()

    def generate(self, client, prompt: str) -> str:
        """
        Generate text using the provided client and prompt.

        Args:
            client: The LLM client.
            prompt: The prompt to use for generation.

        Returns:
            The generated text.
        """
        try:
            response = client.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return ""
