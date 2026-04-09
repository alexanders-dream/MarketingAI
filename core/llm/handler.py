"""
LLM provider handling and management
"""
import logging
from typing import Optional, Union, Tuple, List
import requests
import time

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import AppConfig, get_settings

logger = logging.getLogger(__name__)

class LLMProviderHandler:
    """Handles multiple LLM providers with unified interface"""

    SUPPORTED_PROVIDERS = ["GROQ", "OPENAI", "GEMINI", "OLLAMA"]

    # We use a primitive dict instead of lru_cache because we only want to cache locally
    # based on endpoint and provider, but API keys are sensitive and shouldn't be long-lived in cache.
    _models_cache = {}

    @classmethod
    def fetch_models(cls, provider: str, endpoint: str, api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
        """
        Fetch available models for a provider with a simple TTL cache.

        Args:
            provider: Provider name (GROQ, OPENAI, GEMINI, OLLAMA)
            endpoint: API endpoint
            api_key: API key (not required for Ollama)

        Returns:
            Tuple of (list_of_model_names, error_string_if_any)
        """
        provider = provider.upper()
        cache_key = f"{provider}_{endpoint}_{api_key}"
        
        # Check cache (15 minutes TTL)
        if cache_key in cls._models_cache:
            models, timestamp = cls._models_cache[cache_key]
            if time.time() - timestamp < 900:
                return models, None

        try:
            if provider == "GROQ":
                if not api_key:
                    return [], "API key is required for Groq."
                url = f"{endpoint}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    models = [model['id'] for model in response.json()['data']]
                    cls._models_cache[cache_key] = (models, time.time())
                    return models, None
                return [], f"Failed to fetch models: {response.text}"

            elif provider == "OPENAI":
                if not api_key:
                    return [], "API key is required for OpenAI."
                url = f"{endpoint}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    models = [model['id'] for model in response.json()['data']]
                    cls._models_cache[cache_key] = (models, time.time())
                    return models, None
                return [], f"Failed to fetch models: {response.text}"

            elif provider == "GEMINI":
                # Gemini models are predefined
                models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
                cls._models_cache[cache_key] = (models, time.time())
                return models, None

            elif provider == "OLLAMA":
                url = f"{endpoint}/api/tags"
                response = requests.get(url)
                if response.status_code == 200:
                    models = [model['name'] for model in response.json()['models']]
                    cls._models_cache[cache_key] = (models, time.time())
                    return models, None
                return [], f"Failed to connect to Ollama at {endpoint}"

            else:
                logger.warning(f"Unsupported provider: {provider}")
                return [], f"Unsupported provider: {provider}"

        except Exception as e:
            logger.error(f"Error fetching models for {provider}: {str(e)}")
            return [], f"An unexpected error occurred: {str(e)}"

    @classmethod
    def create_client(cls, provider: str, model: str, api_key: Optional[str] = None,
                     endpoint: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> Optional[Union[ChatGroq, ChatOllama, ChatOpenAI, ChatGoogleGenerativeAI]]:
        
        settings = get_settings()
        if temperature is None:
            temperature = settings.default_temperature
        if max_tokens is None:
            max_tokens = settings.default_max_tokens

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
                  endpoint: Optional[str] = None, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> Optional[Union[ChatGroq, ChatOllama, ChatOpenAI, ChatGoogleGenerativeAI]]:
        
        settings = get_settings()
        if temperature is None:
            temperature = settings.default_temperature
        if max_tokens is None:
            max_tokens = settings.default_max_tokens

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
        self._clients.clear()

    def generate(self, client, prompt: str) -> str:
        try:
            response = client.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return ""

def get_llm_for_agent(agent_type: str) -> Union[ChatGroq, ChatOllama, ChatOpenAI, ChatGoogleGenerativeAI]:
    """Factory to get the appropriate LLM for a given agent type. 
    Defaults to groq for fast reasoning unless overridden."""
    settings = get_settings()
    manager = LLMManager()

    # In production, this might map specific agents to specific models/providers
    if settings.groq_api_key:
        return manager.get_client("GROQ", "llama3-8b-8192", settings.groq_api_key)
    elif settings.openai_api_key:
        return manager.get_client("OPENAI", "gpt-3.5-turbo", settings.openai_api_key)
    else:
        raise ValueError("No valid LLM configuration found in environment for agents.")
