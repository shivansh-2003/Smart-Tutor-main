"""
LLM Factory for centralized model initialization
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LLMFactory:
    """Factory for creating LLM instances"""
    
    _instances: Dict[str, Any] = {}
    
    @staticmethod
    def create_llm(
        provider: LLMProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Create LLM instance based on provider"""
        
        cache_key = f"{provider}_{model}_{temperature}"
        
        if cache_key in LLMFactory._instances:
            return LLMFactory._instances[cache_key]
        
        if provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs
            )
        
        elif provider == LLMProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                **kwargs
            )
        
        elif provider == LLMProvider.OLLAMA:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        LLMFactory._instances[cache_key] = llm
        return llm
    
    @staticmethod
    def get_embedding_model(provider: LLMProvider = LLMProvider.OPENAI):
        """Create embedding model instance"""
        
        if provider == LLMProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        elif provider == LLMProvider.GOOGLE:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def clear_cache():
        """Clear cached LLM instances"""
        LLMFactory._instances.clear()


def get_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
):
    """Convenience function for getting LLM"""
    
    provider_enum = LLMProvider(provider.lower())
    
    # Default models
    default_models = {
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.GOOGLE: "gemini-2.0-flash-exp",
        LLMProvider.OLLAMA: "gpt-oss:20b"
    }
    
    model = model or default_models[provider_enum]
    
    return LLMFactory.create_llm(
        provider=provider_enum,
        model=model,
        temperature=temperature,
        **kwargs
    )