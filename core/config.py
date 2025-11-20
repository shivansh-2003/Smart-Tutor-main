"""
Centralized configuration for Smart Tutor
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Provider-specific models
    openai_model: str = "gpt-4o-mini"
    google_model: str = "gemini-2.0-flash-exp"
    ollama_model: str = "gpt-oss:20b"


@dataclass
class RAGConfig:
    """RAG system configuration"""
    index_name: str = "smart-tutor"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 20
    mmr_k: int = 10
    final_k: int = 5
    namespace_default: str = "documents"


@dataclass
class RouterConfig:
    """Intent router configuration"""
    confidence_threshold: float = 0.7
    use_context: bool = True
    max_retries: int = 3
    router_model: str = "gpt-4o-mini"
    router_temperature: float = 0.1


@dataclass
class ContextConfig:
    """Context manager configuration"""
    max_history: int = 20
    context_window: int = 10
    store_embeddings: bool = False
    session_timeout: int = 3600  # seconds


@dataclass
class ModuleConfig:
    """Module-specific configurations"""
    
    # Chat module
    chat_modes: list = field(default_factory=lambda: ["learn", "hint", "quiz", "eli5", "custom"])
    chat_default_mode: str = "learn"
    
    # Math module
    math_model: str = "gemini-2.0-flash-exp"
    math_temperature: float = 0.0
    
    # Mermaid module
    mermaid_model: str = "gpt-4o"
    mermaid_temperature: float = 0.1
    diagram_types: list = field(default_factory=lambda: [
        "flowchart", "sequence", "class", "er_diagram",
        "state", "graph", "journey", "gitgraph", "c4"
    ])
    
    # Mindmap module
    mindmap_model: str = "gpt-oss:20b"
    mindmap_temperature: float = 0.1


@dataclass
class PathConfig:
    """File path configuration"""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    uploads_dir: Path = field(default_factory=lambda: Path("data/uploads"))
    vectorstore_dir: Path = field(default_factory=lambda: Path("data/vectorstore"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.data_dir, self.uploads_dir, self.vectorstore_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """API keys and endpoints"""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    
    def validate(self) -> bool:
        """Check if required API keys are present"""
        required = {
            "openai": self.openai_api_key,
            "pinecone": self.pinecone_api_key
        }
        missing = [name for name, key in required.items() if not key]
        
        if missing:
            print(f"⚠️  Missing API keys: {', '.join(missing)}")
            return False
        return True


@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling modules"""
    enable_chat: bool = True
    enable_math: bool = True
    enable_mermaid: bool = True
    enable_mindmap: bool = True
    enable_rag: bool = True
    enable_context: bool = True
    debug_mode: bool = False


class Config:
    """Main configuration class"""
    
    _instance: Optional['Config'] = None
    
    def __init__(self):
        self.llm = LLMConfig()
        self.rag = RAGConfig()
        self.router = RouterConfig()
        self.context = ContextConfig()
        self.modules = ModuleConfig()
        self.paths = PathConfig()
        self.api = APIConfig()
        self.features = FeatureFlags()
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "llm": self.llm.__dict__,
            "rag": self.rag.__dict__,
            "router": self.router.__dict__,
            "context": self.context.__dict__,
            "modules": self.modules.__dict__,
            "features": self.features.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update config from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def validate(self) -> bool:
        """Validate configuration"""
        return self.api.validate()
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for specific provider"""
        provider = provider or self.llm.provider
        
        model_map = {
            "openai": self.llm.openai_model,
            "google": self.llm.google_model,
            "ollama": self.llm.ollama_model
        }
        
        return {
            "provider": provider,
            "model": model_map.get(provider, self.llm.model),
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens
        }
    
    def get_module_llm_config(self, module_name: str) -> Dict[str, Any]:
        """Get LLM config for specific module"""
        module_configs = {
            "math": {
                "provider": "google",
                "model": self.modules.math_model,
                "temperature": self.modules.math_temperature
            },
            "mermaid": {
                "provider": "openai",
                "model": self.modules.mermaid_model,
                "temperature": self.modules.mermaid_temperature
            },
            "mindmap": {
                "provider": "ollama",
                "model": self.modules.mindmap_model,
                "temperature": self.modules.mindmap_temperature
            },
            "chat": self.get_llm_config()
        }
        
        return module_configs.get(module_name, self.get_llm_config())
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Check if module is enabled"""
        return getattr(self.features, f"enable_{module_name}", False)


def get_config() -> Config:
    """Get global config instance"""
    return Config.get_instance()


# Environment-based config presets
def load_development_config(config: Config):
    """Load development configuration"""
    config.features.debug_mode = True
    config.llm.temperature = 0.0
    config.router.confidence_threshold = 0.6


def load_production_config(config: Config):
    """Load production configuration"""
    config.features.debug_mode = False
    config.llm.temperature = 0.7
    config.router.confidence_threshold = 0.75


def load_config_from_env():
    """Load configuration based on environment"""
    config = get_config()
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        load_production_config(config)
    else:
        load_development_config(config)
    
    return config