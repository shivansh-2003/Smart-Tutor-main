"""
Centralized configuration for Smart Tutor
"""

import importlib
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
def _load_env():
    try:
        return importlib.import_module("dotenv").load_dotenv()
    except ImportError:  # pragma: no cover - optional dependency
        return None


_load_env()


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    context_window: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = -1
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None


@dataclass
class LLMConfig:
    """Local model configuration with task-based model selection."""
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434"))
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Complex task models (20B-32B)
    complex_model: str = field(default_factory=lambda: os.getenv("LLM_COMPLEX_MODEL", "gpt-oss:20b"))
    complex_model_fallback: str = field(default_factory=lambda: os.getenv("LLM_COMPLEX_FALLBACK", "deepseek-r1:32b"))

    # Simple task models (2B-3B)
    simple_model: str = field(default_factory=lambda: os.getenv("LLM_SIMPLE_MODEL", "qwen2.5:3b"))
    simple_model_fallback: str = field(default_factory=lambda: os.getenv("LLM_SIMPLE_FALLBACK", "gemma2:2b"))

    # Embeddings
    embedding_model: str = field(default_factory=lambda: os.getenv("LLM_EMBEDDING_MODEL", "nomic-embed-text:latest"))
    embedding_dimension: int = 768

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "gpt-oss:20b": ModelConfig(
            name="gpt-oss:20b",
            context_window=32768,
            temperature=0.7,
            top_p=0.9,
        ),
        "deepseek-r1:32b": ModelConfig(
            name="deepseek-r1:32b",
            context_window=16384,
            temperature=0.8,
            top_p=0.95,
        ),
        "qwen2.5:3b": ModelConfig(
            name="qwen2.5:3b",
            context_window=32768,
            temperature=0.7,
            top_p=0.9,
        ),
        "gemma2:2b": ModelConfig(
            name="gemma2:2b",
            context_window=8192,
            temperature=0.7,
            top_p=0.9,
        ),
        "phi3:3.8b": ModelConfig(
            name="phi3:3.8b",
            context_window=4096,
            temperature=0.7,
            top_p=0.9,
        ),
    })

    # Task complexity mapping
    task_complexity_map: Dict[str, TaskComplexity] = field(default_factory=lambda: {
        # Router & context tasks
        "intent_classification": TaskComplexity.COMPLEX,
        "context_analysis": TaskComplexity.COMPLEX,

        # Math module
        "math_solve_problem": TaskComplexity.COMPLEX,
        "math_prove_theorem": TaskComplexity.COMPLEX,
        "math_analyze": TaskComplexity.COMPLEX,
        "math_hint": TaskComplexity.SIMPLE,
        "math_validate": TaskComplexity.SIMPLE,

        # Mermaid module
        "mermaid_generate": TaskComplexity.COMPLEX,
        "mermaid_query_analysis": TaskComplexity.MODERATE,
        "mermaid_synthesis": TaskComplexity.COMPLEX,
        "mermaid_validation": TaskComplexity.SIMPLE,

        # Mindmap module
        "mindmap_create": TaskComplexity.COMPLEX,
        "mindmap_expand": TaskComplexity.MODERATE,

        # Chat module
        "chat_rag_synthesis": TaskComplexity.COMPLEX,
        "chat_casual": TaskComplexity.SIMPLE,
        "chat_follow_up": TaskComplexity.SIMPLE,
        "chat_greeting": TaskComplexity.SIMPLE,

        # RAG tasks
        "rag_rerank": TaskComplexity.MODERATE,
        "rag_synthesis": TaskComplexity.COMPLEX,
        "rag_query_expansion": TaskComplexity.SIMPLE,
    })

    # Performance settings
    simple_task_timeout: int = 30
    complex_task_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 5

    def get_model_for_task(self, task_name: str) -> str:
        """Select appropriate model for a task."""
        complexity = self.task_complexity_map.get(task_name, TaskComplexity.MODERATE)

        if complexity == TaskComplexity.COMPLEX:
            return self.complex_model

        return self.simple_model

    def get_model_config(self, model_name: Optional[str] = None) -> ModelConfig:
        """Get configuration for a specific model."""
        target = model_name or self.simple_model
        return self.models.get(target, self.models[self.simple_model])

    def get_timeout_for_task(self, task_name: str) -> int:
        """Return timeout window based on task complexity."""
        complexity = self.task_complexity_map.get(task_name, TaskComplexity.MODERATE)
        return self.complex_task_timeout if complexity == TaskComplexity.COMPLEX else self.simple_task_timeout


@dataclass
class RAGConfig:
    """RAG system configuration"""
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "smart-tutor"))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    ))
    embedding_device: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_DEVICE", "cpu"))
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("HUGGINGFACE_EMBED_DIM", "384"))
    )
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
    diagram_types: list = field(default_factory=lambda: [
        "flowchart", "sequence", "class", "er_diagram",
        "state", "graph", "journey", "gitgraph", "c4"
    ])


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
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    
    def validate(self) -> bool:
        """Check if required API keys are present"""
        required = {
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
    
    def get_llm_config(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get local model configuration for a specific task"""
        resolved_task = task_name or "chat_casual"
        model_name = self.llm.get_model_for_task(resolved_task)
        model_config = self.llm.get_model_config(model_name)

        return {
            "model": model_name,
            "base_url": self.llm.base_url,
            "timeout": self.llm.get_timeout_for_task(resolved_task),
            "task": resolved_task,
            "options": {
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "top_k": model_config.top_k,
                "repeat_penalty": model_config.repeat_penalty,
                "num_predict": model_config.num_predict,
            },
        }
    
    def get_module_llm_config(self, module_name: str, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM config for specific module (task-aware)"""
        module_task_fallbacks = {
            "math": "math_solve_problem",
            "mermaid": "mermaid_generate",
            "mindmap": "mindmap_create",
            "chat": "chat_casual",
        }
        resolved_task = task_name or module_task_fallbacks.get(module_name, f"{module_name}_default")
        return self.get_llm_config(task_name=resolved_task)
    
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