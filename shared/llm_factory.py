"""
LLM Factory utilities for local model management.
"""

import importlib
import logging
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Dict, Any, Iterable, TYPE_CHECKING

def _load_env():
    try:
        return importlib.import_module("dotenv").load_dotenv()
    except ImportError:  # pragma: no cover - optional dependency
        return None

if TYPE_CHECKING:
    from core.config import LLMConfig, ModelConfig

try:
    import ollama as client_lib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    client_lib = None

_load_env()

logger = logging.getLogger(__name__)


class LLM:
    """Wrapper for interacting with local model chat endpoints"""

    def __init__(
        self,
        client: "client_lib.Client",
        model_name: str,
        config: "ModelConfig",
        timeout: int,
    ):
        self.client = client
        self.model_name = model_name
        self.config = config
        self.timeout = timeout

    def _build_messages(self, prompt: str, system: Optional[str]) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = overrides or {}
        return {
            "temperature": overrides.get("temperature", self.config.temperature),
            "top_p": overrides.get("top_p", self.config.top_p),
            "top_k": overrides.get("top_k", self.config.top_k),
            "repeat_penalty": overrides.get("repeat_penalty", self.config.repeat_penalty),
            "num_predict": overrides.get("num_predict", self.config.num_predict),
        }

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Generate a non-streamed response"""
        messages = self._build_messages(prompt, system)
        options = self._build_options(kwargs)

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=False,
        )

        return response["message"]["content"]

    def stream_generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterable[str]:
        """Stream generation chunks"""
        messages = self._build_messages(prompt, system)
        options = self._build_options(kwargs)

        for chunk in self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=True,
        ):
            yield chunk["message"]["content"]

    def invoke(self, prompt: str, system: Optional[str] = None, **kwargs):
        """
        Compatibility helper to mimic standard ChatModel interface.
        Returns an object with a .content attribute.
        """
        content = self.generate(prompt, system=system, **kwargs)
        return SimpleNamespace(content=content)


class LLMFactory:
    """Factory for creating and managing local LLMs"""

    def __init__(self, config: "LLMConfig"):
        if client_lib is None:
            raise ImportError(
                "The 'ollama' package is required for LLMFactory. Install it via `pip install ollama`."
            )

        self.config = config
        self.client = client_lib.Client(host=config.base_url)
        self._check_models()

    def _check_models(self):
        """Warn if required models are missing locally"""
        try:
            available = [model["name"] for model in self.client.list().get("models", [])]
        except Exception as exc:  # pragma: no cover - network error
            logger.warning("Unable to inspect local models: %s", exc)
            return

        required = {
            self.config.complex_model,
            self.config.simple_model,
            self.config.embedding_model,
        }

        missing = [model for model in required if model not in available]

        if missing:
            logger.warning(
                "Missing local models: %s. Pull them via `ollama pull <model>`.",
                ", ".join(missing),
            )

    @lru_cache(maxsize=10)
    def get_llm(self, task_name: str) -> LLM:
        model_name = self.config.get_model_for_task(task_name)
        model_config = self.config.get_model_config(model_name)
        timeout = self.config.get_timeout_for_task(task_name)

        logger.info(
            "Task '%s' -> Model '%s'",
            task_name,
            model_name,
        )

        return LLM(
            client=self.client,
            model_name=model_name,
            config=model_config,
            timeout=timeout,
        )

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via local embedding endpoint"""
        embeddings: list[list[float]] = []

        for text in texts:
            response = self.client.embeddings(
                model=self.config.embedding_model,
                prompt=text,
            )
            embeddings.append(response["embedding"])

        return embeddings


_FACTORY: Optional[LLMFactory] = None


def _get_factory() -> LLMFactory:
    """Lazy-init singleton for LLMFactory"""
    global _FACTORY

    if _FACTORY:
        return _FACTORY

    from core.config import Config

    config = Config.get_instance()
    _FACTORY = LLMFactory(config.llm)
    return _FACTORY


def reset_factory():
    """Reset cached factory (helpful for tests)"""
    global _FACTORY
    _FACTORY = None


def get_task_llm(task_name: str):
    """Retrieve an LLM tuned for a specific task."""
    factory = _get_factory()
    return factory.get_llm(task_name)