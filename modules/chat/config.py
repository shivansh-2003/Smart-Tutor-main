"""
Chat module configuration helpers for task-based routing.
"""

from typing import Dict

from core.config import Config
from shared.llm_factory import get_task_llm


class ChatModuleConfig:
    """Determines which model to use for each chat mode."""

    def __init__(self):
        self.config = Config.get_instance()
        self.task_models: Dict[str, str] = {
            "casual_chat": "chat_casual",
            "rag_search": "chat_rag_synthesis",
            "rag_synthesis": "chat_rag_synthesis",
            "greeting": "chat_greeting",
            "follow_up": "chat_follow_up",
        }
    def get_llm_for_mode(self, mode: str):
        """
        Retrieve an LLM based on chat mode.
        RAG modes are routed to complex models automatically.
        """
        normalized_mode = mode.lower()
        resolved_task = self.task_models.get(normalized_mode, "chat_casual")
        return get_task_llm(resolved_task)

