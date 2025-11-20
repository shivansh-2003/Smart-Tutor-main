"""
Mermaid module configuration for task-specific routing.
"""

from typing import Dict

from core.config import Config
from shared.llm_factory import get_task_llm


class MermaidModuleConfig:
    """Resolves Mermaid pipeline stages to appropriately sized models."""

    def __init__(self):
        self.config = Config.get_instance()
        self.task_models: Dict[str, str] = {
            "query_analysis": "mermaid_query_analysis",
            "rag_synthesis": "mermaid_synthesis",
            "diagram_generation": "mermaid_generate",
            "validation": "mermaid_validation",
        }
    def get_llm_for_task(self, task_name: str):
        """Return an LLM tuned for the supplied Mermaid task."""
        resolved_task = self.task_models.get(task_name, "mermaid_generate")
        return get_task_llm(resolved_task)

