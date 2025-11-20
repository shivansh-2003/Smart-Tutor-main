"""
Math module configuration utilities for task-aware LLM routing.
"""

from typing import Dict

from core.config import Config
from shared.llm_factory import get_task_llm


class MathModuleConfig:
    """Map math sub-tasks to task-specific LLM instances."""

    def __init__(self):
        self.config = Config.get_instance()
        self.task_models: Dict[str, str] = {
            "input_processing": "math_input_processing",
            "intent_classification": "math_intent_classification",
            "problem_analysis": "math_analyze",
            "solution_generation": "math_solve_problem",
            "hint_generation": "math_hint",
            "validation": "math_validate",
        }
    def get_llm_for_agent(self, agent_name: str):
        """Return an LLM instance tailored for the given math agent."""
        task_name = self.task_models.get(agent_name, "math_solve_problem")
        return get_task_llm(task_name)

