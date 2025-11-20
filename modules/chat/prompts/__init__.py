"""
Prompt loader for chat module
"""

from pathlib import Path
from typing import Optional


class PromptLoader:
    """Load and format prompt templates for chat modes"""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize prompt loader"""
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.prompts_dir = prompts_dir
        self._cache = {}
    
    def _load_prompt(self, mode: str) -> str:
        """Load prompt template for mode"""
        if mode in self._cache:
            return self._cache[mode]
        
        # Map mode to filename
        mode_file_map = {
            'learn': 'learn.txt',
            'hint': 'hint.txt',
            'quiz': 'quiz.txt',
            'eli5': 'eli5.txt',
            'custom': 'default.txt'
        }
        
        filename = mode_file_map.get(mode, 'learn.txt')
        prompt_path = self.prompts_dir / filename
        
        if not prompt_path.exists():
            # Fallback to default
            prompt_path = self.prompts_dir / 'learn.txt'
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._cache[mode] = content
            return content
        except Exception as e:
            # Return a basic fallback prompt
            return self._get_fallback_prompt(mode)
    
    def _get_fallback_prompt(self, mode: str) -> str:
        """Get fallback prompt if file loading fails"""
        return f"""You are a helpful AI tutor in {mode.upper()} mode.

CONVERSATION HISTORY:
{{history}}

CONTEXT FROM STUDY MATERIALS:
{{context}}

USER INPUT: {{question}}

RESPONSE:"""
    
    def format(
        self,
        mode: str,
        context: str = "",
        question: str = "",
        history: str = "",
        custom_instructions: str = ""
    ) -> str:
        """Format prompt template with variables"""
        template = self._load_prompt(mode)
        
        # Format template
        formatted = template.format(
            context=context or "No context available.",
            question=question,
            history=history or "No previous conversation.",
            custom_instructions=custom_instructions or "No custom instructions provided."
        )
        
        return formatted


__all__ = ['PromptLoader']

