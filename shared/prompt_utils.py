"""
Prompt utilities for formatting and template management
"""

from typing import Dict, Any, List, Optional
from string import Template


class PromptFormatter:
    """Utilities for prompt formatting and template management"""
    
    @staticmethod
    def format_template(template: str, variables: Dict[str, Any]) -> str:
        """Format template with variables, handling missing keys gracefully"""
        try:
            return template.format(**variables)
        except KeyError as e:
            missing_key = str(e).strip("'")
            variables[missing_key] = f"[{missing_key} not provided]"
            return template.format(**variables)
    
    @staticmethod
    def safe_format(template: str, **kwargs) -> str:
        """Safe template formatting using string.Template"""
        t = Template(template)
        return t.safe_substitute(**kwargs)
    
    @staticmethod
    def format_context(
        context: str,
        max_length: Optional[int] = None,
        truncate_message: str = "\n...[truncated]..."
    ) -> str:
        """Format context with optional truncation"""
        if max_length and len(context) > max_length:
            truncate_at = max_length - len(truncate_message)
            return context[:truncate_at] + truncate_message
        return context
    
    @staticmethod
    def format_chat_history(
        messages: List[Dict[str, str]],
        max_messages: Optional[int] = None,
        format_template: str = "{role}: {content}\n"
    ) -> str:
        """Format chat history into string"""
        if max_messages:
            messages = messages[-max_messages:]
        
        formatted = []
        for msg in messages:
            formatted.append(format_template.format(
                role=msg.get("role", "unknown").title(),
                content=msg.get("content", "")
            ))
        
        return "".join(formatted)
    
    @staticmethod
    def format_list_items(
        items: List[str],
        format_type: str = "numbered",
        indent: int = 0
    ) -> str:
        """Format list items as numbered or bulleted"""
        indent_str = " " * indent
        
        if format_type == "numbered":
            return "\n".join(
                f"{indent_str}{i+1}. {item}" for i, item in enumerate(items)
            )
        elif format_type == "bullet":
            return "\n".join(f"{indent_str}• {item}" for item in items)
        elif format_type == "dash":
            return "\n".join(f"{indent_str}- {item}" for item in items)
        else:
            return "\n".join(f"{indent_str}{item}" for item in items)
    
    @staticmethod
    def format_dict_as_context(
        data: Dict[str, Any],
        exclude_keys: Optional[List[str]] = None,
        max_value_length: Optional[int] = 200
    ) -> str:
        """Format dictionary as readable context string"""
        exclude_keys = exclude_keys or []
        
        lines = []
        for key, value in data.items():
            if key in exclude_keys:
                continue
            
            if isinstance(value, (dict, list)):
                value = str(value)
            
            if max_value_length and len(str(value)) > max_value_length:
                value = str(value)[:max_value_length] + "..."
            
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def clean_json_from_markdown(text: str) -> str:
        """Extract and clean JSON from markdown code blocks"""
        text = text.strip()
        
        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Remove language identifier
        if text.startswith("json"):
            text = text[4:].strip()
        
        return text.strip()
    
    @staticmethod
    def format_system_prompt(
        role: str,
        instructions: List[str],
        constraints: Optional[List[str]] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Format a structured system prompt"""
        prompt_parts = [f"You are {role}."]
        
        if instructions:
            prompt_parts.append("\nINSTRUCTIONS:")
            prompt_parts.append(PromptFormatter.format_list_items(instructions, "numbered"))
        
        if constraints:
            prompt_parts.append("\nCONSTRAINTS:")
            prompt_parts.append(PromptFormatter.format_list_items(constraints, "bullet"))
        
        if examples:
            prompt_parts.append("\nEXAMPLES:")
            prompt_parts.append(PromptFormatter.format_list_items(examples, "dash"))
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def format_few_shot_examples(
        examples: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output"
    ) -> str:
        """Format few-shot learning examples"""
        formatted = []
        
        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {example.get(input_key, '')}")
            formatted.append(f"Output: {example.get(output_key, '')}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    @staticmethod
    def escape_special_chars(text: str, chars_to_escape: str = "{}") -> str:
        """Escape special characters in text"""
        for char in chars_to_escape:
            text = text.replace(char, f"\\{char}")
        return text
    
    @staticmethod
    def wrap_in_xml_tags(content: str, tag: str) -> str:
        """Wrap content in XML tags"""
        return f"<{tag}>\n{content}\n</{tag}>"
    
    @staticmethod
    def create_rag_prompt(
        question: str,
        context: str,
        instruction: str = "Answer the question based on the context provided.",
        context_label: str = "Context",
        question_label: str = "Question"
    ) -> str:
        """Create a standard RAG prompt"""
        return f"""{instruction}

{context_label}:
{context}

{question_label}: {question}

Answer:"""