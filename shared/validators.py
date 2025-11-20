"""
Input and output validators for Smart Tutor
"""

import re
import json
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ValidationType(str, Enum):
    TEXT = "text"
    JSON = "json"
    CODE = "code"
    MATH = "math"
    URL = "url"
    FILE_PATH = "file_path"


class ValidationResult:
    """Validation result container"""
    
    def __init__(self, is_valid: bool, message: str = "", sanitized_value: Any = None):
        self.is_valid = is_valid
        self.message = message
        self.sanitized_value = sanitized_value
    
    def __bool__(self):
        return self.is_valid
    
    def __repr__(self):
        return f"ValidationResult(is_valid={self.is_valid}, message='{self.message}')"


class InputValidator:
    """Validators for user input"""
    
    @staticmethod
    def validate_text(
        text: str,
        min_length: int = 1,
        max_length: int = 10000,
        allow_empty: bool = False
    ) -> ValidationResult:
        """Validate text input"""
        if not isinstance(text, str):
            return ValidationResult(False, "Input must be a string")
        
        if not allow_empty and not text.strip():
            return ValidationResult(False, "Input cannot be empty")
        
        if len(text) < min_length:
            return ValidationResult(False, f"Input too short (min: {min_length} chars)")
        
        if len(text) > max_length:
            return ValidationResult(False, f"Input too long (max: {max_length} chars)")
        
        return ValidationResult(True, "Valid text input", text.strip())
    
    @staticmethod
    def validate_json(text: str) -> ValidationResult:
        """Validate JSON string"""
        try:
            parsed = json.loads(text)
            return ValidationResult(True, "Valid JSON", parsed)
        except json.JSONDecodeError as e:
            return ValidationResult(False, f"Invalid JSON: {str(e)}")
    
    @staticmethod
    def validate_math_expression(expression: str) -> ValidationResult:
        """Validate mathematical expression"""
        if not expression or not expression.strip():
            return ValidationResult(False, "Expression cannot be empty")
        
        # Basic validation for common math patterns
        valid_chars = set("0123456789+-*/^()=.,xyzabcdefghijklmnopqrstuvw \t\n")
        
        if not all(c.lower() in valid_chars for c in expression):
            return ValidationResult(False, "Expression contains invalid characters")
        
        # Check balanced parentheses
        if expression.count('(') != expression.count(')'):
            return ValidationResult(False, "Unbalanced parentheses")
        
        return ValidationResult(True, "Valid math expression", expression.strip())
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate URL format"""
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        if url_pattern.match(url):
            return ValidationResult(True, "Valid URL", url)
        else:
            return ValidationResult(False, "Invalid URL format")
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = False) -> ValidationResult:
        """Validate file path"""
        import os
        
        if not path or not path.strip():
            return ValidationResult(False, "Path cannot be empty")
        
        # Check for dangerous path traversal
        if ".." in path or path.startswith("/"):
            return ValidationResult(False, "Path contains invalid characters")
        
        if must_exist and not os.path.exists(path):
            return ValidationResult(False, f"File does not exist: {path}")
        
        return ValidationResult(True, "Valid file path", path)
    
    @staticmethod
    def validate_file_extension(
        filename: str,
        allowed_extensions: List[str]
    ) -> ValidationResult:
        """Validate file extension"""
        import os
        
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in allowed_extensions:
            return ValidationResult(True, f"Valid extension: {ext}", ext)
        else:
            return ValidationResult(
                False,
                f"Invalid extension. Allowed: {', '.join(allowed_extensions)}"
            )
    
    @staticmethod
    def validate_enum_value(value: str, enum_class: type[Enum]) -> ValidationResult:
        """Validate enum value"""
        try:
            enum_value = enum_class(value)
            return ValidationResult(True, "Valid enum value", enum_value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            return ValidationResult(
                False,
                f"Invalid value. Must be one of: {', '.join(valid_values)}"
            )
    
    @staticmethod
    def sanitize_text(
        text: str,
        remove_html: bool = True,
        remove_scripts: bool = True
    ) -> str:
        """Sanitize text input by removing potentially harmful content"""
        if remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        if remove_scripts:
            text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def validate_mode(mode: str, available_modes: List[str]) -> ValidationResult:
        """Validate chatbot mode selection"""
        if mode.lower() in [m.lower() for m in available_modes]:
            return ValidationResult(True, "Valid mode", mode.lower())
        else:
            return ValidationResult(
                False,
                f"Invalid mode. Available: {', '.join(available_modes)}"
            )


class OutputValidator:
    """Validators for system output"""
    
    @staticmethod
    def validate_json_structure(
        data: Dict[str, Any],
        required_keys: List[str]
    ) -> ValidationResult:
        """Validate JSON structure has required keys"""
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return ValidationResult(
                False,
                f"Missing required keys: {', '.join(missing_keys)}"
            )
        
        return ValidationResult(True, "Valid structure")
    
    @staticmethod
    def validate_mermaid_syntax(mermaid_code: str) -> ValidationResult:
        """Basic validation of Mermaid diagram syntax"""
        if not mermaid_code or not mermaid_code.strip():
            return ValidationResult(False, "Mermaid code cannot be empty")
        
        # Check for valid diagram type declarations
        valid_starts = [
            'flowchart', 'graph', 'sequenceDiagram', 'classDiagram',
            'erDiagram', 'stateDiagram', 'journey', 'gantt', 'pie',
            'gitgraph', 'C4Context'
        ]
        
        first_line = mermaid_code.strip().split('\n')[0].strip()
        if not any(first_line.startswith(start) for start in valid_starts):
            return ValidationResult(False, "Invalid Mermaid diagram type")
        
        return ValidationResult(True, "Valid Mermaid syntax", mermaid_code)
    
    @staticmethod
    def validate_response_length(
        response: str,
        min_length: int = 10,
        max_length: int = 5000
    ) -> ValidationResult:
        """Validate response length"""
        length = len(response)
        
        if length < min_length:
            return ValidationResult(False, f"Response too short ({length} chars)")
        
        if length > max_length:
            return ValidationResult(False, f"Response too long ({length} chars)")
        
        return ValidationResult(True, "Valid response length")
    
    @staticmethod
    def validate_list_output(
        data: List[Any],
        min_items: int = 1,
        max_items: Optional[int] = None
    ) -> ValidationResult:
        """Validate list output"""
        if not isinstance(data, list):
            return ValidationResult(False, "Output must be a list")
        
        if len(data) < min_items:
            return ValidationResult(False, f"List too short (min: {min_items} items)")
        
        if max_items and len(data) > max_items:
            return ValidationResult(False, f"List too long (max: {max_items} items)")
        
        return ValidationResult(True, "Valid list output")
    
    @staticmethod
    def validate_math_solution(solution: Dict[str, Any]) -> ValidationResult:
        """Validate math solution structure"""
        required_keys = ['step_by_step_solution', 'final_answer']
        
        result = OutputValidator.validate_json_structure(solution, required_keys)
        if not result.is_valid:
            return result
        
        steps = solution.get('step_by_step_solution', [])
        if not isinstance(steps, list) or len(steps) == 0:
            return ValidationResult(False, "Solution must contain at least one step")
        
        return ValidationResult(True, "Valid math solution structure")
    
    @staticmethod
    def sanitize_output(
        output: str,
        remove_sensitive: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """Sanitize output before returning to user"""
        if remove_sensitive:
            # Remove potential API keys or tokens
            output = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[REDACTED]', output)
        
        if max_length and len(output) > max_length:
            output = output[:max_length] + "...[truncated]"
        
        return output
    
    @staticmethod
    def validate_diagram_type(diagram_type: str) -> ValidationResult:
        """Validate diagram type"""
        valid_types = [
            'flowchart', 'sequence', 'class', 'er_diagram',
            'state', 'graph', 'journey', 'gitgraph', 'c4'
        ]
        
        if diagram_type.lower() in valid_types:
            return ValidationResult(True, "Valid diagram type", diagram_type.lower())
        else:
            return ValidationResult(
                False,
                f"Invalid diagram type. Valid: {', '.join(valid_types)}"
            )