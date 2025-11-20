import json
import base64
from typing import Union
from ..models.schemas import ProcessedInput, InputType
from ..prompts.input_processor import IMAGE_PROCESSING_PROMPT
from ..config import MathModuleConfig

class InputAgent:
    def __init__(self):
        self.config = MathModuleConfig()
        self.llm = self.config.get_llm_for_agent("input_processing")
    
    def process_input(self, content: Union[str, bytes], input_type: InputType) -> ProcessedInput:
        """Process text or image input and extract mathematical content"""
        
        if input_type == InputType.TEXT:
            return ProcessedInput(
                content=content,
                input_type=input_type,
                extracted_text=content,
                metadata={"source": "direct_text"}
            )
        
        elif input_type == InputType.IMAGE:
            return self._process_image(content)
    
    def _process_image(self, image_data: bytes) -> ProcessedInput:
        """Process image input using local model"""
        
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create prompt with image description request
        # Note: Local models may not support direct image input
        # This is a text-based workaround
        prompt = f"{IMAGE_PROCESSING_PROMPT}\n\nImage data (base64): {base64_image[:100]}... [truncated]"
        
        response = self.llm.invoke(prompt)
        
        try:
            # Clean response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            extracted_data = json.loads(content)
            
            return ProcessedInput(
                content=extracted_data.get("extracted_content", ""),
                input_type=InputType.IMAGE,
                extracted_text=extracted_data.get("extracted_content", ""),
                metadata={
                    "content_type": extracted_data.get("content_type"),
                    "latex_expressions": extracted_data.get("latex_expressions", []),
                    "detected_elements": extracted_data.get("detected_elements", []),
                    "confidence": extracted_data.get("confidence", 0.0),
                    "notes": extracted_data.get("notes", "")
                }
            )
        
        except json.JSONDecodeError:
            return ProcessedInput(
                content=response.content,
                input_type=InputType.IMAGE,
                extracted_text=response.content,
                metadata={"source": "image", "error": "json_decode_failed"}
            )