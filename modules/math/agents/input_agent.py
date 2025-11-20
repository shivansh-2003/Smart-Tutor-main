import json
import base64
from typing import Union
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from models.schemas import ProcessedInput, InputType
from prompts.input_processor import IMAGE_PROCESSING_PROMPT
from dotenv import load_dotenv

load_dotenv()

class InputAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            max_output_tokens=1000
        )
    
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
        """Process image input using Gemini Flash"""
        
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create message with image for Gemini
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_PROCESSING_PROMPT},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        )
        
        response = self.llm.invoke([message])
        
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