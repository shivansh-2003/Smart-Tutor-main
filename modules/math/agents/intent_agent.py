import json
from ..models.schemas import ProcessedInput, IntentClassification, Intent, MathDomain, DifficultyLevel
from ..prompts.intent_classifier import INTENT_CLASSIFICATION_PROMPT
from ..config import MathModuleConfig

class IntentAgent:
    def __init__(self):
        self.config = MathModuleConfig()
        self.llm = self.config.get_llm_for_agent("intent_classification")
    
    def classify_intent(self, processed_input: ProcessedInput) -> IntentClassification:
        """Classify the user's intent and mathematical domain"""
        
        prompt = INTENT_CLASSIFICATION_PROMPT.replace("{user_input}", processed_input.extracted_text or processed_input.content)
        
        response = self.llm.invoke(prompt)
        
        try:
            # Clean response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            classification_data = json.loads(content)
            
            return IntentClassification(
                primary_intent=Intent(classification_data["primary_intent"]),
                math_domain=MathDomain(classification_data["math_domain"]),
                difficulty_level=DifficultyLevel(classification_data["difficulty_level"]),
                confidence=classification_data["confidence"],
                reasoning=classification_data["reasoning"]
            )
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Return default classification on error
            return IntentClassification(
                primary_intent=Intent.SOLVE_PROBLEM,
                math_domain=MathDomain.ALGEBRA,
                difficulty_level=DifficultyLevel.HIGH_SCHOOL,
                confidence=0.5,
                reasoning=f"Error in classification: {str(e)}. Using default values."
            )