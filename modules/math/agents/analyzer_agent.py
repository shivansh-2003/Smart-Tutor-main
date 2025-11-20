import json
from ..models.schemas import ProcessedInput, IntentClassification, ProblemAnalysis
from ..prompts.problem_analyzer import PROBLEM_ANALYSIS_PROMPT
from ..config import MathModuleConfig

class ProblemAnalyzerAgent:
    def __init__(self):
        self.config = MathModuleConfig()
        self.llm = self.config.get_llm_for_agent("problem_analysis")
    
    def analyze_problem(
        self, 
        processed_input: ProcessedInput, 
        intent_classification: IntentClassification
    ) -> ProblemAnalysis:
        """Analyze the mathematical problem and create solution strategy"""
        
        prompt = PROBLEM_ANALYSIS_PROMPT.replace("{problem_content}", processed_input.extracted_text or processed_input.content) \
                                       .replace("{intent}", intent_classification.primary_intent.value) \
                                       .replace("{math_domain}", intent_classification.math_domain.value) \
                                       .replace("{difficulty_level}", intent_classification.difficulty_level.value)
        
        response = self.llm.invoke(prompt)
        
        try:
            # Clean response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            analysis_data = json.loads(content)
            
            return ProblemAnalysis(
                problem_type=analysis_data["problem_type"],
                required_concepts=analysis_data["required_concepts"],
                solution_steps=analysis_data["solution_steps"],
                complexity_level=analysis_data["complexity_level"],
                estimated_time=analysis_data["estimated_time"],
                prerequisites=analysis_data["prerequisites"]
            )
        
        except (json.JSONDecodeError, KeyError) as e:
            # Return default analysis on error
            return ProblemAnalysis(
                problem_type="general_mathematical_problem",
                required_concepts=["basic_mathematics"],
                solution_steps=["analyze_problem", "apply_methodology", "verify_solution"],
                complexity_level="medium",
                estimated_time="5-10 minutes",
                prerequisites=["basic_math_knowledge"]
            )