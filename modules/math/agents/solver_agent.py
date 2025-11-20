import json
import requests
from ..models.schemas import ProcessedInput, IntentClassification, ProblemAnalysis, MathSolution
from ..prompts.math_solver import MATH_SOLVER_PROMPT
from ..config import MathModuleConfig

class MathSolverAgent:
    def __init__(self):
        self.config = MathModuleConfig()
        self.llm = self.config.get_llm_for_agent("solution_generation")
    
    def solve_problem(
        self, 
        processed_input: ProcessedInput,
        intent_classification: IntentClassification,
        problem_analysis: ProblemAnalysis
    ) -> MathSolution:
        """Solve the mathematical problem with comprehensive explanation"""
        
        # Get additional context from Wikipedia/MathWorld if needed
        external_context = self._get_external_context(
            problem_analysis.required_concepts,
            intent_classification.math_domain.value
        )
        
        prompt = MATH_SOLVER_PROMPT.replace("{problem_content}", processed_input.extracted_text or processed_input.content) \
                                  .replace("{problem_type}", problem_analysis.problem_type) \
                                  .replace("{required_concepts}", ", ".join(problem_analysis.required_concepts)) \
                                  .replace("{solution_steps}", ", ".join(problem_analysis.solution_steps)) \
                                  .replace("{intent}", intent_classification.primary_intent.value)
        
        # Add external context if available
        if external_context:
            prompt += f"\n\nAdditional Context from External Sources:\n{external_context}"
        
        response = self.llm.invoke(prompt)
        
        try:
            # Clean response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            solution_data = json.loads(content)
            
            return MathSolution(
                step_by_step_solution=solution_data["step_by_step_solution"],
                final_answer=solution_data["final_answer"],
                alternative_methods=solution_data["alternative_methods"],
                common_mistakes=solution_data["common_mistakes"],
                conceptual_insights=solution_data["conceptual_insights"],
                verification=solution_data.get("verification")
            )
        
        except (json.JSONDecodeError, KeyError) as e:
            # Return basic solution on error
            return MathSolution(
                step_by_step_solution=[{
                    "step_number": "1",
                    "description": "Analysis in progress",
                    "calculation": response.content[:200],
                    "reasoning": "Processing mathematical solution"
                }],
                final_answer="Solution requires further analysis",
                alternative_methods=["Standard approach"],
                common_mistakes=["Check calculation steps"],
                conceptual_insights=["Mathematical reasoning required"],
                verification="Verify calculations independently"
            )
    
    def _get_external_context(self, concepts: list, domain: str) -> str:
        """Get relevant context from Wikipedia for mathematical concepts"""
        context_parts = []
        
        for concept in concepts[:2]:  # Limit to first 2 concepts
            try:
                # Wikipedia API search
                search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + concept.replace(" ", "_")
                response = requests.get(search_url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'extract' in data:
                        context_parts.append(f"{concept}: {data['extract'][:200]}...")
            
            except:
                continue
        
        return "\n".join(context_parts) if context_parts else ""