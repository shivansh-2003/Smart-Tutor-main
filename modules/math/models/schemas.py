from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

class Intent(str, Enum):
    SOLVE_PROBLEM = "solve_problem"
    LEARN_CONCEPT = "learn_concept"
    PRACTICE_MODE = "practice_mode"
    VERIFY_SOLUTION = "verify_solution"

class MathDomain(str, Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    LINEAR_ALGEBRA = "linear_algebra"
    TRIGONOMETRY = "trigonometry"
    ARITHMETIC = "arithmetic"

class DifficultyLevel(str, Enum):
    ELEMENTARY = "elementary"
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"

class ProcessedInput(BaseModel):
    content: str
    input_type: InputType
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IntentClassification(BaseModel):
    primary_intent: Intent
    math_domain: MathDomain
    difficulty_level: DifficultyLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class ProblemAnalysis(BaseModel):
    problem_type: str
    required_concepts: List[str]
    solution_steps: List[str]
    complexity_level: str
    estimated_time: str
    prerequisites: List[str]

class MathSolution(BaseModel):
    step_by_step_solution: List[Dict[str, str]]
    final_answer: str
    alternative_methods: List[str]
    common_mistakes: List[str]
    conceptual_insights: List[str]
    verification: Optional[str] = None

class MathAssistantResponse(BaseModel):
    processed_input: ProcessedInput
    intent_classification: IntentClassification
    problem_analysis: ProblemAnalysis
    solution: MathSolution
    processing_time: float