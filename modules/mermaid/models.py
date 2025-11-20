"""
Pydantic Models for Mermaid Diagram Generation Pipeline
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


# ================== ENUMS ==================

class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    ER_DIAGRAM = "er_diagram"
    GRAPH = "graph"
    GITGRAPH = "gitgraph"
    C4 = "c4"
    STATE = "state"
    JOURNEY = "journey"


class QueryType(str, Enum):
    COMPONENT = "component"
    RELATIONSHIP = "relationship"
    PROCESS = "process"
    TECHNICAL = "technical"
    PATTERN = "pattern"
    CONTEXT = "context"
    DETAIL = "detail"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class QualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    FAIR = "fair"
    POOR = "poor"


# ================== QUERY ANALYSIS MODELS ==================

class ExtractedEntities(BaseModel):
    """Entities extracted from user query"""
    components: List[str] = Field(default_factory=list, description="Main components/services mentioned")
    technologies: List[str] = Field(default_factory=list, description="Technologies, frameworks, tools")
    patterns: List[str] = Field(default_factory=list, description="Architectural patterns")
    process_steps: List[str] = Field(default_factory=list, description="Sequential actions or workflows")
    relationships: List[str] = Field(default_factory=list, description="Connections and dependencies")


class QueryContext(BaseModel):
    """Context analysis of the query"""
    domain: str = Field(description="Domain area (Software Engineering, Business Process, etc.)")
    complexity: ComplexityLevel = Field(description="Complexity level assessment")
    scope: str = Field(description="System-level, Component-level, Process-level")
    perspective: str = Field(description="Technical, Business, User-focused")


class KeyConcepts(BaseModel):
    """Key concepts identified in the query"""
    primary_focus: str = Field(description="Main subject of the query")
    secondary_elements: List[str] = Field(default_factory=list, description="Supporting components")
    implicit_requirements: List[str] = Field(default_factory=list, description="Implied but not stated needs")


class QueryAnalysisResult(BaseModel):
    """Complete query analysis result"""
    entities: ExtractedEntities
    context: QueryContext
    key_concepts: KeyConcepts


# ================== INTENT CLASSIFICATION MODELS ==================

class AlternativeIntent(BaseModel):
    """Alternative diagram type option"""
    diagram_type: DiagramType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class DiagramCharacteristics(BaseModel):
    """Characteristics of the intended diagram"""
    complexity: ComplexityLevel
    interactivity: Literal["static", "dynamic"]
    hierarchy: Literal["flat", "hierarchical", "nested"]


class IntentClassificationResult(BaseModel):
    """Intent classification result"""
    primary_intent: DiagramType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternative_options: List[AlternativeIntent] = Field(default_factory=list)
    diagram_characteristics: DiagramCharacteristics


# ================== QUERY GENERATOR MODELS ==================

class GeneratedQuery(BaseModel):
    """Individual generated query for RAG search"""
    query_type: QueryType
    query_text: str = Field(min_length=1)
    purpose: str = Field(description="What information this query aims to retrieve")
    keywords: List[str] = Field(default_factory=list, description="Key terms to search")


class SearchStrategy(BaseModel):
    """Overall search strategy"""
    primary_focus: str = Field(description="Main area of information gathering")
    coverage_areas: List[str] = Field(description="Different areas to be covered")
    expected_document_types: List[str] = Field(description="Types of documents expected")


class QueryGeneratorResult(BaseModel):
    """Query generator result"""
    queries: List[GeneratedQuery] = Field(min_items=3, max_items=10)
    search_strategy: SearchStrategy

    @validator('queries')
    def validate_query_types(cls, v):
        query_types = [q.query_type for q in v]
        if len(set(query_types)) < len(query_types) * 0.6:  # At least 60% should be unique
            raise ValueError("Query types should be diverse")
        return v


# ================== RAG SEARCH MODELS ==================

class DocumentSource(BaseModel):
    """Source document metadata"""
    source_file: str
    file_type: str
    file_path: str
    chunk_id: Optional[int] = None
    total_chunks: Optional[int] = None


class RetrievedDocument(BaseModel):
    """Retrieved document from RAG search"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None
    query_type: Optional[QueryType] = None


class RAGSearchResult(BaseModel):
    """Result from RAG search"""
    query: GeneratedQuery
    documents: List[RetrievedDocument]
    total_retrieved: int
    search_success: bool = True
    error_message: Optional[str] = None


# ================== INFORMATION SYNTHESIS MODELS ==================

class Component(BaseModel):
    """Identified component from synthesis"""
    name: str
    type: str = Field(description="service, database, interface, process, etc.")
    description: str
    properties: List[str] = Field(default_factory=list)
    source_query: QueryType


class Relationship(BaseModel):
    """Identified relationship between components"""
    source: str
    target: str
    relationship_type: str = Field(description="depends_on, calls, contains, inherits, etc.")
    description: str
    properties: List[str] = Field(default_factory=list)
    source_query: QueryType


class Process(BaseModel):
    """Identified process or workflow"""
    name: str
    steps: List[str]
    decision_points: List[str] = Field(default_factory=list)
    parallel_flows: List[str] = Field(default_factory=list)
    source_query: QueryType


class DataFlow(BaseModel):
    """Identified data flow between components"""
    source: str
    target: str
    data_type: str
    direction: Literal["bidirectional", "unidirectional"]
    transformation: Optional[str] = None
    source_query: QueryType


class Hierarchy(BaseModel):
    """Identified hierarchical structure"""
    parent: str
    children: List[str]
    hierarchy_type: str = Field(description="contains, inherits, layers, etc.")
    source_query: QueryType


class StructuredInformation(BaseModel):
    """Structured information extracted from documents"""
    components: List[Component] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    processes: List[Process] = Field(default_factory=list)
    data_flows: List[DataFlow] = Field(default_factory=list)
    hierarchies: List[Hierarchy] = Field(default_factory=list)


class QueryCoverage(BaseModel):
    """Coverage analysis for a specific query type"""
    coverage: Literal["complete", "partial", "none"]
    key_findings: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)


class DiagramReadiness(BaseModel):
    """Assessment of diagram readiness"""
    structural_completeness: Literal["ready", "needs_enhancement", "insufficient"]
    relationship_clarity: Literal["clear", "partially_clear", "unclear"]
    missing_critical_information: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SynthesisSummary(BaseModel):
    """Summary of synthesis process"""
    total_documents_analyzed: int = Field(ge=0)
    key_themes_identified: List[str] = Field(default_factory=list)
    information_coverage: Literal["complete", "partial", "limited"]
    confidence_level: ConfidenceLevel


class InformationSynthesisResult(BaseModel):
    """Complete information synthesis result"""
    synthesis_summary: SynthesisSummary
    structured_information: StructuredInformation
    query_coverage_analysis: Dict[QueryType, QueryCoverage]
    diagram_readiness: DiagramReadiness


# ================== LLM GENERATION MODELS ==================

class DiagramAnalysis(BaseModel):
    """Analysis of the generated diagram"""
    complexity_assessment: ComplexityLevel
    components_included: int = Field(ge=0)
    relationships_mapped: int = Field(ge=0)
    diagram_focus: str
    coverage_assessment: Literal["comprehensive", "partial", "basic"]


class DiagramExplanation(BaseModel):
    """Explanation of the diagram"""
    main_purpose: str
    key_components: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list)
    design_decisions: List[str] = Field(default_factory=list)


class TechnicalNotes(BaseModel):
    """Technical notes about the diagram"""
    syntax_features_used: List[str] = Field(default_factory=list)
    styling_applied: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    enhancement_opportunities: List[str] = Field(default_factory=list)


class MermaidGenerationResult(BaseModel):
    """Result from Mermaid diagram generation"""
    diagram_analysis: DiagramAnalysis
    mermaid_code: str = Field(min_length=1, description="Complete Mermaid diagram code")
    diagram_explanation: DiagramExplanation
    technical_notes: TechnicalNotes

    @validator('mermaid_code')
    def validate_mermaid_syntax(cls, v):
        if not v.strip().startswith(('```mermaid', 'flowchart', 'sequenceDiagram', 'classDiagram', 'erDiagram', 'graph', 'gitgraph', 'C4Context', 'stateDiagram', 'journey')):
            raise ValueError("Mermaid code must start with valid diagram declaration")
        return v


# ================== REFINEMENT MODELS ==================

class IdentifiedIssue(BaseModel):
    """Issue identified during review"""
    issue_type: Literal["missing_component", "incorrect_relationship", "syntax_error", "clarity_issue"]
    severity: Literal["critical", "major", "minor"]
    description: str
    affected_elements: List[str] = Field(default_factory=list)
    suggested_fix: str


class EnhancementOpportunity(BaseModel):
    """Enhancement opportunity identified"""
    category: Literal["structure", "styling", "content", "organization"]
    priority: Literal["high", "medium", "low"]
    description: str
    expected_benefit: str


class MissingElements(BaseModel):
    """Missing elements in the diagram"""
    components: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    processes: List[str] = Field(default_factory=list)
    details: List[str] = Field(default_factory=list)


class EnhancementRecommendations(BaseModel):
    """Recommendations for enhancement"""
    structural_improvements: List[str] = Field(default_factory=list)
    styling_suggestions: List[str] = Field(default_factory=list)
    content_additions: List[str] = Field(default_factory=list)
    organization_changes: List[str] = Field(default_factory=list)


class ReviewSummary(BaseModel):
    """Summary of review process"""
    overall_quality: QualityLevel
    completeness_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    requires_enhancement: bool


class InitialGenerationReviewResult(BaseModel):
    """Result from initial generation review"""
    review_summary: ReviewSummary
    identified_issues: List[IdentifiedIssue] = Field(default_factory=list)
    enhancement_opportunities: List[EnhancementOpportunity] = Field(default_factory=list)
    missing_elements: MissingElements
    enhancement_recommendations: EnhancementRecommendations


class StructuralChange(BaseModel):
    """Structural change made during enhancement"""
    change_type: Literal["addition", "modification", "removal", "reorganization"]
    description: str
    rationale: str


class ContentImprovement(BaseModel):
    """Content improvement made"""
    improvement_type: Literal["labeling", "detail", "accuracy", "completeness"]
    description: str
    impact: str


class StylingEnhancement(BaseModel):
    """Styling enhancement applied"""
    enhancement_type: Literal["color", "shape", "grouping", "layout"]
    description: str
    purpose: str


class EnhancementDetails(BaseModel):
    """Details of enhancements made"""
    structural_changes: List[StructuralChange] = Field(default_factory=list)
    content_improvements: List[ContentImprovement] = Field(default_factory=list)
    styling_enhancements: List[StylingEnhancement] = Field(default_factory=list)


class QualityMetrics(BaseModel):
    """Quality metrics for enhanced diagram"""
    estimated_completeness: float = Field(ge=0.0, le=1.0)
    estimated_accuracy: float = Field(ge=0.0, le=1.0)
    estimated_clarity: float = Field(ge=0.0, le=1.0)
    diagram_readiness: Literal["production_ready", "needs_minor_tweaks", "needs_further_work"]


class EnhancedDiagram(BaseModel):
    """Enhanced diagram result"""
    mermaid_code: str = Field(min_length=1)
    diagram_analysis: DiagramAnalysis


class EnhancementSummary(BaseModel):
    """Summary of enhancement process"""
    changes_made: List[str] = Field(default_factory=list)
    issues_resolved: List[str] = Field(default_factory=list)
    improvements_added: List[str] = Field(default_factory=list)
    enhancement_level: Literal["minor", "moderate", "major", "complete_overhaul"]


class DiagramEnhancementResult(BaseModel):
    """Result from diagram enhancement"""
    enhancement_summary: EnhancementSummary
    enhanced_diagram: EnhancedDiagram
    enhancement_details: EnhancementDetails
    quality_metrics: QualityMetrics


# ================== VALIDATION MODELS ==================

class ValidationCheck(BaseModel):
    """Individual validation check result"""
    status: Literal["passed", "failed"]
    issues_found: List[str] = Field(default_factory=list)
    corrections_needed: List[str] = Field(default_factory=list)


class ContentValidation(ValidationCheck):
    """Content validation specific fields"""
    accuracy_score: float = Field(ge=0.0, le=1.0)
    missing_elements: List[str] = Field(default_factory=list)
    incorrect_elements: List[str] = Field(default_factory=list)


class QualityValidation(ValidationCheck):
    """Quality validation specific fields"""
    clarity_score: float = Field(ge=0.0, le=1.0)
    organization_score: float = Field(ge=0.0, le=1.0)
    improvement_suggestions: List[str] = Field(default_factory=list)


class CompletenessValidation(ValidationCheck):
    """Completeness validation specific fields"""
    coverage_score: float = Field(ge=0.0, le=1.0)
    gaps_identified: List[str] = Field(default_factory=list)
    enhancement_opportunities: List[str] = Field(default_factory=list)


class DetailedValidation(BaseModel):
    """Detailed validation results"""
    syntax_check: ValidationCheck
    content_check: ContentValidation
    quality_check: QualityValidation
    completeness_check: CompletenessValidation


class FinalRecommendations(BaseModel):
    """Final recommendations from validation"""
    required_fixes: List[str] = Field(default_factory=list)
    optional_improvements: List[str] = Field(default_factory=list)
    usage_notes: List[str] = Field(default_factory=list)
    deployment_readiness: Literal["ready", "needs_fixes", "needs_improvements"]


class ValidationSummary(BaseModel):
    """Validation summary"""
    diagram_quality: QualityLevel
    meets_requirements: bool
    user_satisfaction_estimate: Literal["high", "medium", "low"]
    recommended_action: Literal["deploy", "fix_and_redeploy", "major_revision_needed"]


class ValidationResult(BaseModel):
    """Validation result"""
    overall_status: Literal["passed", "failed", "passed_with_warnings"]
    syntax_valid: bool
    content_accurate: bool
    quality_acceptable: bool
    completeness_sufficient: bool
    production_ready: bool


class DiagramValidationResult(BaseModel):
    """Complete diagram validation result"""
    validation_result: ValidationResult
    detailed_validation: DetailedValidation
    final_recommendations: FinalRecommendations
    validation_summary: ValidationSummary


# ================== PIPELINE MODELS ==================

class PipelineStep(BaseModel):
    """Individual pipeline step result"""
    step_name: str
    success: bool
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class MermaidDiagramPipelineResult(BaseModel):
    """Complete pipeline result"""
    user_query: str
    diagram_type: DiagramType
    
    # Step results
    query_analysis: QueryAnalysisResult
    intent_classification: IntentClassificationResult
    query_generation: QueryGeneratorResult
    rag_search_results: List[RAGSearchResult]
    information_synthesis: InformationSynthesisResult
    initial_generation: MermaidGenerationResult
    review_result: Optional[InitialGenerationReviewResult] = None
    enhancement_result: Optional[DiagramEnhancementResult] = None
    validation_result: Optional[DiagramValidationResult] = None
    
    # Final output
    final_mermaid_code: str
    final_diagram_quality: QualityLevel
    
    # Pipeline metadata
    pipeline_steps: List[PipelineStep] = Field(default_factory=list)
    total_execution_time: Optional[float] = None
    success: bool = True
    
    @validator('final_mermaid_code')
    def validate_final_code(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Final Mermaid code cannot be empty or too short")
        return v