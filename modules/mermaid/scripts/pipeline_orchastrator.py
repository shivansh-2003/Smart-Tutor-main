"""
LangGraph-based Pipeline Orchestrator for Mermaid Diagram Generation
mermaid/scripts/pipeline_orchestrator.py - MODIFIED FOR LANGGRAPH
"""

import logging
import json
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports (ensure project src is on sys.path)
sys.path.append(str(Path(__file__).resolve().parents[2]))

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# LangChain imports - UPDATED FOR MODERN PATTERNS
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Local imports
from modules.mermaid.config import MermaidModuleConfig
from ..models import *
from .rag_service import RAGService

# Import existing prompt templates
from ..prompts.query_analysis import QUERY_ANALYSIS_PROMPT, INTENT_CLASSIFICATION_PROMPT
from ..prompts.query_generator import QUERY_GENERATOR_PROMPT, DIAGRAM_SPECIFIC_GUIDANCE
from ..prompts.information_synthesis import INFORMATION_SYNTHESIS_PROMPT
from ..prompts.LLM import MERMAID_GENERATION_PROMPT, DIAGRAM_SPECIFIC_INSTRUCTIONS
from ..prompts.multi_step import (
    INITIAL_GENERATION_REVIEW_PROMPT, 
    ENHANCEMENT_PROMPT, 
    VALIDATION_PROMPT
)

logger = logging.getLogger(__name__)


# LangGraph State Schema
class PipelineState(TypedDict):
    """State schema for the LangGraph pipeline"""
    # Input
    user_query: str
    
    # Stage outputs
    query_analysis: Dict[str, Any]
    intent_classification: Dict[str, Any] 
    generated_queries: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    structured_info: Dict[str, Any]
    initial_diagram: Dict[str, Any]
    enhanced_diagram: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    
    # Final output
    final_mermaid_code: str
    
    # Pipeline metadata
    current_stage: str
    refinement_needed: bool
    validation_needed: bool
    error_occurred: bool
    error_message: Optional[str]
    
    # Messages for debugging/logging
    messages: Annotated[List[BaseMessage], add_messages]


class QueryAnalysisOutputParser(BaseOutputParser[Dict[str, Any]]):
    """Parse query analysis output into structured format"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            import json
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in output")
        except Exception as e:
            logger.error(f"Failed to parse query analysis: {e}")
            return {"entities": {}, "context": {}, "key_concepts": []}


class IntentClassificationOutputParser(BaseOutputParser[Dict[str, Any]]):
    """Parse intent classification output into structured format"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            import json
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in output")
        except Exception as e:
            logger.error(f"Failed to parse intent classification: {e}")
            return {"primary_intent": "flowchart", "confidence": 0.5, "alternative_options": []}


class MermaidDiagramPipeline:
    """LangGraph-based pipeline orchestrator for Mermaid diagram generation"""
    
    def __init__(self, namespace: str = "documents"):
        self.config = MermaidModuleConfig()
        self.rag_service = RAGService(namespace=namespace)
        self.namespace = namespace
        self._llm_cache = {}
        
        # Initialize output parsers (keeping existing)
        self.query_analysis_parser = QueryAnalysisOutputParser()
        self.intent_classification_parser = IntentClassificationOutputParser()
        
        # Create LangChain chains (modified to work with LangGraph)
        self._create_chains()
        
        # Create LangGraph workflow
        self.workflow = self._create_langgraph_workflow()
        
        logger.info(f"LangGraph pipeline initialized with local models, Namespace: {namespace}")
    
    def _get_llm(self, task_name: str):
        """Get LLM for specific task"""
        if task_name not in self._llm_cache:
            self._llm_cache[task_name] = self.config.get_llm_for_task(task_name)
        return self._llm_cache[task_name]
    
    def _create_chains(self):
        """Create modern LangChain runnables using | operator"""
        
        # Create LangChain-compatible LLM wrapper
        try:
            from langchain_ollama import ChatOllama
            from core.config import get_config
            config = get_config()
            # Use a single LLM instance for all chains (can be optimized later)
            self.llm = ChatOllama(
                model=config.llm.complex_model,
                base_url=config.llm.base_url,
                temperature=0
            )
        except ImportError:
            logger.error("langchain_ollama not available. Install with: pip install langchain-ollama")
            raise
        
        def _escape_prompt_keep_vars(template_str: str, keep_vars: List[str]) -> str:
            """Escape all braces in prompt except placeholders in keep_vars"""
            escaped = template_str.replace("{", "{{").replace("}", "}}")
            for var in keep_vars:
                escaped = escaped.replace("{{" + var + "}}", "{" + var + "}")
            return escaped
        
        # Query Analysis Chain - Modern pattern
        query_analysis_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(QUERY_ANALYSIS_PROMPT, ["user_query"])
        )
        self.query_analysis_chain = (
            query_analysis_prompt 
            | self.llm 
            | self.query_analysis_parser
        )
        
        # Intent Classification Chain
        intent_classification_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(INTENT_CLASSIFICATION_PROMPT, ["user_query", "entities", "context"])
        )
        self.intent_classification_chain = (
            intent_classification_prompt
            | self.llm 
            | self.intent_classification_parser
        )
        
        # Query Generation Chain
        query_generation_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(QUERY_GENERATOR_PROMPT, [
                "user_query", "entities", "intent", "diagram_type", "diagram_specific_guidance"
            ])
        )
        self.query_generation_chain = (
            query_generation_prompt
            | self.llm
            | RunnableLambda(self._parse_json_output)
        )
        
        # Information Synthesis Chain
        synthesis_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(INFORMATION_SYNTHESIS_PROMPT, [
                "user_query", "diagram_type", "generated_queries", "retrieved_documents"
            ])
        )
        self.synthesis_chain = (
            synthesis_prompt
            | self.llm
            | RunnableLambda(self._parse_json_output)
        )
        
        # Diagram Generation Chain
        diagram_generation_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(MERMAID_GENERATION_PROMPT, [
                "diagram_type", "user_query", "intent_info", "synthesized_info", "diagram_specific_instructions"
            ])
        )
        self.diagram_generation_chain = (
            diagram_generation_prompt
            | self.llm
            | RunnableLambda(self._parse_json_output)
        )
        
        # Enhancement Chain
        enhancement_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(ENHANCEMENT_PROMPT, [
                "user_query", "initial_diagram", "review_feedback", "enhancement_recommendations", "synthesized_info"
            ])
        )
        self.enhancement_chain = (
            enhancement_prompt
            | self.llm
            | RunnableLambda(self._parse_json_output)
        )
        
        # Validation Chain
        validation_prompt = ChatPromptTemplate.from_template(
            _escape_prompt_keep_vars(VALIDATION_PROMPT, [
                "user_query", "enhanced_diagram", "enhancement_summary", "synthesized_info", "diagram_type"
            ])
        )
        self.validation_chain = (
            validation_prompt
            | self.llm
            | RunnableLambda(self._parse_json_output)
        )
    
    def _compact_rag_results(self, rag_results: List[Dict[str, Any]], max_docs_per_query: int = 3, max_chars: int = 600) -> List[Dict[str, Any]]:
        """Reduce size of RAG results to fit within LLM context limits"""
        compacted: List[Dict[str, Any]] = []
        for result in rag_results or []:
            docs = result.get("documents", [])[:max_docs_per_query]
            compact_docs = []
            for d in docs:
                content = d.get("content", "")
                if len(content) > max_chars:
                    content = content[:max_chars] + "..."
                compact_docs.append({
                    "content": content,
                    "metadata": {k: d.get("metadata", {}).get(k) for k in ["source", "title", "pipeline_rank", "chunk_id"] if k in d.get("metadata", {})},
                    "relevance_score": d.get("relevance_score")
                })
            compacted.append({
                "query": result.get("query", {}),
                "documents": compact_docs,
                "total_retrieved": len(compact_docs),
                "search_success": result.get("search_success", True)
            })
        return compacted
    
    def _create_langgraph_workflow(self) -> StateGraph:
        """Create LangGraph workflow with conditional routing"""
        
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("query_analysis", self._query_analysis_node)
        workflow.add_node("intent_classification", self._intent_classification_node)
        workflow.add_node("query_generation", self._query_generation_node)
        workflow.add_node("rag_search", self._rag_search_node)
        workflow.add_node("information_synthesis", self._information_synthesis_node)
        workflow.add_node("diagram_generation", self._diagram_generation_node)
        workflow.add_node("enhancement", self._enhancement_node)
        workflow.add_node("validation", self._validation_node)
        
        # Set entry point
        workflow.set_entry_point("query_analysis")
        
        # Add edges (linear flow with conditional branches)
        workflow.add_edge("query_analysis", "intent_classification")
        workflow.add_edge("intent_classification", "query_generation")
        workflow.add_edge("query_generation", "rag_search")
        workflow.add_edge("rag_search", "information_synthesis")
        workflow.add_edge("information_synthesis", "diagram_generation")
        workflow.add_edge("diagram_generation", "enhancement")
        
        # Conditional edges for refinement
        workflow.add_conditional_edges(
            "enhancement",
            self._should_validate,
            {
                "validate": "validation",
                "finish": END
            }
        )
        
        workflow.add_conditional_edges(
            "validation",
            self._needs_further_refinement,
            {
                "refine": "enhancement",
                "finish": END
            }
        )
        
        return workflow.compile()
    
    def _parse_json_output(self, ai_message) -> Dict[str, Any]:
        """Helper method to parse JSON output from LLM"""
        try:
            # Extract content from AIMessage
            content = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
            
            # Strip code fences if present
            if content.strip().startswith("```"):
                # Remove starting fence with optional language and ending fence
                content_stripped = content.strip()
                if content_stripped.startswith("```json"):
                    content_stripped = content_stripped[len("```json"):]
                elif content_stripped.startswith("```" ):
                    content_stripped = content_stripped[len("```"):]
                if content_stripped.endswith("```"):
                    content_stripped = content_stripped[:-3]
                content = content_stripped.strip()
            
            # Clean control characters except whitespace controls
            content = "".join(ch for ch in content if ch >= " " or ch in "\n\r\t")
            
            # If Mermaid fenced block exists, extract it
            import re
            mermaid_match = re.search(r"```mermaid\s*([\s\S]*?)```", content, re.IGNORECASE)
            if mermaid_match:
                mermaid_code = mermaid_match.group(1).strip()
                return {"mermaid_code": mermaid_code}
            
            # Find JSON in the content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                # Return fallback structure
                return {"result": content}
        except Exception as e:
            logger.error(f"Failed to parse JSON output: {e}")
            return {"error": str(e), "raw_content": str(ai_message)}
    
    # LangGraph Node Functions
    def _query_analysis_node(self, state: PipelineState) -> PipelineState:
        """Query Analysis Node"""
        try:
            logger.info("🔍 Executing Query Analysis Node")
            
            # Use modern chain with | operator
            analysis_result = self.query_analysis_chain.invoke({
                "user_query": state["user_query"]
            })
            
            return {
                **state,
                "query_analysis": analysis_result,
                "current_stage": "query_analysis_completed"
            }
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"Query analysis failed: {str(e)}"
            }
    
    def _intent_classification_node(self, state: PipelineState) -> PipelineState:
        """Intent Classification Node"""
        try:
            logger.info("🎯 Executing Intent Classification Node")
            
            # Use modern chain with | operator
            intent_result = self.intent_classification_chain.invoke({
                "user_query": state["user_query"],
                "entities": state["query_analysis"].get("entities", {}),
                "context": state["query_analysis"].get("context", {})
            })
            
            return {
                **state,
                "intent_classification": intent_result,
                "current_stage": "intent_classification_completed"
            }
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"Intent classification failed: {str(e)}"
            }
    
    def _query_generation_node(self, state: PipelineState) -> PipelineState:
        """Query Generation Node"""
        try:
            logger.info("🔗 Executing Query Generation Node")
            
            diagram_type = state["intent_classification"].get("primary_intent", "flowchart")
            diagram_specific_guidance = DIAGRAM_SPECIFIC_GUIDANCE.get(diagram_type, "")
            
            # Use modern chain with | operator
            query_gen_result = self.query_generation_chain.invoke({
                "user_query": state["user_query"],
                "entities": state["query_analysis"].get("entities", {}),
                "intent": state["intent_classification"],
                "diagram_type": diagram_type,
                "diagram_specific_guidance": diagram_specific_guidance
            })
            
            # Handle the result
            if isinstance(query_gen_result, dict) and "queries" in query_gen_result:
                queries = query_gen_result["queries"]
            else:
                # Fallback structure
                queries = [{"query_type": "general", "query_text": state["user_query"]}]
            
            return {
                **state,
                "generated_queries": queries,
                "current_stage": "query_generation_completed"
            }
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"Query generation failed: {str(e)}"
            }
    
    def _rag_search_node(self, state: PipelineState) -> PipelineState:
        """RAG Search Node"""
        try:
            logger.info("🔍 Executing RAG Search Node")
            
            # Convert generated queries to GeneratedQuery objects
            generated_queries = []
            for q in state["generated_queries"]:
                if isinstance(q, dict):
                    query_obj = GeneratedQuery(
                        query_type=QueryType(q.get("query_type", "general")),
                        query_text=q.get("query_text", ""),
                        purpose=q.get("purpose", ""),
                        keywords=q.get("keywords", [])
                    )
                    generated_queries.append(query_obj)
            
            # Perform RAG search
            rag_results = self.rag_service.search_documents(generated_queries)
            
            # Convert results back to dict format for state
            rag_results_dict = []
            for result in rag_results:
                result_dict = {
                    "query": {
                        "query_type": result.query.query_type.value,
                        "query_text": result.query.query_text,
                        "purpose": result.query.purpose
                    },
                    "documents": [
                        {
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "relevance_score": doc.relevance_score
                        } for doc in result.documents
                    ],
                    "total_retrieved": result.total_retrieved,
                    "search_success": result.search_success
                }
                rag_results_dict.append(result_dict)
            
            return {
                **state,
                "rag_results": rag_results_dict,
                "current_stage": "rag_search_completed"
            }
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"RAG search failed: {str(e)}"
            }
    
    def _information_synthesis_node(self, state: PipelineState) -> PipelineState:
        """Information Synthesis Node"""
        try:
            logger.info("🧠 Executing Information Synthesis Node")
            
            diagram_type = state["intent_classification"].get("primary_intent", "flowchart")
            compact_results = self._compact_rag_results(state["rag_results"], max_docs_per_query=3, max_chars=600)
            
            # Use modern chain with | operator
            synthesis_result = self.synthesis_chain.invoke({
                "user_query": state["user_query"],
                "diagram_type": diagram_type,
                "generated_queries": state["generated_queries"],
                "retrieved_documents": compact_results
            })
            
            return {
                **state,
                "structured_info": synthesis_result,
                "current_stage": "information_synthesis_completed"
            }
        except Exception as e:
            logger.error(f"Information synthesis failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"Information synthesis failed: {str(e)}"
            }
    
    def _diagram_generation_node(self, state: PipelineState) -> PipelineState:
        """Diagram Generation Node"""
        try:
            logger.info("🎨 Executing Diagram Generation Node")
            
            diagram_type = state["intent_classification"].get("primary_intent", "flowchart")
            diagram_specific_instructions = DIAGRAM_SPECIFIC_INSTRUCTIONS.get(diagram_type, "")
            
            # Use modern chain with | operator
            diagram_result = self.diagram_generation_chain.invoke({
                "user_query": state["user_query"],
                "diagram_type": diagram_type,
                "intent_info": state["intent_classification"],
                "synthesized_info": state["structured_info"],
                "diagram_specific_instructions": diagram_specific_instructions
            })
            
            return {
                **state,
                "initial_diagram": diagram_result,
                "current_stage": "diagram_generation_completed"
            }
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return {
                **state,
                "error_occurred": True,
                "error_message": f"Diagram generation failed: {str(e)}"
            }
    
    def _enhancement_node(self, state: PipelineState) -> PipelineState:
        """Enhancement Node"""
        try:
            logger.info("⚡ Executing Enhancement Node")
            
            # Simple review feedback for enhancement
            review_feedback = {
                "review_summary": {"requires_enhancement": True},
                "enhancement_recommendations": {
                    "structural_improvements": ["Add more detail", "Improve organization"],
                    "styling_suggestions": ["Apply consistent styling", "Add colors"],
                    "content_additions": ["Include missing components"]
                }
            }
            
            # Use modern chain with | operator
            enhanced_result = self.enhancement_chain.invoke({
                "user_query": state["user_query"],
                "initial_diagram": state["initial_diagram"],
                "review_feedback": review_feedback,
                "enhancement_recommendations": review_feedback["enhancement_recommendations"],
                "synthesized_info": state["structured_info"]
            })
            
            # Set final diagram code
            final_code = ""
            if isinstance(enhanced_result, dict) and "enhanced_diagram" in enhanced_result:
                if "mermaid_code" in enhanced_result["enhanced_diagram"]:
                    final_code = enhanced_result["enhanced_diagram"]["mermaid_code"]
            elif isinstance(state["initial_diagram"], dict) and "mermaid_code" in state["initial_diagram"]:
                final_code = state["initial_diagram"]["mermaid_code"]
            
            return {
                **state,
                "enhanced_diagram": enhanced_result,
                "final_mermaid_code": final_code,
                "current_stage": "enhancement_completed"
            }
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            # Fallback to initial diagram
            final_code = state["initial_diagram"].get("mermaid_code", "") if isinstance(state["initial_diagram"], dict) else ""
            return {
                **state,
                "enhanced_diagram": state["initial_diagram"],
                "final_mermaid_code": final_code,
                "current_stage": "enhancement_failed_fallback"
            }
    
    def _validation_node(self, state: PipelineState) -> PipelineState:
        """Validation Node"""
        try:
            logger.info("✅ Executing Validation Node")
            
            diagram_type = state["intent_classification"].get("primary_intent", "flowchart")
            
            # Use modern chain with | operator
            validation_result = self.validation_chain.invoke({
                "user_query": state["user_query"],
                "enhanced_diagram": state["enhanced_diagram"],
                "enhancement_summary": state["enhanced_diagram"].get("enhancement_summary", {}) if isinstance(state["enhanced_diagram"], dict) else {},
                "synthesized_info": state["structured_info"],
                "diagram_type": diagram_type
            })
            
            return {
                **state,
                "validation_result": validation_result,
                "current_stage": "validation_completed"
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                **state,
                "validation_result": {"validation_result": {"overall_status": "passed"}},
                "current_stage": "validation_failed_fallback"
            }
    
    # Conditional Edge Functions
    def _should_validate(self, state: PipelineState) -> str:
        """Determine if validation is needed"""
        try:
            # Check quality metrics from enhancement
            quality_metrics = state.get("enhanced_diagram", {}).get("quality_metrics", {})
            diagram_readiness = quality_metrics.get("diagram_readiness", "production_ready")
            
            if diagram_readiness == "needs_further_work":
                return "validate"
            else:
                return "finish"
        except:
            return "finish"
    
    def _needs_further_refinement(self, state: PipelineState) -> str:
        """Determine if further refinement is needed after validation"""
        try:
            validation_result = state.get("validation_result", {})
            recommended_action = validation_result.get("validation_summary", {}).get("recommended_action", "deploy")
            
            if recommended_action == "major_revision_needed":
                return "refine"
            else:
                return "finish"
        except:
            return "finish"
    
    # Public Methods
    def generate_diagram(self, user_query: str) -> Dict[str, Any]:
        """Generate Mermaid diagram using LangGraph workflow"""
        try:
            logger.info(f"🚀 Starting LangGraph pipeline for query: {user_query}")
            
            # Initialize state
            initial_state = PipelineState(
                user_query=user_query,
                query_analysis={},
                intent_classification={},
                generated_queries=[],
                rag_results=[],
                structured_info={},
                initial_diagram={},
                enhanced_diagram=None,
                validation_result=None,
                final_mermaid_code="",
                current_stage="initialized",
                refinement_needed=False,
                validation_needed=False,
                error_occurred=False,
                error_message=None,
                messages=[]
            )
            
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            logger.info(f"✅ LangGraph pipeline completed. Final stage: {final_state.get('current_stage')}")
            
            return {
                "success": not final_state.get("error_occurred", False),
                "final_diagram": final_state.get("final_mermaid_code", ""),
                "pipeline_state": final_state,
                "query": user_query
            }
            
        except Exception as e:
            logger.error(f"LangGraph pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": user_query
            }
    
    def generate_diagram_with_agent(self, user_query: str) -> str:
        """Legacy method for compatibility"""
        result = self.generate_diagram(user_query)
        return result.get("final_diagram", f"Error: {result.get('error', 'Unknown error')}")
    
    def generate_diagram_with_chains(self, user_query: str) -> Dict[str, Any]:
        """Legacy method for compatibility"""
        result = self.generate_diagram(user_query)
        if result["success"]:
            return result["pipeline_state"]
        else:
            return {"error": result["error"]}


# Simple usage example
if __name__ == "__main__":
    # Initialize LangGraph pipeline
    pipeline = MermaidDiagramPipeline()
    
    # Test query
    test_query = "please create a flow for a rag pipeline"
    
    print("🚀 Testing LangGraph Mermaid Pipeline")
    print("=" * 50)
    print(f"Query: {test_query}")
    print()
    
    # Generate diagram
    result = pipeline.generate_diagram(test_query)
    
    if result["success"]:
        print("✅ Diagram generated successfully!")
        print("\n🎨 Generated Diagram:")
        print("-" * 30)
        print(result["final_diagram"])
        print(f"\nPipeline completed at stage: {result['pipeline_state']['current_stage']}")
    else:
        print("❌ Diagram generation failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")