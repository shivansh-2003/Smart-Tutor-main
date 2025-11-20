"""
Simplified Main Interface for LangChain-based Mermaid Diagram Generation
mermaid/scripts/main.py
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.pipeline_orchastrator import MermaidDiagramPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MermaidDiagramAPI:
    """Simplified API interface for LangChain-based Mermaid diagram generation"""
    
    def __init__(self, llm_model: str = "gpt-4", namespace: str = "documents"):
        self.pipeline = MermaidDiagramPipeline(llm_model=llm_model, namespace=namespace)
        logger.info(f"Mermaid Diagram API initialized with model: {llm_model}")
    
    def generate_diagram(self, user_query: str, method: str = "agent") -> Dict[str, Any]:
        """
        Generate Mermaid diagram from user query
        
        Args:
            user_query: The user's request for diagram generation
            method: "agent" for ReAct agent or "chains" for individual chains
            
        Returns:
            Dictionary containing the generated diagram and metadata
        """
        try:
            if method == "agent":
                diagram = self.pipeline.generate_diagram_with_agent(user_query)
                return {
                    "success": True,
                    "method": "agent",
                    "diagram": diagram,
                    "query": user_query
                }
            elif method == "chains":
                result = self.pipeline.generate_diagram_with_chains(user_query)
                return {
                    "success": True,
                    "method": "chains",
                    "result": result,
                    "query": user_query
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method}. Use 'agent' or 'chains'"
                }
                
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": user_query
            }
    
    def get_available_methods(self) -> list:
        """Get list of available generation methods"""
        return ["agent", "chains"]


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Mermaid diagrams using LangChain")
    parser.add_argument("query", help="Query for diagram generation")
    parser.add_argument("--method", choices=["agent", "chains"], default="agent", 
                       help="Generation method to use")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    parser.add_argument("--namespace", default="documents", help="RAG namespace")
    
    args = parser.parse_args()
    
    # Initialize API
    api = MermaidDiagramAPI(llm_model=args.model, namespace=args.namespace)
    
    # Generate diagram
    result = api.generate_diagram(args.query, method=args.method)
    
    if result["success"]:
        print("✅ Diagram generated successfully!")
        print(f"Method: {result['method']}")
        print("\n🎨 Generated Diagram:")
        print("-" * 40)
        if result["method"] == "agent":
            print(result["diagram"])
        else:
            print(result["result"].get("diagram", "No diagram found"))
    else:
        print("❌ Diagram generation failed!")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()