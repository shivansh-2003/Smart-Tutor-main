"""
LangChain-based Example Usage and Test Scripts
mermaid/scripts/examples.py
"""

import logging
import json
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.pipeline_orchastrator import MermaidDiagramPipeline
from scripts.main import MermaidDiagramAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_langchain_pipeline():
    """Test the LangChain-based pipeline"""
    try:
        # Initialize LangChain pipeline
        pipeline = MermaidDiagramPipeline(llm_model="gpt-4", namespace="documents")
        
        # Test query
        test_query = "Create a flowchart for user authentication process"
        
        print("🚀 Testing LangChain-based Mermaid Diagram Pipeline")
        print("=" * 60)
        print(f"Query: {test_query}")
        print()
        
        # Method 1: Agent-based generation
        print("🤖 Method 1: ReAct Agent Generation")
        print("-" * 40)
        agent_result = pipeline.generate_diagram_with_agent(test_query)
        print("🎨 Agent-generated Diagram:")
        print("-" * 30)
        print(agent_result)
        
        print("\n" + "=" * 60)
        
        # Method 2: Chain-based generation
        print("🔗 Method 2: Individual Chains Generation")
        print("-" * 40)
        chains_result = pipeline.generate_diagram_with_chains(test_query)
        print("🎨 Chain-generated Result:")
        print("-" * 30)
        print(json.dumps(chains_result, indent=2)[:500] + "...")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")


def test_langchain_chains():
    """Test individual LangChain chains"""
    try:
        pipeline = MermaidDiagramPipeline()
        
        print("🔗 Testing Individual LangChain Chains")
        print("=" * 50)
        
        # Test Query Analysis Chain
        print("1. Query Analysis Chain")
        query_analysis = pipeline.query_analysis_chain.run(
            user_query="Create a system architecture diagram"
        )
        print(f"   Result: {query_analysis[:100]}...")
        
        # Test Intent Classification Chain
        print("2. Intent Classification Chain")
        intent_result = pipeline.intent_classification_chain.run(
            user_query="Create a system architecture diagram",
            entities={"system": "architecture"},
            context={"domain": "software"}
        )
        print(f"   Result: {intent_result[:100]}...")
        
        print("✅ All chains tested successfully!")
        
    except Exception as e:
        logger.error(f"Chain test failed: {e}")
        print(f"❌ Chain test failed: {e}")


def test_langchain_tools():
    """Test LangChain tools"""
    try:
        pipeline = MermaidDiagramPipeline()
        
        print("🛠️ Testing LangChain Tools")
        print("=" * 40)
        
        # Test RAG Search Tool
        print("1. RAG Search Tool")
        rag_result = pipeline.tools[0].func("system architecture")
        print(f"   Result: {rag_result[:100]}...")
        
        # Test Diagram Validation Tool
        print("2. Diagram Validation Tool")
        validation_result = pipeline.tools[1].func("graph TD\nA[Start] --> B[End]")
        print(f"   Result: {validation_result}")
        
        print("✅ All tools tested successfully!")
        
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        print(f"❌ Tool test failed: {e}")


def test_api_interface():
    """Test the simplified API interface"""
    try:
        api = MermaidDiagramAPI()
        
        print("🌐 Testing API Interface")
        print("=" * 40)
        
        # Test agent method
        print("1. Agent Method")
        agent_result = api.generate_diagram("Create a simple flowchart", method="agent")
        print(f"   Success: {agent_result['success']}")
        if agent_result['success']:
            print(f"   Diagram: {agent_result['diagram'][:100]}...")
        
        # Test chains method
        print("2. Chains Method")
        chains_result = api.generate_diagram("Create a simple flowchart", method="chains")
        print(f"   Success: {chains_result['success']}")
        if chains_result['success']:
            print(f"   Result keys: {list(chains_result['result'].keys())}")
        
        print("✅ API interface tested successfully!")
        
    except Exception as e:
        logger.error(f"API test failed: {e}")
        print(f"❌ API test failed: {e}")


def main():
    """Main function to run LangChain tests"""
    print("🎯 LangChain Mermaid Diagram Pipeline Tests")
    print("=" * 60)
    
    # Test individual components
    print("\n1. Testing LangChain Tools...")
    test_langchain_tools()
    
    print("\n2. Testing LangChain Chains...")
    test_langchain_chains()
    
    print("\n3. Testing API Interface...")
    test_api_interface()
    
    print("\n4. Testing Full LangChain Pipeline...")
    test_langchain_pipeline()
    
    print("\n🎉 All LangChain tests completed!")


if __name__ == "__main__":
    main()