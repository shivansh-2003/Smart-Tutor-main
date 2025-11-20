"""
Simplified RAG Retrieval System
Pipeline: Search -> MMR -> Re-ranking -> RetrievalQA
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.example_selectors import  MaxMarginalRelevanceExampleSelector

# Pinecone imports
from pinecone import Pinecone

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetrieval:
    """Simplified RAG retrieval with search, MMR, re-ranking, and QA"""
    
    def __init__(self):
        # Configuration
        self.index_name = "smart-tutor"
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize components
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_reranker()
    
    def _setup_embeddings(self):
        """Initialize OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("Embeddings initialized")
    
    def _setup_vector_store(self):
        """Initialize Pinecone vector store"""
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.vector_store = PineconeVectorStore(
            index=pc.Index(self.index_name),
            embedding=self.embeddings
        )
        logger.info("Vector store initialized")
    
    def _setup_reranker(self):
        """Initialize FlashRank reranker"""
        self.reranker = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2"
        )
        logger.info("Reranker initialized")
    
    def search_documents(
        self, 
        query: str, 
        namespace: str = "documents", 
        k: int = 20
    ) -> List[Document]:
        """Step 1: Basic similarity search"""
        try:
            # Use the namespace parameter directly in similarity_search
            docs = self.vector_store.similarity_search(
                query=query, 
                k=k, 
                namespace=namespace  # Pass namespace directly here
            )
            logger.info(f"Found {len(docs)} documents from similarity search in namespace '{namespace}'")
            return docs
        except Exception as e:
            logger.error(f"Search failed in namespace '{namespace}': {e}")
            return []
    
    def apply_mmr(
        self, 
        query: str, 
        documents: List[Document], 
        k: int = 10,
        lambda_mult: float = 0.7,
        namespace: str = "documents"
    ) -> List[Document]:
        """Simplified MMR for debugging"""
        if len(documents) <= k:
            return documents
        
        try:
            # Use namespace parameter directly in max_marginal_relevance_search
            mmr_docs = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=len(documents),
                lambda_mult=lambda_mult,
                namespace=namespace  # Pass namespace directly
            )
            logger.info(f"MMR selected {len(mmr_docs)} diverse documents")
            return mmr_docs
        except Exception as e:
            logger.warning(f"MMR failed, using top-k selection: {e}")
            return documents[:k]
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        k: int = 5
    ) -> List[Document]:
        """Step 3: Re-rank documents using FlashRank"""
        if not self.reranker or not documents:
            return documents[:k]
        
        # Create a simple retriever for the documents
        class SimpleRetriever(BaseRetriever):
            def __init__(self, docs):
                super().__init__()
                self._docs = docs
            
            def _get_relevant_documents(self, query, *, run_manager=None):
                return self._docs
        
        # Create compression retriever with reranker
        base_retriever = SimpleRetriever(documents)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=base_retriever
        )
        
        # Get reranked documents
        reranked_docs = compression_retriever.get_relevant_documents(query)
        logger.info(f"Reranked to {len(reranked_docs[:k])} top documents")
        return reranked_docs[:k]
    
    def rag_pipeline(
        self, 
        query: str, 
        namespace: str = "documents",
        search_k: int = 20,
        mmr_k: int = 10, 
        final_k: int = 5
    ) -> List[Document]:
        """
        Complete RAG pipeline: Search -> MMR -> Re-ranking
        """
        logger.info(f"Starting RAG pipeline for query: '{query}' in namespace: '{namespace}'")
        
        # Step 1: Search documents
        search_docs = self.search_documents(query, namespace, search_k)
        logger.info(f"Search returned {len(search_docs)} documents")
        
        if not search_docs:
            logger.warning(f"No documents found for query: '{query}' in namespace: '{namespace}'")
            logger.info("Tip: Check if documents are indexed in the correct namespace")
            return []
        
        # Debug: Print first document
        if search_docs:
            logger.info(f"First document preview: {search_docs[0].page_content[:100]}...")
        
        # Step 2: Apply MMR for diversity (pass namespace)
        mmr_docs = self.apply_mmr(query, search_docs, mmr_k, namespace=namespace)
        logger.info(f"MMR returned {len(mmr_docs)} documents")
        
        # Step 3: Re-rank documents
        final_docs = self.rerank_documents(query, mmr_docs, final_k)
        logger.info(f"Re-ranking returned {len(final_docs)} documents")
        
        # Add pipeline metadata
        for i, doc in enumerate(final_docs):
            doc.metadata.update({
                'pipeline_rank': i + 1,
                'query': query,
                'namespace': namespace
            })
        
        logger.info(f"RAG pipeline completed with {len(final_docs)} final documents")
        return final_docs
    
    def create_retrieval_qa(
        self, 
        namespace: str = "documents",
        model_name: str = "gpt-3.5-turbo",
        search_k: int = 10
    ) -> RetrievalQA:
        """Create RetrievalQA chain with custom retriever"""
        # Set up LLM
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create custom retriever that uses our RAG pipeline
        class CustomRAGRetriever(BaseRetriever):
            def __init__(self, rag_system, namespace, k):
                super().__init__()
                self._rag_system = rag_system
                self._namespace = namespace
                self._k = k
            
            def _get_relevant_documents(self, query, *, run_manager=None):
                return self._rag_system.rag_pipeline(
                    query=query,
                    namespace=self._namespace,
                    final_k=self._k
                )
        
        # Create custom retriever
        custom_retriever = CustomRAGRetriever(self, namespace, search_k)
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=custom_retriever,
            return_source_documents=True,
            verbose=True
        )
        
        logger.info(f"Created RetrievalQA chain with {model_name}")
        return qa_chain
    
    def answer_question(
        self, 
        question: str, 
        namespace: str = "documents",
        model_name: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """
        Complete Q&A pipeline: RAG retrieval + LLM generation
        """
        # Create QA chain
        qa_chain = self.create_retrieval_qa(namespace, model_name)
        
        # Get answer
        result = qa_chain({"query": question})
        
        # Format response
        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "num_sources": len(result["source_documents"]),
            "namespace": namespace
        }
        
        logger.info(f"Generated answer using {response['num_sources']} sources")
        return response
    
    def get_vector_store(self, namespace: str = "documents") -> PineconeVectorStore:
        """Get vector store with specific namespace"""
        # Note: Namespace is passed as parameter to search methods, not set as attribute
        return self.vector_store


# Example usage
if __name__ == "__main__":
    # Initialize RAG retrieval system
    rag = RAGRetrieval()
    
    # Test the complete pipeline
    test_query = "please tell me about RAG"
    
    print("=== Testing RAG Pipeline ===")
    
    # Test individual pipeline steps
    docs = rag.rag_pipeline(
        query=test_query,
        namespace="documents",
        search_k=20,
        mmr_k=10,
        final_k=5
    )
    
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source_file', 'Unknown')}")
        print("-" * 50)
    
    print("\n=== Testing Q&A Pipeline ===")
    
    # Test complete Q&A
    response = rag.answer_question(
        question=test_query,
        namespace="documents",
        model_name="gpt-3.5-turbo"
    )
    
    print(f"Question: {response['question']}")
    print(f"Answer: {response['answer']}")
    print(f"Sources used: {response['num_sources']}")
    
    # Show source documents
    print("\nSource Documents:")
    for i, doc in enumerate(response['source_documents'], 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
        print("-" * 50)