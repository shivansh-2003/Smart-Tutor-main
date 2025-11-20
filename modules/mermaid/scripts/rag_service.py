"""
RAG Service for Mermaid Diagram Generation Pipeline
mermaid/scripts/rag_service.py
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path to import existing RAG modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.retrieval import RAGRetrieval
from ..models import GeneratedQuery, RAGSearchResult, RetrievedDocument, QueryType

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for document retrieval"""
    
    def __init__(self, namespace: str = "documents"):
        self.rag_retrieval = RAGRetrieval()
        self.namespace = namespace
        logger.info(f"RAG Service initialized with namespace: {namespace}")
    
    def search_documents(self, queries: List[GeneratedQuery], 
                        search_k: int = 20, mmr_k: int = 10, 
                        final_k: int = 5) -> List[RAGSearchResult]:
        """Search documents using generated queries"""
        results = []
        
        for query in queries:
            try:
                logger.info(f"Searching for query: {query.query_text} (type: {query.query_type})")
                
                # Use the complete RAG pipeline from retrieval.py
                documents = self.rag_retrieval.rag_pipeline(
                    query=query.query_text,
                    namespace=self.namespace,
                    search_k=search_k,
                    mmr_k=mmr_k,
                    final_k=final_k
                )
                
                # Convert to RetrievedDocument format
                retrieved_docs = []
                for doc in documents:
                    retrieved_doc = RetrievedDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        relevance_score=doc.metadata.get('pipeline_rank'),
                        query_type=query.query_type
                    )
                    retrieved_docs.append(retrieved_doc)
                
                # Create search result
                search_result = RAGSearchResult(
                    query=query,
                    documents=retrieved_docs,
                    total_retrieved=len(retrieved_docs),
                    search_success=True
                )
                
                results.append(search_result)
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query type: {query.query_type}")
                
            except Exception as e:
                logger.error(f"Search failed for query '{query.query_text}': {e}")
                
                # Create failed search result
                search_result = RAGSearchResult(
                    query=query,
                    documents=[],
                    total_retrieved=0,
                    search_success=False,
                    error_message=str(e)
                )
                results.append(search_result)
        
        return results
    
    def get_all_documents_from_results(self, search_results: List[RAGSearchResult]) -> List[Dict[str, Any]]:
        """Extract all documents from search results for synthesis"""
        all_documents = []
        
        for result in search_results:
            if result.search_success:
                for doc in result.documents:
                    doc_dict = {
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'relevance_score': doc.relevance_score,
                        'query_type': doc.query_type.value if doc.query_type else None,
                        'source_query': result.query.query_text
                    }
                    all_documents.append(doc_dict)
        
        # Sort by relevance score if available
        all_documents.sort(
            key=lambda x: x.get('relevance_score', 999), 
            reverse=False  # Lower rank numbers are better
        )
        
        logger.info(f"Collected {len(all_documents)} total documents for synthesis")
        return all_documents
    
    def get_documents_by_query_type(self, search_results: List[RAGSearchResult]) -> Dict[QueryType, List[Dict[str, Any]]]:
        """Group documents by query type for analysis"""
        grouped_docs = {}
        
        for result in search_results:
            if result.search_success and result.query.query_type:
                if result.query.query_type not in grouped_docs:
                    grouped_docs[result.query.query_type] = []
                
                for doc in result.documents:
                    doc_dict = {
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'relevance_score': doc.relevance_score,
                        'source_query': result.query.query_text
                    }
                    grouped_docs[result.query.query_type].append(doc_dict)
        
        return grouped_docs
    
    def get_search_statistics(self, search_results: List[RAGSearchResult]) -> Dict[str, Any]:
        """Get statistics about search results"""
        total_queries = len(search_results)
        successful_queries = sum(1 for r in search_results if r.search_success)
        total_documents = sum(len(r.documents) for r in search_results if r.search_success)
        
        query_type_stats = {}
        for result in search_results:
            query_type = result.query.query_type.value
            if query_type not in query_type_stats:
                query_type_stats[query_type] = {
                    'attempted': 0,
                    'successful': 0,
                    'documents_retrieved': 0
                }
            
            query_type_stats[query_type]['attempted'] += 1
            if result.search_success:
                query_type_stats[query_type]['successful'] += 1
                query_type_stats[query_type]['documents_retrieved'] += len(result.documents)
        
        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'failed_queries': total_queries - successful_queries,
            'total_documents': total_documents,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'avg_documents_per_query': total_documents / successful_queries if successful_queries > 0 else 0,
            'query_type_breakdown': query_type_stats
        }
    
    def filter_high_quality_documents(self, search_results: List[RAGSearchResult], 
                                    min_relevance_rank: int = 3) -> List[Dict[str, Any]]:
        """Filter documents to only include high-quality results"""
        high_quality_docs = []
        
        for result in search_results:
            if result.search_success:
                for doc in result.documents:
                    # Include document if it has a good relevance rank
                    if (doc.relevance_score is not None and 
                        doc.relevance_score <= min_relevance_rank):
                        doc_dict = {
                            'content': doc.content,
                            'metadata': doc.metadata,
                            'relevance_score': doc.relevance_score,
                            'query_type': doc.query_type.value if doc.query_type else None,
                            'source_query': result.query.query_text
                        }
                        high_quality_docs.append(doc_dict)
        
        logger.info(f"Filtered to {len(high_quality_docs)} high-quality documents")
        return high_quality_docs