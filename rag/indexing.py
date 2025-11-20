import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGIndexer:
    """Simplified RAG document indexer with file upload and Pinecone storage"""
    
    def __init__(self):
        # Configuration
        self.index_name = "smart-tutor"
        self.dimension = 1536
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Initialize components
        self._setup_pinecone()
        self._setup_embeddings()
        self._setup_text_splitter()
    
    def _setup_pinecone(self):
        """Initialize Pinecone client and index"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create index if it doesn't exist
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def _setup_embeddings(self):
        """Initialize OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("OpenAI embeddings initialized")
    
    def _setup_text_splitter(self):
        """Initialize recursive character text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("Text splitter initialized")
    
    def upload_and_load_file(self, file_path: str) -> List[Document]:
        """
        Upload and load a single file using appropriate LangChain loader
        Supports: PDF, DOCX, PPTX, TXT, MD
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        documents = []
        
        # Select appropriate loader based on file type
        if file_extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_extension == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif file_extension in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(str(file_path))
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load documents
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'source_file': file_path.name,
                'file_type': file_extension[1:],
                'file_path': str(file_path)
            })
        
        logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter"""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        logger.info(f"Split documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Document], namespace: str = "default") -> PineconeVectorStore:
        """Create Pinecone vector store from document chunks"""
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Create vector store
        vector_store = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            index_name=self.index_name,
            namespace=namespace
        )
        
        logger.info(f"Created vector store with {len(texts)} vectors in namespace '{namespace}'")
        return vector_store
    
    def index_file(self, file_path: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Complete indexing pipeline for a single file
        Upload -> Load -> Split -> Embed -> Store in Pinecone
        """
        logger.info(f"Starting indexing pipeline for: {file_path}")
        
        # Step 1: Upload and load file
        documents = self.upload_and_load_file(file_path)
        
        # Step 2: Split into chunks
        chunks = self.split_documents(documents)
        
        # Step 3: Create vector store (embeds and stores in Pinecone)
        vector_store = self.create_vector_store(chunks, namespace)
        
        # Return success result
        result = {
            "success": True,
            "file": Path(file_path).name,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "namespace": namespace,
            "index_name": self.index_name
        }
        
        logger.info(f"Indexing completed successfully: {result}")
        return result
    
    def index_multiple_files(self, file_paths: List[str], namespace: str = "default") -> List[Dict[str, Any]]:
        """Index multiple files"""
        results = []
        for file_path in file_paths:
            result = self.index_file(file_path, namespace)
            results.append(result)
        return results
    
    def get_vector_store(self, namespace: str = "default") -> PineconeVectorStore:
        """Get existing vector store for querying"""
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )
    
    def search_documents(self, query: str, namespace: str = "default", k: int = 5) -> List[Document]:
        """Search documents using similarity search"""
        vector_store = self.get_vector_store(namespace)
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents for query: '{query}'")
        return results
    
    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace"""
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted namespace: {namespace}")
        return True


# Example usage
if __name__ == "__main__":
    # Initialize the indexer
    indexer = RAGIndexer()
    
    # Index a single file
    result = indexer.index_file("/Users/shivanshmahajan/Developer/SmartTutor-main/test/context.pdf", namespace="documents")
    print("Indexing result:", result)
        
    # Search documents
    search_results = indexer.search_documents("What is the main topic of the document?", namespace="documents")
    for doc in search_results:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)