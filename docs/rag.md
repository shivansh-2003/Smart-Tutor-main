# Smart Tutor RAG Layer

## Overview
The `rag/` package implements document indexing and retrieval primitives shared across chat, math, and mermaid workflows.

## Indexing Pipeline (`rag/indexing.py`)
`RAGIndexer` orchestrates:
1. **Pinecone Setup** – creates/attaches to the `smart-tutor` cosine index (Serverless, AWS `us-east-1`) using `PINECONE_API_KEY`.
2. **Embedding Setup** – OpenAI `text-embedding-3-small` (1536-d) as default generator; swap via `RAGConfig`.
3. **Text Splitter** – `RecursiveCharacterTextSplitter` with 1000-char chunks / 200 overlap and multiple separators for stable chunking.
4. **Loaders** – pdf/docx/ppt/txt/md via document loaders; metadata (file name/type/path) added to every document.
5. **Vector Store** – `PineconeVectorStore.from_texts(...)` writes chunk embeddings into a namespace (default `"default"`, pass custom).

### Public Methods
- `index_file(path, namespace="default")` → runs upload → split → embed → store; returns summary dict.
- `index_multiple_files([...])` → convenience loop.
- `search_documents(query, namespace, k)` → direct similarity search.
- `delete_namespace(namespace)` → wipes vectors.

## Retrieval Pipeline (`rag/retrieval.py`)
`RAGRetrieval` layers multiple post-processing stages:
1. **Similarity Search** – `PineconeVectorStore.similarity_search(query, k, namespace)`.
2. **MMR Diversification** – `max_marginal_relevance_search` for broader coverage (`lambda_mult=0.7`).
3. **Re-ranking** – `FlashrankRerank` + `ContextualCompressionRetriever` to keep the top N (`final_k=5`).

Additional utilities:
- `rag_pipeline(query, namespace, search_k, mmr_k, final_k)` returning final `Document` list tagged with `pipeline_rank`.
- `create_retrieval_qa(namespace, model_name)` builds a `RetrievalQA` chain that plugs the custom pipeline into a chat model answerer.
- `answer_question(question, namespace, model_name)` runs the full QA pipeline and returns answer + source docs.

## Configuration & Environment
- Ensure `OPENAI_API_KEY` + `PINECONE_API_KEY` are set.
- Configure index/namespace defaults via `core.config.RAGConfig`.
- Set `LLM_EMBEDDING_MODEL` if moving embeddings to a local workflow; update `RAGIndexer._setup_embeddings` accordingly.

## Usage Example
```python
from rag.indexing import RAGIndexer
from rag.retrieval import RAGRetrieval

indexer = RAGIndexer()
indexer.index_file("/path/to/manual.pdf", namespace="documents")

retriever = RAGRetrieval()
docs = retriever.rag_pipeline("Explain the flow control", namespace="documents")
```

## Extending
- Replace Pinecone with another vector DB by swapping vector store adapter.
- Add document loaders (CSV, HTML, etc.) by extending `upload_and_load_file`.
- Leverage `CacheManager` in modules to store frequently used retrieval outputs keyed by query+namespace.

