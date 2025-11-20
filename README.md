## Smart Tutor

Smart Tutor is a Streamlit-based teaching assistant that combines conversational tutoring, a multi-agent math solver, and diagram generation. An intent router decides which module should handle an input, optionally grounding responses with Retrieval-Augmented Generation (RAG) over user-provided documents.

---

## Features

- **Unified UI**: Single Streamlit app with sidebar navigation and an auto-routing mode.
- **Chat Assistant**: Multi-mode chatbot (`learn`, `hint`, `quiz`, `eli5`, `custom`) with conversation memory and optional RAG context.
- **Math Assistant**: Pipeline of agents (input, intent, analyzer, solver) that produce structured step-by-step solutions and metadata.
- **Diagram Generator**: LangGraph-based orchestrator that generates, enhances, and validates Mermaid diagrams, with optional RAG support.
- **Knowledge Base Uploads**: Upload PDFs/DOCX/PPT/TXT/MD through the sidebar and index them into Pinecone; chats can then use this context.
- **Configurable LLM Stack**: Ollama for task LLMs, Hugging Face embeddings for RAG, all driven by environment variables.
- **Pinecone Vector Store**: Automatic index creation and namespace management for document embeddings.

---

## Project Structure

```text
Smart-tutor/
├── api/                    # FastAPI routes (if used outside Streamlit)
├── core/
│   ├── config.py           # Global configuration (LLMs, RAG, paths, flags)
│   ├── context_manager.py  # Session and conversation context
│   └── router.py           # IntentRouter for CHAT / MATH / MERMAID / MINDMAP
├── modules/
│   ├── chat/               # ChatBot, prompts, chat memory
│   ├── math/               # MathAssistant + agents + prompts + schemas
│   ├── mermaid/            # Diagram models, prompts, LangGraph pipeline
│   └── mindmap/            # (Future) mindmap generation
├── rag/
│   ├── indexing.py         # RAGIndexer: file → chunks → embeddings → Pinecone
│   └── retrieval.py        # RAGRetrieval: search → MMR → rerank → QA
├── shared/                 # LLM factory, validators, prompt utilities
├── ui/
│   ├── app.py              # Main Streamlit app
│   ├── components/
│   │   ├── sidebar.py      # Sidebar, settings, KB uploads
│   │   ├── chat_interface.py
│   │   ├── math_interface.py
│   │   └── mermaid_interface.py
│   └── utils.py            # Session helpers, error display
├── data/
│   ├── uploads/            # Uploaded documents (for RAG indexing)
│   ├── vectorstore/        # Vector store cache (if used)
│   └── cache/              # Any local caches
├── main.py                 # Entry point: `streamlit run main.py`
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+ recommended
- [Ollama](https://ollama.com/) running locally for task LLMs (or adapt `shared/llm_factory.py`)
- Pinecone account and API key for vector storage

---

## Installation

```bash
git clone <your-repo-url> smart-tutor
cd smart-tutor

python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root. Example:

```ini
# -----------------------------------------------------------------------------
# LLM Configuration (Ollama)
# -----------------------------------------------------------------------------

LLM_BASE_URL=http://localhost:11434
LLM_COMPLEX_MODEL=gpt-oss:20b
LLM_COMPLEX_FALLBACK=deepseek-r1:32b
LLM_SIMPLE_MODEL=qwen2.5:3b
LLM_SIMPLE_FALLBACK=gemma2:2b

# -----------------------------------------------------------------------------
# Embeddings (Hugging Face)
# -----------------------------------------------------------------------------

# SentenceTransformers model used for RAG
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Device can be cpu or cuda
HUGGINGFACE_DEVICE=cpu
# Embedding dimension must match the model (384 for all-MiniLM-L6-v2)
HUGGINGFACE_EMBED_DIM=384

# -----------------------------------------------------------------------------
# Vector Database (Pinecone)
# -----------------------------------------------------------------------------

PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=smart-tutor

# -----------------------------------------------------------------------------
# Application Environment
# -----------------------------------------------------------------------------

ENVIRONMENT=development  # or production
```

`core/config.py` reads these environment variables via `get_config()` and exposes:

- LLM model selection (simple vs complex tasks)
- RAG index name and embedding model/device
- Router thresholds and feature flags
- Path configuration for `data/uploads`, `data/vectorstore`, etc.

---

## Running the App

From the project root:

```bash
streamlit run main.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

---

## UI Overview

### Sidebar

- **Modules**
  - **Auto Mode**: Send a message; the `IntentRouter` chooses Chat / Math / Mermaid.
  - **Chat Assistant**: Direct access to the chat module.
  - **Math Assistant**: Direct access to the math solver UI.
  - **Diagram Generator**: Direct access to the Mermaid pipeline UI.

- **Settings**
  - **Chat Mode**: `learn`, `hint`, `quiz`, `eli5`, `custom`.
  - **Use RAG**: Toggle RAG context when chatting.

- **📂 Knowledge Base (RAG)**
  - **Namespace** input (e.g. `documents`, `course-notes`).
  - **File uploader** for `pdf`, `docx`, `ppt/pptx`, `txt`, `md`.
  - **Index Documents** button:
    - Saves files to `data/uploads/`.
    - Runs `RAGIndexer.index_file(path, namespace)` (see `rag/indexing.py`).
    - Embeds with Hugging Face and stores in Pinecone under the given namespace.

- **Session**
  - Message count metric.
  - Clear History / Reset Session buttons (using `core.context_manager`).

- **Debug**
  - Optional debug info: active module, last intent, message count, session id.

### Chat Assistant

Located in `ui/components/chat_interface.py`:

- Displays chat history using `ChatMemory`.
- Uses `ChatBot.chat(...)` with:
  - Selected mode (learn/hint/quiz/eli5/custom).
  - RAG toggle to decide whether to call `rag.retrieval.RAGRetrieval`.
- Custom mode allows user-defined system instructions.

### Math Assistant

Located in `ui/components/math_interface.py`:

- Uses `MathAssistant.process_query(content, input_type)` from `modules/math/math_assistant.py`.
- Displays:
  - Step-by-step solution.
  - Intent classification (intent, domain, difficulty, reasoning).
  - Problem analysis (type, required concepts, complexity, prerequisites).
  - Additional insights (alternative methods, common mistakes, key insights).

Internally, `MathAssistant` orchestrates:

1. `InputAgent` – normalize input (text or image) to `ProcessedInput`.
2. `IntentAgent` – classify intent/domain/difficulty.
3. `ProblemAnalyzerAgent` – infer problem structure and concepts.
4. `MathSolverAgent` – generate solution JSON.

All structures are defined in `modules/math/models/schemas.py`.

### Diagram Generator

Located in `ui/components/mermaid_interface.py`:

- Uses `MermaidDiagramPipeline` from `modules/mermaid/scripts/pipeline_orchastrator.py`.
- User provides a natural language description of a diagram.
- Pipeline stages:
  1. Query analysis and intent classification (diagram type, entities, context).
  2. Query generation for RAG.
  3. RAG search via `RAGService` (which calls `rag.retrieval.RAGRetrieval`).
  4. Information synthesis.
  5. Initial Mermaid code generation.
  6. Enhancement and validation loops.
- UI attempts to render Mermaid:
  - If `streamlit-mermaid` is installed, it shows the diagram directly.
  - Otherwise, shows mermaid code and an optional HTML preview.

---

## RAG Architecture

### Indexing (`rag/indexing.py`)

- Uses `RecursiveCharacterTextSplitter` from `langchain_text_splitters` to split documents (see LangChain splitters docs).
- Supported loaders (`langchain_community.document_loaders`):
  - `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredPowerPointLoader`, `TextLoader`.
- Indexing pipeline:
  1. Load file(s) from disk.
  2. Add basic metadata (source filename, type, path).
  3. Split into text chunks.
  4. Embed chunks with `HuggingFaceEmbeddings`.
  5. Store embeddings in Pinecone via `PineconeVectorStore.from_texts`.

### Retrieval (`rag/retrieval.py`)

- `RAGRetrieval` pipeline:
  1. Similarity search in a namespace.
  2. Optional diversity via max marginal relevance (MMR).
  3. Optional reranking with FlashRank (if available).
  4. Attach pipeline metadata to each document (rank, query, namespace).
- Can also create a RetrievalQA chain (if `RetrievalQA` is available in your LangChain setup) that uses the RAG pipeline as a custom retriever.

The chat module (`ChatBot`) uses `RAGRetrieval.rag_pipeline` to pull context when `use_rag` is enabled.

---

## Configuration (`core/config.py`)

Key configuration sections:

- **LLMConfig**: model URLs, names, and per-task complexity mapping.
- **RAGConfig**: index name, embedding model/device/dimension, chunking parameters, search/MRR/final K, and default namespace.
- **RouterConfig**: confidence thresholds and context usage.
- **ContextConfig**: history length, context window, session timeout.
- **FeatureFlags**: enable/disable modules and RAG features.

Access everything via:

```python
from core.config import get_config
config = get_config()
```

---

## Development Notes

- **Local LLMs**: The system assumes Ollama is available. If you want to use OpenAI, Anthropic, etc., adapt `shared/llm_factory.py` and `core/config.py`.
- **Pinecone**: Ensure your index name and dimensions match: `embedding_dimension` in `RAGConfig` must match the Hugging Face model.
- **Embeddings**: For custom HF models, set:
  - `HUGGINGFACE_EMBEDDING_MODEL`
  - `HUGGINGFACE_EMBED_DIM`
  - `HUGGINGFACE_DEVICE`
- **Optional components**: Some RAG reranking and RetrievalQA utilities are optional; the code logs warnings and falls back to simpler behavior if imports fail.

---

## Contributing

1. Fork and create a feature branch.
2. Install dependencies and run `streamlit run main.py`.
3. Make changes and keep modules decoupled (chat, math, mermaid, rag).
4. Update docs in `docs/` and this `README.md` as needed.
5. Open a PR describing:
   - What you changed
   - How to reproduce / test it

---

## Troubleshooting

- **RAG indexer initialization failed**:
  - Check `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.
  - Ensure `HUGGINGFACE_*` vars and `sentence-transformers` are installed.
  - Verify `HUGGINGFACE_EMBED_DIM` matches your embedding model.

- **ContextualCompressionRetriever / Flashrank warnings**:
  - These are optional enhancements. To enable:
    - Install `flashrank`.
    - Ensure `langchain-community` matches the versions in `requirements.txt`.

- **Diagram pipeline errors**:
  - Make sure LangGraph and LangChain versions match `requirements.txt`.
  - Check terminal logs; they include the current pipeline stage and errors.

If you run into specific stack traces, include the error message and module in your issue/PR so we can iterate quickly.


