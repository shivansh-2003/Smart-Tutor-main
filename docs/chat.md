# Chat Module

## Overview
The chat module (`modules/chat/`) handles conversational tutoring, multi-mode chat (learn/hint/quiz/etc.), and RAG-enabled assistance. While `chatbot.py` currently serves as a placeholder, the surrounding infrastructure defines how chat sessions should behave.

## Components
- `config.py` – `ChatModuleConfig` maps chat modes to task names:
  - Casual / greetings / follow-ups → `chat_casual` (small model tier).
  - RAG-powered tutoring → `chat_rag_synthesis` (complex model tier).
- `memory.py` – intended to manage chat-specific memory (short-term notes, user preferences) atop `core.context_manager`.
- `prompts/` – reserve system prompts per mode (learn, hint, quiz, etc.) to keep instructions reusable.
- `modules/__init__.py` – use `get_chat_module()` and module factory helpers to register the chat implementation once available.

## Expected Flow
1. API layer (`api/routes/chat.py`) receives a user query and loads conversation context via `core.context_manager`.
2. Router decides if the request stays in chat or should hop to math/mermaid.
3. Chat service selects prompt + LLM using `ChatModuleConfig.get_llm_for_mode(mode)`.
4. Optional RAG augmentation: call `rag.rag_pipeline` and inject retrieved snippets into the final prompt.
5. Persist assistant reply through `ContextManager.add_message(...)` and update any chat memory structures.

## RAG Modes
- For document-grounded conversations, store uploaded files under `data/uploads/` and index via `rag/RAGIndexer`.
- Use `chat_rag_synthesis` task name so the factory assigns a large reasoning model (`qwen2.5:32b` by default).

## UI Integration
- `ui/components/chat_interface.py` renders the Streamlit-based chat window.
- Streamlit app (`ui/app.py`) should read `Config.modules.chat_modes` to populate mode selectors, dispatching to the backend route accordingly.

## Roadmap Ideas
- Build chain-of-thought or tool-use memory: `memory.py` can track citations, hints delivered, and follow-up reminders.
- Add guardrails using `shared.validators.InputValidator.validate_text` to reject empty or oversized messages.
- Implement conversation analytics (turn counts, mode switches) via `ConversationContext.get_context_summary()`.

