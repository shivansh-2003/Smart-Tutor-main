# Smart Tutor Core Layer

## Overview
The core package orchestrates global configuration, context, and routing for every Smart Tutor feature.

- `core.config` centralizes environment-driven settings (LLM, RAG, modules, paths, API keys, feature flags) and exposes helpers such as `get_config()` and `Config.get_module_llm_config()`.
- `core.router.IntentRouter` classifies incoming queries into module intents (chat, math, mermaid, mindmap) through a hybrid keyword/LLM strategy.
- `core.context_manager` stores multi-session conversation state, module contexts, message history, and user preferences.

## Configuration Highlights (`core/config.py`)
- **LLMConfig** focuses on local model deployments with task-to-model routing (simple vs complex tasks), fallback models, embedding selection, and timeout rules.
- **RAGConfig** defines Pinecone index info, chunking sizes, and embedding metadata shared across modules.
- **ModuleConfig** keeps per-module defaults plus chat modes. Modules may still override these by instantiating their own configs.
- **PathConfig** auto-creates data/cache directories on startup; CacheManager relies on these.
- **APIConfig** loads keys and warns whenever mandatory ones (OpenAI/Pinecone) are missing.
- `load_config_from_env()` selects development/production presets via `ENVIRONMENT`.

### Task-Based LLM Selection
`LLMConfig.get_model_for_task(task_name)` and `Config.get_module_llm_config(module, task)` feed `shared.llm_factory` so each pipeline can request either small (chat, hints, validation) or large (math proof, RAG synthesis) models transparently.

## Intent Routing (`core/router.py`)
1. **Quick Pattern Pass** – regex/keyword heuristics capture obvious math/mindmap/diagram requests with capped confidence (≤0.85).
2. **LLM Pass** – fallback to a structured JSON prompt via `shared.llm_factory.get_task_llm("intent_classification")`.
3. **RouteResult** aggregates intent, reasoning, metadata, and respects confidence thresholds; helper `route_with_fallback` enforces allowed intents and low-confidence fallbacks.

## Conversation Context (`core/context_manager.py`)
- Tracks messages in a bounded deque (`ContextConfig.max_history`), module state maps, and recency metadata per session.
- Provides helpers to add/filter messages, format histories for prompts, maintain module-specific state, and summarize sessions.
- `ContextManager` handles multi-session lifecycle (create, cleanup expired, export/import).

## Usage Patterns
```python
from core.router import IntentRouter
from core.context_manager import ContextManager

router = IntentRouter()
contexts = ContextManager()
session = contexts.get_context("user-123")

result = router.route("Generate a flowchart for OAuth", context=session.get_context_summary())
session.add_message("user", "Generate a flowchart...", intent=result.intent, module=result.intent.value)
```

## Extensibility Checklist
- Add new intents by extending `Intent` enum, updating `INTENT_PATTERNS`, and documenting module capabilities.
- Register new modules via `modules/__init__.py` registry helpers and map tasks inside `LLMConfig.task_complexity_map`.
- Keep `ContextConfig` and `FeatureFlags` tuned to desired history depth and enabled modules.

## Diagnostics
- Enable debug mode via `ENVIRONMENT=development` to lower router confidence threshold.
- Use `Config.to_dict()` dumps for quick config sanity checks.
- Monitor router logs (warnings on LLM errors, parse fallbacks).

