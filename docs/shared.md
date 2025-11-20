# Smart Tutor Shared Utilities

The `shared` package provides reusable infrastructure for every module: LLM access, prompt formatting, validation, and caching.

## LLM Factory (`shared/llm_factory.py`)
- `LLMFactory` manages local model clients, caches task-specific `LLM` wrappers, and exposes:
  - `get_task_llm(task_name)` → pre-configured chat client using `core.config.LLMConfig` task map.
  - `get_embeddings(texts)` → local embedding model (or whatever `embedding_model` defines).
  - `reset_factory()` for tests/hot reloads.

### Usage
```python
from shared.llm_factory import get_task_llm
math_solver = get_task_llm("math_solve_problem")
response = math_solver.generate("Solve x^2 - 4 = 0")
```

## Prompt Utilities (`shared/prompt_utils.py`)
- Safe string formatting (`format_template`, `safe_format`), history/context pretty printers, list/dict formatting helpers.
- JSON cleaning for LLM responses and structured system prompt builders, including few-shot composers and RAG prompt templates.

## Validators (`shared/validators.py`)
- `InputValidator` checks plain text, JSON, math expressions, URLs, file paths, enum values, and chatbot modes.
- `OutputValidator` ensures JSON structures, Mermaid syntax, list sizes, math solutions, and sanitizes outputs to remove secrets or enforce length budgets.
- Both return a `ValidationResult` object with `is_valid`, message, and sanitized value for easy gating.

## Cache Layer (`shared/cache.py`)
- Disk-backed namespaces (`rag`, `llm`, `embeddings`, `default`) keyed by hashed identifiers with TTL support.
- Integrates with `core.config.PathConfig` for per-namespace directories and TTL values from `config.cache`.
- Useful helpers: `set`, `get`, `delete`, `exists`, `clear_namespace`, `cleanup_expired`.

## Package Exports (`shared/__init__.py`)
Exports all key helpers so modules can import from `shared` directly:
```python
from shared import get_task_llm, PromptFormatter, InputValidator, CacheManager
```

## Best Practices
- Reuse validators before invoking agents or external services.
- Use `CacheManager(cache_namespace="rag")` to persist expensive retrieval results between requests.
- When adding new task names, update `core.config.LLMConfig.task_complexity_map` to ensure `get_task_llm()` resolves the right model.

