# Mindmap Module

## Status
The mindmap module (`modules/mindmap/`) is a planned feature for generating hierarchical concept maps and structured summaries. Current files (`mindmap.py`, `config.py`, `prompts/`) act as scaffolding for future development.

## Intended Responsibilities
- **High-Level Flow**
  1. Accept topic prompts or retrieved document snippets.
  2. Classify sub-topics, prerequisites, and relationships (parent/child).
  3. Produce machine-readable structures (JSON/mindmap DSL) and natural language explanations.
  4. Support export to diagramming formats or the Mermaid module for visualization.

- **Configuration (`modules/mindmap/config.py`)**
  - Should mirror other module configs by instantiating `MindmapModuleConfig`.
  - Recommended task mapping:
    - `mindmap_create` → complex model (`qwen2.5:32b`)
    - `mindmap_expand` → moderate complexity
    - `mindmap_validate` → simple model
  - Pull LLM instances through `shared.llm_factory.get_task_llm`.

- **Prompts Directory**
  - Reserve templates for topic decomposition, summarization, and validation loops.
  - Encourage using `PromptFormatter` to keep instructions dynamic (topic name, depth limit, persona).

## Integration Notes
- Router already defines `Intent.MINDMAP` and pattern matches terms like “mind map”, “concept map”, “organize concepts”. Once implementation lands, register the module through `modules/__init__.py`.
- UI placeholder should live in `ui/components/mindmap_interface.py` (to be created) mirroring the existing chat/math components.
- API route stub exists at `api/routes/mindmap.py`; connect it to the future mindmap service entrypoint.

## Next Steps Checklist
1. Define Pydantic schemas describing mindmap nodes, edges, metadata (difficulty tags, prerequisites).
2. Implement retrieval hooks (reuse `rag` or let users upload outlines).
3. Add tests and docs describing available modes (e.g., “overview”, “detailed syllabus”, “flashcards”).
4. Update `Config.modules` (mindmap defaults) once the module is functional.

