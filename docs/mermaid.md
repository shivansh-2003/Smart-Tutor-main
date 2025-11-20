# Mermaid Diagram Module

## Scope
Generates, critiques, and enhances Mermaid diagrams (flowcharts, sequence diagrams, ERDs, etc.) leveraging a multi-stage pipeline with structured Pydantic models. Source files live in `modules/mermaid/`.

## Key Pieces
- `generator.py` (placeholder for orchestrator logic) – expected to stitch together query analysis, RAG, generation, review, and validation.
- `models.py` – comprehensive Pydantic schemas covering every stage:
  - Query analysis (`ExtractedEntities`, `QueryContext`)
  - Intent classification & diagram characteristics
  - Query generation strategies for RAG search
  - Structured information synthesis (components, relationships, processes, data flows, hierarchies)
  - Mermaid generation + enhancement + validation summaries
  - Pipeline progress tracking via `MermaidDiagramPipelineResult`.
- `config.py` – exposes `MermaidModuleConfig` to fetch task-specific LLMs (`mermaid_query_analysis`, `mermaid_synthesis`, etc.).
- `prompts/` – curated instructions for each pipeline step (query analysis, generator, multi-step orchestration, information synthesis, LLM loops).
- `scripts/` – experimental executables (`pipeline_orchastrator.py`, `rag_service.py`, etc.) useful for iterating on the pipeline outside the API layer.

## Typical Flow
1. **Query Analysis** – parse user input into entities/context (DiagramType, QueryType classification).
2. **Intent Classification** – pick diagram type, provide reasoning, fallback options, and structural characteristics.
3. **Query Generation** – craft RAG sub-queries with coverage strategy; run retrieval through shared RAG layer.
4. **Information Synthesis** – convert documents into structured components/relationships/processes; compute coverage metrics and readiness.
5. **Initial Generation** – produce Mermaid code, explanation, and technical notes.
6. **Review & Enhancement** – detect issues, propose improvements, produce enhanced diagram + quality metrics.
7. **Validation** – run syntax/content/quality/completeness checks and final recommendations.

## LLM Routing
Use `MermaidModuleConfig.get_llm_for_task("diagram_generation")`, etc., to ensure complex steps hit `qwen2.5:32b` while lighter validation uses `qwen2.5:3b`.

## Integration Tips
- When hooking into API routes, cache intermediate artifacts (analysis, RAG docs) to speed up refinement loops.
- Most models in `models.py` enforce fields via validators (e.g., unique query types, minimum mermaid code length) to guard against malformed outputs; surface those errors back to the user with actionable hints.
- Align retrieval with `rag/rag_pipeline` to reuse deduplicated and re-ranked sources.

## Future Work
- Flesh out `generator.py` to instantiate every model sequentially, persisting intermediate results for audit trails.
- Provide UI affordances (in Streamlit) to display pipeline step statuses using `pipeline_steps` metadata.

