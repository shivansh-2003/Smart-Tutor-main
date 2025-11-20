# Math Module

## Purpose
Provides a multi-agent pipeline for solving mathematics problems, generating hints, and validating answers. Code lives under `modules/math/`.

## Architecture
`math_assistant.MathAssistant` wires together four agents:

| Agent | File | Responsibility |
|-------|------|----------------|
| `InputAgent` | `agents/input_agent.py` | Normalizes raw user text or images (Gemini Flash for OCR) into structured `ProcessedInput`. |
| `IntentAgent` | `agents/intent_agent.py` | Classifies math intent/domain/difficulty so downstream steps can tailor reasoning. |
| `ProblemAnalyzerAgent` | `agents/analyzer_agent.py` | Extracts problem type, required concepts, complexity estimates. |
| `MathSolverAgent` | `agents/solver_agent.py` | Generates step-by-step solutions, alternative methods, pitfalls, and verification. |

Structured responses are defined in `models/schemas.py` (Pydantic models such as `MathAssistantResponse`, `ProcessedInput`, `ProblemAnalysis`, etc.).

## LLM & Task Routing
- Real-time agents can leverage task routing via `modules/math/config.MathModuleConfig`. Each sub-task (`math_input_processing`, `math_solve_problem`, etc.) maps to `LLMConfig` complexity tiers.
- Update agent constructors to call `MathModuleConfig.get_llm_for_agent("input_processing")` for task-based model selection.

## Processing Flow
1. **process_query(content, input_type)** – calls `InputAgent.process_input`; handles text vs image pipeline, including base64 encoding and JSON cleaning.
2. **intent_agent.classify_intent** – returns domain, level, and reasoning metadata used by analyzer/solver.
3. **Problem analysis** – identifies required concepts, estimated time, prerequisites.
4. **Solver** – produces `step_by_step_solution`, `final_answer`, optional alternative methods, conceptual insights, common mistakes, and verification statements.
5. Response envelope includes total processing time for logging/perf dashboards.

## Prompts
Located under `modules/math/prompts/`:
- `input_processor.py`, `intent_classifier.py`, `problem_analyzer.py`, `math_solver.py` supply base prompt templates; edit to adjust persona or rubric.

## CLI & Testing
`math_assistant.py` exposes:
- `main()` – interactive terminal wizard supporting text/image inputs and API key prompts (`GOOGLE_API_KEY`).
- `test_mode()` – automated test using sample equation.

## Integrating With Router
`core.router.Intent.MATH` routes here. Modules should:
```python
from modules.math.math_assistant import MathAssistant
assistant = MathAssistant(openai_api_key=...)
result = assistant.process_query("Solve 2x + 5 = 13", InputType.TEXT)
```

## Extensibility
- Implement caching around analyzer/solver outputs using `shared.CacheManager`.
- Expand `MathModuleConfig.task_models` when adding new agents (e.g., `grading`, `proof_verification`).
- Add safeguard validators via `shared.validators.InputValidator.validate_math_expression`.
- Replace agent constructors with `get_llm_for_agent(...)` for task-based routing.

