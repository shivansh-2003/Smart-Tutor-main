"""
Query Analysis & Intent Classification Prompt Templates
"""

QUERY_ANALYSIS_PROMPT = """
You are an expert system analyst tasked with analyzing user queries for diagram generation.

USER QUERY: "{user_query}"

Your task is to extract structured information from this query. Analyze the query and provide:

1. **ENTITY EXTRACTION**:
   - Key Components/Services: Identify main systems, services, components, or entities mentioned
   - Technologies: Programming languages, frameworks, databases, tools, platforms
   - Architectural Patterns: Microservices, monolith, event-driven, layered, etc.
   - Process Steps: Any sequential actions, workflows, or procedures mentioned
   - Relationships: Connections, dependencies, interactions between entities

2. **CONTEXT ANALYSIS**:
   - Domain: (Software Engineering, Business Process, Data Architecture, etc.)
   - Complexity Level: (Simple, Moderate, Complex)
   - Scope: (System-level, Component-level, Process-level)
   - Perspective: (Technical, Business, User-focused)

3. **KEY CONCEPTS**:
   - Primary Focus: What is the main subject of the query?
   - Secondary Elements: Supporting components or concepts
   - Implicit Requirements: What might be needed but not explicitly stated?

Return your analysis in the following JSON format:
```json
{
  "entities": {
    "components": ["component1", "component2"],
    "technologies": ["tech1", "tech2"],
    "patterns": ["pattern1"],
    "process_steps": ["step1", "step2"],
    "relationships": ["relationship1", "relationship2"]
  },
  "context": {
    "domain": "domain_name",
    "complexity": "level",
    "scope": "scope_type",
    "perspective": "perspective_type"
  },
  "key_concepts": {
    "primary_focus": "main_subject",
    "secondary_elements": ["element1", "element2"],
    "implicit_requirements": ["req1", "req2"]
  }
}
```

Be thorough and extract even implied entities and relationships. Think about what someone would need to know to create a comprehensive diagram.
"""

INTENT_CLASSIFICATION_PROMPT = """
You are an expert diagram type classifier. Analyze the user query and determine the most appropriate Mermaid diagram type.

USER QUERY: "{user_query}"
EXTRACTED ENTITIES: {entities}
CONTEXT: {context}

Based on the query content, classify the intent into one of these Mermaid diagram types:

**DIAGRAM TYPES:**

1. **FLOWCHART** - For processes, algorithms, decision trees, workflows
   - Keywords: process, steps, flow, decision, algorithm, procedure
   - Use when: Sequential processes, conditional logic, decision points

2. **SEQUENCE** - For interactions, API calls, user journeys, time-based flows
   - Keywords: interaction, API, request/response, user journey, timeline
   - Use when: Time-based interactions, message passing, user workflows

3. **CLASS** - For software architecture, object relationships, system structure
   - Keywords: classes, objects, inheritance, system architecture, components
   - Use when: Object-oriented design, system components, relationships

4. **ER_DIAGRAM** - For database schemas, entity relationships
   - Keywords: database, entities, tables, relationships, schema, data model
   - Use when: Data modeling, database design, entity relationships

5. **GRAPH** - For networks, concept maps, general relationships
   - Keywords: network, topology, connections, graph, nodes, relationships
   - Use when: Network diagrams, concept mapping, general relationships

6. **GITGRAPH** - For deployment pipelines, branching strategies, version control
   - Keywords: deployment, pipeline, git, branching, CI/CD, version control
   - Use when: Development workflows, deployment processes

7. **C4** - For system architecture context, containers, components
   - Keywords: system context, containers, architecture overview, high-level design
   - Use when: System architecture, context diagrams, container views

8. **STATE** - For state machines, status workflows, lifecycle diagrams
   - Keywords: states, transitions, lifecycle, status, state machine
   - Use when: State transitions, workflow states, lifecycle processes

9. **JOURNEY** - For user experience flows, customer journeys
   - Keywords: user experience, customer journey, user flow, experience mapping
   - Use when: User experience design, customer interactions

**ANALYSIS CRITERIA:**
- Primary purpose of the diagram
- Type of relationships being modeled
- Temporal vs structural representation
- Level of technical detail required

Provide your classification with confidence score and reasoning:

```json
{
  "primary_intent": "diagram_type",
  "confidence": 0.95,
  "reasoning": "Detailed explanation of why this diagram type is most appropriate",
  "alternative_options": [
    {
      "diagram_type": "alternative_type",
      "confidence": 0.75,
      "reasoning": "Why this could also work"
    }
  ],
  "diagram_characteristics": {
    "complexity": "simple|moderate|complex",
    "interactivity": "static|dynamic",
    "hierarchy": "flat|hierarchical|nested"
  }
}
```

Consider the full context and be precise in your classification.
"""