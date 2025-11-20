"""
Query Generator Prompt Templates for RAG Search
"""

QUERY_GENERATOR_PROMPT = """
You are an expert query generator for retrieving relevant information from a vector database containing technical documents, architecture guides, and process documentation.

**INPUT:**
- USER QUERY: "{user_query}"
- EXTRACTED ENTITIES: {entities}
- INTENT: {intent}
- DIAGRAM TYPE: {diagram_type}

**TASK:**
Generate 5-7 targeted search queries that will retrieve comprehensive information needed to create a {diagram_type} diagram. Each query should focus on different aspects of the requirements.

**QUERY TYPES TO GENERATE:**

1. **COMPONENT QUERY**: Focus on main components, services, or systems
2. **RELATIONSHIP QUERY**: Focus on connections, dependencies, interactions
3. **PROCESS QUERY**: Focus on workflows, sequences, procedures (if applicable)
4. **TECHNICAL QUERY**: Focus on technologies, frameworks, implementation details
5. **PATTERN QUERY**: Focus on architectural patterns, best practices
6. **CONTEXT QUERY**: Focus on broader system context, environment
7. **DETAIL QUERY**: Focus on specific implementation or configuration details

**GENERATION GUIDELINES:**

- Make queries specific and targeted to retrieve relevant documents
- Use technical terminology from the entities extracted
- Include synonyms and related terms to broaden search
- Ensure queries are complementary, not overlapping
- Focus on information that would be essential for diagram creation
- Consider both high-level and detailed perspectives

**QUERY OPTIMIZATION FOR {diagram_type}:**
{diagram_specific_guidance}

Generate queries in this JSON format:
```json
{
  "queries": [
    {
      "query_type": "component",
      "query_text": "specific search query text",
      "purpose": "what information this query aims to retrieve",
      "keywords": ["key", "terms", "to", "search"]
    },
    {
      "query_type": "relationship",
      "query_text": "specific search query text", 
      "purpose": "what information this query aims to retrieve",
      "keywords": ["key", "terms", "to", "search"]
    },
    // ... more queries
  ],
  "search_strategy": {
    "primary_focus": "main area of information gathering",
    "coverage_areas": ["area1", "area2", "area3"],
    "expected_document_types": ["type1", "type2"]
  }
}
```

Ensure comprehensive coverage while maintaining query specificity.
"""

DIAGRAM_SPECIFIC_GUIDANCE = {
    "flowchart": """
    - Focus on process steps, decision points, and workflow sequences
    - Search for procedures, algorithms, and step-by-step guides
    - Include queries about conditions, branches, and process flows
    - Look for workflow documentation and process descriptions
    """,
    
    "sequence": """
    - Focus on interactions, API calls, and message exchanges
    - Search for request/response patterns and communication flows
    - Include queries about protocols, interfaces, and interaction patterns
    - Look for user journey documentation and interaction sequences
    """,
    
    "class": """
    - Focus on system components, objects, and their relationships
    - Search for class structures, inheritance patterns, and component designs
    - Include queries about system architecture and component interactions
    - Look for design patterns and object-oriented structures
    """,
    
    "er_diagram": """
    - Focus on data entities, relationships, and database schema
    - Search for data models, entity definitions, and database designs
    - Include queries about foreign keys, relationships, and data flow
    - Look for database documentation and data architecture guides
    """,
    
    "graph": """
    - Focus on nodes, connections, and network topologies
    - Search for network structures, relationship mappings, and connections
    - Include queries about graph structures and network architectures
    - Look for topology documentation and relationship mappings
    """,
    
    "gitgraph": """
    - Focus on branching strategies, deployment flows, and version control
    - Search for CI/CD pipelines, deployment processes, and git workflows
    - Include queries about release processes and deployment strategies
    - Look for DevOps documentation and deployment guides
    """,
    
    "c4": """
    - Focus on system context, containers, and high-level architecture
    - Search for system boundaries, external dependencies, and container relationships
    - Include queries about system integration and architectural context
    - Look for system architecture documentation and context diagrams
    """,
    
    "state": """
    - Focus on states, transitions, and lifecycle processes
    - Search for state machines, status workflows, and transition rules
    - Include queries about state management and lifecycle documentation
    - Look for workflow states and transition logic documentation
    """,
    
    "journey": """
    - Focus on user steps, touchpoints, and experience flows
    - Search for user experience documentation and customer journey maps
    - Include queries about user interactions and experience touchpoints
    - Look for UX documentation and user flow descriptions
    """
}