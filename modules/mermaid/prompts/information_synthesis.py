"""
Information Synthesis Prompt Templates
"""

INFORMATION_SYNTHESIS_PROMPT = """
You are an expert information synthesizer tasked with analyzing retrieved documents and extracting structured information for Mermaid diagram generation.

**INPUT:**
- USER QUERY: "{user_query}"
- DIAGRAM TYPE: {diagram_type}
- GENERATED QUERIES: {generated_queries}
- RETRIEVED DOCUMENTS: {retrieved_documents}

**TASK:**
Analyze all retrieved documents and extract structured information that directly relates to creating a {diagram_type} diagram. Map the information back to the original generated queries to ensure comprehensive coverage.

**EXTRACTION REQUIREMENTS:**

1. **COMPONENT RELATIONSHIPS**:
   - Identify all components, services, or entities mentioned
   - Map their relationships, dependencies, and interactions
   - Note hierarchy levels and groupings
   - Extract connection types (depends on, calls, contains, inherits, etc.)

2. **DATA FLOWS**:
   - Identify data movement between components
   - Note direction and types of data flow
   - Extract transformation points and processing steps
   - Map input/output relationships

3. **PROCESS SEQUENCES**:
   - Extract step-by-step procedures and workflows
   - Identify decision points and branching logic
   - Note parallel processes and synchronization points
   - Map temporal relationships and dependencies

4. **DEPENDENCIES AND CONNECTIONS**:
   - System dependencies and external integrations
   - Service-to-service connections
   - Infrastructure dependencies
   - Protocol and interface specifications

5. **HIERARCHICAL STRUCTURES**:
   - System layers and abstraction levels
   - Component containment relationships
   - Organizational structures and groupings
   - Inheritance and composition patterns

**SYNTHESIS GUIDELINES:**
- Cross-reference information across documents for consistency
- Resolve conflicts by prioritizing more recent or authoritative sources
- Fill information gaps using logical inference
- Maintain traceability to source documents
- Focus on information relevant to {diagram_type} creation

**OUTPUT FORMAT:**
```json
{
  "synthesis_summary": {
    "total_documents_analyzed": 0,
    "key_themes_identified": ["theme1", "theme2"],
    "information_coverage": "complete|partial|limited",
    "confidence_level": "high|medium|low"
  },
  "structured_information": {
    "components": [
      {
        "name": "component_name",
        "type": "service|database|interface|process",
        "description": "component description",
        "properties": ["prop1", "prop2"],
        "source_query": "originating_query_type"
      }
    ],
    "relationships": [
      {
        "source": "component1",
        "target": "component2", 
        "relationship_type": "depends_on|calls|contains|inherits",
        "description": "relationship description",
        "properties": ["prop1", "prop2"],
        "source_query": "originating_query_type"
      }
    ],
    "processes": [
      {
        "name": "process_name",
        "steps": ["step1", "step2", "step3"],
        "decision_points": ["decision1", "decision2"],
        "parallel_flows": ["flow1", "flow2"],
        "source_query": "originating_query_type"
      }
    ],
    "data_flows": [
      {
        "source": "source_component",
        "target": "target_component",
        "data_type": "data_description",
        "direction": "bidirectional|unidirectional",
        "transformation": "transformation_description",
        "source_query": "originating_query_type"
      }
    ],
    "hierarchies": [
      {
        "parent": "parent_component",
        "children": ["child1", "child2"],
        "hierarchy_type": "contains|inherits|layers",
        "source_query": "originating_query_type"
      }
    ]
  },
  "query_coverage_analysis": {
    "component_query": {
      "coverage": "complete|partial|none",
      "key_findings": ["finding1", "finding2"],
      "missing_information": ["gap1", "gap2"]
    },
    "relationship_query": {
      "coverage": "complete|partial|none", 
      "key_findings": ["finding1", "finding2"],
      "missing_information": ["gap1", "gap2"]
    },
    // ... analysis for each query type
  },
  "diagram_readiness": {
    "structural_completeness": "ready|needs_enhancement|insufficient",
    "relationship_clarity": "clear|partially_clear|unclear",
    "missing_critical_information": ["gap1", "gap2"],
    "recommendations": ["rec1", "rec2"]
  }
}
```

**CRITICAL REQUIREMENTS:**
- Extract ALL relevant information, even if it seems minor
- Maintain clear traceability between findings and source queries
- Identify and flag any missing critical information
- Provide specific, actionable information for diagram generation
- Ensure information is structured appropriately for {diagram_type}

Focus on completeness and accuracy - this information will directly drive diagram generation quality.
"""

CONFLICT_RESOLUTION_PROMPT = """
You have identified conflicting information across documents. Apply these resolution strategies:

**CONFLICT RESOLUTION HIERARCHY:**
1. **Recency**: Prefer more recent documentation
2. **Authority**: Prefer official documentation over informal notes
3. **Specificity**: Prefer specific implementation details over general descriptions
4. **Consistency**: Prefer information that aligns with the broader context
5. **Source Quality**: Prefer information from primary sources

**CONFLICT TYPES:**
- Component naming inconsistencies
- Relationship direction conflicts
- Process step variations
- Technology stack differences
- Architectural pattern discrepancies

For each conflict, document:
- Conflicting sources
- Resolution applied
- Reasoning for choice
- Confidence in resolution
"""