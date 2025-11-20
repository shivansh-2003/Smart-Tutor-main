"""
LLM-Powered Diagram Generation Prompt Templates
"""

MERMAID_GENERATION_PROMPT = """
You are an expert Mermaid diagram generator. Create a comprehensive {diagram_type} diagram based on the synthesized information.

**INPUT:**
- ORIGINAL QUERY: "{user_query}"
- DIAGRAM TYPE: {diagram_type}
- INTENT CLASSIFICATION: {intent_info}
- SYNTHESIZED INFORMATION: {synthesized_info}

**TASK:**
Generate a well-structured Mermaid {diagram_type} diagram that accurately represents the information provided. Follow Mermaid syntax precisely and create a diagram that is both comprehensive and visually clear.

**DIAGRAM REQUIREMENTS:**

1. **STRUCTURAL ACCURACY**:
   - Include all identified components from synthesized information
   - Represent all relationships and connections accurately
   - Maintain logical flow and hierarchy
   - Ensure diagram completeness based on available information

2. **VISUAL CLARITY**:
   - Use clear, descriptive labels
   - Apply logical grouping and organization
   - Include appropriate styling and formatting
   - Ensure readability and professional appearance

3. **MERMAID SYNTAX COMPLIANCE**:
   - Follow correct {diagram_type} syntax
   - Use proper node types and connection styles
   - Include valid styling and formatting directives
   - Ensure the diagram will render correctly

**{diagram_type} SPECIFIC GUIDELINES:**
{diagram_specific_instructions}

**OUTPUT FORMAT:**
```json
{
  "diagram_analysis": {
    "complexity_assessment": "simple|moderate|complex",
    "components_included": 0,
    "relationships_mapped": 0,
    "diagram_focus": "primary focus of the diagram",
    "coverage_assessment": "comprehensive|partial|basic"
  },
  "mermaid_code": "```mermaid\n{diagram_type}\n// Your complete Mermaid diagram code here\n```",
  "diagram_explanation": {
    "main_purpose": "what this diagram represents",
    "key_components": ["component1", "component2"],
    "key_relationships": ["relationship1", "relationship2"],
    "design_decisions": ["decision1", "decision2"]
  },
  "technical_notes": {
    "syntax_features_used": ["feature1", "feature2"],
    "styling_applied": ["style1", "style2"],
    "limitations": ["limitation1", "limitation2"],
    "enhancement_opportunities": ["opportunity1", "opportunity2"]
  }
}
```

**CRITICAL REQUIREMENTS:**
- Generate valid, syntactically correct Mermaid code
- Ensure all components from synthesized information are represented
- Create logical, easy-to-follow diagram structure
- Use appropriate Mermaid features for the diagram type
- Maintain consistency with the original user intent

Create a production-ready diagram that effectively communicates the intended information.
"""

DIAGRAM_SPECIFIC_INSTRUCTIONS = {
    "flowchart": """
**FLOWCHART SPECIFIC REQUIREMENTS:**
- Use appropriate node shapes: rectangles for processes, diamonds for decisions, circles for start/end
- Implement proper flow direction (TD, LR, or as appropriate)
- Include all decision points with Yes/No branches
- Use descriptive labels for each step
- Group related processes using subgraphs if beneficial
- Apply consistent styling for similar node types

**SYNTAX EXAMPLE:**
```mermaid
flowchart TD
    A[Start Process] --> B{Decision Point}
    B -->|Yes| C[Process Step]
    B -->|No| D[Alternative Step]
    C --> E[End Process]
    D --> E
```
""",

    "sequence": """
**SEQUENCE DIAGRAM SPECIFIC REQUIREMENTS:**
- Include all participants (actors, systems, services)
- Show message flow with proper arrow types
- Include activation/deactivation boxes where relevant
- Add notes for complex interactions
- Use proper message types: ->, ->>, -x, ->>
- Include loop and alternative flows as needed

**SYNTAX EXAMPLE:**
```mermaid
sequenceDiagram
    participant U as User
    participant S as System
    participant D as Database
    
    U->>S: Request Data
    activate S
    S->>D: Query Database
    D-->>S: Return Results
    S-->>U: Send Response
    deactivate S
```
""",

    "class": """
**CLASS DIAGRAM SPECIFIC REQUIREMENTS:**
- Define all classes with proper attributes and methods
- Show inheritance with proper arrow types
- Include composition and aggregation relationships
- Use proper visibility indicators (+, -, #, ~)
- Group related classes using namespaces if beneficial
- Include interface implementations where applicable

**SYNTAX EXAMPLE:**
```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +makeSound()
    }
    class Dog {
        +String breed
        +bark()
    }
    Animal <|-- Dog
```
""",

    "er_diagram": """
**ER DIAGRAM SPECIFIC REQUIREMENTS:**
- Define all entities with proper attributes
- Show relationships with correct cardinality
- Use appropriate relationship types (one-to-one, one-to-many, many-to-many)
- Include primary and foreign key indicators
- Use proper ER notation and symbols
- Group related entities logically

**SYNTAX EXAMPLE:**
```mermaid
erDiagram
    CUSTOMER {
        int customer_id PK
        string name
        string email
    }
    ORDER {
        int order_id PK
        int customer_id FK
        date order_date
    }
    CUSTOMER ||--o{ ORDER : places
```
""",

    "graph": """
**GRAPH DIAGRAM SPECIFIC REQUIREMENTS:**
- Include all nodes with descriptive labels
- Show connections with appropriate edge types
- Use directional arrows where relationships have direction
- Group related nodes using subgraphs if beneficial
- Apply consistent styling for node types
- Include edge labels where relationships need clarification

**SYNTAX EXAMPLE:**
```mermaid
graph TD
    A[Node 1] --> B[Node 2]
    A --> C[Node 3]
    B --> D[Node 4]
    C --> D
```
""",

    "gitgraph": """
**GITGRAPH SPECIFIC REQUIREMENTS:**
- Show proper branching strategy
- Include merge points and commit flows
- Use descriptive branch names
- Show proper git workflow patterns
- Include important commits and milestones
- Represent deployment flows accurately

**SYNTAX EXAMPLE:**
```mermaid
gitgraph
    commit id: "Initial"
    branch develop
    commit id: "Feature 1"
    checkout main
    merge develop
    commit id: "Release"
```
""",

    "c4": """
**C4 DIAGRAM SPECIFIC REQUIREMENTS:**
- Show system boundaries clearly
- Include external systems and users
- Use proper C4 notation and containers
- Show system interactions and dependencies
- Include technology choices where relevant
- Maintain appropriate abstraction level

**SYNTAX EXAMPLE:**
```mermaid
C4Context
    title System Context diagram
    Person(customer, "Customer", "Uses the system")
    System(system, "System", "Provides functionality")
    System_Ext(external, "External System", "External dependency")
    
    Rel(customer, system, "Uses")
    Rel(system, external, "Calls")
```
""",

    "state": """
**STATE DIAGRAM SPECIFIC REQUIREMENTS:**
- Define all states clearly
- Show state transitions with triggers/conditions
- Include initial and final states
- Use proper state notation
- Show concurrent states if applicable
- Include state actions where relevant

**SYNTAX EXAMPLE:**
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : start
    Processing --> Complete : finish
    Processing --> Error : fail
    Error --> Idle : reset
    Complete --> [*]
```
""",

    "journey": """
**USER JOURNEY SPECIFIC REQUIREMENTS:**
- Include all journey steps/phases
- Show user emotions and satisfaction levels
- Include touchpoints and interactions
- Use appropriate journey structure
- Show pain points and opportunities
- Include stakeholder perspectives

**SYNTAX EXAMPLE:**
```mermaid
journey
    title User Journey
    section Discovery
      User Research: 5: User
      Compare Options: 3: User
    section Purchase
      Make Decision: 4: User
      Complete Purchase: 2: User
```
"""
}