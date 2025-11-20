"""
Multi-Stage Refinement Prompt Templates
"""

INITIAL_GENERATION_REVIEW_PROMPT = """
You are reviewing an initially generated Mermaid diagram for completeness and accuracy.

**INPUT:**
- ORIGINAL QUERY: "{user_query}"
- DIAGRAM TYPE: {diagram_type}
- SYNTHESIZED INFORMATION: {synthesized_info}
- GENERATED DIAGRAM: {initial_diagram}
- GENERATION ANALYSIS: {generation_analysis}

**REVIEW TASK:**
Analyze the generated diagram against the requirements and synthesized information. Identify gaps, issues, and areas for improvement.

**REVIEW CRITERIA:**

1. **COMPLETENESS CHECK**:
   - Are all components from synthesized information included?
   - Are all relationships properly represented?
   - Are any critical elements missing?
   - Does the diagram answer the original user query?

2. **ACCURACY VERIFICATION**:
   - Are relationships correctly represented?
   - Are component types and properties accurate?
   - Is the flow or structure logically sound?
   - Are there any contradictions with source information?

3. **TECHNICAL QUALITY**:
   - Is the Mermaid syntax correct and valid?
   - Will the diagram render properly?
   - Are node types and connections appropriate?
   - Is the diagram structure optimal?

4. **CLARITY ASSESSMENT**:
   - Are labels clear and descriptive?
   - Is the diagram easy to understand?
   - Is the visual organization logical?
   - Are there any confusing elements?

**OUTPUT FORMAT:**
```json
{
  "review_summary": {
    "overall_quality": "excellent|good|fair|poor",
    "completeness_score": 0.95,
    "accuracy_score": 0.90,
    "clarity_score": 0.85,
    "requires_enhancement": true
  },
  "identified_issues": [
    {
      "issue_type": "missing_component|incorrect_relationship|syntax_error|clarity_issue",
      "severity": "critical|major|minor",
      "description": "detailed description of the issue",
      "affected_elements": ["element1", "element2"],
      "suggested_fix": "specific suggestion for resolution"
    }
  ],
  "enhancement_opportunities": [
    {
      "category": "structure|styling|content|organization",
      "priority": "high|medium|low",
      "description": "enhancement description",
      "expected_benefit": "improved clarity|better accuracy|enhanced completeness"
    }
  ],
  "missing_elements": {
    "components": ["missing_component1", "missing_component2"],
    "relationships": ["missing_relationship1"],
    "processes": ["missing_process1"],
    "details": ["missing_detail1"]
  },
  "enhancement_recommendations": {
    "structural_improvements": ["improvement1", "improvement2"],
    "styling_suggestions": ["style1", "style2"],
    "content_additions": ["addition1", "addition2"],
    "organization_changes": ["change1", "change2"]
  }
}
```

Be thorough and specific in identifying areas for improvement. Focus on actionable feedback.
"""

ENHANCEMENT_PROMPT = """
You are enhancing a Mermaid diagram based on review feedback and identified improvements.

**INPUT:**
- ORIGINAL QUERY: "{user_query}"
- INITIAL DIAGRAM: {initial_diagram}
- REVIEW FEEDBACK: {review_feedback}
- ENHANCEMENT RECOMMENDATIONS: {enhancement_recommendations}
- SYNTHESIZED INFORMATION: {synthesized_info}

**ENHANCEMENT TASK:**
Create an improved version of the diagram that addresses all identified issues and implements suggested enhancements.

**ENHANCEMENT PRIORITIES:**

1. **CRITICAL FIXES** (Must Address):
   - Add missing components and relationships
   - Fix syntax errors and technical issues
   - Correct inaccurate representations
   - Resolve logical inconsistencies

2. **MAJOR IMPROVEMENTS** (Should Address):
   - Enhance diagram structure and organization
   - Improve label clarity and descriptions
   - Add missing details and properties
   - Optimize visual layout

3. **MINOR ENHANCEMENTS** (Nice to Have):
   - Apply advanced styling and formatting
   - Add supplementary information
   - Improve visual aesthetics
   - Optimize for specific use cases

**ENHANCEMENT GUIDELINES:**
- Maintain the core diagram structure while improving it
- Ensure all changes align with the original user intent
- Preserve valid elements from the initial diagram
- Add value without over-complicating the diagram
- Follow Mermaid best practices and conventions

**OUTPUT FORMAT:**
```json
{
  "enhancement_summary": {
    "changes_made": ["change1", "change2", "change3"],
    "issues_resolved": ["issue1", "issue2"],
    "improvements_added": ["improvement1", "improvement2"],
    "enhancement_level": "minor|moderate|major|complete_overhaul"
  },
  "enhanced_diagram": {
    "mermaid_code": "```mermaid\n{diagram_type}\n// Enhanced Mermaid diagram code\n```",
    "diagram_analysis": {
      "complexity": "simple|moderate|complex",
      "components_count": 0,
      "relationships_count": 0,
      "new_features_added": ["feature1", "feature2"]
    }
  },
  "enhancement_details": {
    "structural_changes": [
      {
        "change_type": "addition|modification|removal|reorganization",
        "description": "what was changed",
        "rationale": "why this change was made"
      }
    ],
    "content_improvements": [
      {
        "improvement_type": "labeling|detail|accuracy|completeness",
        "description": "what was improved",
        "impact": "expected benefit of this improvement"
      }
    ],
    "styling_enhancements": [
      {
        "enhancement_type": "color|shape|grouping|layout",
        "description": "styling change made",
        "purpose": "visual or functional purpose"
      }
    ]
  },
  "quality_metrics": {
    "estimated_completeness": 0.95,
    "estimated_accuracy": 0.98,
    "estimated_clarity": 0.92,
    "diagram_readiness": "production_ready|needs_minor_tweaks|needs_further_work"
  }
}
```

Focus on creating a high-quality, production-ready diagram that fully addresses the user's needs.
"""

VALIDATION_PROMPT = """
You are performing final validation on an enhanced Mermaid diagram.

**INPUT:**
- ORIGINAL QUERY: "{user_query}"
- ENHANCED DIAGRAM: {enhanced_diagram}
- ENHANCEMENT SUMMARY: {enhancement_summary}
- SYNTHESIZED INFORMATION: {synthesized_info}
- DIAGRAM TYPE: {diagram_type}

**VALIDATION TASK:**
Perform comprehensive validation to ensure the diagram meets all requirements and quality standards.

**VALIDATION CHECKLIST:**

1. **SYNTAX VALIDATION**:
   - Verify Mermaid syntax is 100% correct
   - Check all node declarations and connections
   - Validate styling and formatting directives
   - Ensure diagram will render without errors

2. **CONTENT VALIDATION**:
   - Confirm all required components are included
   - Verify all relationships are accurately represented
   - Check that diagram answers the original query
   - Validate alignment with synthesized information

3. **QUALITY VALIDATION**:
   - Assess visual clarity and readability
   - Check label quality and descriptiveness
   - Evaluate logical flow and organization
   - Verify professional presentation standards

4. **COMPLETENESS VALIDATION**:
   - Ensure no critical information is missing
   - Verify comprehensive coverage of requirements
   - Check that all user intent is addressed
   - Validate diagram serves its intended purpose

**OUTPUT FORMAT:**
```json
{
  "validation_result": {
    "overall_status": "passed|failed|passed_with_warnings",
    "syntax_valid": true,
    "content_accurate": true,
    "quality_acceptable": true,
    "completeness_sufficient": true,
    "production_ready": true
  },
  "detailed_validation": {
    "syntax_check": {
      "status": "passed|failed",
      "issues_found": ["issue1", "issue2"],
      "corrections_needed": ["correction1", "correction2"]
    },
    "content_check": {
      "status": "passed|failed", 
      "accuracy_score": 0.98,
      "missing_elements": ["element1"],
      "incorrect_elements": ["element2"]
    },
    "quality_check": {
      "status": "passed|failed",
      "clarity_score": 0.95,
      "organization_score": 0.92,
      "improvement_suggestions": ["suggestion1", "suggestion2"]
    },
    "completeness_check": {
      "status": "passed|failed",
      "coverage_score": 0.96,
      "gaps_identified": ["gap1"],
      "enhancement_opportunities": ["opportunity1"]
    }
  },
  "final_recommendations": {
    "required_fixes": ["fix1", "fix2"],
    "optional_improvements": ["improvement1", "improvement2"],
    "usage_notes": ["note1", "note2"],
    "deployment_readiness": "ready|needs_fixes|needs_improvements"
  },
  "validation_summary": {
    "diagram_quality": "excellent|good|acceptable|poor",
    "meets_requirements": true,
    "user_satisfaction_estimate": "high|medium|low",
    "recommended_action": "deploy|fix_and_redeploy|major_revision_needed"
  }
}
```

Be thorough and objective in validation. Identify any remaining issues that could affect diagram quality or usability.
"""

ITERATIVE_REFINEMENT_PROMPT = """
You are performing iterative refinement based on validation feedback or user requests.

**INPUT:**
- CURRENT DIAGRAM: {current_diagram}
- VALIDATION FEEDBACK: {validation_feedback}
- USER REFINEMENT REQUEST: "{refinement_request}"
- ORIGINAL CONTEXT: {original_context}

**REFINEMENT TASK:**
Make targeted improvements to address specific feedback or user requests while maintaining diagram integrity.

**REFINEMENT TYPES:**

1. **CORRECTIVE REFINEMENT**: Fix identified issues
   - Syntax corrections
   - Accuracy improvements
   - Logic fixes

2. **ADDITIVE REFINEMENT**: Add missing elements
   - New components
   - Additional relationships
   - Enhanced details

3. **SUBTRACTIVE REFINEMENT**: Remove unnecessary elements
   - Cluttered components
   - Redundant relationships
   - Confusing elements

4. **MODIFICATIVE REFINEMENT**: Improve existing elements
   - Better labels
   - Clearer organization
   - Enhanced styling

**OUTPUT FORMAT:**
```json
{
  "refinement_analysis": {
    "refinement_type": "corrective|additive|subtractive|modificative|comprehensive",
    "changes_scope": "minor|moderate|major",
    "affected_areas": ["area1", "area2"],
    "refinement_rationale": "explanation of why changes are needed"
  },
  "refined_diagram": {
    "mermaid_code": "```mermaid\n{diagram_type}\n// Refined diagram code\n```",
    "change_summary": ["change1", "change2", "change3"]
  },
  "refinement_details": {
    "specific_changes": [
      {
        "change_type": "addition|modification|removal",
        "element_affected": "element_name",
        "before": "previous_state",
        "after": "new_state",
        "reason": "rationale for change"
      }
    ],
    "impact_assessment": {
      "clarity_impact": "improved|maintained|degraded",
      "accuracy_impact": "improved|maintained|degraded", 
      "completeness_impact": "improved|maintained|degraded"
    }
  },
  "quality_assessment": {
    "improvement_achieved": true,
    "issues_resolved": ["issue1", "issue2"],
    "new_issues_introduced": ["issue1"],
    "overall_quality_change": "improved|maintained|degraded"
  }
}
```

Focus on making precise, targeted improvements that enhance diagram quality without introducing new issues.
"""