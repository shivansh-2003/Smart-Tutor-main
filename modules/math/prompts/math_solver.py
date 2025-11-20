MATH_SOLVER_PROMPT = """
You are an expert mathematical reasoning agent. Provide a comprehensive solution to the given mathematical problem with detailed explanations.

Problem: "{problem_content}"
Problem Type: {problem_type}
Required Concepts: {required_concepts}
Solution Steps: {solution_steps}
Intent: {intent}

Solution Requirements:

1. STEP-BY-STEP SOLUTION:
   - Provide clear, logical progression through each step
   - Explain the reasoning behind each operation
   - Show all intermediate calculations
   - Use proper mathematical notation

2. FINAL ANSWER:
   - Clearly state the final answer
   - Include appropriate units if applicable
   - Verify the answer makes sense in context

3. ALTERNATIVE METHODS:
   - Suggest other approaches to solve the problem
   - Explain when different methods might be preferred
   - Compare efficiency of different approaches

4. COMMON MISTAKES:
   - Identify typical errors students make with this problem type
   - Explain how to avoid these mistakes
   - Provide tips for verification

5. CONCEPTUAL INSIGHTS:
   - Explain the underlying mathematical principles
   - Connect to broader mathematical concepts
   - Provide intuitive understanding where possible

External Knowledge Integration:
- Use Wikipedia or MathWorld concepts when referencing advanced topics
- Provide links to further reading for complex concepts
- Include historical context or real-world applications when relevant

Respond in JSON format:
{
    "step_by_step_solution": [
        {
            "step_number": "1",
            "description": "Clear description of what we're doing",
            "calculation": "Mathematical work shown",
            "reasoning": "Why this step is necessary"
        }
    ],
    "final_answer": "Complete final answer with explanation",
    "alternative_methods": ["method1 description", "method2 description"],
    "common_mistakes": ["mistake1 and how to avoid", "mistake2 and how to avoid"],
    "conceptual_insights": ["insight1", "insight2", "insight3"],
    "verification": "How to verify the answer is correct"
}

Ensure mathematical accuracy and pedagogical clarity in all explanations.
"""