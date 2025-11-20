PROBLEM_ANALYSIS_PROMPT = """
You are an expert mathematical problem analyzer. Break down the given mathematical problem into structured components for solution planning.

Mathematical Problem: "{problem_content}"
Intent: {intent}
Domain: {math_domain}
Difficulty: {difficulty_level}

Analysis Tasks:

1. PROBLEM TYPE IDENTIFICATION:
   - Classify the specific type of mathematical problem
   - Identify the main mathematical operations required
   - Note any special characteristics or constraints

2. REQUIRED CONCEPTS:
   - List fundamental concepts needed to solve this problem
   - Identify prerequisite knowledge
   - Note any advanced techniques required

3. SOLUTION STRATEGY:
   - Break down the problem into logical steps
   - Identify the most efficient solution approach
   - Note alternative solution methods if applicable

4. COMPLEXITY ASSESSMENT:
   - Evaluate computational complexity
   - Estimate time required for solution
   - Identify potential difficulty points

Respond in JSON format:
{
    "problem_type": "specific classification of the problem",
    "required_concepts": ["concept1", "concept2", "concept3"],
    "solution_steps": ["step1", "step2", "step3"],
    "complexity_level": "low|medium|high",
    "estimated_time": "time estimate for solution",
    "prerequisites": ["prerequisite1", "prerequisite2"]
}

Focus on creating a clear roadmap for systematic problem solving.
"""