import time
import json
from typing import Union
from .agents.input_agent import InputAgent
from .agents.intent_agent import IntentAgent
from .agents.analyzer_agent import ProblemAnalyzerAgent
from .agents.solver_agent import MathSolverAgent
from .models.schemas import InputType, MathAssistantResponse

class MathAssistant:
    def __init__(self):
        self.input_agent = InputAgent()
        self.intent_agent = IntentAgent()
        self.analyzer_agent = ProblemAnalyzerAgent()
        self.solver_agent = MathSolverAgent()
    
    def process_query(self, content: Union[str, bytes], input_type: InputType) -> MathAssistantResponse:
        """Main processing pipeline for math queries"""
        start_time = time.time()
        
        # Step 1: Process Input
        processed_input = self.input_agent.process_input(content, input_type)
        
        # Step 2: Classify Intent
        intent_classification = self.intent_agent.classify_intent(processed_input)
        
        # Step 3: Analyze Problem
        problem_analysis = self.analyzer_agent.analyze_problem(processed_input, intent_classification)
        
        # Step 4: Solve Problem
        solution = self.solver_agent.solve_problem(processed_input, intent_classification, problem_analysis)
        
        processing_time = time.time() - start_time
        
        return MathAssistantResponse(
            processed_input=processed_input,
            intent_classification=intent_classification,
            problem_analysis=problem_analysis,
            solution=solution,
            processing_time=processing_time
        )

def main():
    print("🧮 AI Math Assistant")
    print("=" * 50)
    print("Get step-by-step solutions and explanations for mathematical problems")
    print()
    
    math_assistant = MathAssistant()
    
    while True:
        print("\n" + "=" * 50)
        print("Choose input method:")
        print("1. Text input")
        print("2. Image file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            user_input = input("\nEnter your mathematical problem: ").strip()
            if user_input:
                print("\n🔄 Processing your mathematical problem...")
                try:
                    response = math_assistant.process_query(user_input, InputType.TEXT)
                    display_response(response)
                except Exception as e:
                    print(f"❌ Error processing problem: {e}")
            else:
                print("❌ Please enter a mathematical problem.")
        
        elif choice == "2":
            image_path = input("\nEnter path to image file: ").strip()
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                print("\n🔄 Analyzing image and solving problem...")
                response = math_assistant.process_query(image_data, InputType.IMAGE)
                display_response(response)
            except FileNotFoundError:
                print(f"❌ File not found: {image_path}")
            except Exception as e:
                print(f"❌ Error processing image: {e}")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def display_response(response: MathAssistantResponse):
    """Display the assistant's response in a formatted way"""
    
    print(f"✅ Processed in {response.processing_time:.2f} seconds")
    print()
    
    # Display extracted input
    print("📝 Processed Input")
    print("-" * 20)
    print(f"Content: {response.processed_input.content}")
    if response.processed_input.metadata:
        print(f"Metadata: {json.dumps(response.processed_input.metadata, indent=2)}")
    print()
    
    # Display intent classification
    print("🎯 Intent Analysis")
    print("-" * 20)
    print(f"Intent: {response.intent_classification.primary_intent.value}")
    print(f"Domain: {response.intent_classification.math_domain.value}")
    print(f"Level: {response.intent_classification.difficulty_level.value}")
    print(f"Confidence: {response.intent_classification.confidence:.2f}")
    print(f"Reasoning: {response.intent_classification.reasoning}")
    print()
    
    # Display problem analysis
    print("🔍 Problem Analysis")
    print("-" * 20)
    print(f"Problem Type: {response.problem_analysis.problem_type}")
    print(f"Complexity: {response.problem_analysis.complexity_level}")
    print(f"Estimated Time: {response.problem_analysis.estimated_time}")
    print()
    
    print("Required Concepts:")
    for concept in response.problem_analysis.required_concepts:
        print(f"• {concept}")
    print()
    
    print("Prerequisites:")
    for prereq in response.problem_analysis.prerequisites:
        print(f"• {prereq}")
    print()
    
    # Display solution
    print("📚 Solution")
    print("=" * 50)
    
    # Step-by-step solution
    print("Step-by-Step Solution:")
    print("-" * 30)
    for step in response.solution.step_by_step_solution:
        print(f"\nStep {step['step_number']}: {step['description']}")
        if step.get('calculation'):
            print(f"Calculation: {step['calculation']}")
        print(f"Reasoning: {step['reasoning']}")
        print("-" * 30)
    
    # Final answer
    print(f"\n🎯 Final Answer: {response.solution.final_answer}")
    print()
    
    # Additional information
    if response.solution.alternative_methods:
        print("Alternative Methods:")
        for method in response.solution.alternative_methods:
            print(f"• {method}")
        print()
    
    if response.solution.conceptual_insights:
        print("Key Insights:")
        for insight in response.solution.conceptual_insights:
            print(f"• {insight}")
        print()
    
    if response.solution.common_mistakes:
        print("Common Mistakes to Avoid:")
        for mistake in response.solution.common_mistakes:
            print(f"• {mistake}")
        print()
    
    if response.solution.verification:
        print(f"Verification: {response.solution.verification}")
        print()

def test_mode():
    """Test mode that runs a sample problem without interactive input"""
    print("🧮 AI Math Assistant - Test Mode")
    print("=" * 50)
    
    math_assistant = MathAssistant()
    
    # Test with a simple math problem
    test_problem = "Solve for x: 2x + 5 = 13"
    print(f"Testing with problem: {test_problem}")
    print()
    
    try:
        response = math_assistant.process_query(test_problem, InputType.TEXT)
        display_response(response)
    except Exception as e:
        print(f"❌ Error in test mode: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode()
    else:
        main()