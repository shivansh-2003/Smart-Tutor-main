"""
Math interface component for Smart Tutor UI
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.math.math_assistant import MathAssistant
from modules.math.models.schemas import InputType
from ui.utils import (
    get_conversation_context,
    display_error
)


def render_math_interface():
    """Render the math assistant interface"""
    
    # Initialize math assistant
    if 'math_assistant' not in st.session_state:
        st.session_state.math_assistant = MathAssistant()
    
    math_assistant = st.session_state.math_assistant
    context = get_conversation_context()
    
    st.subheader("🧮 Math Assistant")
    st.markdown("Enter a mathematical problem and get step-by-step solutions.")
    
    # Input method selection
    input_method = st.radio(
        "Input Method:",
        options=["Text", "Image (Coming Soon)"],
        horizontal=True
    )
    
    # Problem input
    if input_method == "Text":
        problem_input = st.text_area(
            "Enter your mathematical problem:",
            height=100,
            placeholder="e.g., Solve for x: 2x + 5 = 13"
        )
        
        if st.button("Solve", type="primary", use_container_width=True):
            if problem_input.strip():
                with st.spinner("Analyzing problem and generating solution..."):
                    try:
                        # Process the problem
                        response = math_assistant.process_query(
                            content=problem_input,
                            input_type=InputType.TEXT
                        )
                        
                        # Update context
                        context.add_message(
                            role="user",
                            content=problem_input,
                            intent=None,
                            module="math"
                        )
                        
                        # Display results
                        display_math_response(response, context)
                        
                    except Exception as e:
                        display_error(e, "in math processing")
                        st.error("Failed to process the problem. Please try again.")
            else:
                st.warning("Please enter a mathematical problem.")
    
    else:
        st.info("📸 Image input will be available in a future update.")
        uploaded_file = st.file_uploader(
            "Upload an image of your math problem",
            type=['png', 'jpg', 'jpeg'],
            disabled=True
        )
        if uploaded_file:
            st.warning("Image processing is not yet implemented.")


def display_math_response(response, context):
    """Display the math assistant response"""
    
    # Processing time
    st.success(f"✅ Processed in {response.processing_time:.2f} seconds")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Solution",
        "🎯 Analysis",
        "🔍 Problem Details",
        "💡 Insights"
    ])
    
    with tab1:
        st.subheader("Step-by-Step Solution")
        
        # Display steps
        for step in response.solution.step_by_step_solution:
            step_num = step.get('step_number', 0)
            description = step.get('description', '')
            calculation = step.get('calculation', '')
            reasoning = step.get('reasoning', '')
            
            with st.expander(f"Step {step_num}: {description}", expanded=(step_num == 1)):
                if calculation:
                    st.code(calculation, language='python')
                if reasoning:
                    st.write(f"**Reasoning:** {reasoning}")
        
        # Final answer
        st.markdown("---")
        st.markdown(f"### 🎯 Final Answer")
        st.success(f"**{response.solution.final_answer}**")
        
        # Verification
        if response.solution.verification:
            st.info(f"**Verification:** {response.solution.verification}")
    
    with tab2:
        st.subheader("Intent Classification")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intent", response.intent_classification.primary_intent.value.replace('_', ' ').title())
        with col2:
            st.metric("Domain", response.intent_classification.math_domain.value.title())
        with col3:
            st.metric("Difficulty", response.intent_classification.difficulty_level.value.replace('_', ' ').title())
        
        st.write(f"**Confidence:** {response.intent_classification.confidence:.2%}")
        st.write(f"**Reasoning:** {response.intent_classification.reasoning}")
        
        st.markdown("---")
        st.subheader("Problem Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Problem Type:** {response.problem_analysis.problem_type}")
            st.write(f"**Complexity:** {response.problem_analysis.complexity_level}")
        with col2:
            st.write(f"**Estimated Time:** {response.problem_analysis.estimated_time}")
        
        st.markdown("---")
        st.write("**Required Concepts:**")
        for concept in response.problem_analysis.required_concepts:
            st.write(f"• {concept}")
        
        st.write("**Prerequisites:**")
        for prereq in response.problem_analysis.prerequisites:
            st.write(f"• {prereq}")
    
    with tab3:
        st.subheader("Processed Input")
        st.write(f"**Content:** {response.processed_input.content}")
        if response.processed_input.metadata:
            st.json(response.processed_input.metadata)
    
    with tab4:
        st.subheader("Additional Information")
        
        if response.solution.alternative_methods:
            st.write("**Alternative Methods:**")
            for method in response.solution.alternative_methods:
                st.write(f"• {method}")
            st.markdown("---")
        
        if response.solution.conceptual_insights:
            st.write("**Key Insights:**")
            for insight in response.solution.conceptual_insights:
                st.write(f"• {insight}")
            st.markdown("---")
        
        if response.solution.common_mistakes:
            st.write("**Common Mistakes to Avoid:**")
            for mistake in response.solution.common_mistakes:
                st.write(f"• {mistake}")
    
    # Add assistant response to context
    solution_summary = f"Solved: {response.solution.final_answer}"
    context.add_message(
        role="assistant",
        content=solution_summary,
        intent=None,
        module="math"
    )

