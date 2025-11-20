"""
Mermaid interface component for Smart Tutor UI
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.mermaid.scripts.pipeline_orchastrator import MermaidDiagramPipeline
from ui.utils import (
    get_conversation_context,
    display_error
)


def render_mermaid_interface():
    """Render the mermaid diagram generator interface"""
    
    # Initialize mermaid pipeline
    if 'mermaid_pipeline' not in st.session_state:
        with st.spinner("Initializing diagram generator..."):
            try:
                st.session_state.mermaid_pipeline = MermaidDiagramPipeline()
            except Exception as e:
                st.error(f"Failed to initialize diagram generator: {e}")
                st.stop()
    
    pipeline = st.session_state.mermaid_pipeline
    context = get_conversation_context()
    
    st.subheader("📊 Diagram Generator")
    st.markdown("Describe the diagram you want to create, and I'll generate Mermaid code for it.")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - "Create a flowchart for a user login process"
        - "Draw a sequence diagram for an API request flow"
        - "Generate a class diagram for an e-commerce system"
        - "Show me a state diagram for a vending machine"
        - "Create an ER diagram for a library management system"
        """)
    
    # Query input
    query = st.text_area(
        "Describe your diagram:",
        height=100,
        placeholder="e.g., Create a flowchart showing the steps of a user registration process"
    )
    
    if st.button("Generate Diagram", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Generating diagram... This may take a moment."):
                try:
                    # Generate diagram
                    result = pipeline.generate_diagram(query)
                    
                    # Update context
                    context.add_message(
                        role="user",
                        content=query,
                        intent=None,
                        module="mermaid"
                    )
                    
                    # Display results
                    display_mermaid_result(result, context)
                    
                except Exception as e:
                    display_error(e, "in diagram generation")
                    st.error("Failed to generate diagram. Please try again.")
        else:
            st.warning("Please enter a description for your diagram.")


def display_mermaid_result(result, context):
    """Display the mermaid generation result"""
    
    if result.get("success", False):
        st.success("✅ Diagram generated successfully!")
        
        mermaid_code = result.get("final_diagram", "")
        pipeline_state = result.get("pipeline_state", {})
        
        if mermaid_code:
            # Display diagram
            st.markdown("### 📊 Generated Diagram")
            
            # Try to render with streamlit-mermaid if available, otherwise show code
            try:
                import streamlit_mermaid as st_mermaid
                st_mermaid.st_mermaid(mermaid_code)
            except ImportError:
                # Fallback: display as code block
                st.info("💡 Install `streamlit-mermaid` for diagram rendering. Showing code:")
                st.code(mermaid_code, language="mermaid")
                
                # Also try to show in HTML if possible
                try:
                    st.markdown("---")
                    st.markdown("### Preview (HTML)")
                    html_content = f"""
                    <div style="background: white; padding: 20px; border-radius: 5px;">
                        <pre class="mermaid">
{mermaid_code}
                        </pre>
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                    <script>mermaid.initialize({{startOnLoad:true}});</script>
                    """
                    import streamlit.components.v1 as components
                    components.html(html_content, height=400, scrolling=True)
                except Exception:
                    pass  # If HTML rendering fails, just show code
            
            # Code display with copy
            st.markdown("---")
            st.markdown("### 📝 Mermaid Code")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(mermaid_code, language="mermaid")
            with col2:
                st.button("📋 Copy Code", key="copy_mermaid", use_container_width=True)
            
            # Pipeline status (if debug mode)
            if st.session_state.get('show_debug', False):
                st.markdown("---")
                with st.expander("🔧 Pipeline Debug Info"):
                    st.json({
                        "current_stage": pipeline_state.get("current_stage", "unknown"),
                        "error_occurred": pipeline_state.get("error_occurred", False),
                        "error_message": pipeline_state.get("error_message"),
                        "refinement_needed": pipeline_state.get("refinement_needed", False),
                        "validation_needed": pipeline_state.get("validation_needed", False)
                    })
            
            # Add assistant response to context
            context.add_message(
                role="assistant",
                content=f"Generated diagram: {mermaid_code[:100]}...",
                intent=None,
                module="mermaid"
            )
        else:
            st.warning("Diagram code is empty. Check debug info for details.")
            if st.session_state.get('show_debug', False):
                st.json(pipeline_state)
    else:
        error_msg = result.get("error", "Unknown error")
        st.error(f"❌ Diagram generation failed: {error_msg}")
        
        if st.session_state.get('show_debug', False):
            with st.expander("Error Details"):
                st.json(result)

