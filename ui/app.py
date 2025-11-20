"""
Main Streamlit application for Smart Tutor
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.utils import (
    initialize_session_state,
    get_intent_router,
    get_conversation_context,
    get_module_from_intent,
    display_error
)
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface
from ui.components.math_interface import render_math_interface
from ui.components.mermaid_interface import render_mermaid_interface
from core.router import Intent


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Smart Tutor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Get current module selection
    current_module = st.session_state.get('current_module', 'auto')
    
    # Main content area
    if current_module == 'auto':
        render_auto_mode()
    elif current_module == 'chat':
        render_chat_interface()
    elif current_module == 'math':
        render_math_interface()
    elif current_module == 'mermaid':
        render_mermaid_interface()
    else:
        st.error(f"Unknown module: {current_module}")


def render_auto_mode():
    """Render auto mode with intent routing"""
    
    st.subheader("🤖 Auto Mode")
    st.markdown("I'll automatically detect what you need and route to the appropriate module.")
    
    # Display conversation history from context
    context = get_conversation_context()
    recent_messages = context.get_messages(limit=20)
    
    # Display recent messages
    if recent_messages:
        st.markdown("### Recent Conversation")
        for msg in recent_messages[-10:]:  # Show last 10 messages
            role = msg.role
            content = msg.content
            module = msg.module or "auto"
            
            with st.chat_message(role):
                st.write(content)
                if st.session_state.get('show_debug', False):
                    st.caption(f"Module: {module} | Intent: {msg.intent.value if msg.intent else 'N/A'}")
    
    # Unified input
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Route intent
        router = get_intent_router()
        
        # Get routing context
        routing_context = {
            "active_module": context.active_module,
            "last_intent": context.last_intent.value if context.last_intent else None,
            "recent_messages": context.get_recent_context(window=3)
        }
        
        # Route the query
        with st.spinner("Analyzing your query..."):
            try:
                route_result = router.route(user_input, routing_context)
                
                # Display routing info if debug mode
                if st.session_state.get('show_debug', False):
                    with st.expander("🔍 Routing Info"):
                        st.json(route_result.to_dict())
                
                # Route to appropriate module
                detected_module = get_module_from_intent(route_result.intent)
                
                # Add user message to context
                context.add_message(
                    role="user",
                    content=user_input,
                    intent=route_result.intent,
                    module=detected_module
                )
                
                # Display routing decision
                st.info(f"🎯 Detected: **{route_result.intent.value.upper()}** ({route_result.confidence:.0%} confidence)")
                
                # Route to module
                if route_result.intent == Intent.CHAT:
                    handle_chat_query(user_input, context)
                elif route_result.intent == Intent.MATH:
                    handle_math_query(user_input, context)
                elif route_result.intent == Intent.MERMAID:
                    handle_mermaid_query(user_input, context)
                else:
                    # Fallback to chat
                    handle_chat_query(user_input, context)
                    
            except Exception as e:
                display_error(e, "in routing")
                # Fallback to chat
                handle_chat_query(user_input, context)


def handle_chat_query(user_input: str, context):
    """Handle chat query in auto mode"""
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from modules.chat.chatbot import ChatBot
    from ui.utils import display_error
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    
    chatbot = st.session_state.chatbot
    session_id = st.session_state.get('session_id', 'default')
    use_rag = st.session_state.get('use_rag', True)
    
    # Set chat mode
    mode = st.session_state.get('chat_mode', 'learn')
    if mode == 'custom':
        chatbot.set_custom_mode(st.session_state.get('custom_instructions', ''))
    else:
        chatbot.set_mode(mode)
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot.chat(
                    user_input=user_input,
                    session_id=session_id,
                    use_rag=use_rag
                )
                
                st.write(response)
                
                # Update context
                context.add_message(
                    role="assistant",
                    content=response,
                    intent=None,
                    module="chat"
                )
                
            except Exception as e:
                display_error(e, "in chat")
                st.error("Failed to generate response. Please try again.")


def handle_math_query(user_input: str, context):
    """Handle math query in auto mode"""
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from modules.math.math_assistant import MathAssistant
    from modules.math.models.schemas import InputType
    from ui.utils import display_error
    
    # Initialize math assistant
    if 'math_assistant' not in st.session_state:
        st.session_state.math_assistant = MathAssistant()
    
    math_assistant = st.session_state.math_assistant
    
    with st.spinner("Solving your math problem..."):
        try:
            response = math_assistant.process_query(
                content=user_input,
                input_type=InputType.TEXT
            )
            
            # Import display function
            from ui.components.math_interface import display_math_response
            display_math_response(response, context)
            
        except Exception as e:
            display_error(e, "in math processing")
            st.error("Failed to process the math problem. Please try again.")


def handle_mermaid_query(user_input: str, context):
    """Handle mermaid query in auto mode"""
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from modules.mermaid.scripts.pipeline_orchastrator import MermaidDiagramPipeline
    from ui.utils import display_error
    
    # Initialize pipeline
    if 'mermaid_pipeline' not in st.session_state:
        with st.spinner("Initializing diagram generator..."):
            try:
                st.session_state.mermaid_pipeline = MermaidDiagramPipeline()
            except Exception as e:
                st.error(f"Failed to initialize diagram generator: {e}")
                return
    
    pipeline = st.session_state.mermaid_pipeline
    
    with st.spinner("Generating diagram..."):
        try:
            result = pipeline.generate_diagram(user_input)
            
            # Import display function
            from ui.components.mermaid_interface import display_mermaid_result
            display_mermaid_result(result, context)
            
        except Exception as e:
            display_error(e, "in diagram generation")
            st.error("Failed to generate diagram. Please try again.")


if __name__ == "__main__":
    main()

