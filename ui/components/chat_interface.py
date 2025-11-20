"""
Chat interface component for Smart Tutor UI
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.chat.chatbot import ChatBot
from ui.utils import (
    get_session_id,
    get_conversation_context,
    display_error
)


def render_chat_interface():
    """Render the chat interface"""
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    
    chatbot = st.session_state.chatbot
    session_id = get_session_id()
    context = get_conversation_context()
    
    # Set chat mode
    mode = st.session_state.get('chat_mode', 'learn')
    if mode == 'custom':
        custom_instructions = st.text_area(
            "Custom Instructions:",
            value=st.session_state.get('custom_instructions', ''),
            key='custom_instructions_input',
            help="Provide custom instructions for the chatbot behavior"
        )
        st.session_state.custom_instructions = custom_instructions
        chatbot.set_custom_mode(custom_instructions)
    else:
        chatbot.set_mode(mode)
    
    # Display chat history
    st.subheader("💬 Chat Assistant")
    
    # Get conversation history
    history = chatbot.get_history(session_id)
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        if not history:
            st.info("👋 Start a conversation! Ask me anything or upload documents for RAG-powered answers.")
        else:
            for message in history:
                if hasattr(message, 'content'):
                    content = message.content
                    if hasattr(message, '__class__'):
                        from langchain_core.messages import HumanMessage, AIMessage
                        if isinstance(message, HumanMessage):
                            with st.chat_message("user"):
                                st.write(content)
                        elif isinstance(message, AIMessage):
                            with st.chat_message("assistant"):
                                st.write(content)
                else:
                    # Fallback for dict format
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    with st.chat_message(role):
                        st.write(content)
    
    # Input area
    st.markdown("---")
    
    # RAG toggle
    use_rag = st.session_state.get('use_rag', True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to display immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Update context
        context.add_message(
            role="user",
            content=user_input,
            intent=None,
            module="chat"
        )
        
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

