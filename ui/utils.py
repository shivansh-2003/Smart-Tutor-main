"""
Utility functions for Streamlit UI
"""

import streamlit as st
import uuid
from typing import Optional, Dict, Any
from core.context_manager import ContextManager, ConversationContext
from core.router import Intent, IntentRouter


def get_session_id() -> str:
    """Get or create unique session ID for Streamlit session"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_context_manager() -> ContextManager:
    """Get or create ContextManager instance"""
    if 'context_manager' not in st.session_state:
        st.session_state.context_manager = ContextManager()
    return st.session_state.context_manager


def get_conversation_context() -> ConversationContext:
    """Get conversation context for current session"""
    cm = get_context_manager()
    session_id = get_session_id()
    return cm.get_context(session_id)


def get_intent_router() -> IntentRouter:
    """Get or create IntentRouter instance"""
    if 'intent_router' not in st.session_state:
        st.session_state.intent_router = IntentRouter()
    return st.session_state.intent_router


def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_module = 'auto'  # 'auto', 'chat', 'math', 'mermaid'
        st.session_state.chat_mode = 'learn'
        st.session_state.use_rag = True
        st.session_state.show_debug = False


def format_error_message(error: Exception) -> str:
    """Format error message for display"""
    error_type = type(error).__name__
    error_msg = str(error)
    return f"**{error_type}**: {error_msg}"


def display_error(error: Exception, context: str = ""):
    """Display error in Streamlit with expandable details"""
    st.error(f"❌ Error {context}: {format_error_message(error)}")
    with st.expander("Error Details"):
        import traceback
        st.code(traceback.format_exc())


def get_module_display_name(module: str) -> str:
    """Get display name for module"""
    names = {
        'chat': '💬 Chat Assistant',
        'math': '🧮 Math Assistant',
        'mermaid': '📊 Diagram Generator',
        'auto': '🤖 Auto Mode'
    }
    return names.get(module, module.title())


def clear_module_history(module: Optional[str] = None):
    """Clear history for specific module or all modules"""
    context = get_conversation_context()
    context.clear_history(module)
    st.success(f"History cleared for {module or 'all modules'}")


def get_module_from_intent(intent: Intent) -> str:
    """Map intent to module name"""
    mapping = {
        Intent.CHAT: 'chat',
        Intent.MATH: 'math',
        Intent.MERMAID: 'mermaid',
        Intent.MINDMAP: 'mindmap'
    }
    return mapping.get(intent, 'chat')

