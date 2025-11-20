"""
Sidebar component for Smart Tutor UI
"""

import streamlit as st
from ui.utils import (
    get_conversation_context,
    clear_module_history,
    get_module_display_name
)


def render_sidebar():
    """Render the sidebar with module selection and settings"""
    
    with st.sidebar:
        st.title("🎓 Smart Tutor")
        st.markdown("---")
        
        # Module Selection
        st.subheader("📚 Modules")
        current_module = st.radio(
            "Select Module:",
            options=['auto', 'chat', 'math', 'mermaid'],
            format_func=get_module_display_name,
            index=['auto', 'chat', 'math', 'mermaid'].index(
                st.session_state.get('current_module', 'auto')
            ),
            key='module_selector'
        )
        st.session_state.current_module = current_module
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        # Chat-specific settings
        if current_module in ['auto', 'chat']:
            chat_mode = st.selectbox(
                "Chat Mode:",
                options=['learn', 'hint', 'quiz', 'eli5', 'custom'],
                index=['learn', 'hint', 'quiz', 'eli5', 'custom'].index(
                    st.session_state.get('chat_mode', 'learn')
                ),
                key='chat_mode_selector'
            )
            st.session_state.chat_mode = chat_mode
            
            use_rag = st.checkbox(
                "Use RAG (Retrieval Augmented Generation)",
                value=st.session_state.get('use_rag', True),
                key='rag_toggle'
            )
            st.session_state.use_rag = use_rag
        
        st.markdown("---")
        
        # Session Management
        st.subheader("💾 Session")
        
        context = get_conversation_context()
        message_count = len(context.messages)
        st.metric("Messages", message_count)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_module_history()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['initialized', 'session_id']:
                        del st.session_state[key]
                clear_module_history()
                st.rerun()
        
        st.markdown("---")
        
        # Debug Mode
        st.subheader("🔧 Debug")
        show_debug = st.checkbox(
            "Show Debug Info",
            value=st.session_state.get('show_debug', False),
            key='debug_toggle'
        )
        st.session_state.show_debug = show_debug
        
        if show_debug:
            st.markdown("---")
            st.subheader("📊 Debug Info")
            st.json({
                "session_id": st.session_state.get('session_id', 'N/A'),
                "current_module": current_module,
                "active_module": context.active_module,
                "last_intent": context.last_intent.value if context.last_intent else None,
                "message_count": message_count
            })
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        Smart Tutor is an AI-powered educational assistant with:
        - 💬 **Chat**: General Q&A with RAG support
        - 🧮 **Math**: Step-by-step problem solving
        - 📊 **Diagrams**: Mermaid diagram generation
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit & LangChain")

