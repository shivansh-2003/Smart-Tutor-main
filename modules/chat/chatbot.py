"""Multi-mode chatbot with RAG support"""

import sys
from pathlib import Path
from typing import Literal, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.llm_factory import get_task_llm
from rag.retrieval import RAGRetrieval
from .memory import ChatMemory
from .prompts import PromptLoader
from .config import ChatModuleConfig


class ChatBot:
    """Conversational chatbot with RAG and multiple modes"""
    
    MODES = ["learn", "hint", "quiz", "eli5", "custom"]
    
    def __init__(self, 
                 namespace: str = "general",
                 mode: str = "learn"):
        """Initialize chatbot with RAG and LLM"""
        
        # Initialize components
        self.rag = RAGRetrieval()
        self.config = ChatModuleConfig()
        self.memory = ChatMemory()
        self.prompt_loader = PromptLoader()
        
        # Settings
        self.namespace = namespace
        self.current_mode = mode
        self.custom_instructions = ""
        self._llm = None
    
    def set_mode(self, mode: Literal["learn", "hint", "quiz", "eli5", "custom"]):
        """Change chatbot mode"""
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode. Must be one of: {self.MODES}")
        self.current_mode = mode
        self._llm = None  # Reset LLM to get new one for mode
    
    def set_custom_mode(self, instructions: str):
        """Set custom mode with user instructions"""
        self.current_mode = "custom"
        self.custom_instructions = instructions
    
    def _get_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from RAG"""
        docs = self.rag.rag_pipeline(
            query=query,
            namespace=self.namespace,
            final_k=k
        )
        
        # Format context with sources
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts)
    
    def chat(self, 
             user_input: str, 
             session_id: str = "default",
             use_rag: bool = True) -> str:
        """Process user input and generate response"""
        
        # Get conversation history
        history_str = self.memory.get_history_str(session_id, last_n=10)
        
        # Get RAG context if enabled
        context = ""
        if use_rag:
            context = self._get_context(user_input)
        
        # Build prompt
        prompt = self.prompt_loader.format(
            mode=self.current_mode,
            context=context,
            question=user_input,
            history=history_str,
            custom_instructions=self.custom_instructions
        )
        
        # Get LLM for current mode
        if self._llm is None:
            self._llm = self.config.get_llm_for_mode(self.current_mode)
        
        # Generate response
        response = self._llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Update memory
        self.memory.add_user_message(session_id, user_input)
        self.memory.add_ai_message(session_id, response_text)
        
        return response_text
    
    def clear_history(self, session_id: str = "default"):
        """Clear conversation history"""
        self.memory.clear_session(session_id)
    
    def get_history(self, session_id: str = "default"):
        """Get conversation history"""
        return self.memory.get_history(session_id)
    
    def process(self, user_input: str, context: dict = None) -> dict:
        """Process input (module interface compatibility)"""
        session_id = context.get('session_id', 'default') if context else 'default'
        use_rag = context.get('use_rag', True) if context else True
        
        response = self.chat(user_input, session_id, use_rag)
        
        return {
            'response': response,
            'mode': self.current_mode,
            'session_id': session_id
        }
    
    def can_handle(self, user_input: str, context: dict = None) -> bool:
        """Check if this module can handle the input"""
        # Chat module is the fallback, can handle anything
        return True
    
    def get_capabilities(self) -> list:
        """Get module capabilities"""
        return [
            "general_conversation",
            "rag_based_qa",
            "multi_mode_learning",
            "conversation_memory"
        ]
    
    def get_module_name(self) -> str:
        """Get module name"""
        return "chat"