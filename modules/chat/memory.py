"""Chat memory management using LangChain"""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver


class ChatMemory:
    """Manages conversation history for chat sessions"""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        self._sessions: Dict[str, List[BaseMessage]] = {}
    
    def add_user_message(self, session_id: str, message: str):
        """Add user message to session"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(HumanMessage(content=message))
    
    def add_ai_message(self, session_id: str, message: str):
        """Add AI message to session"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(AIMessage(content=message))
    
    def get_history(self, session_id: str, last_n: int = None) -> List[BaseMessage]:
        """Get conversation history for session"""
        messages = self._sessions.get(session_id, [])
        if last_n:
            return messages[-last_n:]
        return messages
    
    def get_history_str(self, session_id: str, last_n: int = 10) -> str:
        """Get formatted conversation history as string"""
        messages = self.get_history(session_id, last_n)
        history_str = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"
        return history_str
    
    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def clear_all(self):
        """Clear all sessions"""
        self._sessions.clear()
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self._sessions)