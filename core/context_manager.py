"""
Context manager for conversation state and history
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from core.config import get_config
from core.router import Intent


@dataclass
class Message:
    """Conversation message"""
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[Intent] = None
    module: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "intent": self.intent.value if self.intent else None,
            "module": self.module,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ModuleContext:
    """Context for a specific module"""
    module_name: str
    active: bool = False
    state: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def update(self, **kwargs):
        """Update module state"""
        self.state.update(kwargs)
        self.last_updated = time.time()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        return self.state.get(key, default)
    
    def clear(self):
        """Clear module state"""
        self.state.clear()
        self.last_updated = time.time()


class ConversationContext:
    """Manages conversation context and state"""
    
    def __init__(self, session_id: str = "default"):
        self.config = get_config()
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Message history
        self.messages: deque = deque(maxlen=self.config.context.max_history)
        
        # Active module and intent
        self.active_module: Optional[str] = None
        self.last_intent: Optional[Intent] = None
        
        # Module-specific contexts
        self.module_contexts: Dict[str, ModuleContext] = {
            "chat": ModuleContext("chat"),
            "math": ModuleContext("math"),
            "mermaid": ModuleContext("mermaid"),
            "mindmap": ModuleContext("mindmap")
        }
        
        # User preferences
        self.user_preferences: Dict[str, Any] = {}
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def add_message(
        self,
        role: str,
        content: str,
        intent: Optional[Intent] = None,
        module: Optional[str] = None,
        **metadata
    ):
        """Add message to conversation history"""
        message = Message(
            role=role,
            content=content,
            intent=intent,
            module=module,
            metadata=metadata
        )
        
        self.messages.append(message)
        self.last_activity = time.time()
        
        # Update active module and intent
        if module:
            self.active_module = module
        if intent:
            self.last_intent = intent
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        module: Optional[str] = None
    ) -> List[Message]:
        """Get messages with optional filtering"""
        messages = list(self.messages)
        
        if role:
            messages = [m for m in messages if m.role == role]
        
        if module:
            messages = [m for m in messages if m.module == module]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_recent_context(
        self,
        window: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages as context"""
        window = window or self.config.context.context_window
        recent = list(self.messages)[-window:]
        return [m.to_dict() for m in recent]
    
    def get_formatted_history(
        self,
        max_messages: Optional[int] = None,
        format_template: str = "{role}: {content}\n"
    ) -> str:
        """Get formatted conversation history"""
        messages = self.get_messages(limit=max_messages)
        
        formatted = []
        for msg in messages:
            formatted.append(format_template.format(
                role=msg.role.title(),
                content=msg.content
            ))
        
        return "".join(formatted)
    
    def set_active_module(self, module: str):
        """Set active module"""
        # Deactivate all modules
        for mod_ctx in self.module_contexts.values():
            mod_ctx.active = False
        
        # Activate target module
        if module in self.module_contexts:
            self.module_contexts[module].active = True
            self.active_module = module
    
    def get_module_context(self, module: str) -> Optional[ModuleContext]:
        """Get context for specific module"""
        return self.module_contexts.get(module)
    
    def update_module_context(self, module: str, **state):
        """Update module context state"""
        if module in self.module_contexts:
            self.module_contexts[module].update(**state)
    
    def clear_module_context(self, module: str):
        """Clear module context"""
        if module in self.module_contexts:
            self.module_contexts[module].clear()
    
    def set_user_preference(self, key: str, value: Any):
        """Set user preference"""
        self.user_preferences[key] = value
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        return self.user_preferences.get(key, default)
    
    def clear_history(self, module: Optional[str] = None):
        """Clear conversation history"""
        if module:
            # Clear only messages from specific module
            self.messages = deque(
                [m for m in self.messages if m.module != module],
                maxlen=self.config.context.max_history
            )
        else:
            # Clear all history
            self.messages.clear()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context"""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "active_module": self.active_module,
            "last_intent": self.last_intent.value if self.last_intent else None,
            "session_duration": time.time() - self.created_at,
            "last_activity": time.time() - self.last_activity,
            "module_states": {
                name: {
                    "active": ctx.active,
                    "state_keys": list(ctx.state.keys())
                }
                for name, ctx in self.module_contexts.items()
            }
        }
    
    def is_session_active(self) -> bool:
        """Check if session is still active"""
        timeout = self.config.context.session_timeout
        return (time.time() - self.last_activity) < timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "messages": [m.to_dict() for m in self.messages],
            "active_module": self.active_module,
            "last_intent": self.last_intent.value if self.last_intent else None,
            "user_preferences": self.user_preferences,
            "metadata": self.metadata
        }


class ContextManager:
    """Manages multiple conversation contexts (sessions)"""
    
    def __init__(self):
        self.config = get_config()
        self.contexts: Dict[str, ConversationContext] = {}
    
    def get_context(self, session_id: str = "default") -> ConversationContext:
        """Get or create context for session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(session_id)
        
        return self.contexts[session_id]
    
    def delete_context(self, session_id: str):
        """Delete context for session"""
        if session_id in self.contexts:
            del self.contexts[session_id]
    
    def cleanup_inactive_sessions(self):
        """Remove inactive sessions"""
        inactive = [
            sid for sid, ctx in self.contexts.items()
            if not ctx.is_session_active()
        ]
        
        for sid in inactive:
            del self.contexts[sid]
        
        return len(inactive)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return [
            sid for sid, ctx in self.contexts.items()
            if ctx.is_session_active()
        ]
    
    def get_all_contexts(self) -> Dict[str, ConversationContext]:
        """Get all contexts"""
        return self.contexts.copy()
    
    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[Intent] = None,
        module: Optional[str] = None,
        **metadata
    ):
        """Convenience method to add message to session"""
        context = self.get_context(session_id)
        context.add_message(role, content, intent, module, **metadata)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for specific session"""
        if session_id in self.contexts:
            return self.contexts[session_id].get_context_summary()
        return None
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data"""
        if session_id in self.contexts:
            return self.contexts[session_id].to_dict()
        return None
    
    def import_session(self, session_data: Dict[str, Any]):
        """Import session data"""
        session_id = session_data.get("session_id", "imported")
        context = ConversationContext(session_id)
        
        # Restore basic fields
        context.created_at = session_data.get("created_at", time.time())
        context.last_activity = session_data.get("last_activity", time.time())
        context.active_module = session_data.get("active_module")
        context.user_preferences = session_data.get("user_preferences", {})
        context.metadata = session_data.get("metadata", {})
        
        # Restore intent
        last_intent = session_data.get("last_intent")
        if last_intent:
            context.last_intent = Intent(last_intent)
        
        # Restore messages
        for msg_dict in session_data.get("messages", []):
            intent = Intent(msg_dict["intent"]) if msg_dict.get("intent") else None
            message = Message(
                role=msg_dict["role"],
                content=msg_dict["content"],
                intent=intent,
                module=msg_dict.get("module"),
                timestamp=msg_dict.get("timestamp", time.time()),
                metadata=msg_dict.get("metadata", {})
            )
            context.messages.append(message)
        
        self.contexts[session_id] = context
        return session_id