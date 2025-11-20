"""
Smart Tutor Modules Package
Provides specialized functionality modules for the tutor system
"""

from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod

from core.router import Intent
from core.context_manager import ConversationContext


class ModuleInterface(Protocol):
    """Protocol defining the interface all modules must implement"""
    
    def process(self, user_input: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Process user input and return response
        
        Args:
            user_input: User's query or request
            context: Conversation context
            
        Returns:
            Dict containing response and metadata
        """
        ...
    
    def can_handle(self, user_input: str, context: ConversationContext) -> bool:
        """
        Check if module can handle the request
        
        Args:
            user_input: User's query
            context: Conversation context
            
        Returns:
            True if module can handle, False otherwise
        """
        ...
    
    def get_capabilities(self) -> list:
        """
        Get list of module capabilities
        
        Returns:
            List of capability descriptions
        """
        ...
    
    def get_module_name(self) -> str:
        """
        Get module name
        
        Returns:
            Module identifier
        """
        ...


class BaseModule(ABC):
    """Base class for all modules"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._initialized = False
    
    @abstractmethod
    def process(self, user_input: str, context: ConversationContext) -> Dict[str, Any]:
        """Process user input"""
        pass
    
    @abstractmethod
    def can_handle(self, user_input: str, context: ConversationContext) -> bool:
        """Check if module can handle request"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> list:
        """Get module capabilities"""
        pass
    
    def get_module_name(self) -> str:
        """Get module name"""
        return self.module_name
    
    def initialize(self):
        """Initialize module resources"""
        self._initialized = True
    
    def cleanup(self):
        """Cleanup module resources"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if module is initialized"""
        return self._initialized
    
    def validate_input(self, user_input: str) -> bool:
        """Validate input before processing"""
        return bool(user_input and user_input.strip())
    
    def format_response(
        self,
        content: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format standard module response"""
        return {
            "success": success,
            "content": content,
            "module": self.module_name,
            "metadata": metadata or {}
        }
    
    def format_error(self, error_message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "content": error_message,
            "module": self.module_name,
            "error": True
        }


# Module registry for dynamic loading
_MODULE_REGISTRY: Dict[str, type] = {}


def register_module(intent: Intent, module_class: type):
    """Register a module class for an intent"""
    _MODULE_REGISTRY[intent.value] = module_class


def get_module(intent: Intent) -> Optional[type]:
    """Get module class for intent"""
    return _MODULE_REGISTRY.get(intent.value)


def get_registered_modules() -> Dict[str, type]:
    """Get all registered modules"""
    return _MODULE_REGISTRY.copy()


# Lazy imports to avoid circular dependencies
def get_chat_module():
    """Get chat module instance"""
    from modules.chat.chatbot import ChatModule
    return ChatModule()


def get_math_module():
    """Get math module instance"""
    from modules.math.math_assistant import MathModule
    return MathModule()


def get_mermaid_module():
    """Get mermaid module instance"""
    from modules.mermaid.generator import MermaidModule
    return MermaidModule()


def get_mindmap_module():
    """Get mindmap module instance"""
    from modules.mindmap.mindmap import MindmapModule
    return MindmapModule()


# Module factory
class ModuleFactory:
    """Factory for creating module instances"""
    
    _instances: Dict[str, Any] = {}
    
    @staticmethod
    def get_module(intent: Intent, force_new: bool = False):
        """
        Get or create module instance
        
        Args:
            intent: Intent to get module for
            force_new: Force creation of new instance
            
        Returns:
            Module instance
        """
        module_key = intent.value
        
        if force_new or module_key not in ModuleFactory._instances:
            module_getters = {
                Intent.CHAT: get_chat_module,
                Intent.MATH: get_math_module,
                Intent.MERMAID: get_mermaid_module,
                Intent.MINDMAP: get_mindmap_module
            }
            
            getter = module_getters.get(intent)
            if getter:
                ModuleFactory._instances[module_key] = getter()
        
        return ModuleFactory._instances.get(module_key)
    
    @staticmethod
    def clear_cache():
        """Clear cached module instances"""
        ModuleFactory._instances.clear()
    
    @staticmethod
    def get_all_modules() -> Dict[Intent, Any]:
        """Get all available modules"""
        return {
            Intent.CHAT: ModuleFactory.get_module(Intent.CHAT),
            Intent.MATH: ModuleFactory.get_module(Intent.MATH),
            Intent.MERMAID: ModuleFactory.get_module(Intent.MERMAID),
            Intent.MINDMAP: ModuleFactory.get_module(Intent.MINDMAP)
        }


__all__ = [
    'ModuleInterface',
    'BaseModule',
    'ModuleFactory',
    'register_module',
    'get_module',
    'get_registered_modules',
    'get_chat_module',
    'get_math_module',
    'get_mermaid_module',
    'get_mindmap_module'
]