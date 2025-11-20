"""
Intent router for classifying and routing user queries
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass

from shared.llm_factory import get_llm
from shared.validators import ValidationResult
from core.config import get_config


class Intent(str, Enum):
    """Available intents for routing"""
    CHAT = "chat"
    MATH = "math"
    MERMAID = "mermaid"
    MINDMAP = "mindmap"
    UNKNOWN = "unknown"


@dataclass
class RouteResult:
    """Result of intent routing"""
    intent: Intent
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    
    def __bool__(self):
        return self.confidence >= get_config().router.confidence_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


class IntentRouter:
    """Intent classifier and router"""
    
    # Keyword patterns for quick classification
    INTENT_PATTERNS = {
        Intent.MATH: {
            "keywords": [
                "solve", "calculate", "compute", "equation", "derivative",
                "integral", "matrix", "algebra", "calculus", "geometry",
                "x =", "y =", "f(x)", "∫", "∑", "√", "^2", "sin", "cos", "tan"
            ],
            "patterns": [
                r'\d+\s*[+\-*/^]\s*\d+',
                r'[a-z]\s*=\s*\d+',
                r'\b(solve|find|calculate)\s+(for|x|y)',
                r'\b\d+x\b',
                r'[=<>≤≥≠]'
            ]
        },
        Intent.MERMAID: {
            "keywords": [
                "diagram", "flowchart", "visualize", "chart", "graph",
                "sequence diagram", "class diagram", "draw", "sketch",
                "architecture", "flow", "process diagram", "er diagram",
                "state diagram", "show flow", "create diagram"
            ],
            "patterns": [
                r'\b(create|generate|make|draw|show)\s+(a\s+)?(diagram|flowchart|chart)',
                r'\bvisuali[zs]e\b',
                r'\bdiagram\s+for\b'
            ]
        },
        Intent.MINDMAP: {
            "keywords": [
                "mindmap", "mind map", "concept map", "summarize structure",
                "knowledge graph", "map out", "organize concepts", "break down",
                "hierarchical structure", "topic breakdown"
            ],
            "patterns": [
                r'\b(mindmap|mind\s+map|concept\s+map)\b',
                r'\bmap\s+out\b',
                r'\b(break\s+down|organize)\s+.+\s+(concept|topic|structure)'
            ]
        }
    }
    
    CLASSIFICATION_PROMPT = """You are an intent classifier for Smart Tutor. Classify the user's query into one of these intents:

1. MATH - Mathematical problems, equations, calculations, proofs
2. MERMAID - Diagram requests (flowcharts, sequence diagrams, architecture diagrams, etc.)
3. MINDMAP - Mindmap/concept map generation, knowledge structure visualization
4. CHAT - General questions, explanations, learning assistance, anything else

USER QUERY: "{query}"

CONTEXT (previous conversation):
{context}

Analyze the query and respond in JSON format:
{{
    "intent": "CHAT|MATH|MERMAID|MINDMAP",
    "confidence": 0.95,
    "reasoning": "Brief explanation of classification decision",
    "indicators": ["list", "of", "key", "indicators"]
}}

Consider:
- Primary action requested
- Keywords and phrases
- Mathematical notation or equations
- Diagram-related terminology
- Context from previous messages
"""
    
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm(
            model=self.config.router.router_model,
            temperature=self.config.router.router_temperature
        )
    
    def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RouteResult:
        """Route query to appropriate module"""
        
        # Quick pattern-based classification
        quick_result = self._quick_classify(query)
        if quick_result and quick_result.confidence >= 0.9:
            return quick_result
        
        # LLM-based classification for complex cases
        llm_result = self._llm_classify(query, context)
        
        # Combine results if both available
        if quick_result and llm_result:
            if quick_result.intent == llm_result.intent:
                # Both agree, boost confidence
                combined_confidence = min(
                    (quick_result.confidence + llm_result.confidence) / 2 + 0.1,
                    1.0
                )
                return RouteResult(
                    intent=quick_result.intent,
                    confidence=combined_confidence,
                    reasoning=llm_result.reasoning,
                    metadata={
                        "quick_match": True,
                        "llm_match": True,
                        **llm_result.metadata
                    }
                )
        
        return llm_result or quick_result or RouteResult(
            intent=Intent.CHAT,
            confidence=0.5,
            reasoning="Fallback to general chat",
            metadata={}
        )
    
    def _quick_classify(self, query: str) -> Optional[RouteResult]:
        """Fast pattern-based classification"""
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in Intent if intent != Intent.UNKNOWN}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            # Check keywords
            keyword_matches = sum(
                1 for keyword in patterns["keywords"]
                if keyword.lower() in query_lower
            )
            
            # Check regex patterns
            pattern_matches = sum(
                1 for pattern in patterns["patterns"]
                if re.search(pattern, query_lower)
            )
            
            # Calculate score
            total_indicators = len(patterns["keywords"]) + len(patterns["patterns"])
            scores[intent] = (keyword_matches + pattern_matches * 2) / total_indicators
        
        # Get best match
        best_intent = max(scores.items(), key=lambda x: x[1])
        
        if best_intent[1] > 0.3:  # Minimum threshold
            return RouteResult(
                intent=best_intent[0],
                confidence=min(best_intent[1], 0.85),  # Cap at 0.85 for pattern matching
                reasoning=f"Pattern-based classification: {best_intent[0].value}",
                metadata={"method": "pattern", "score": best_intent[1]}
            )
        
        return None
    
    def _llm_classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[RouteResult]:
        """LLM-based classification"""
        try:
            # Format context
            context_str = ""
            if context and self.config.router.use_context:
                context_str = self._format_context(context)
            
            # Create prompt
            prompt = self.CLASSIFICATION_PROMPT.format(
                query=query,
                context=context_str or "No previous context"
            )
            
            # Get classification
            response = self.llm.invoke(prompt)
            result = self._parse_classification_response(response.content)
            
            return RouteResult(
                intent=Intent(result["intent"].lower()),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                metadata={
                    "method": "llm",
                    "indicators": result.get("indicators", [])
                }
            )
        
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None
    
    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM classification response"""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate
            if "intent" not in data or "confidence" not in data:
                raise ValueError("Missing required fields")
            
            return data
        
        except Exception as e:
            print(f"Response parsing error: {e}")
            return {
                "intent": "CHAT",
                "confidence": 0.5,
                "reasoning": "Parse error, defaulting to chat",
                "indicators": []
            }
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt"""
        parts = []
        
        if "active_module" in context:
            parts.append(f"Active module: {context['active_module']}")
        
        if "last_intent" in context:
            parts.append(f"Last intent: {context['last_intent']}")
        
        if "recent_messages" in context:
            messages = context["recent_messages"][-3:]  # Last 3 messages
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                parts.append(f"{role}: {content}")
        
        return "\n".join(parts) if parts else "No context"
    
    def route_with_fallback(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        allowed_intents: Optional[List[Intent]] = None
    ) -> RouteResult:
        """Route with fallback logic"""
        result = self.route(query, context)
        
        # Check if intent is allowed
        if allowed_intents and result.intent not in allowed_intents:
            # Fallback to chat
            return RouteResult(
                intent=Intent.CHAT,
                confidence=0.6,
                reasoning=f"Intent {result.intent.value} not allowed, fallback to chat",
                metadata={"original_intent": result.intent.value}
            )
        
        # Check confidence threshold
        if result.confidence < self.config.router.confidence_threshold:
            # Fallback to chat for low confidence
            return RouteResult(
                intent=Intent.CHAT,
                confidence=0.6,
                reasoning=f"Low confidence ({result.confidence}), fallback to chat",
                metadata={"original_intent": result.intent.value}
            )
        
        return result
    
    def get_module_capabilities(self, intent: Intent) -> List[str]:
        """Get capabilities of a module"""
        capabilities = {
            Intent.MATH: [
                "Solve equations and mathematical problems",
                "Explain mathematical concepts",
                "Provide step-by-step solutions",
                "Verify calculations"
            ],
            Intent.MERMAID: [
                "Generate flowcharts and diagrams",
                "Create sequence diagrams",
                "Visualize system architectures",
                "Draw ER diagrams and state machines"
            ],
            Intent.MINDMAP: [
                "Generate concept mindmaps",
                "Organize knowledge hierarchically",
                "Create knowledge graphs",
                "Break down complex topics"
            ],
            Intent.CHAT: [
                "Answer general questions",
                "Explain concepts with examples",
                "Provide learning guidance",
                "Engage in educational conversations"
            ]
        }
        
        return capabilities.get(intent, [])