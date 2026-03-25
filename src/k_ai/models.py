# src/k_ai/models.py
"""
Pydantic models for k-ai, ensuring data consistency and validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class ToolCall(BaseModel):
    id: str
    function_name: str
    arguments: Dict[str, Any]

class Message(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class CompletionChunk(BaseModel):
    delta_content: str = ""
    delta_thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None

class LLMConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True

class UIStyle(str, Enum):
    CHAT = "chat"
    FANCY = "fancy"
    RAW = "raw"
