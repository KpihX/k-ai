# src/k_ai/models.py
"""
Pydantic models for k-ai, ensuring data consistency and validation.
"""
import json
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum


class MessageRole(str, Enum):
    """Valid roles for a chat message, matching the OpenAI/LiteLLM convention."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A fully resolved tool call emitted by the model (non-streaming)."""

    id: str
    """Unique identifier assigned by the model for this call."""
    function_name: str
    """Name of the function/tool to invoke."""
    arguments: Dict[str, Any]
    """Decoded JSON arguments (always a dict, never a raw string)."""


class Message(BaseModel):
    """A single turn in the conversation history."""

    role: MessageRole
    content: str = ""
    name: Optional[str] = None
    """Optional name label (used for tool result messages)."""
    tool_call_id: Optional[str] = None
    """When role=tool, the id of the ToolCall this result is responding to."""
    tool_calls: Optional[List[ToolCall]] = None
    """When role=assistant and the model requested tool calls, they go here."""

    def to_litellm(self) -> Dict[str, Any]:
        """
        Serialize this message to the OpenAI/LiteLLM wire format.

        Pydantic's model_dump() does not produce the correct shape for
        tool-related messages (assistant requesting tools, or tool results).
        This method handles all four cases:
          - system / user / plain assistant → standard {"role": ..., "content": ...}
          - assistant with tool_calls       → adds "tool_calls" array in OpenAI format
          - tool result                     → adds "tool_call_id" and "name"
        """
        if self.tool_calls:
            # Assistant message that requested one or more tool calls.
            # The OpenAI spec requires "content" to be present (even as null)
            # when "tool_calls" is set — do NOT filter it out.
            return {
                "role": self.role.value,
                # null content is valid and expected by OpenAI-compatible APIs
                "content": self.content if self.content else None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function_name,
                            # LiteLLM expects arguments as a JSON string, not a dict.
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in self.tool_calls
                ],
            }

        d: Dict[str, Any] = {"role": self.role.value}

        if self.role == MessageRole.TOOL:
            # Tool result returned to the model after executing a function.
            d["content"] = self.content
            if self.tool_call_id:
                d["tool_call_id"] = self.tool_call_id
            if self.name:
                d["name"] = self.name
        else:
            # Standard system / user / assistant text message.
            d["content"] = self.content
            if self.name:
                d["name"] = self.name

        # Strip None values for non-tool-call messages (tool_calls handled above).
        return {k: v for k, v in d.items() if v is not None}


class TokenUsage(BaseModel):
    """Aggregated token counts for a single completion or an entire session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChunk(BaseModel):
    """
    A single streaming chunk from the LLM.

    Fields are additive deltas — the consumer accumulates them to build the
    full response.  At least one field will be non-empty per chunk.
    """

    delta_content: str = ""
    """Incremental visible text from the model."""
    delta_thought: Optional[str] = None
    """Incremental chain-of-thought text extracted from <think>…</think> tags."""
    tool_calls: Optional[List[ToolCall]] = None
    """Completed tool calls (populated only on the finish chunk, finish_reason='tool_calls')."""
    finish_reason: Optional[str] = None
    """Why the model stopped: 'stop', 'tool_calls', 'length', or None while streaming."""
    usage: Optional[TokenUsage] = None
    """Token counts — typically non-None only on the last chunk."""


class LLMConfig(BaseModel):
    """Per-call overrides for the global ConfigManager temperature/max_tokens settings."""

    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True


class UIStyle(str, Enum):
    """Display style for rendered responses (reserved for future use)."""

    CHAT = "chat"
    FANCY = "fancy"
    RAW = "raw"


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class SessionMetadata(BaseModel):
    """Persistent metadata for a single chat session."""

    id: str
    session_type: str = "classic"
    title: str = ""
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""
    provider: str = ""
    model: str = ""
    message_count: int = 0
    total_tokens: int = 0
    themes: List[str] = []


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    """A single entry in the internal memory store."""

    id: int
    text: str
    created_at: str = ""


# ---------------------------------------------------------------------------
# Tool system
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Result returned by an internal tool after execution."""
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    message: str
    data: Optional[Any] = None


class ToolProposal(BaseModel):
    """A tool call proposed by the LLM, pending user validation."""

    tool_name: str
    arguments: Dict[str, Any]
    justification: str = ""
