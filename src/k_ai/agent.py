# src/k_ai/agent.py
"""
LangGraph-powered agentic loop for k-ai.

Implements a state machine:
    User Message → LLM → [tool_calls?] → Execute Tools → LLM → ... → Final Response

The graph handles:
  - Streaming LLM responses
  - Internal tool execution with human-in-the-loop
  - Automatic looping until no more tool calls
  - Session lifecycle actions (exit, new, load)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import StateGraph, END

from .models import Message, MessageRole, ToolCall, CompletionChunk, ToolResult, TokenUsage
from .tools.base import ToolRegistry, ToolContext


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Mutable state flowing through the LangGraph nodes."""
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    final_content: str = ""
    usage: Optional[TokenUsage] = None
    round: int = 0
    max_rounds: int = 10
    done: bool = False
    # Lifecycle signals from tools
    exit_requested: bool = False
    new_session_requested: bool = False
    load_session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def call_llm(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Call the LLM with current messages and tools. Stream the response."""
    llm = config["llm"]
    tool_registry: ToolRegistry = config["tool_registry"]
    on_chunk = config.get("on_chunk")  # callback for streaming UI

    all_messages = [Message(role=MessageRole.SYSTEM, content=state.system_prompt)] + state.messages
    tools = tool_registry.to_openai_tools()

    content = ""
    pending: List[ToolCall] = []
    usage = None

    async for chunk in llm.chat_stream(all_messages, tools=tools):
        if on_chunk:
            on_chunk(chunk)
        content += chunk.delta_content or ""
        if chunk.tool_calls:
            pending.extend(chunk.tool_calls)
        if chunk.usage:
            usage = chunk.usage

    # Record assistant message
    msg = Message(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=pending if pending else None,
    )
    state.messages.append(msg)
    state.pending_tool_calls = pending
    state.final_content = content
    state.usage = usage
    state.round += 1

    return state


async def execute_tools(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Execute all pending internal tool calls with human-in-the-loop."""
    tool_registry: ToolRegistry = config["tool_registry"]
    tool_ctx: ToolContext = config["tool_ctx"]
    confirm_fn = config.get("confirm_tool")  # (name, args) -> bool
    on_tool_result = config.get("on_tool_result")  # (name, result) -> None

    state.tool_results = []

    for tc in state.pending_tool_calls:
        tool = tool_registry.get(tc.function_name)
        if not tool:
            result = ToolResult(success=False, message=f"Unknown tool: {tc.function_name}")
        else:
            # Human-in-the-loop
            if tool.requires_approval and confirm_fn:
                approved = confirm_fn(tc.function_name, tc.arguments)
                if not approved:
                    result = ToolResult(success=False, message="User rejected.")
                else:
                    result = await tool.execute(tc.arguments, tool_ctx)
            else:
                result = await tool.execute(tc.arguments, tool_ctx)

        if on_tool_result:
            on_tool_result(tc.function_name, result)

        state.tool_results.append(result)

        # Append tool result message
        state.messages.append(Message(
            role=MessageRole.TOOL,
            content=result.message,
            tool_call_id=tc.id,
            name=tc.function_name,
        ))

        # Check lifecycle signals
        if tool_ctx.request_exit and hasattr(tool_ctx, '_exit_check'):
            pass  # Handled via state flags set by tool callbacks

    state.pending_tool_calls = []
    return state


def should_continue(state: AgentState) -> str:
    """Router: continue to tools or end."""
    if state.done or state.exit_requested or state.new_session_requested or state.load_session_id:
        return END
    if not state.pending_tool_calls:
        return END
    if state.round >= state.max_rounds:
        return END
    return "execute_tools"


def after_tools(state: AgentState) -> str:
    """After tool execution: loop back to LLM or end."""
    if state.done or state.exit_requested or state.new_session_requested or state.load_session_id:
        return END
    if state.round >= state.max_rounds:
        return END
    return "call_llm"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph state machine for the agentic loop.

    Graph:
        call_llm → [tool_calls?] → execute_tools → call_llm → ... → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)

    graph.set_entry_point("call_llm")

    graph.add_conditional_edges("call_llm", should_continue, {
        "execute_tools": "execute_tools",
        END: END,
    })

    graph.add_conditional_edges("execute_tools", after_tools, {
        "call_llm": "call_llm",
        END: END,
    })

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

async def run_agent_loop(
    messages: List[Message],
    system_prompt: str,
    llm: Any,
    tool_registry: ToolRegistry,
    tool_ctx: ToolContext,
    on_chunk=None,
    confirm_tool=None,
    on_tool_result=None,
    max_rounds: int = 10,
) -> AgentState:
    """
    Run the full agentic loop and return the final state.

    Args:
        messages: Current conversation history (mutated in place).
        system_prompt: The system prompt text.
        llm: The LLMProvider instance.
        tool_registry: Registry of internal tools.
        tool_ctx: Context passed to tools.
        on_chunk: Callback for each streaming chunk.
        confirm_tool: Callback (name, args) -> bool for human-in-the-loop.
        on_tool_result: Callback (name, result) for displaying results.
        max_rounds: Safety limit.

    Returns:
        Final AgentState with all messages and results.
    """
    graph = build_agent_graph()

    initial_state = AgentState(
        messages=messages,
        system_prompt=system_prompt,
        max_rounds=max_rounds,
    )

    config = {
        "llm": llm,
        "tool_registry": tool_registry,
        "tool_ctx": tool_ctx,
        "on_chunk": on_chunk,
        "confirm_tool": confirm_tool,
        "on_tool_result": on_tool_result,
    }

    result = await graph.ainvoke(initial_state, config={"configurable": config})
    return result
