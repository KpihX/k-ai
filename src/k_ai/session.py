# src/k_ai/session.py
"""
Manages an interactive chat session with consciousness:
  - Session persistence (auto-save every message)
  - Boot flow (proactive greeting, session resume detection)
  - Agentic tool loop with human-in-the-loop confirmation
  - Context compaction when approaching the context window
  - Memory integration (external read-only + internal read-write)
"""
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.prompt import Confirm
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter, ConditionalCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style as PTStyle

from .config import ConfigManager
from .llm_core import get_provider, LLMProvider
from .models import (
    Message, MessageRole, ToolCall, TokenUsage, SessionMetadata, ToolResult,
)
from .memory import MemoryStore, load_external_memory
from .session_store import SessionStore
from .tools import create_registry, ToolRegistry
from .tools.base import ToolContext
from .ui import StreamingRenderer, render_sessions_table, render_tool_result
from .commands import CommandHandler, SLASH_COMMANDS
from .exceptions import LLMError, ProviderAuthenticationError, ContextLengthExceededError


# Max rounds in the agentic tool loop per user message
_MAX_TOOL_ROUNDS = 10


class ChatSession:
    """
    Encapsulates a complete interactive chat session.

    Can be used as a standalone CLI (via ``await session.start()``) or driven
    programmatically by calling ``await session.send(message)``.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.cm = config_manager
        self.console = Console()

        # LLM
        self.llm: LLMProvider = get_provider(self.cm, provider=provider, model=model)
        self.history: List[Message] = []
        self.system_prompt: Optional[str] = None
        self.total_usage = TokenUsage()

        # Persistence
        sessions_dir = self.cm.get_nested("sessions", "directory", default="~/.k-ai/sessions")
        self.session_store = SessionStore(sessions_dir)
        self.session_store.init()

        # Memory
        mem_path = self.cm.get_nested("memory", "internal_file", default="~/.k-ai/MEMORY.json")
        self.memory = MemoryStore(Path(mem_path))
        self.memory.load()

        ext_path = self.cm.get_nested("memory", "external_file", default="")
        self.external_memory = load_external_memory(ext_path)

        # Active session ID (set during boot or when creating/loading)
        self._session_id: Optional[str] = None

        # Session lifecycle flags (set by tools via callbacks)
        self._exit_requested: bool = False
        self._new_session_requested: bool = False
        self._load_session_id: Optional[str] = None
        self._compact_requested: bool = False

        # Tool system
        self._tool_ctx = ToolContext(
            config=self.cm,
            memory=self.memory,
            session_store=self.session_store,
            console=self.console,
            get_history=lambda: self.history,
            set_history=lambda h: setattr(self, "history", h),
            get_session_id=lambda: self._session_id,
            set_session_id=lambda sid: setattr(self, "_session_id", sid),
            get_system_prompt=lambda: self.system_prompt,
            reload_provider=lambda **kw: self.reload_provider(**kw),
            request_exit=lambda: setattr(self, "_exit_requested", True),
            request_new_session=self._handle_new_session,
            request_load_session=lambda sid: setattr(self, "_load_session_id", sid),
            request_compact=lambda: setattr(self, "_compact_requested", True),
        )
        self.tool_registry: ToolRegistry = create_registry(self._tool_ctx)
        self.command_handler = CommandHandler(self)

    # ------------------------------------------------------------------
    # Tool definitions (filter disabled tools)
    # ------------------------------------------------------------------

    def _get_active_tools(self) -> list:
        """Return OpenAI tool definitions excluding disabled tools."""
        disabled = set()
        tools_cfg = self.cm.get_nested("tools", default={})
        if isinstance(tools_cfg, dict):
            for name, cfg in tools_cfg.items():
                if isinstance(cfg, dict) and not cfg.get("enabled", True):
                    disabled.add(name)
        # Map config names to tool names (e.g. exa_search, python_exec)
        return [
            t.to_openai_tool() for t in self.tool_registry.list_tools()
            if t.name not in disabled
        ]

    @property
    def _debug(self) -> bool:
        return bool(self.cm.get_nested("cli", "debug", default=False))

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def reload_provider(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        effective_provider = provider or self.llm.provider_name
        self.llm = get_provider(self.cm, provider=effective_provider, model=model)

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, include_sessions: bool = False) -> str:
        parts: List[str] = []

        identity = self.cm.get_nested("prompts", "identity", default=(
            "You are k-ai, an intelligent CLI chat assistant."
        ))
        parts.append(identity)

        if self.external_memory:
            parts.append(f"## User Context\n{self.external_memory}")

        if self.memory.entries:
            entries_text = "\n".join(f"- {e.text}" for e in self.memory.entries)
            parts.append(f"## Remembered Facts\n{entries_text}")

        if include_sessions:
            max_recent = self.cm.get_nested("sessions", "max_recent", default=10)
            sessions = self.session_store.list_sessions(limit=max_recent)
            if sessions:
                lines = []
                for s in sessions:
                    title = s.title if s.title != s.id else "(untitled)"
                    summary_part = f" - {s.summary}" if s.summary else ""
                    lines.append(
                        f"- [{s.id[:8]}] \"{title}\"{summary_part} "
                        f"({s.message_count} msgs, {s.updated_at[:10]})"
                    )
                parts.append(
                    "## Recent Sessions\n"
                    + "\n".join(lines)
                    + "\n\nYou may suggest resuming a session if relevant."
                )

        if self.system_prompt:
            parts.append(f"## Custom Instructions\n{self.system_prompt}")

        return "\n\n".join(parts)

    def _messages_with_system(self, include_sessions: bool = False) -> List[Message]:
        system_text = self._build_system_prompt(include_sessions=include_sessions)
        return [Message(role=MessageRole.SYSTEM, content=system_text)] + self.history

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the interactive REPL with consciousness."""
        if self.cm.get_nested("cli", "show_welcome_panel", default=True):
            self._print_welcome_panel()

        max_recent = self.cm.get_nested("sessions", "max_recent", default=10)
        recent = self.session_store.list_sessions(limit=max_recent)
        if recent:
            render_sessions_table(self.console, recent)

        # Boot greeting (ephemeral)
        await self._boot_greeting(recent)

        # Build prompt session with styled prompt and slash autocompletion
        slash_completer = FuzzyCompleter(
            WordCompleter(SLASH_COMMANDS, sentence=True)
        )

        pt_style = PTStyle.from_dict({
            "prompt": "bold ansicyan",
        })

        prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            completer=slash_completer,
            complete_while_typing=True,
            reserve_space_for_menu=4,
            style=pt_style,
        )

        @Condition
        def _is_slash():
            buf = prompt_session.app.current_buffer
            return buf.text.lstrip().startswith("/")

        prompt_session.completer = ConditionalCompleter(slash_completer, filter=_is_slash)

        while True:
            if self._exit_requested:
                break

            if self._new_session_requested:
                self._new_session_requested = False
                self._do_new_session()
                continue

            if self._load_session_id:
                sid = self._load_session_id
                self._load_session_id = None
                self._do_load_session(sid)
                continue

            if self._compact_requested:
                self._compact_requested = False
                await self._do_compact()
                continue

            try:
                user_input = await prompt_session.prompt_async(
                    [("class:prompt", "You: ")]
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    result = await self.command_handler.handle(user_input)
                    if result is False:
                        break
                    continue

                first_message = not self._session_id
                if first_message:
                    self._do_new_session()

                await self._process_message(user_input)

                # Generate title after first real exchange
                if first_message and self._session_id and len(self.history) >= 2:
                    await self._auto_generate_title()

                await self._maybe_compact()

            except (EOFError, KeyboardInterrupt):
                break

        if self._session_id:
            await self._auto_rename_on_exit()
        self.console.print("\n[bold green]Goodbye![/bold green]")

    # ------------------------------------------------------------------
    # Boot greeting
    # ------------------------------------------------------------------

    async def _boot_greeting(self, recent: List[SessionMetadata]) -> None:
        if recent:
            boot_instruction = self.cm.get_nested(
                "prompts", "boot_with_sessions",
                default="[SESSION_BOOT] Greet the user and suggest resuming a session.",
            )
        else:
            boot_instruction = self.cm.get_nested(
                "prompts", "boot_no_sessions",
                default="[SESSION_BOOT] Greet the user warmly.",
            )

        boot_messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_system_prompt(include_sessions=True)),
            Message(role=MessageRole.USER, content=boot_instruction),
        ]

        try:
            tools = self._get_active_tools() if recent else None
            pending_tool_calls: List[ToolCall] = []

            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            with StreamingRenderer(self.console, self.llm.model_name, render_mode=render_mode) as renderer:
                async for chunk in self.llm.chat_stream(boot_messages, tools=tools):
                    renderer.update(chunk)
                    if chunk.tool_calls:
                        pending_tool_calls.extend(chunk.tool_calls)

            for tc in pending_tool_calls:
                if self.tool_registry.is_internal(tc.function_name):
                    await self._execute_internal_tool(tc)

        except Exception as e:
            self.console.print(f"[dim]Boot greeting skipped: {e}[/dim]")

    # ------------------------------------------------------------------
    # Agentic message processing loop
    # ------------------------------------------------------------------

    async def _process_message(self, user_input: str) -> None:
        """
        Full agentic loop: send user message, execute tools, loop back
        to the LLM until no more tool calls are pending.

        Flow:
          1. Append user message
          2. Call LLM with tools
          3. If tool_calls: execute each, append results, goto 2
          4. If no tool_calls: display final response, done
        """
        self.history.append(Message(role=MessageRole.USER, content=user_input))
        self._persist_message(self.history[-1])

        render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
        tools = self._get_active_tools()

        try:
            for _round in range(_MAX_TOOL_ROUNDS):
                messages = self._messages_with_system()

                # Debug: show raw prompt
                if self._debug:
                    self.console.print(RichPanel(
                        "\n".join(f"[{m.role.value}] {m.content[:200]}" for m in messages),
                        title="[dim]DEBUG: Prompt[/dim]",
                        border_style="dim",
                    ))

                # Stream LLM response
                pending_tool_calls: List[ToolCall] = []
                with StreamingRenderer(self.console, self.llm.model_name, render_mode=render_mode) as renderer:
                    async for chunk in self.llm.chat_stream(messages, tools=tools):
                        renderer.update(chunk)
                        if chunk.tool_calls:
                            pending_tool_calls.extend(chunk.tool_calls)
                    full_content = renderer.full_content

                # Accumulate usage
                if renderer.last_usage:
                    u = renderer.last_usage
                    self.total_usage.prompt_tokens += u.prompt_tokens
                    self.total_usage.completion_tokens += u.completion_tokens
                    self.total_usage.total_tokens += u.total_tokens
                    if self.cm.get_nested("cli", "show_token_usage", default=True):
                        self.console.print(
                            f"[dim]  {u.prompt_tokens} in / {u.completion_tokens} out[/dim]",
                            justify="right",
                        )

                # Record assistant message
                if full_content or pending_tool_calls:
                    msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                        tool_calls=pending_tool_calls if pending_tool_calls else None,
                    )
                    self.history.append(msg)
                    self._persist_message(msg)

                # No tool calls → done
                if not pending_tool_calls:
                    return

                # Execute tool calls and loop back
                any_executed = False
                for tc in pending_tool_calls:
                    if self.tool_registry.is_internal(tc.function_name):
                        result = await self._execute_internal_tool(tc)
                        tool_msg = Message(
                            role=MessageRole.TOOL,
                            content=result.message,
                            tool_call_id=tc.id,
                            name=tc.function_name,
                        )
                        self.history.append(tool_msg)
                        self._persist_message(tool_msg)
                        any_executed = True

                        # If a lifecycle action was triggered, stop the loop
                        if self._exit_requested or self._new_session_requested or self._load_session_id:
                            return

                if not any_executed:
                    return  # All tool calls were unknown — stop

                # Loop continues: LLM will see tool results and respond

        except ContextLengthExceededError:
            self.history.pop()
            self.console.print(
                "[bold red]Context length exceeded.[/bold red] "
                "Use [cyan]/compact[/cyan] or [cyan]/clear[/cyan]."
            )
        except ProviderAuthenticationError as e:
            self.history.pop()
            self.console.print(f"[bold red]Auth error:[/bold red] {e}")
        except LLMError as e:
            self.history.pop()
            self.console.print(f"[bold red]LLM error:[/bold red] {e}")

    # ------------------------------------------------------------------
    # Internal tool execution with inline confirmation
    # ------------------------------------------------------------------

    async def _execute_internal_tool(self, tc: ToolCall) -> ToolResult:
        """Execute an internal tool with inline human-in-the-loop."""
        tool = self.tool_registry.get(tc.function_name)
        if not tool:
            return ToolResult(success=False, message=f"Unknown tool: {tc.function_name}")

        # Human-in-the-loop: show tool call with proper rendering
        if tool.requires_approval:
            from rich.syntax import Syntax
            from rich.console import Group

            parts = []
            for k, v in (tc.arguments or {}).items():
                if k == "code" and isinstance(v, str) and len(v) > 40:
                    # Render code with syntax highlighting
                    lang = "python" if tc.function_name == "python_exec" else "bash"
                    parts.append(Syntax(v, lang, theme="monokai", line_numbers=True, word_wrap=True))
                elif k == "command" and isinstance(v, str):
                    parts.append(Syntax(v, "bash", theme="monokai", word_wrap=True))
                else:
                    parts.append(Text(f"{k}: {v}"))

            content = Group(*parts) if parts else Text(tc.function_name)
            self.console.print(RichPanel(
                content,
                title=f"[bold yellow]{tc.function_name}[/bold yellow]",
                border_style="yellow",
                expand=True,
                padding=(0, 1),
            ))
            approved = Confirm.ask("  Execute?", console=self.console, default=True)
            if not approved:
                self.console.print("  [dim]Skipped.[/dim]")
                return ToolResult(success=False, message="User rejected.")

        result = await tool.execute(tc.arguments, self._tool_ctx)
        max_len = int(self.cm.get_nested("cli", "tool_result_max_display", default=500))
        render_tool_result(self.console, tc.function_name, result, max_display_length=max_len)
        return result

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _auto_generate_title(self) -> None:
        """Generate a title for the session after the first real exchange (best-effort)."""
        if not self._session_id or len(self.history) < 2:
            return
        meta = self.session_store.get_session(self._session_id)
        if not meta or (meta.title and meta.title != meta.id):
            return
        try:
            first_user = next((m.content[:200] for m in self.history if m.role == MessageRole.USER), "")
            title_prompt = self.cm.get_nested(
                "prompts", "exit_title",
                default="Generate a short title (max 60 chars) for this chat.",
            )
            title = ""
            msgs = [
                Message(role=MessageRole.SYSTEM, content=title_prompt),
                Message(role=MessageRole.USER, content=first_user),
            ]
            async for chunk in self.llm.chat_stream(msgs):
                title += chunk.delta_content
            title = title.strip().strip('"').strip("'")[:60]
            if title:
                self.session_store.rename_session(self._session_id, title)
        except Exception:
            pass  # Best effort

    def _do_new_session(self) -> None:
        meta = self.session_store.create_session(
            provider=self.llm.provider_name,
            model=self.llm.model_name,
        )
        self._session_id = meta.id
        self.history = []

    def _do_load_session(self, session_id: str) -> None:
        meta = self.session_store.get_session(session_id)
        if not meta:
            self.console.print(f"[red]Session {session_id} not found.[/red]")
            return
        messages = self.session_store.load_messages(meta.id)
        self.history = messages
        self._session_id = meta.id
        self.console.print(
            f"[green]Resumed[/green] [bold]{meta.title}[/bold] "
            f"({len(messages)} messages)"
        )

        # Show the last N messages for context
        keep_n = int(self.cm.get_nested("compaction", "keep_last_n", default=10))
        recent = messages[-keep_n:] if len(messages) > keep_n else messages
        if recent:
            self.console.print("[dim]--- Recent history ---[/dim]")
            for m in recent:
                if m.role == MessageRole.SYSTEM:
                    continue
                role_style = "cyan" if m.role == MessageRole.USER else "green"
                label = "You" if m.role == MessageRole.USER else self.llm.model_name
                preview = m.content[:150].replace("\n", " ")
                if len(m.content) > 150:
                    preview += "..."
                self.console.print(f"  [{role_style}]{label}:[/{role_style}] {preview}")
            self.console.print("[dim]--- End of history ---[/dim]")

    def _persist_message(self, message: Message) -> None:
        if self._session_id:
            self.session_store.save_message(self._session_id, message)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        trigger_pct = self.cm.get_nested("compaction", "trigger_percent", default=80)
        if trigger_pct <= 0:
            return
        ctx_window = self.llm.provider_config.get("context_window", 128000)
        estimated_tokens = sum(len(m.content) for m in self.history) // 4
        threshold = int(ctx_window * trigger_pct / 100)
        if estimated_tokens > threshold:
            self.console.print("[dim]Context approaching limit, auto-compacting...[/dim]")
            await self._do_compact()

    async def _do_compact(self) -> None:
        keep_n = self.cm.get_nested("compaction", "keep_last_n", default=10)
        if len(self.history) <= keep_n:
            self.console.print("[dim]History too short to compact.[/dim]")
            return

        old_messages = self.history[:-keep_n]
        recent_messages = self.history[-keep_n:]

        compact_instruction = self.cm.get_nested(
            "prompts", "compact_summarize",
            default="Summarize the following conversation concisely.",
        )
        summary_prompt = compact_instruction + "\n\n"
        for m in old_messages:
            summary_prompt += f"[{m.role.value}]: {m.content[:500]}\n"

        try:
            summary = ""
            summary_msgs = [
                Message(role=MessageRole.SYSTEM, content=compact_instruction),
                Message(role=MessageRole.USER, content=summary_prompt),
            ]
            async for chunk in self.llm.chat_stream(summary_msgs):
                summary += chunk.delta_content

            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[Compacted context]\n{summary}",
            )
            self.history = [summary_msg] + recent_messages
            self.console.print(
                f"[green]Compacted:[/green] {len(old_messages)} messages summarized, "
                f"{keep_n} recent kept."
            )

            if self.cm.get_nested("compaction", "auto_rename", default=True):
                if self._session_id:
                    title = summary[:80].replace("\n", " ").strip()
                    if title:
                        self.session_store.rename_session(self._session_id, title)
                        self.session_store.update_summary(self._session_id, summary[:300])

        except Exception as e:
            self.console.print(f"[red]Compaction failed:[/red] {e}")

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def _auto_rename_on_exit(self) -> None:
        if not self._session_id or len(self.history) < 2:
            return

        meta = self.session_store.get_session(self._session_id)
        if not meta or (meta.title and meta.title != meta.id):
            return

        try:
            summary_input = "\n".join(
                f"[{m.role.value}]: {m.content[:200]}" for m in self.history[:10]
            )
            title_prompt = self.cm.get_nested(
                "prompts", "exit_title",
                default="Generate a short title (max 60 chars) for this chat.",
            )
            title = ""
            msgs = [
                Message(role=MessageRole.SYSTEM, content=title_prompt),
                Message(role=MessageRole.USER, content=summary_input),
            ]
            async for chunk in self.llm.chat_stream(msgs):
                title += chunk.delta_content

            title = title.strip().strip('"').strip("'")[:60]
            if title:
                self.session_store.rename_session(self._session_id, title)
                summary_prompt = self.cm.get_nested(
                    "prompts", "exit_summary",
                    default="Summarize this chat in 2-3 sentences.",
                )
                summary = ""
                msgs2 = [
                    Message(role=MessageRole.SYSTEM, content=summary_prompt),
                    Message(role=MessageRole.USER, content=summary_input),
                ]
                async for chunk in self.llm.chat_stream(msgs2):
                    summary += chunk.delta_content
                if summary.strip():
                    self.session_store.update_summary(self._session_id, summary.strip()[:500])

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Programmatic API
    # ------------------------------------------------------------------

    async def send(self, message: str) -> str:
        self.history.append(Message(role=MessageRole.USER, content=message))
        messages = self._messages_with_system()
        full_content = ""
        async for chunk in self.llm.chat_stream(messages):
            full_content += chunk.delta_content
        self.history.append(Message(role=MessageRole.ASSISTANT, content=full_content))
        return full_content

    async def send_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[ToolCall], Awaitable[str]],
        max_tool_rounds: int = 10,
    ) -> str:
        self.history.append(Message(role=MessageRole.USER, content=message))
        for _round in range(max_tool_rounds):
            messages = self._messages_with_system()
            full_content = ""
            pending_tool_calls: List[ToolCall] = []
            async for chunk in self.llm.chat_stream(messages, tools=tools):
                full_content += chunk.delta_content
                if chunk.tool_calls:
                    pending_tool_calls.extend(chunk.tool_calls)
            if not pending_tool_calls:
                self.history.append(Message(role=MessageRole.ASSISTANT, content=full_content))
                return full_content
            self.history.append(Message(
                role=MessageRole.ASSISTANT, content=full_content,
                tool_calls=pending_tool_calls,
            ))
            for tc in pending_tool_calls:
                try:
                    result = await tool_executor(tc)
                except Exception as exc:
                    result = f"Error executing tool '{tc.function_name}': {exc}"
                self.history.append(Message(
                    role=MessageRole.TOOL, content=result,
                    tool_call_id=tc.id, name=tc.function_name,
                ))
        return full_content  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_new_session(self) -> None:
        self._new_session_requested = True

    def _print_welcome_panel(self) -> None:
        from rich.panel import Panel
        from rich.text import Text

        title = Text("k-ai  |  Unified LLM Chat", style="bold cyan", justify="center")
        body = Text.from_markup(
            f"[dim]Provider :[/dim] [bold white]{self.llm.provider_name}[/bold white]\n"
            f"[dim]Model    :[/dim] [bold white]{self.llm.model_name}[/bold white]\n"
            f"[dim]Temp     :[/dim] [bold white]{self.cm.get('temperature')}[/bold white]  "
            f"[dim]Max tokens:[/dim] [bold white]{self.cm.get('max_tokens')}[/bold white]"
        )
        self.console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))
        self.console.print("[dim]Type /help for commands, or just chat.[/dim]")
