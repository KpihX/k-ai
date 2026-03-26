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
from contextlib import contextmanager
import json
import os
import select
import signal
import sys
import termios
import threading
import tty
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
from .ui import (
    StreamingRenderer,
    render_assistant_panel,
    render_notice,
    render_runtime_panel,
    render_sessions_table,
    render_tool_proposal,
    render_tool_result,
    render_user_panel,
)
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

        initial_model = model if model is not None else (self.cm.get("model") or None)
        self.llm: LLMProvider = get_provider(self.cm, provider=provider, model=initial_model)
        self.history: List[Message] = []
        self.system_prompt: Optional[str] = None
        self.total_usage = TokenUsage()

        sessions_dir = self.cm.get_nested("sessions", "directory", default="~/.k-ai/sessions")
        self.session_store = SessionStore(sessions_dir)
        self.session_store.init()

        mem_path = self.cm.get_nested("memory", "internal_file", default="~/.k-ai/MEMORY.json")
        self.memory = MemoryStore(Path(mem_path))
        self.memory.load()

        ext_path = self.cm.get_nested("memory", "external_file", default="")
        self.external_memory = load_external_memory(ext_path)

        self._session_id: Optional[str] = None
        self._exit_requested: bool = False
        self._new_session_requested: bool = False
        self._load_session_id: Optional[str] = None
        self._load_session_last_n: Optional[int] = None
        self._compact_requested: bool = False
        self._interrupt_requested: bool = False
        self._prompt_interrupt_count: int = 0

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
            request_load_session=self._queue_load_session,
            request_compact=lambda: setattr(self, "_compact_requested", True),
            apply_config_change=self.apply_config_change,
            generate_session_digest=self.generate_session_digest,
            get_runtime_snapshot=self.get_runtime_snapshot,
            is_interrupt_requested=lambda: self._interrupt_requested,
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
        effective_model = model if model is not None else (self.cm.get("model") or None)
        self.llm = get_provider(self.cm, provider=effective_provider, model=effective_model)

    def _estimate_context_tokens(self, messages: Optional[List[Message]] = None) -> int:
        payload = messages if messages is not None else self.history
        return sum(len(m.content) for m in payload) // 4

    def _assistant_completion_estimate(self) -> int:
        return sum(len(m.content) for m in self.history if m.role == MessageRole.ASSISTANT) // 4

    def get_token_snapshot(self) -> Dict[str, Any]:
        actual_prompt = int(self.total_usage.prompt_tokens or 0)
        actual_completion = int(self.total_usage.completion_tokens or 0)
        actual_total = int(self.total_usage.total_tokens or 0)
        estimated_prompt = self._estimate_context_tokens(self._messages_with_system())
        estimated_completion = self._assistant_completion_estimate()
        estimated_total = self._estimate_context_tokens(self.history)

        if actual_total > 0:
            return {
                "prompt_tokens": actual_prompt,
                "completion_tokens": actual_completion,
                "total_tokens": actual_total,
                "token_source": "provider",
            }
        return {
            "prompt_tokens": estimated_prompt,
            "completion_tokens": estimated_completion,
            "total_tokens": estimated_total,
            "token_source": "estimated",
        }

    def _sync_session_totals(self) -> None:
        if not self._session_id:
            return
        snapshot = self.get_token_snapshot()
        self.session_store.update_meta(
            self._session_id,
            total_tokens=int(snapshot.get("total_tokens", 0) or 0),
        )

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        ctx_window = int(self.llm.provider_config.get("context_window", 128000) or 128000)
        used = self._estimate_context_tokens()
        compaction_pct = int(self.cm.get_nested("compaction", "trigger_percent", default=80) or 0)
        threshold = int(ctx_window * compaction_pct / 100) if compaction_pct > 0 else 0
        remaining = max(ctx_window - used, 0)
        percent = round((used / ctx_window) * 100, 1) if ctx_window else 0.0
        session_meta = self.session_store.get_session(self._session_id) if self._session_id else None
        token_snapshot = self.get_token_snapshot()
        persist_path = (
            str(self.cm.override_path.expanduser())
            if self.cm.override_path
            else str(Path(self.cm.get_nested("config", "persist_path", default="~/.k-ai/config.yaml")).expanduser())
        )
        return {
            "provider": self.llm.provider_name,
            "model": self.llm.model_name,
            "auth_mode": self.llm.auth_mode or "n/a",
            "temperature": self.cm.get("temperature"),
            "max_tokens": self.cm.get("max_tokens"),
            "stream": self.cm.get("stream"),
            "render_mode": self.cm.get_nested("cli", "render_mode", default="rich"),
            "session_id": self._session_id or "",
            "session_summary": session_meta.summary if session_meta else "",
            "session_themes": list(session_meta.themes) if session_meta else [],
            "history_messages": len(self.history),
            "context_window": ctx_window,
            "estimated_context_tokens": used,
            "remaining_context_tokens": remaining,
            "context_percent": percent,
            "compaction_trigger_percent": compaction_pct,
            "compaction_trigger_tokens": threshold,
            "tool_result_max_display": self.cm.get_nested("cli", "tool_result_max_display", default=500),
            "tool_result_max_history": self.cm.get_nested("cli", "tool_result_max_history", default=4000),
            "confirm_all_tools": self.cm.get_nested("cli", "confirm_all_tools", default=True),
            "show_runtime_stats": self.cm.get_nested("cli", "show_runtime_stats", default=True),
            "runtime_stats_mode": self.cm.get_nested("cli", "runtime_stats_mode", default="compact"),
            "prompt_tokens": token_snapshot["prompt_tokens"],
            "completion_tokens": token_snapshot["completion_tokens"],
            "total_tokens": token_snapshot["total_tokens"],
            "token_source": token_snapshot["token_source"],
            "provider_prompt_tokens": self.total_usage.prompt_tokens,
            "provider_completion_tokens": self.total_usage.completion_tokens,
            "provider_total_tokens": self.total_usage.total_tokens,
            "persist_path": persist_path,
        }

    def apply_config_change(self, key: str, value: Any, persist: bool = False) -> Dict[str, Any]:
        current = self.cm.get_path(key)
        applied = self.cm.set(key, value)
        saved_to: Optional[str] = None
        try:
            if key in {"provider", "model"}:
                provider_name = self.cm.get("provider")
                model_name = self.cm.get("model") or None
                self.reload_provider(provider=provider_name, model=model_name)
            if persist:
                saved_to = str(self.cm.save_active_yaml())
        except Exception:
            self.cm.set(key, current)
            if key in {"provider", "model"}:
                self.reload_provider(
                    provider=self.cm.get("provider"),
                    model=self.cm.get("model") or None,
                )
            raise
        return {"key": key, "old_value": current, "value": applied, "saved_to": saved_to}

    def save_active_config(self, path: Optional[str] = None) -> str:
        return str(self.cm.save_active_yaml(path))

    def _print_runtime_snapshot(self, title: str = "Runtime Transparency") -> None:
        if not self.cm.get_nested("cli", "show_runtime_stats", default=True):
            return
        mode = self.cm.get_nested("cli", "runtime_stats_mode", default="compact")
        self.console.print(render_runtime_panel(self.get_runtime_snapshot(), title=title, mode=mode))

    @contextmanager
    def _interrupt_scope(self, allow_escape: bool = True):
        self._interrupt_requested = False
        stop_event = threading.Event()
        watcher: Optional[threading.Thread] = None
        fd: Optional[int] = None
        old_attrs = None
        old_handler = None

        def _trigger_interrupt() -> None:
            self._interrupt_requested = True

        if threading.current_thread() is threading.main_thread():
            old_handler = signal.getsignal(signal.SIGINT)

            def _handler(signum, frame):
                _trigger_interrupt()

            signal.signal(signal.SIGINT, _handler)

        if allow_escape and sys.stdin.isatty():
            try:
                fd = sys.stdin.fileno()
                old_attrs = termios.tcgetattr(fd)
                tty.setcbreak(fd)

                def _watch_input() -> None:
                    while not stop_event.is_set():
                        ready, _, _ = select.select([fd], [], [], 0.1)
                        if not ready:
                            continue
                        char = os.read(fd, 1)
                        if char in (b"\x1b", b"\x03"):
                            _trigger_interrupt()
                            break

                watcher = threading.Thread(target=_watch_input, daemon=True)
                watcher.start()
            except Exception:
                fd = None
                old_attrs = None

        try:
            yield
        finally:
            stop_event.set()
            if watcher and watcher.is_alive():
                watcher.join(timeout=0.2)
            if fd is not None and old_attrs is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                except Exception:
                    pass
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)

    def _handle_prompt_interrupt(self) -> bool:
        self._prompt_interrupt_count += 1
        if self._prompt_interrupt_count >= 2:
            return True
        render_notice(
            self.console,
            "Saisie interrompue. Appuyez une seconde fois sur Ctrl+C pour quitter, ou continuez à écrire.",
            level="warning",
            title="Interruption",
        )
        return False

    def _reset_prompt_interrupts(self) -> None:
        self._prompt_interrupt_count = 0

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
                    summary = s.summary or (s.title if s.title != s.id else "(untitled)")
                    themes = f" | themes: {', '.join(s.themes[:4])}" if s.themes else ""
                    lines.append(
                        f"- [{s.id[:8]}] \"{summary}\"{themes} "
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

    def _boot_tools(self) -> Optional[List[Dict[str, Any]]]:
        tools = self._get_active_tools()
        boot_tools = [
            tool for tool in tools
            if tool.get("function", {}).get("name") == "load_session"
        ]
        return boot_tools or None

    def _tool_rationale(self, assistant_content: str) -> str:
        """Extract a concise justification from assistant text for the tool proposal."""
        if not assistant_content.strip():
            return ""
        lines = []
        for raw_line in assistant_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```") or line.startswith("import ") or line.startswith("from "):
                break
            if line.startswith("#") or line.startswith("for ") or line.startswith("while "):
                break
            if line.startswith("Je vais exécuter"):
                lines.append(line)
                break
            if len(line) > 180:
                line = line[:177] + "..."
            lines.append(line)
            if len(lines) >= 2:
                break
        return " ".join(lines).strip()

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
                last_n = self._load_session_last_n
                self._load_session_id = None
                self._load_session_last_n = None
                self._do_load_session(sid, last_n=last_n)
                continue

            if self._compact_requested:
                self._compact_requested = False
                await self._do_compact()
                continue

            try:
                user_input = await prompt_session.prompt_async(
                    [("class:prompt", "You: ")]
                )
                self._reset_prompt_interrupts()
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

                self.console.print(render_user_panel(user_input))
                await self._process_message(user_input)

                if first_message and self._session_id and len(self.history) >= 2:
                    await self._auto_generate_session_summary()

                await self._maybe_compact()

            except EOFError:
                break
            except KeyboardInterrupt:
                if self._handle_prompt_interrupt():
                    break
                continue

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
            tools = self._boot_tools() if recent else None
            pending_tool_calls: List[ToolCall] = []

            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            with self._interrupt_scope(allow_escape=True):
                with StreamingRenderer(self.console, self.llm.model_name, render_mode=render_mode) as renderer:
                    async for chunk in self.llm.chat_stream(boot_messages, tools=tools):
                        if self._interrupt_requested:
                            raise KeyboardInterrupt
                        renderer.update(chunk)
                        if chunk.tool_calls:
                            pending_tool_calls.extend(chunk.tool_calls)

            for tc in pending_tool_calls:
                if self.tool_registry.is_internal(tc.function_name):
                    await self._execute_internal_tool(tc)

        except KeyboardInterrupt:
            render_notice(self.console, "Boot interrompu. Retour au prompt.", level="warning", title="Interruption")
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
        turn_start = len(self.history)
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

                pending_tool_calls: List[ToolCall] = []
                full_content = ""
                with self._interrupt_scope(allow_escape=True):
                    with StreamingRenderer(self.console, self.llm.model_name, render_mode=render_mode) as renderer:
                        async for chunk in self.llm.chat_stream(messages, tools=tools):
                            if self._interrupt_requested:
                                raise KeyboardInterrupt
                            renderer.update(chunk)
                            if chunk.tool_calls:
                                pending_tool_calls.extend(chunk.tool_calls)
                        full_content = renderer.full_content

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

                if full_content or pending_tool_calls:
                    msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                        tool_calls=pending_tool_calls if pending_tool_calls else None,
                    )
                    self.history.append(msg)
                    self._persist_message(msg)

                if not pending_tool_calls:
                    self._sync_session_totals()
                    self._print_runtime_snapshot()
                    return

                rationale = self._tool_rationale(full_content)
                any_executed = False
                for tc in pending_tool_calls:
                    if self.tool_registry.is_internal(tc.function_name):
                        result = await self._execute_internal_tool(tc, rationale=rationale)
                        if result.data and isinstance(result.data, dict) and result.data.get("interrupted"):
                            self._rollback_turn(turn_start)
                            render_notice(self.console, "Action interrompue. Retour au prompt.", level="warning", title="Interruption")
                            return
                        tool_content = self._normalize_tool_result_for_history(result.message)
                        tool_msg = Message(
                            role=MessageRole.TOOL,
                            content=tool_content,
                            tool_call_id=tc.id,
                            name=tc.function_name,
                        )
                        self.history.append(tool_msg)
                        self._persist_message(tool_msg)
                        any_executed = True

                        if self._exit_requested or self._new_session_requested or self._load_session_id:
                            return

                if not any_executed:
                    return

        except ContextLengthExceededError:
            self._rollback_turn(turn_start)
            self.console.print(
                "[bold red]Context length exceeded.[/bold red] "
                "Use [cyan]/compact[/cyan] or [cyan]/clear[/cyan]."
            )
        except KeyboardInterrupt:
            self._rollback_turn(turn_start)
            render_notice(self.console, "Génération interrompue. La main vous est rendue.", level="warning", title="Interruption")
        except ProviderAuthenticationError as e:
            self._rollback_turn(turn_start)
            self.console.print(f"[bold red]Auth error:[/bold red] {e}")
        except LLMError as e:
            self._rollback_turn(turn_start)
            self.console.print(f"[bold red]LLM error:[/bold red] {e}")

    # ------------------------------------------------------------------
    # Internal tool execution with inline confirmation
    # ------------------------------------------------------------------

    async def _execute_internal_tool(self, tc: ToolCall, rationale: str = "") -> ToolResult:
        """Execute an internal tool with inline human-in-the-loop."""
        tool = self.tool_registry.get(tc.function_name)
        if not tool:
            return ToolResult(success=False, message=f"Unknown tool: {tc.function_name}")

        confirm_all_tools = bool(self.cm.get_nested("cli", "confirm_all_tools", default=True))
        needs_confirmation = confirm_all_tools or tool.requires_approval

        render_tool_proposal(
            self.console,
            tool.display_spec(),
            tool.proposal_sections(tc.arguments or {}, self._tool_ctx),
            rationale=rationale,
            requires_approval=needs_confirmation,
        )

        if needs_confirmation:
            try:
                approved = Confirm.ask("Approve tool execution?", console=self.console, default=True)
            except KeyboardInterrupt:
                return ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
            if not approved:
                self.console.print("  [dim]Skipped.[/dim]")
                return ToolResult(success=False, message="User rejected.")

        try:
            with self._interrupt_scope(allow_escape=True):
                result = await tool.execute(tc.arguments, self._tool_ctx)
                if self._interrupt_requested:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            return ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
        max_len = int(self.cm.get_nested("cli", "tool_result_max_display", default=500))
        render_tool_result(
            self.console,
            tool.display_spec(),
            result,
            tool.result_renderable(result, max_display_length=max_len, ctx=self._tool_ctx),
        )
        if result.success and tool.category == "config":
            self._print_runtime_snapshot(title="Runtime Updated")
        return result

    def _normalize_tool_result_for_history(self, content: str) -> str:
        max_len = int(self.cm.get_nested("cli", "tool_result_max_history", default=4000))
        if len(content) <= max_len:
            return content
        return content[:max_len] + "\n...(truncated for history)"

    def _rollback_turn(self, turn_start: int) -> None:
        """Rollback any in-memory and persisted messages added during the current turn."""
        if turn_start < 0 or turn_start > len(self.history):
            return
        self.history = self.history[:turn_start]
        if self._session_id:
            self.session_store.rewrite_messages(self._session_id, self.history)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _auto_generate_session_summary(self) -> None:
        """Generate a one-line summary after the first real exchange (best-effort)."""
        if not self._session_id or len(self.history) < 2:
            return
        meta = self.session_store.get_session(self._session_id)
        if not meta or meta.summary:
            return
        try:
            digest = await self.generate_session_digest(self._session_id, persist=True)
            if digest.get("summary"):
                self.session_store.update_digest(self._session_id, digest["summary"], digest.get("themes", []))
        except Exception:
            pass  # Best effort

    def _do_new_session(self) -> None:
        meta = self.session_store.create_session(
            provider=self.llm.provider_name,
            model=self.llm.model_name,
        )
        self._session_id = meta.id
        self.history = []

    def _do_load_session(self, session_id: str, last_n: Optional[int] = None) -> None:
        meta = self.session_store.get_session(session_id)
        if not meta:
            render_notice(self.console, f"Session {session_id} not found.", level="error")
            return
        load_default = int(self.cm.get_nested("sessions", "load_last_n", default=0))
        effective_last_n = last_n if last_n is not None else (load_default if load_default > 0 else None)
        messages = self.session_store.load_messages(meta.id, last_n=effective_last_n)
        self.history = messages
        self._session_id = meta.id
        render_notice(
            self.console,
            f"Session [bold]{meta.summary or meta.title}[/bold] resumed with {len(messages)} loaded messages.",
            level="success",
            title="Session Resumed",
        )

        # Show the last N messages with proper rendering (same as live chat)
        keep_n = int(self.cm.get_nested("sessions", "preview_last_n", default=10))
        recent = messages[-keep_n:] if len(messages) > keep_n else messages
        if recent:
            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            from .ui.markdown import render_content
            skipped = len(messages) - len(recent)
            if skipped > 0:
                self.console.print(f"[dim]  ({skipped} older messages not shown)[/dim]")
            for m in recent:
                if m.role == MessageRole.SYSTEM:
                    continue
                elif m.role == MessageRole.USER:
                    self.console.print(render_user_panel(m.content))
                elif m.role == MessageRole.ASSISTANT:
                    content_renderable = render_content(m.content, render_mode) if m.content else Text("[dim](empty)[/dim]")
                    if m.content:
                        self.console.print(render_assistant_panel(m.content, self.llm.model_name, render_mode=render_mode))
                    else:
                        self.console.print(RichPanel(
                            content_renderable,
                            title=f"[bold green]Assistant[/bold green] [dim]{self.llm.model_name}[/dim]",
                            border_style="green",
                        ))
                elif m.role == MessageRole.TOOL:
                    name = m.name or "tool"
                    border = "green"
                    self.console.print(RichPanel(
                        m.content[:300] + ("..." if len(m.content) > 300 else ""),
                        title=f"[bold {border}]Agent[/bold {border}] [dim]{name}[/dim]",
                        border_style=border,
                        expand=False,
                        padding=(0, 1),
                    ))

    def _persist_message(self, message: Message) -> None:
        if self._session_id:
            self.session_store.save_message(self._session_id, message)

    def _queue_load_session(self, session_id: str, last_n: Optional[int] = None) -> None:
        self._load_session_id = session_id
        self._load_session_last_n = last_n

    def reset_history(self) -> None:
        """Clear the active in-memory history and persist the cleared state."""
        self.history = []
        if self._session_id:
            self.session_store.rewrite_messages(self._session_id, [])

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        trigger_pct = self.cm.get_nested("compaction", "trigger_percent", default=80)
        if trigger_pct <= 0:
            return
        ctx_window = self.llm.provider_config.get("context_window", 128000)
        estimated_tokens = self._estimate_context_tokens()
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
            if self._session_id:
                self.session_store.rewrite_messages(self._session_id, self.history)
            self.console.print(
                f"[green]Compacted:[/green] {len(old_messages)} messages summarized, "
                f"{keep_n} recent kept."
            )

            if self.cm.get_nested("compaction", "auto_rename", default=True):
                if self._session_id:
                    digest = await self.generate_session_digest(self._session_id, persist=True)
                    if digest.get("summary"):
                        self.session_store.update_digest(self._session_id, digest["summary"], digest.get("themes", []))

        except Exception as e:
            self.console.print(f"[red]Compaction failed:[/red] {e}")

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def _auto_rename_on_exit(self) -> None:
        if not self._session_id or len(self.history) < 2:
            return

        meta = self.session_store.get_session(self._session_id)
        if not meta:
            return

        try:
            digest = await self.generate_session_digest(self._session_id, persist=True)
            if digest.get("summary"):
                self.session_store.update_digest(self._session_id, digest["summary"], digest.get("themes", []))

        except Exception:
            pass

    async def generate_session_digest(
        self,
        session_id: Optional[str] = None,
        persist: bool = False,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a digest sentence and key themes for a session."""
        sid = session_id or self._session_id
        if not sid:
            return {"summary": "", "themes": []}

        messages = self.session_store.load_messages(sid, offset=offset, limit=limit)
        if not messages:
            return {"summary": "", "themes": []}

        digest_prompt = self.cm.get_nested(
            "prompts", "session_digest",
            default=(
                "Return strict JSON with keys summary and themes. "
                "summary must be one short sentence describing the discussion. "
                "themes must be a short list of key topics."
            ),
        )
        transcript = "\n".join(
            f"[{m.role.value}] {m.content[:300]}"
            for m in messages
            if m.role != MessageRole.SYSTEM
        )
        raw = ""
        msgs = [
            Message(role=MessageRole.SYSTEM, content=digest_prompt),
            Message(role=MessageRole.USER, content=transcript[:6000]),
        ]
        async for chunk in self.llm.chat_stream(msgs):
            raw += chunk.delta_content

        summary = ""
        themes: List[str] = []
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            payload = json.loads(raw[start:end + 1] if start != -1 and end != -1 else raw)
            summary = str(payload.get("summary", "")).strip().replace("\n", " ")[:120]
            raw_themes = payload.get("themes", [])
            if isinstance(raw_themes, list):
                themes = [str(theme).strip()[:40] for theme in raw_themes if str(theme).strip()][:8]
        except Exception:
            summary = raw.strip().replace("\n", " ")[:120]
            themes = []

        digest = {"summary": summary, "themes": themes}
        if persist and summary:
            self.session_store.update_digest(sid, summary, themes)
        return digest

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
        self.console.print(render_runtime_panel(self.get_runtime_snapshot(), title="k-ai  |  Unified LLM Chat", mode="welcome"))
        self.console.print("[dim]Type /help for commands, or just chat.[/dim]")
