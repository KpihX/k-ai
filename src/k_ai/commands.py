# src/k_ai/commands.py
"""
Handles /slash commands for the interactive chat.

Every command either maps directly to an internal tool (same action the LLM
can propose) or is a UI-only shortcut.  This ensures uniform behaviour:
``/compact`` and "please compact the discussion" do the exact same thing.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List

from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from .config import ConfigManager
from .ui_theme import resolve_syntax_theme

if TYPE_CHECKING:
    from .session import ChatSession


# ---------------------------------------------------------------------------
# Slash commands for autocompletion (prompt-toolkit)
# ---------------------------------------------------------------------------

SLASH_COMMANDS = [
    "/help", "/exit", "/quit", "/bye",
    "/new", "/init", "/load", "/sessions", "/rename", "/delete",
    "/compact", "/clear", "/reset",
    "/digest", "/extract",
    "/history", "/model", "/provider", "/system",
    "/cwd", "/focus",
    "/set", "/settings", "/status",
    "/tools", "/tools capabilities", "/tools enable", "/tools disable",
    "/config show", "/config save", "/config get", "/config sections", "/config edit",
    "/save", "/tokens",
    "/memory list", "/memory add", "/memory remove",
    "/skills", "/skills list", "/skills show", "/skills reload", "/skills active",
    "/hooks", "/hooks list", "/hooks reload",
    "/mcp", "/mcp list", "/mcp tools", "/mcp resources", "/mcp templates", "/mcp prompts", "/mcp reload",
    "/mcp probe", "/mcp install", "/mcp add-stdio", "/mcp add-http", "/mcp enable", "/mcp disable", "/mcp remove",
    "/doctor", "/doctor reset",
    "/qmd query", "/qmd search", "/qmd vsearch",
    "/qmd get", "/qmd ls", "/qmd collections",
    "/qmd status", "/qmd update", "/qmd embed", "/qmd cleanup",
    "/sandbox list", "/sandbox add", "/sandbox remove",
]


# ---------------------------------------------------------------------------
# Help registry
# ---------------------------------------------------------------------------

_HELP: dict[str, tuple[str, str, str]] = {
    "/help": ("Show this help table.", "-", "/help"),
    "/exit, /quit, /bye": ("Exit the interactive session safely.", "-", "/exit"),
    "/new [classic|meta]": (
        "Start a fresh session. Use [cyan]classic[/cyan] for a real task/topic, [cyan]meta[/cyan] for admin/config work.",
        "type=classic|meta",
        "/new meta",
    ),
    "/init": (
        "Start or restart onboarding: choose the assistant name, choose how you want to be addressed, then receive a guided capability overview.",
        "-",
        "/init",
    ),
    "/load <id> [last_n]": (
        "Resume a previous session by exact ID or prefix, optionally loading only the last N messages.",
        "id + optional last_n window",
        "/load a3b5b372 20",
    ),
    "/sessions [recent|oldest] [classic|meta]": (
        "List known sessions with order and optional type filter. The table already includes summary, themes, count, and date.",
        "order=recent|oldest, type=classic|meta",
        "/sessions oldest meta",
    ),
    "/rename <title>": ("Rename the current session title/summary anchor.", "free text title", "/rename Debug OAuth Google"),
    "/delete <id>": ("Permanently delete one session by ID or prefix.", "target session id", "/delete 0d193bea"),
    "/compact": ("Compact the in-memory context and persist a summary of old turns.", "-", "/compact"),
    "/clear": ("Clear the terminal screen only. Session history remains unchanged.", "-", "/clear"),
    "/reset": ("Clear the current conversation history inside this session.", "-", "/reset"),
    "/digest [id]": (
        "Generate or refresh summary, themes, and session_type for the current or a past session.",
        "optional session id",
        "/digest a3b5b372",
    ),
    "/extract <id> [offset] [limit]": (
        "Extract an explicit message window from a session without loading the whole chat.",
        "id, offset>=0, limit>0",
        "/extract a3b5b372 20 15",
    ),
    "/history": ("Show current in-context history size and first/last message previews.", "-", "/history"),
    "/model [name|default]": ("Reset to the current provider default model when omitted or set to default, or switch to a specific model live.", "optional model name", "/model mistral-large-latest"),
    "/provider [name] [model]": (
        "Show current provider or switch provider, optionally changing model at the same time.",
        "provider + optional model",
        "/provider mistral mistral-medium-latest",
    ),
    "/system [prompt|off]": ("Show, set, or disable the session-specific system prompt.", "free text or off", "/system You are concise."),
    "/cwd [path]": (
        "Show the current working directory, or set it for chat, shell blocks, python blocks, and runtime tools.",
        "optional path",
        "/cwd ~/Work/AI/k_ai",
    ),
    "/focus <shell|python>": (
        "Focus a persistent local runner. Keystrokes go directly to it and are not persisted; press Ctrl+] to return to chat.",
        "shell|python",
        "/focus shell",
    ),
    "/set <key> <value>": ("Change one runtime config key live using dot-notation.", "dot.key + value", "/set cli.theme mono"),
    "/settings [prefix]": ("List current active config keys, optionally filtered by prefix.", "optional prefix", "/settings cli"),
    "/status": ("Show full runtime transparency: provider, tokens, context, limits, approvals, paths.", "-", "/status"),
    "/tools": (
        "Inspect tool approval policies or live-enable/disable capability groups such as exa, python, shell, qmd, and mcp. Protected admin rules stay YAML-only.",
        "show|capabilities|enable|disable|ask|auto|reset ...",
        "/tools capabilities",
    ),
    "/config show [key|section:<name> ...]": (
        "Inspect active config values or built-in config sections from inside chat.",
        "key or one/many section:<name>",
        "/config show section:models section:governance",
    ),
    "/config save [path]": ("Persist the current merged runtime config to disk.", "optional output path", "/config save ~/.k-ai/config.yaml"),
    "/config get [path] [section ...]": (
        "Export the built-in default config, optionally restricted to chosen sections.",
        "optional path + optional sections",
        "/config get prompts.yaml ui",
    ),
    "/config sections": ("List all built-in config sections and what each one contains.", "-", "/config sections"),
    "/config edit [all|models|ui|sessions|governance|skills|hooks|mcp|interaction]": (
        "Open the active config file or one built-in fragment in the configured editor.",
        "optional target section",
        "/config edit mcp",
    ),
    "/save [filename]": ("Export the current chat history to a JSON file.", "optional filename", "/save debug_chat.json"),
    "/tokens": ("Show cumulative token accounting for the current session.", "-", "/tokens"),
    "/memory list": ("List persistent memory entries.", "-", "/memory list"),
    "/memory add <text>": ("Add one persistent memory entry.", "free text", "/memory add User prefers French."),
    "/memory remove <id>": ("Remove one persistent memory entry by ID.", "memory id", "/memory remove 3"),
    "/skills [list|show|reload|active]": (
        "Inspect discovered skills, view one full SKILL.md body, refresh the skill catalog, or show the session skill context.",
        "optional subcommand",
        "/skills show kpihx-workspace",
    ),
    "/hooks [list|reload]": (
        "Inspect discovered hook matchers or refresh hook configs from disk.",
        "optional subcommand",
        "/hooks list",
    ),
    "/mcp [list|tools|resources|templates|prompts|reload|probe|install|add-stdio|add-http|enable|disable|remove]": (
        "Inspect or administer MCP servers, imported tools, resources, prompts, installation, and enablement from chat.",
        "optional subcommand",
        "/mcp tools",
    ),
    "/doctor": (
        "Run the same environment diagnostic as [cyan]k-ai doctor[/cyan], including config coherence, tool/catalog alignment, and recovery paths.",
        "optional: reset <config|memory|sessions|all>",
        "/doctor reset config",
    ),
    "/qmd query <text>": ("Run the recommended hybrid semantic search over QMD.", "free text query", "/qmd query diagonalisation matrice"),
    "/qmd search <text>": ("Run keyword-first BM25 search in QMD.", "free text query", "/qmd search Francis Ngannou"),
    "/qmd vsearch <text>": ("Run vector similarity search in QMD.", "free text query", "/qmd vsearch startup camerounaise"),
    "/qmd get <file> [N]": ("Open a QMD document, optionally truncating to N lines.", "file path + optional line count", "/qmd get qmd://k-ai/abc.jsonl 80"),
    "/qmd ls [collection]": ("List indexed files, optionally restricted to one collection.", "optional collection", "/qmd ls k-ai"),
    "/qmd collections": ("List all QMD collections currently available.", "-", "/qmd collections"),
    "/qmd status": ("Show QMD index health, counts, and sync status.", "-", "/qmd status"),
    "/qmd update [--pull]": ("Refresh the QMD index, optionally pulling sources first.", "optional --pull", "/qmd update --pull"),
    "/qmd embed [-f]": ("Refresh or force-refresh vector embeddings.", "optional -f", "/qmd embed -f"),
    "/qmd cleanup": ("Vacuum caches and cleanup QMD index artifacts.", "-", "/qmd cleanup"),
    "/sandbox list": ("List installed packages inside the python_exec sandbox.", "-", "/sandbox list"),
    "/sandbox add <package...>": ("Install one or more extra packages into the python_exec sandbox.", "one or many package specifiers", "/sandbox add polars pyarrow"),
    "/sandbox remove <package...>": ("Remove one or more non-core packages from the python_exec sandbox.", "one or many package names", "/sandbox remove polars"),
}


class CommandHandler:
    def __init__(self, session: "ChatSession"):
        self.session = session
        self.console = session.console

    async def _run_internal_tool(self, tool_name: str, arguments: Dict[str, object]) -> bool:
        tool = self.session.tool_registry.get(tool_name)
        if not tool:
            from .ui import render_notice
            render_notice(self.console, f"Tool {tool_name} not found.", level="error")
            return True

        from .models import ToolCall
        tool_call = ToolCall(id=f"cmd_{tool_name}", function_name=tool_name, arguments=dict(arguments))
        await self.session._execute_internal_tool(tool_call, rationale="Triggered explicitly via slash command.")
        return True

    async def handle(self, text: str) -> bool:
        """
        Dispatch a slash command.
        Returns False when the session should exit, True otherwise.
        """
        parts = text.lstrip("/").split()
        if not parts:
            return True

        cmd = parts[0].lower()
        args: List[str] = parts[1:]

        dispatch: Dict[str, object] = {
            "help": self._help,
            "exit": self._exit,
            "quit": self._exit,
            "bye": self._exit,
            "q": self._exit,
            "new": self._new,
            "init": self._init,
            "load": self._load,
            "sessions": self._sessions,
            "rename": self._rename,
            "delete": self._delete_session,
            "compact": self._compact,
            "clear": self._clear,
            "cls": self._clear,
            "reset": self._reset,
            "digest": self._digest,
            "extract": self._extract,
            "history": self._history,
            "model": self._model,
            "provider": self._provider,
            "system": self._system,
            "cwd": self._cwd,
            "focus": self._focus,
            "set": self._set,
            "settings": self._settings,
            "status": self._status,
            "tools": self._tools,
            "config": self._config,
            "save": self._save,
            "tokens": self._tokens,
            "memory": self._memory,
            "skills": self._skills,
            "hooks": self._hooks,
            "mcp": self._mcp,
            "doctor": self._doctor,
            "qmd": self._qmd,
            "sandbox": self._sandbox,
        }

        handler = dispatch.get(cmd)
        if handler is None:
            self.console.print(
                f"[bold red]Unknown command:[/bold red] /{cmd}  "
                "(type [cyan]/help[/cyan] for a list)"
            )
            return True

        return await handler(args)

    # ------------------------------------------------------------------
    # Session lifecycle commands (delegate to tools via callbacks)
    # ------------------------------------------------------------------

    async def _exit(self, args: List[str]) -> bool:
        return False

    async def _new(self, args: List[str]) -> bool:
        seed = {}
        if args and args[0].lower() in {"classic", "meta"}:
            seed["session_type"] = args[0].lower()
        await self.session._finalize_active_session()
        self.session._do_new_session(seed=seed or None)
        session_type = seed.get("session_type", "classic")
        self.console.print(f"[green]New {session_type} session started.[/green]")
        return True

    async def _init(self, args: List[str]) -> bool:
        return await self._run_internal_tool("init_system", {})

    async def _load(self, args: List[str]) -> bool:
        if not args:
            self.console.print("[yellow]Usage:[/yellow] /load <session_id> [last_n]")
            return True
        try:
            last_n = int(args[1]) if len(args) > 1 else None
        except ValueError:
            self.console.print("[red]last_n must be a number.[/red]")
            return True
        self.session._do_load_session(args[0], last_n=last_n)
        return True

    async def _sessions(self, args: List[str]) -> bool:
        from .ui import render_sessions_table
        max_recent = self.session.cm.get_nested("sessions", "max_recent", default=10)
        order = "recent"
        session_type = None
        for arg in args:
            lowered = arg.lower()
            if lowered in {"oldest", "last", "bottom", "recent"}:
                order = "oldest" if lowered in {"oldest", "last", "bottom"} else "recent"
            elif lowered in {"classic", "meta"}:
                session_type = lowered
        title = "Sessions" if order == "recent" else "Sessions (Oldest First)"
        if session_type:
            title += f" [{session_type}]"
        raw_limit = self.session.session_store.session_count() if session_type else max_recent
        sessions = self.session.session_store.list_sessions(limit=raw_limit, order=order)
        if session_type:
            sessions = [session for session in sessions if session.session_type == session_type][:max_recent]
        render_sessions_table(self.console, sessions, title=title)
        return True

    async def _rename(self, args: List[str]) -> bool:
        if not args:
            self.console.print("[yellow]Usage:[/yellow] /rename <new title>")
            return True
        title = " ".join(args)
        if self.session._session_id:
            self.session.session_store.rename_session(self.session._session_id, title)
            self.console.print(f"[green]Session renamed to:[/green] {title}")
        else:
            self.console.print("[dim]No active session.[/dim]")
        return True

    async def _delete_session(self, args: List[str]) -> bool:
        if not args:
            self.console.print("[yellow]Usage:[/yellow] /delete <session_id>")
            return True
        return await self._run_internal_tool("delete_session", {"session_id": args[0]})

    async def _compact(self, args: List[str]) -> bool:
        await self.session._do_compact()
        return True

    async def _clear(self, args: List[str]) -> bool:
        self.console.clear()
        return True

    async def _reset(self, args: List[str]) -> bool:
        self.session.reset_history()
        from .ui import render_notice
        render_notice(self.console, "Active conversation history cleared.", level="success", title="History Reset")
        return True

    async def _digest(self, args: List[str]) -> bool:
        session_id = args[0] if args else None
        return await self._run_internal_tool(
            "session_digest",
            {"session_id": session_id} if session_id else {},
        )

    async def _cwd(self, args: List[str]) -> bool:
        if not args:
            self.console.print(f"[cyan]{self.session.current_cwd()}[/cyan]")
            return True
        target = " ".join(args).strip()
        try:
            resolved = self.session.set_current_cwd(target)
        except Exception as exc:
            self.console.print(f"[red]{exc}[/red]")
            return True
        self.console.print(f"[green]Working directory:[/green] {resolved}")
        return True

    async def _focus(self, args: List[str]) -> bool:
        if not args or args[0].lower() not in {"shell", "python"}:
            self.console.print("[yellow]Usage:[/yellow] /focus <shell|python>")
            return True
        try:
            await self.session.focus_runner(args[0].lower())
        except Exception as exc:
            self.console.print(f"[red]{exc}[/red]")
        return True

    async def _extract(self, args: List[str]) -> bool:
        if not args:
            self.console.print("[yellow]Usage:[/yellow] /extract <session_id> [offset] [limit]")
            return True
        payload: Dict[str, object] = {"session_id": args[0]}
        try:
            if len(args) > 1:
                payload["offset"] = int(args[1])
            if len(args) > 2:
                payload["limit"] = int(args[2])
        except ValueError:
            self.console.print("[red]offset and limit must be numbers.[/red]")
            return True
        return await self._run_internal_tool("session_extract", payload)

    # ------------------------------------------------------------------
    # QMD commands
    # ------------------------------------------------------------------

    async def _qmd(self, args: List[str]) -> bool:
        if not args:
            self.console.print(
                "[yellow]Usage:[/yellow]\n"
                "  /qmd query <text>        Hybrid semantic search\n"
                "  /qmd search <text>       BM25 keyword search\n"
                "  /qmd vsearch <text>      Vector similarity search\n"
                "  /qmd get <file> [lines]  View a document\n"
                "  /qmd ls [collection]     List indexed files\n"
                "  /qmd collections         List collections\n"
                "  /qmd status              Index health\n"
                "  /qmd update [--pull]     Re-index collections\n"
                "  /qmd embed [-f]          Refresh embeddings\n"
                "  /qmd cleanup             Clear caches"
            )
            return True

        sub = args[0].lower()
        rest = args[1:]
        # Map subcommand to tool name + arguments
        tool_map = {
            "query":       ("qmd_query",   lambda: {"query": " ".join(rest)}),
            "search":      ("qmd_search",  lambda: {"query": " ".join(rest)}),
            "vsearch":     ("qmd_vsearch", lambda: {"query": " ".join(rest)}),
            "get":         ("qmd_get",     lambda: {"file": rest[0] if rest else "", "lines": int(rest[1]) if len(rest) > 1 else None}),
            "ls":          ("qmd_ls",      lambda: {"path": rest[0] if rest else ""}),
            "collections": ("qmd_collection_list", lambda: {}),
            "status":      ("qmd_status",  lambda: {}),
            "update":      ("qmd_update",  lambda: {"pull": "--pull" in rest}),
            "embed":       ("qmd_embed",   lambda: {"force": "-f" in rest}),
            "cleanup":     ("qmd_cleanup", lambda: {}),
        }

        if sub not in tool_map:
            self.console.print(f"[red]Unknown qmd subcommand:[/red] {sub}. Type /qmd for help.")
            return True

        tool_name, build_args = tool_map[sub]
        try:
            arguments = build_args()
        except (IndexError, ValueError) as e:
            self.console.print(f"[red]Invalid arguments:[/red] {e}")
            return True

        # Filter None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        return await self._run_internal_tool(tool_name, arguments)

    # ------------------------------------------------------------------
    # Memory commands
    # ------------------------------------------------------------------

    async def _memory(self, args: List[str]) -> bool:
        if not args:
            self.console.print(
                "[yellow]Usage:[/yellow] /memory list | /memory add <text> | /memory remove <id>"
            )
            return True

        sub = args[0].lower()
        if sub == "list":
            entries = self.session.memory.list_entries()
            if not entries:
                self.console.print("[dim]Memory is empty.[/dim]")
            else:
                for e in entries:
                    self.console.print(f"  [cyan]#{e.id}[/cyan] {e.text}")
        elif sub == "add" and len(args) > 1:
            return await self._run_internal_tool("memory_add", {"text": " ".join(args[1:])})
        elif sub == "remove" and len(args) > 1:
            try:
                return await self._run_internal_tool("memory_remove", {"entry_id": int(args[1])})
            except ValueError:
                self.console.print("[red]ID must be a number.[/red]")
        else:
            self.console.print(
                "[yellow]Usage:[/yellow] /memory list | /memory add <text> | /memory remove <id>"
            )
        return True

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    async def _skills(self, args: List[str]) -> bool:
        from rich.table import Table

        sub = args[0].lower() if args else "list"
        if sub in {"reload", "refresh"}:
            self.session.skill_manager.refresh()
            self.console.print(
                f"[green]{self.session.skill_manager.config_text('skills', 'runtime', 'ui', 'reload_message', default='Skill catalog refreshed.')}[/green]"
            )
            return True

        if sub == "active":
            active = list(self.session._session_skill_names)
            if not active:
                self.console.print(
                    f"[dim]{self.session.skill_manager.config_text('skills', 'runtime', 'ui', 'active_none_message', default='No session skill context is currently retained.')}[/dim]"
                )
                return True
            for name in active:
                self.console.print(
                    self.session.skill_manager.config_text(
                        "skills",
                        "runtime",
                        "ui",
                        "active_item_template",
                        default="  [magenta]{skill_name}[/magenta]",
                        skill_name=name,
                    )
                )
            return True

        if sub == "show":
            if len(args) < 2:
                self.console.print(
                    self.session.skill_manager.config_text(
                        "skills",
                        "runtime",
                        "ui",
                        "show_usage_message",
                        default="[yellow]Usage:[/yellow] /skills show <skill_name>",
                    )
                )
                return True
            document = self.session.skill_manager.load_named_skill(args[1].strip().lower())
            if document is None:
                self.console.print(
                    self.session.skill_manager.config_text(
                        "skills",
                        "runtime",
                        "ui",
                        "unknown_message",
                        default="[red]Unknown skill:[/red] {skill_name}",
                        skill_name=args[1],
                    )
                )
                return True
            syntax_theme = resolve_syntax_theme(self.session.cm.get_nested("cli", "theme", default="default"))
            body = f"---\nname: {document.summary.name}\ndescription: {document.summary.description}\n---\n\n{document.body}"
            self.console.print(
                Panel(
                    Syntax(body, "markdown", theme=syntax_theme, word_wrap=True),
                    title=self.session.skill_manager.config_text(
                        "skills",
                        "runtime",
                        "ui",
                        "panel_title",
                        default="[bold magenta]Skill[/bold magenta] [dim]{skill_name}[/dim]",
                        skill_name=document.summary.name,
                    ),
                    border_style="magenta",
                )
            )
            return True

        skills = self.session.skill_manager.catalog(force_refresh=False)
        issues = self.session.skill_manager.issues()
        if not skills:
            self.console.print(
                f"[dim]{self.session.skill_manager.config_text('skills', 'runtime', 'ui', 'list_none_message', default='No skills discovered.')}[/dim]"
            )
            return True

        table = Table(
            title=self.session.skill_manager.config_text(
                "skills",
                "runtime",
                "ui",
                "table_title",
                default="[bold magenta]Skills[/bold magenta]",
            ),
            show_header=True,
            header_style="bold",
        )
        table.add_column(
            self.session.skill_manager.config_text("skills", "runtime", "ui", "columns", "name", default="Name"),
            style="magenta",
            no_wrap=True,
        )
        table.add_column(
            self.session.skill_manager.config_text("skills", "runtime", "ui", "columns", "scope", default="Scope"),
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            self.session.skill_manager.config_text("skills", "runtime", "ui", "columns", "description", default="Description")
        )
        for skill in skills:
            table.add_row(skill.name, skill.scope, skill.description)
        self.console.print(table)
        if issues:
            self.console.print(
                self.session.skill_manager.config_text(
                    "skills",
                    "runtime",
                    "ui",
                    "issues_message",
                    default="[dim]{issue_count} discovery issue(s) detected. Use /skills reload after fixes.[/dim]",
                    issue_count=str(len(issues)),
                )
            )
        return True

    # ------------------------------------------------------------------
    # Doctor
    # ------------------------------------------------------------------

    async def _doctor(self, args: List[str]) -> bool:
        from .doctor import run_doctor
        reset = []
        if args and args[0].lower() == "reset":
            reset = args[1:] or ["all"]
        run_doctor(
            self.session.cm,
            self.session.memory,
            self.session.session_store,
            self.console,
            reset=reset,
        )
        return True

    # ------------------------------------------------------------------
    # Existing commands (kept from previous implementation)
    # ------------------------------------------------------------------

    async def _help(self, args: List[str]) -> bool:
        t = Table(title="[bold cyan]k-ai Commands[/bold cyan]", show_header=True, header_style="bold")
        t.add_column("Command", style="cyan", no_wrap=True)
        t.add_column("What It Does")
        t.add_column("Arguments / Values", style="magenta")
        t.add_column("Example", style="green")
        for cmd, (desc, params, example) in _HELP.items():
            t.add_row(cmd, desc, params, example)
        self.console.print(t)
        self.console.print(
            "[dim]Tip:[/dim] shell-level help is also available with "
            "[cyan]k-ai --help[/cyan], [cyan]k-ai chat --help[/cyan], "
            "[cyan]k-ai config --help[/cyan], and [cyan]k-ai doctor --help[/cyan]."
        )
        return True

    async def _history(self, args: List[str]) -> bool:
        hist = self.session.history
        count = len(hist)
        if count == 0:
            self.console.print("[dim]History is empty.[/dim]")
        else:
            self.console.print(f"[dim]History:[/dim] {count} message(s) in context.")
            if count >= 2:
                first, last = hist[0], hist[-1]
                preview = lambda m: m.content[:60].replace("\n", " ") + ("..." if len(m.content) > 60 else "")
                self.console.print(f"  [dim]first ({first.role.value}):[/dim] {preview(first)}")
                self.console.print(f"  [dim]last  ({last.role.value}):[/dim] {preview(last)}")
        return True

    async def _model(self, args: List[str]) -> bool:
        if not args or args[0].lower() == "default":
            default_model = self.session.provider_default_model()
            if not default_model:
                self.console.print("[red]No default model is configured for the current provider.[/red]")
                return True
            return await self._run_internal_tool("set_config", {"key": "model", "value": default_model})
        return await self._run_internal_tool("set_config", {"key": "model", "value": args[0]})

    async def _provider(self, args: List[str]) -> bool:
        if not args:
            return await self._run_internal_tool("runtime_status", {"mode": "compact"})
        target_provider = args[0]
        target_model = args[1] if len(args) > 1 else self.session.provider_default_model(target_provider)

        if (
            self.session._session_id
            and self.session.history
            and target_provider != self.session.llm.provider_name
            and self.session.history_has_nonportable_tool_state()
        ):
            self.console.print(
                "[yellow]This session contains tool-call history from the current provider.[/yellow]"
            )
            start_new = Confirm.ask(
                "Start a new session for this provider change?",
                console=self.console,
                default=True,
            )
            if start_new:
                current_meta = self.session.session_store.get_session(self.session._session_id) if self.session._session_id else None
                seed = None
                if current_meta:
                    seed = {
                        "session_type": current_meta.session_type,
                    }
                await self.session._finalize_active_session()
                self.session._do_new_session(seed=seed)
                self.console.print("[green]Started a new session for the provider switch.[/green]")
            else:
                removed = self.session.preserve_simple_history_only()
                self.console.print(
                    f"[yellow]Kept only simple user/assistant text history ({removed} complex tool messages removed) before switching provider.[/yellow]"
                )

        await self._run_internal_tool("set_config", {"key": "provider", "value": target_provider})
        if target_model:
            await self._run_internal_tool("set_config", {"key": "model", "value": target_model})
        return True

    async def _system(self, args: List[str]) -> bool:
        if not args:
            if self.session.system_prompt:
                self.console.print(
                    Panel(self.session.system_prompt, title="[bold yellow]System Prompt[/bold yellow]", border_style="yellow")
                )
            else:
                self.console.print("[dim]No system prompt set.[/dim]")
        elif args[0].lower() == "off":
            self.session.system_prompt = None
            self.console.print("[green]System prompt disabled.[/green]")
        else:
            self.session.system_prompt = " ".join(args)
            self.console.print("[green]System prompt set.[/green]")
        return True

    async def _set(self, args: List[str]) -> bool:
        if len(args) < 2:
            self.console.print("[yellow]Usage:[/yellow] /set <key> <value>")
            return True
        return await self._run_internal_tool("set_config", {"key": args[0], "value": " ".join(args[1:])})

    async def _settings(self, args: List[str]) -> bool:
        payload: Dict[str, object] = {"limit": 200}
        if args:
            payload["prefix"] = args[0]
        return await self._run_internal_tool("list_config", payload)

    async def _status(self, args: List[str]) -> bool:
        return await self._run_internal_tool("runtime_status", {"mode": "full"})

    async def _tools(self, args: List[str]) -> bool:
        if args and args[0].lower() in {"capabilities", "caps", "enabled"}:
            return await self._run_internal_tool("tool_capability_list", {})

        if not args or args[0].lower() in {"list", "show", "status"}:
            payload: Dict[str, object] = {}
            if args:
                if len(args) > 1 and args[1].lower() in {"ask", "auto"}:
                    payload["policy"] = args[1].lower()
                elif len(args) > 1 and args[1].lower() in {"default", "session", "global", "protected"}:
                    payload["source"] = args[1].lower()
            return await self._run_internal_tool("tool_policy_list", payload)

        sub = args[0].lower()
        if sub in {"enable", "disable"}:
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /tools enable|disable <exa|python|shell|qmd|mcp>")
                return True
            payload = {
                "capability": args[1],
                "enabled": sub == "enable",
            }
            return await self._run_internal_tool("tool_capability_set", payload)

        if sub in {"ask", "auto"}:
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /tools ask|auto <target> [session|global] [tool|category|risk]")
                return True
            payload: Dict[str, object] = {
                "target": args[1],
                "policy": sub,
                "scope": args[2] if len(args) > 2 else "session",
                "target_kind": args[3] if len(args) > 3 else "tool",
            }
            return await self._run_internal_tool("tool_policy_set", payload)

        if sub == "reset":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /tools reset <target> [session|global] [tool|category|risk]")
                return True
            payload = {
                "target": args[1],
                "scope": args[2] if len(args) > 2 else "session",
                "target_kind": args[3] if len(args) > 3 else "tool",
            }
            return await self._run_internal_tool("tool_policy_reset", payload)

        self.console.print(
            "[yellow]Usage:[/yellow]\n"
            "  /tools capabilities\n"
            "  /tools enable <exa|python|shell|qmd|mcp>\n"
            "  /tools disable <exa|python|shell|qmd|mcp>\n"
            "  /tools show [ask|auto|default|session|global|protected]\n"
            "  /tools ask <target> [session|global] [tool|category|risk]\n"
            "  /tools auto <target> [session|global] [tool|category|risk]\n"
            "  /tools reset <target> [session|global] [tool|category|risk]\n"
            "  Protected admin approval rules remain manual YAML edits by design."
        )
        return True

    async def _sandbox(self, args: List[str]) -> bool:
        sub = args[0].lower() if args else "list"
        if sub == "list":
            return await self._run_internal_tool("python_sandbox_list_packages", {})
        if sub == "add":
            packages = [item for item in args[1:] if item.strip()]
            if not packages:
                self.console.print("[yellow]Usage:[/yellow] /sandbox add <package...>")
                return True
            return await self._run_internal_tool("python_sandbox_install_packages", {"packages": packages})
        if sub in {"remove", "rm"}:
            packages = [item for item in args[1:] if item.strip()]
            if not packages:
                self.console.print("[yellow]Usage:[/yellow] /sandbox remove <package...>")
                return True
            return await self._run_internal_tool("python_sandbox_remove_packages", {"packages": packages})
        self.console.print(
            "[yellow]Usage:[/yellow]\n"
            "  /sandbox list\n"
            "  /sandbox add <package...>\n"
            "  /sandbox remove <package...>"
        )
        return True

    async def _config(self, args: List[str]) -> bool:
        sub = args[0].lower() if args else "get"
        known_sections = {item["name"] for item in ConfigManager.list_default_sections()}
        syntax_theme = resolve_syntax_theme(self.session.cm.get_nested("cli", "theme", default="default"))
        if sub == "show":
            extra = args[1:]
            section_names = []
            key = ""
            for item in extra:
                lowered = item.lower()
                if lowered.startswith("section:"):
                    section_names.append(lowered.split(":", 1)[1])
                elif lowered in known_sections:
                    section_names.append(lowered)
                elif not key:
                    key = item
                else:
                    self.console.print("[yellow]Usage:[/yellow] /config show [key] | /config show section:<name> [section:<name> ...]")
                    return True
            if section_names:
                try:
                    yaml_str = self.session.cm.get_default_yaml(sections=section_names)
                    title = "Default Config [" + ", ".join(ConfigManager.normalize_default_sections(section_names)) + "]"
                    self.console.print(Panel(
                        Syntax(yaml_str, "yaml", theme=syntax_theme),
                        title=f"[bold cyan]{title}[/bold cyan]",
                        border_style="cyan",
                    ))
                except Exception as e:
                    self.console.print(f"[bold red]Error:[/bold red] {e}")
                return True
            payload = {"key": key} if key else {}
            return await self._run_internal_tool("get_config", payload)
        elif sub == "save":
            payload = {"path": args[1]} if len(args) > 1 else {}
            return await self._run_internal_tool("save_config", payload)
        elif sub in {"sections", "list"}:
            table = Table(title="Config Sections", header_style="bold cyan")
            table.add_column("Section", style="cyan", no_wrap=True)
            table.add_column("File", style="magenta")
            table.add_column("Description", style="white")
            for item in ConfigManager.list_default_sections():
                table.add_row(item["name"], item["file"], item["description"])
            self.console.print(table)
            return True
        elif sub == "edit":
            if len(args) > 2:
                self.console.print("[yellow]Usage:[/yellow] /config edit [all|models|ui|sessions|governance|skills|hooks|mcp|interaction]")
                return True
            target_name = args[1] if len(args) > 1 else "all"
            try:
                target_path = self.session.cm.resolve_edit_target(target_name)
                editor_cmd = self.session.cm.resolve_editor_command()
                self.console.print(f"[cyan]Opening[/cyan] {target_path} [cyan]with[/cyan] {' '.join(editor_cmd)}")
                subprocess.run([*editor_cmd, str(target_path)], check=True)
            except FileNotFoundError as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
            except subprocess.CalledProcessError as e:
                self.console.print(f"[bold red]Editor exited with error:[/bold red] {e}")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
            return True
        elif sub == "get":
            extra = args[1:]
            target = "config.yaml"
            section_names: List[str] = []
            if extra:
                first = extra[0]
                lowered = first.lower()
                if lowered in known_sections:
                    section_names = [item.lower() for item in extra]
                else:
                    target = first
                    section_names = [item.lower() for item in extra[1:]]
            try:
                if os.path.exists(target):
                    if not Confirm.ask(f"Overwrite '{target}'?", console=self.console, default=False):
                        return True
                with open(target, "w", encoding="utf-8") as f:
                    f.write(self.session.cm.get_default_yaml(sections=section_names))
                suffix = ""
                if section_names:
                    normalized = ConfigManager.normalize_default_sections(section_names)
                    suffix = f" ({', '.join(normalized)})"
                self.console.print(f"[green]Default config saved to '{target}'{suffix}.[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            self.console.print(
                "[yellow]Usage:[/yellow] /config show [key] | /config show section:<name> [section:<name> ...] | "
                "/config save [path] | /config get [path] [section ...] | /config sections | "
                "/config edit [all|models|ui|sessions|governance|skills|hooks|mcp|interaction]"
            )
        return True

    async def _hooks(self, args: List[str]) -> bool:
        sub = args[0].lower() if args else "list"
        if sub in {"reload", "refresh"}:
            self.session.hook_manager.refresh()
            self.console.print(
                f"[green]{self.session.hook_manager.config_text('hooks', 'runtime', 'reload_message', default='Hook catalog refreshed.')}[/green]"
            )
            return True

        catalog = self.session.hook_manager.catalog(force_refresh=False)
        if not catalog.matchers:
            self.console.print(
                self.session.hook_manager.config_text("hooks", "runtime", "discovery_none", default="No hooks discovered.")
            )
            return True
        table = Table(title=self.session.hook_manager.config_text("hooks", "runtime", "table_title", default="[bold cyan]Hooks[/bold cyan]"))
        table.add_column(self.session.hook_manager.config_text("hooks", "runtime", "columns", "event", default="Event"), style="cyan")
        table.add_column(self.session.hook_manager.config_text("hooks", "runtime", "columns", "matcher", default="Matcher"), style="magenta")
        table.add_column(self.session.hook_manager.config_text("hooks", "runtime", "columns", "scope", default="Scope"), style="green")
        table.add_column(self.session.hook_manager.config_text("hooks", "runtime", "columns", "source", default="Source"), style="dim")
        table.add_column(self.session.hook_manager.config_text("hooks", "runtime", "columns", "command", default="Command"))
        for matcher in catalog.matchers:
            for command in matcher.commands:
                text = command.command if len(command.command) <= 72 else command.command[:69] + "..."
                table.add_row(matcher.event, matcher.matcher, matcher.scope, str(matcher.source_file), text)
        self.console.print(table)
        return True

    async def _mcp(self, args: List[str]) -> bool:
        sub = args[0].lower() if args else "list"
        if sub in {"reload", "refresh"}:
            return await self._run_internal_tool("mcp_catalog_reload", {})

        if sub == "probe":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /mcp probe <server_name> [command_or_package]")
                return True
            payload: Dict[str, object] = {"server_name": args[1]}
            if len(args) > 2:
                payload["command"] = args[2]
            return await self._run_internal_tool("mcp_server_probe", payload)

        if sub == "install":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /mcp install <server_name> [package_name] [binary_name]")
                return True
            payload = {"server_name": args[1]}
            if len(args) > 2:
                payload["package_name"] = args[2]
            if len(args) > 3:
                payload["binary_name"] = args[3]
            return await self._run_internal_tool("mcp_server_install", payload)

        if sub == "add-stdio":
            if len(args) < 3:
                self.console.print("[yellow]Usage:[/yellow] /mcp add-stdio <server_name> <command> [cwd]")
                return True
            payload = {
                "server_name": args[1],
                "transport": "stdio",
                "command": args[2],
            }
            if len(args) > 3:
                payload["cwd"] = args[3]
            return await self._run_internal_tool("mcp_server_upsert", payload)

        if sub == "add-http":
            if len(args) < 3:
                self.console.print("[yellow]Usage:[/yellow] /mcp add-http <server_name> <url> [streamable_http|sse]")
                return True
            payload = {
                "server_name": args[1],
                "transport": args[3] if len(args) > 3 else "streamable_http",
                "url": args[2],
            }
            return await self._run_internal_tool("mcp_server_upsert", payload)

        if sub in {"enable", "disable"}:
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /mcp enable|disable <server_name>")
                return True
            return await self._run_internal_tool(
                "mcp_server_set_enabled",
                {"server_name": args[1], "enabled": sub == "enable"},
            )

        if sub == "remove":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/yellow] /mcp remove <server_name>")
                return True
            return await self._run_internal_tool("mcp_server_remove", {"server_name": args[1]})

        await self.session._ensure_mcp_catalog_loaded()
        catalog = self.session.mcp_manager.current_catalog()

        if sub == "tools":
            if not catalog.tools:
                self.console.print(self.session.mcp_manager.config_text("mcp", "runtime", "discovery_none", default="No MCP servers discovered."))
                return True
            table = Table(title=self.session.mcp_manager.config_text("mcp", "runtime", "tools_table_title", default="[bold cyan]MCP Tools[/bold cyan]"))
            table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "server", default="Server"), style="cyan")
            table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "tool_name", default="Tool"), style="magenta")
            table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "qualified_name", default="Qualified Name"), style="green")
            table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "risk", default="Risk"), style="yellow")
            table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "approval", default="Approval"), style="white")
            for tool in catalog.tools:
                wrapped = self.session.tool_registry.get(tool.qualified_name)
                table.add_row(
                    tool.server_title,
                    tool.name,
                    tool.qualified_name,
                    getattr(wrapped, "danger_level", "medium"),
                    "ask" if getattr(wrapped, "requires_approval", True) else "auto",
                )
            self.console.print(table)
            return True

        if sub == "resources":
            payload: Dict[str, object] = {}
            if len(args) > 1:
                payload["server_name"] = args[1]
            return await self._run_internal_tool("mcp_resource_list", payload)

        if sub == "templates":
            payload: Dict[str, object] = {}
            if len(args) > 1:
                payload["server_name"] = args[1]
            return await self._run_internal_tool("mcp_resource_template_list", payload)

        if sub == "prompts":
            payload: Dict[str, object] = {}
            if len(args) > 1:
                payload["server_name"] = args[1]
            return await self._run_internal_tool("mcp_prompt_list", payload)

        if not catalog.servers and not catalog.issues:
            self.console.print(self.session.mcp_manager.config_text("mcp", "runtime", "discovery_none", default="No MCP servers discovered."))
            return True

        table = Table(title=self.session.mcp_manager.config_text("mcp", "runtime", "table_title", default="[bold cyan]MCP Servers[/bold cyan]"))
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "server", default="Server"), style="cyan")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "status", default="Status"), style="magenta")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "transport", default="Transport"), style="green")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "tools", default="Tools"), style="white")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "resources", default="Resources"), style="white")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "prompts", default="Prompts"), style="white")
        table.add_column(self.session.mcp_manager.config_text("mcp", "runtime", "columns", "source", default="Source"), style="dim")
        seen_servers = set()
        for server in catalog.servers:
            seen_servers.add(server.spec.name)
            table.add_row(
                server.server_title,
                self.session.mcp_manager.config_text("mcp", "runtime", "statuses", "ready", default="ready"),
                server.spec.transport,
                str(len(server.tools)),
                str(len(server.resources)),
                str(len(server.prompts)),
                server.spec.command,
            )
        for issue in catalog.issues:
            seen_servers.add(issue.server_name)
            table.add_row(
                issue.server_name,
                self.session.mcp_manager.config_text("mcp", "runtime", "statuses", "error", default="error"),
                "-",
                "0",
                "0",
                "0",
                issue.message,
            )
        configured = dict(self.session.cm.get_nested("mcp", "servers", default={}) or {})
        for name, payload in configured.items():
            if name in seen_servers or not isinstance(payload, dict):
                continue
            table.add_row(
                str(name),
                self.session.mcp_manager.config_text("mcp", "runtime", "statuses", "disabled", default="disabled"),
                str(payload.get("transport", "stdio")),
                "0",
                "0",
                "0",
                str(payload.get("command", payload.get("url", "")) or "-"),
            )
        self.console.print(table)
        return True

    async def _save(self, args: List[str]) -> bool:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = args[0] if args else f"chat_{timestamp}.json"
        try:
            data = {
                "session_id": self.session._session_id,
                "provider": self.session.llm.provider_name,
                "model": self.session.llm.model_name,
                "system_prompt": self.session.system_prompt,
                "messages": [
                    {"role": m.role.value, "content": m.content}
                    for m in self.session.history
                ],
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.console.print(f"[green]Saved to '{filename}'.[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {e}")
        return True

    async def _tokens(self, args: List[str]) -> bool:
        u = self.session.get_token_snapshot()
        t = Table(title="[bold cyan]Token Usage[/bold cyan]")
        t.add_column("Metric", style="cyan")
        t.add_column("Count", justify="right")
        t.add_row("Input", f"{int(u['prompt_tokens']):,}")
        t.add_row("Output", f"{int(u['completion_tokens']):,}")
        t.add_row("Total", f"{int(u['total_tokens']):,}")
        t.add_row("Source", str(u["token_source"]))
        self.console.print(t)
        return True
