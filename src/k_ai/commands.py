# src/k_ai/commands.py
"""
Handles /slash commands for the interactive chat.

Every command either maps directly to an internal tool (same action the LLM
can propose) or is a UI-only shortcut.  This ensures uniform behaviour:
``/compact`` and "please compact the discussion" do the exact same thing.
"""
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

if TYPE_CHECKING:
    from .session import ChatSession


# ---------------------------------------------------------------------------
# Slash commands for autocompletion (prompt-toolkit)
# ---------------------------------------------------------------------------

SLASH_COMMANDS = [
    "/help", "/exit", "/quit", "/bye",
    "/new", "/load", "/sessions", "/rename", "/delete",
    "/compact", "/clear", "/reset",
    "/history", "/model", "/provider", "/system",
    "/set", "/settings", "/status",
    "/config show", "/config get",
    "/save", "/tokens",
    "/memory list", "/memory add", "/memory remove",
    "/doctor",
    "/qmd query", "/qmd search", "/qmd vsearch",
    "/qmd get", "/qmd ls", "/qmd collections",
    "/qmd status", "/qmd update", "/qmd embed", "/qmd cleanup",
]


# ---------------------------------------------------------------------------
# Help registry
# ---------------------------------------------------------------------------

_HELP: dict[str, str] = {
    "/help": "Show this help message.",
    "/exit, /quit, /bye": "Exit the chat session.",
    "/new": "Start a new chat session.",
    "/load <id>": "Resume a previous session by ID (or prefix).",
    "/sessions": "List recent sessions.",
    "/rename <title>": "Rename the current session.",
    "/delete <id>": "Permanently delete a session.",
    "/compact": "Compress conversation history (summarise old messages).",
    "/clear": "Clear the terminal screen (history untouched).",
    "/reset": "Clear conversation history (start over in this session).",
    "/history": "Show message count and first/last messages.",
    "/model [name]": "Show or switch the model.",
    "/provider [name] [model]": "Show or switch provider (+ optional model).",
    "/system [prompt|off]": "Show, set, or disable the system prompt.",
    "/set <key> <value>": "Live-edit any config key.",
    "/settings": "Display all active settings.",
    "/status": "Full runtime status: provider, keys, token usage.",
    "/config show": "Print the full active config as YAML.",
    "/config get [path]": "Export the default config template.",
    "/save [filename]": "Save conversation to a JSON file.",
    "/tokens": "Show cumulative session token usage.",
    "/memory list": "List all memory entries.",
    "/memory add <text>": "Add a memory entry.",
    "/memory remove <id>": "Remove a memory entry by ID.",
    "/doctor": "Run a full diagnostic check.",
    "/qmd query <text>": "Hybrid semantic search (recommended).",
    "/qmd search <text>": "BM25 keyword search (fast).",
    "/qmd vsearch <text>": "Vector similarity search.",
    "/qmd get <file> [N]": "View a document (optional line limit).",
    "/qmd ls [collection]": "List indexed files.",
    "/qmd collections": "List all collections.",
    "/qmd status": "Index health & stats.",
    "/qmd update [--pull]": "Re-index (optionally git pull first).",
    "/qmd embed [-f]": "Refresh vector embeddings.",
    "/qmd cleanup": "Clear caches & vacuum DB.",
}


class CommandHandler:
    def __init__(self, session: "ChatSession"):
        self.session = session
        self.console = session.console

    async def handle(self, text: str) -> bool:
        """
        Dispatch a slash command.
        Returns False when the session should exit, True otherwise.
        """
        parts = text.lstrip("/").split(maxsplit=2)
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
            "load": self._load,
            "sessions": self._sessions,
            "rename": self._rename,
            "delete": self._delete_session,
            "compact": self._compact,
            "clear": self._clear,
            "cls": self._clear,
            "reset": self._reset,
            "history": self._history,
            "model": self._model,
            "provider": self._provider,
            "system": self._system,
            "set": self._set,
            "settings": self._settings,
            "status": self._status,
            "config": self._config,
            "save": self._save,
            "tokens": self._tokens,
            "memory": self._memory,
            "doctor": self._doctor,
            "qmd": self._qmd,
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
        self.session._do_new_session()
        self.console.print("[green]New session started.[/green]")
        return True

    async def _load(self, args: List[str]) -> bool:
        if not args:
            self.console.print("[yellow]Usage:[/yellow] /load <session_id>")
            return True
        self.session._do_load_session(args[0])
        return True

    async def _sessions(self, args: List[str]) -> bool:
        from .ui import render_sessions_table
        max_recent = self.session.cm.get_nested("sessions", "max_recent", default=10)
        sessions = self.session.session_store.list_sessions(limit=max_recent)
        render_sessions_table(self.console, sessions)
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
        ok = self.session.session_store.delete_session(args[0])
        if ok:
            self.console.print(f"[green]Session deleted.[/green]")
        else:
            self.console.print(f"[red]Session '{args[0]}' not found.[/red]")
        return True

    async def _compact(self, args: List[str]) -> bool:
        await self.session._do_compact()
        return True

    async def _clear(self, args: List[str]) -> bool:
        self.console.clear()
        return True

    async def _reset(self, args: List[str]) -> bool:
        self.session.history.clear()
        self.console.print("[green]History cleared.[/green]")
        return True

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
        ctx = self.session._tool_ctx

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
        tool = self.session.tool_registry.get(tool_name)
        if not tool:
            self.console.print(f"[red]Tool {tool_name} not found.[/red]")
            return True

        try:
            arguments = build_args()
        except (IndexError, ValueError) as e:
            self.console.print(f"[red]Invalid arguments:[/red] {e}")
            return True

        # Filter None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await tool.execute(arguments, ctx)
        self.console.print(result.message)
        return True

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
            text = " ".join(args[1:])
            entry = self.session.memory.add(text)
            self.console.print(f"[green]Remembered (#{entry.id}):[/green] {text}")
        elif sub == "remove" and len(args) > 1:
            try:
                entry_id = int(args[1])
                removed = self.session.memory.remove(entry_id)
                if removed:
                    self.console.print(f"[green]Entry #{entry_id} removed.[/green]")
                else:
                    self.console.print(f"[red]Entry #{entry_id} not found.[/red]")
            except ValueError:
                self.console.print("[red]ID must be a number.[/red]")
        else:
            self.console.print(
                "[yellow]Usage:[/yellow] /memory list | /memory add <text> | /memory remove <id>"
            )
        return True

    # ------------------------------------------------------------------
    # Doctor
    # ------------------------------------------------------------------

    async def _doctor(self, args: List[str]) -> bool:
        from .doctor import run_doctor
        run_doctor(self.session.cm, self.session.memory, self.session.session_store, self.console)
        return True

    # ------------------------------------------------------------------
    # Existing commands (kept from previous implementation)
    # ------------------------------------------------------------------

    async def _help(self, args: List[str]) -> bool:
        t = Table(title="[bold cyan]k-ai Commands[/bold cyan]", show_header=True, header_style="bold")
        t.add_column("Command", style="cyan", no_wrap=True)
        t.add_column("Description")
        for cmd, desc in _HELP.items():
            t.add_row(cmd, desc)
        self.console.print(t)
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
        if not args:
            self.console.print(
                f"[dim]Model:[/dim] [bold]{self.session.llm.model_name}[/bold] "
                f"[dim]({self.session.llm.provider_name})[/dim]"
            )
        else:
            try:
                self.session.reload_provider(model=args[0])
                self.console.print(f"[green]Model -> {self.session.llm.model_name}[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        return True

    async def _provider(self, args: List[str]) -> bool:
        if not args:
            self.console.print(
                f"[dim]Provider:[/dim] [bold]{self.session.llm.provider_name}[/bold]  "
                f"[dim]Model:[/dim] [bold]{self.session.llm.model_name}[/bold]"
            )
        else:
            new_provider = args[0]
            new_model = args[1] if len(args) > 1 else None
            try:
                self.session.cm.set("provider", new_provider)
                self.session.reload_provider(provider=new_provider, model=new_model)
                self.console.print(
                    f"[green]Provider -> {self.session.llm.provider_name}  "
                    f"Model -> {self.session.llm.model_name}[/green]"
                )
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
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
        key = args[0]
        value_str = " ".join(args[1:])
        try:
            self.session.cm.set(key, value_str)
            parts = key.split(".")
            coerced = self.session.cm.get_nested(*parts) if len(parts) > 1 else self.session.cm.get(key)
            self.console.print(f"[green]{key}[/green] = [cyan]{coerced!r}[/cyan]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {e}")
        return True

    async def _settings(self, args: List[str]) -> bool:
        cfg = self.session.cm.get_all()
        t = Table(title="[bold cyan]Active Settings[/bold cyan]", show_header=True, header_style="bold")
        t.add_column("Key", style="cyan")
        t.add_column("Value")

        for key, val in cfg.items():
            if not isinstance(val, dict):
                t.add_row(key, str(val))
        for key, val in cfg.get("cli", {}).items():
            if not isinstance(val, dict):
                t.add_row(f"cli.{key}", str(val))

        t.add_row("--- runtime ---", "")
        t.add_row("active_provider", self.session.llm.provider_name)
        t.add_row("active_model", self.session.llm.model_name)
        t.add_row("session_id", self.session._session_id or "(none)")
        if self.session.system_prompt:
            preview = self.session.system_prompt[:60] + ("..." if len(self.session.system_prompt) > 60 else "")
            t.add_row("system_prompt", preview)

        self.console.print(t)
        return True

    async def _status(self, args: List[str]) -> bool:
        from rich.text import Text
        from .secrets import get_all_key_status, get_dotenv_path

        llm = self.session.llm
        cm = self.session.cm
        u = self.session.total_usage

        rt = Table(show_header=False, box=None, padding=(0, 1))
        rt.add_column("key", style="dim")
        rt.add_column("value", style="bold white")
        rt.add_row("Provider", llm.provider_name)
        rt.add_row("Model", llm.model_name)
        rt.add_row("Auth mode", llm.auth_mode or "n/a")
        rt.add_row("Temperature", str(cm.get("temperature")))
        rt.add_row("Max tokens", str(cm.get("max_tokens")))
        rt.add_row("Stream", str(cm.get("stream")))
        rt.add_row("Session ID", self.session._session_id or "(none)")
        rt.add_row("History", f"{len(self.session.history)} message(s)")
        self.console.print(Panel(rt, title="[bold cyan]Runtime[/bold cyan]", border_style="cyan"))

        tt = Table(show_header=False, box=None, padding=(0, 1))
        tt.add_column("metric", style="dim")
        tt.add_column("count", style="bold white", justify="right")
        tt.add_row("Input tokens", f"{u.prompt_tokens:,}")
        tt.add_row("Output tokens", f"{u.completion_tokens:,}")
        tt.add_row("Total tokens", f"{u.total_tokens:,}")
        self.console.print(Panel(tt, title="[bold cyan]Tokens[/bold cyan]", border_style="cyan"))

        dotenv_path = get_dotenv_path()
        dotenv_label = f"[green]{dotenv_path}[/green]" if dotenv_path else "[dim]none[/dim]"
        self.console.print(Panel(
            Text.from_markup(f".env: {dotenv_label}"),
            title="[bold cyan]Environment[/bold cyan]",
            border_style="cyan",
        ))

        key_table = Table(title="[bold cyan]API Keys[/bold cyan]", show_header=True, header_style="bold")
        key_table.add_column("Provider", style="cyan")
        key_table.add_column("Env Var", style="dim")
        key_table.add_column("Status")
        key_table.add_column("Source")

        for entry in get_all_key_status(cm.config):
            st = "[green]set[/green]" if entry["available"] else "[red]missing[/red]"
            key_table.add_row(entry["provider"], entry["env_var"], st, entry["source"])

        self.console.print(key_table)
        return True

    async def _config(self, args: List[str]) -> bool:
        sub = args[0].lower() if args else "get"
        if sub == "show":
            yaml_str = self.session.cm.dump_yaml()
            self.console.print(Panel(
                Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False),
                title="[bold cyan]Active Configuration[/bold cyan]",
                border_style="cyan",
            ))
        elif sub == "get":
            target = args[1] if len(args) > 1 else "config.yaml"
            try:
                if os.path.exists(target):
                    self.console.print(f"[yellow]'{target}' exists. Overwrite? (y/n)[/yellow] ", end="")
                    if input().strip().lower() != "y":
                        return True
                with open(target, "w", encoding="utf-8") as f:
                    f.write(self.session.cm.get_default_yaml())
                self.console.print(f"[green]Default config saved to '{target}'.[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            self.console.print("[yellow]Usage:[/yellow] /config show | /config get [path]")
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
        u = self.session.total_usage
        t = Table(title="[bold cyan]Token Usage[/bold cyan]")
        t.add_column("Metric", style="cyan")
        t.add_column("Count", justify="right")
        t.add_row("Input", f"{u.prompt_tokens:,}")
        t.add_row("Output", f"{u.completion_tokens:,}")
        t.add_row("Total", f"{u.total_tokens:,}")
        self.console.print(t)
        return True
