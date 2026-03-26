# src/k_ai/main.py
"""
Main CLI entry point for k-ai.
"""
import asyncio
import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import ConfigManager
from .session import ChatSession
from .ui_theme import resolve_syntax_theme

console = Console()

app = typer.Typer(
    name="k-ai",
    help="k-ai: The Unified LLM CLI and Library.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------

@app.command()
def chat(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Override the active provider."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the active model."
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to a custom config.yaml."
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Override temperature (0.0-2.0)."
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-n", help="Override max_tokens."
    ),
    system: Optional[str] = typer.Option(
        None, "--system", "-s", help="Set a system prompt for the session."
    ),
):
    """Start an interactive chat session."""
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        cm = ConfigManager(override_path=config_path, **kwargs)
        session = ChatSession(cm, provider=provider, model=model)
        if system:
            session.system_prompt = system
        asyncio.run(session.start())
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    action: str = typer.Argument(
        "get",
        help="Action: [cyan]get[/cyan], [cyan]show[/cyan], or [cyan]sections[/cyan].",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Destination file for config get.",
    ),
    section: List[str] = typer.Option(
        None, "--section", "-s", help="Restrict to one or more built-in config sections.",
    ),
):
    """Manage the k-ai configuration file."""
    try:
        cm = ConfigManager()
        sections = list(section or [])
        syntax_theme = resolve_syntax_theme(cm.get_nested("cli", "theme", default="default"))
        if action == "get":
            target = output or "config.yaml"
            if os.path.exists(target) and not output:
                console.print(f"[yellow]'{target}' already exists.[/yellow]")
                raise typer.Exit(1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(cm.get_default_yaml(sections=sections))
            suffix = f" ({', '.join(ConfigManager.normalize_default_sections(sections))})" if sections else ""
            console.print(f"[green]Default config saved to '{target}'{suffix}.[/green]")
        elif action == "show":
            yaml_str = cm.get_default_yaml(sections=sections)
            title = "Default Config"
            if sections:
                title += " [" + ", ".join(ConfigManager.normalize_default_sections(sections)) + "]"
            console.print(Panel(
                Syntax(yaml_str, "yaml", theme=syntax_theme),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
            ))
        elif action in {"sections", "list"}:
            table = Table(title="Config Sections", header_style="bold cyan")
            table.add_column("Section", style="cyan", no_wrap=True)
            table.add_column("File", style="magenta")
            table.add_column("Description", style="white")
            for item in cm.list_default_sections():
                table.add_row(item["name"], item["file"], item["description"])
            console.print(table)
        else:
            console.print(f"[red]Unknown action:[/red] '{action}'. Use get, show, or sections.")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor_cmd(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to a custom config.yaml.",
    ),
):
    """Run a full diagnostic check."""
    from .doctor import run_doctor
    from .memory import MemoryStore
    from .session_store import SessionStore
    from pathlib import Path

    try:
        cm = ConfigManager(override_path=config_path)

        mem_path = cm.get_nested("memory", "internal_file", default="~/.k-ai/MEMORY.json")
        memory = MemoryStore(Path(mem_path))
        memory.load()

        sessions_dir = cm.get_nested("sessions", "directory", default="~/.k-ai/sessions")
        store = SessionStore(sessions_dir)
        store.init()

        run_doctor(cm, memory, store, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
