# src/k_ai/main.py
"""
Main CLI entry point for k-ai.
"""
import asyncio
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .config import ConfigManager
from .session import ChatSession

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
        help="Action: [cyan]get[/cyan] (export template) or [cyan]show[/cyan] (print).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Destination file for config get.",
    ),
):
    """Manage the k-ai configuration file."""
    try:
        cm = ConfigManager()
        if action == "get":
            target = output or "config.yaml"
            if os.path.exists(target) and not output:
                console.print(f"[yellow]'{target}' already exists.[/yellow]")
                raise typer.Exit(1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(cm.get_default_yaml())
            console.print(f"[green]Default config saved to '{target}'.[/green]")
        elif action == "show":
            yaml_str = cm.get_default_yaml()
            console.print(Panel(
                Syntax(yaml_str, "yaml", theme="monokai"),
                title="[bold cyan]Default Config[/bold cyan]",
                border_style="cyan",
            ))
        else:
            console.print(f"[red]Unknown action:[/red] '{action}'. Use get or show.")
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
