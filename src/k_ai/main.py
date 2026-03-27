# src/k_ai/main.py
"""
Main CLI entry point for k-ai.
"""
import asyncio
import os
import subprocess
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
    help=(
        "Unified LLM CLI and library.\n\n"
        "Main entry points:\n"
        "  • [cyan]k-ai chat[/cyan]     Start the interactive assistant.\n"
        "  • [cyan]k-ai config[/cyan]   Inspect or export the built-in split configuration.\n"
        "  • [cyan]k-ai doctor[/cyan]   Run a full diagnostic of config, memory, providers, and sessions.\n\n"
        "Philosophy:\n"
        "  The CLI shows the available knobs instead of making you remember them.\n"
        "  Use [cyan]--help[/cyan] at any subcommand level for the exact values and workflow."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------

@app.command(
    help=(
        "Start the interactive chat interface.\n\n"
        "This is the main runtime entry point. It loads the merged config, opens the rich UI,\n"
        "shows recent sessions, and lets you control everything from chat or slash commands.\n\n"
        "Typical workflow:\n"
        "  1. Start with defaults: [cyan]k-ai chat[/cyan]\n"
        "  2. Override provider/model for one run: [cyan]k-ai chat -p mistral -m mistral-medium-latest[/cyan]\n"
        "  3. Test a custom config file: [cyan]k-ai chat -c ./config.yaml[/cyan]\n"
        "  4. Force session generation settings: [cyan]k-ai chat -t 0.2 -n 4096[/cyan]\n\n"
        "Inside chat, use [cyan]/help[/cyan] to see all slash commands."
    )
)
def chat(
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help=(
            "Provider override for this run only. Must match a configured provider name under "
            "no_auth, api_key, or oauth. Example: mistral, anthropic, ollama, gemini."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Model override for this run only. Example: mistral-medium-latest, "
            "mistral-large-latest, claude-3-7-sonnet-latest."
        ),
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path to an override YAML config file. This file is merged on top of the built-in "
            "split defaults. Use it when testing a custom provider, prompt, or runtime policy."
        ),
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Sampling temperature override for this run only. Expected range: 0.0 to 2.0.",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        "-n",
        help="Maximum output tokens override for this run only.",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help=(
            "Set a one-off system prompt for the current chat session. This does not rewrite the "
            "default config permanently."
        ),
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

@app.command(
    "config",
    help=(
        "Inspect, export, or enumerate the built-in split configuration.\n\n"
        "Actions:\n"
        "  • [cyan]get[/cyan]       Write the default config to a file.\n"
        "  • [cyan]show[/cyan]      Print the default config or selected sections to the terminal.\n"
        "  • [cyan]sections[/cyan]  List the available built-in config sections and their purpose.\n"
        "  • [cyan]edit[/cyan]      Open either the active override config or one built-in section in your editor.\n\n"
        "Section names currently map to the split default YAML fragments. Use [cyan]k-ai config sections[/cyan]\n"
        "to discover the exact names before exporting only one part.\n\n"
        "Examples:\n"
        "  • [cyan]k-ai config sections[/cyan]\n"
        "  • [cyan]k-ai config show[/cyan]\n"
        "  • [cyan]k-ai config show --section models --section governance[/cyan]\n"
        "  • [cyan]k-ai config get -o config.yaml[/cyan]\n"
        "  • [cyan]k-ai config get -o prompts.yaml -s ui[/cyan]\n"
        "  • [cyan]k-ai config edit all[/cyan]\n"
        "  • [cyan]k-ai config edit models[/cyan]\n"
        "  • [cyan]k-ai config edit mcp[/cyan]"
    ),
)
def config_cmd(
    action: str = typer.Argument(
        "get",
        help=(
            "Config action to run. Allowed values:\n"
            "  • get       Export to a file\n"
            "  • show      Print to the terminal\n"
            "  • sections  List available split config sections\n"
            "  • edit      Open a config file in your configured editor"
        ),
    ),
    target: Optional[str] = typer.Argument(
        None,
        help=(
            "Optional target used mainly by [cyan]config edit[/cyan]. Examples:\n"
            "  • all        -> active/persisted runtime config file\n"
            "  • models     -> 00-models.yaml\n"
            "  • ui         -> 10-ui-prompts.yaml\n"
            "  • sessions   -> 20-sessions-memory.yaml\n"
            "  • governance -> 30-runtime-governance.yaml\n"
            "  • skills     -> 40-skills.yaml\n"
            "  • hooks      -> 50-hooks.yaml\n"
            "  • mcp        -> 60-mcp.yaml"
        ),
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Destination file for [cyan]config get[/cyan]. If omitted, the default output path is "
            "[cyan]config.yaml[/cyan] in the current directory."
        ),
    ),
    section: List[str] = typer.Option(
        None,
        "--section",
        "-s",
        help=(
            "Restrict [cyan]get[/cyan] or [cyan]show[/cyan] to one or more built-in config sections. "
            "Repeat the option to include multiple sections: [cyan]-s models -s governance[/cyan]. "
            "Discover valid names with [cyan]k-ai config sections[/cyan]."
        ),
    ),
):
    """Manage the k-ai configuration file."""
    try:
        cm = ConfigManager()
        sections = list(section or [])
        known_sections = {item["name"] for item in ConfigManager.list_default_sections()}
        if target and not sections and action in {"show", "get"} and target.lower() in known_sections:
            sections = [target.lower()]
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
            syntax_theme = resolve_syntax_theme(cm.get_nested("cli", "theme", default="default"))
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
        elif action == "edit":
            if len(sections) > 1:
                console.print("[red]config edit accepts at most one section.[/red]")
                raise typer.Exit(1)
            target_name = target or (sections[0] if sections else "all")
            target_path = cm.resolve_edit_target(target_name)
            editor_cmd = cm.resolve_editor_command()
            console.print(f"[cyan]Opening[/cyan] {target_path} [cyan]with[/cyan] {' '.join(editor_cmd)}")
            subprocess.run([*editor_cmd, str(target_path)], check=True)
        else:
            console.print(f"[red]Unknown action:[/red] '{action}'. Use get, show, edit, or sections.")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Editor exited with error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

@app.command(
    "doctor",
    help=(
        "Run a full diagnostic of the local k-ai environment.\n\n"
        "The doctor checks the merged config, provider declarations, auth modes, memory store,\n"
        "session store, tool/catalog alignment, and general runtime readability. Use it after\n"
        "changing config files, installing the package, or debugging a provider issue.\n\n"
        "Recovery mode:\n"
        "  • [cyan]--reset config[/cyan]   restore the config to built-in defaults after backup\n"
        "  • [cyan]--reset memory[/cyan]   clear persistent memory after backup\n"
        "  • [cyan]--reset sessions[/cyan] clear persisted sessions after backup\n"
        "  • [cyan]--reset all[/cyan]      backup then reset config + memory + sessions\n\n"
        "Examples:\n"
        "  • [cyan]k-ai doctor[/cyan]\n"
        "  • [cyan]k-ai doctor -c ./config.yaml[/cyan]\n"
        "  • [cyan]k-ai doctor --reset config[/cyan]"
    ),
)
def doctor_cmd(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path to an override YAML config file to validate instead of using only the built-in defaults."
        ),
    ),
    reset: List[str] = typer.Option(
        None,
        "--reset",
        help=(
            "Optional last-resort recovery target. Repeatable. Allowed values: "
            "config, memory, sessions, all. Each reset creates a backup first."
        ),
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

        run_doctor(cm, memory, store, console, reset=reset)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
