# src/k_ai/doctor.py
"""
Comprehensive diagnostic for k-ai: config, providers, memory, sessions,
QMD, tools, and dependencies.
"""
import shutil
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConfigManager
from .memory import MemoryStore, load_external_memory
from .session_store import SessionStore
from .secrets import resolve_secret


def run_doctor(
    cm: ConfigManager,
    memory: MemoryStore,
    session_store: SessionStore,
    console: Console,
) -> None:
    """Run a full diagnostic and print results."""
    console.print("\n[bold cyan]k-ai Doctor[/bold cyan]")
    console.print("[dim]" + "=" * 50 + "[/dim]\n")

    _check_config(cm, console)
    _check_providers(cm, console)
    _check_memory(cm, memory, console)
    _check_sessions(session_store, console)
    _check_qmd(cm, console)
    _check_tools(cm, console)
    _check_dependencies(console)
    console.print()


def _icon(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]ERR[/red]"


def _warn_icon() -> str:
    return "[yellow]WARN[/yellow]"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _check_config(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    table.add_row(_icon(True), f"Default config: {cm.default_config_path}")
    if cm.override_path:
        exists = cm.override_path.exists()
        table.add_row(_icon(exists), f"User config: {cm.override_path}")

    temp = cm.get("temperature", 0.7)
    if isinstance(temp, (int, float)) and temp > 1.5:
        table.add_row(_warn_icon(), f"temperature={temp} (unusually high)")
    else:
        table.add_row(_icon(True), f"temperature={temp}")

    console.print(Panel(table, title="[bold cyan]Config[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _check_providers(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Auth")
    table.add_column("Source")

    providers_by_section = cm.list_providers()
    for section, providers in providers_by_section.items():
        for prov_name in providers:
            cfg = cm.config.get(section, {}).get(prov_name, {})

            if section == "no_auth":
                base_url = cfg.get("base_url", "")
                table.add_row(prov_name, _icon(True), "no_auth", f"[dim]{base_url}[/dim]")
            elif section == "api_key":
                env_var = cfg.get("api_key_env_var", "")
                key, source = resolve_secret(env_var) if env_var else (None, None)
                found = key is not None
                src_label = source or "not found"
                table.add_row(prov_name, _icon(found), env_var, f"[dim]{src_label}[/dim]")
            elif section == "oauth":
                oauth_name = cfg.get("oauth_provider_name", "?")
                table.add_row(prov_name, "[dim]oauth[/dim]", oauth_name, "")

    console.print(Panel(table, title="[bold cyan]Providers[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _check_memory(cm: ConfigManager, memory: MemoryStore, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    # External
    ext_path = cm.get_nested("memory", "external_file", default="")
    if ext_path:
        content = load_external_memory(ext_path)
        if content:
            table.add_row(_icon(True), f"External: {ext_path} ({len(content)} bytes)")
        else:
            table.add_row(_warn_icon(), f"External: {ext_path} (not found or empty)")
    else:
        table.add_row("[dim]--[/dim]", "External: not configured")

    # Internal
    ok, msg = memory.validate()
    count = len(memory.entries)
    table.add_row(_icon(ok), f"Internal: {memory.path} ({count} entries) {msg}")

    console.print(Panel(table, title="[bold cyan]Memory[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _check_sessions(store: SessionStore, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    count = store.session_count()
    size = store.total_size_bytes()
    size_label = f"{size / 1024:.1f} KB" if size < 1048576 else f"{size / 1048576:.1f} MB"
    table.add_row(
        _icon(True),
        f"{store.directory} ({count} sessions, {size_label})",
    )

    console.print(Panel(table, title="[bold cyan]Sessions[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# QMD
# ---------------------------------------------------------------------------

def _check_qmd(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    qmd_path = shutil.which("qmd")
    if qmd_path:
        table.add_row(_icon(True), f"qmd binary: {qmd_path}")
    else:
        table.add_row(_icon(False), "qmd binary: not found in PATH")

    console.print(Panel(table, title="[bold cyan]QMD[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _check_tools(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    # Python
    py = shutil.which("python3")
    table.add_row(_icon(bool(py)), f"python3: {py or 'not found'}")

    # Exa
    exa_cfg = cm.get_nested("tools", "exa_search", default={})
    if exa_cfg.get("enabled", False):
        env_var = exa_cfg.get("api_key_env_var", "EXA_API_KEY")
        key, _ = resolve_secret(env_var)
        table.add_row(_icon(bool(key)), f"exa_search: {env_var} {'found' if key else 'missing'}")
    else:
        table.add_row("[dim]--[/dim]", "exa_search: disabled")

    # Shell
    shell_cfg = cm.get_nested("tools", "shell_exec", default={})
    table.add_row(
        _icon(shell_cfg.get("enabled", True)),
        f"shell_exec: {'enabled' if shell_cfg.get('enabled', True) else 'disabled'}"
    )

    console.print(Panel(table, title="[bold cyan]Tools[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _check_dependencies(console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    for pkg_name in ["rich", "litellm", "prompt_toolkit", "httpx", "pydantic", "yaml"]:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "?")
            table.add_row(_icon(True), f"{pkg_name} {version}")
        except ImportError:
            table.add_row(_icon(False), f"{pkg_name}: not installed")

    table.add_row(_icon(True), f"Python {sys.version.split()[0]}")

    console.print(Panel(table, title="[bold cyan]Dependencies[/bold cyan]", border_style="cyan"))
