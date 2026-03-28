# src/k_ai/doctor.py
"""
Comprehensive diagnostic and recovery helpers for k-ai.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConfigManager
from .memory import MemoryStore, load_external_memory
from .secrets import resolve_secret
from .session_store import SessionStore
from .tools import create_registry
from .tools.base import ToolContext

_RESET_TARGETS = {"config", "memory", "sessions", "all"}


def run_doctor(
    cm: ConfigManager,
    memory: MemoryStore,
    session_store: SessionStore,
    console: Console,
    *,
    reset: Optional[Iterable[str]] = None,
) -> None:
    """Run a full diagnostic, optionally applying last-resort resets."""
    reset_targets = _normalize_reset_targets(reset or [])
    if reset_targets:
        backup_dir = backup_runtime_state(cm, memory, session_store)
        _apply_resets(reset_targets, cm, memory, session_store)
        console.print(
            Panel(
                f"Backup created at: {backup_dir}\nApplied reset targets: {', '.join(reset_targets)}",
                title="[bold yellow]Doctor Recovery[/bold yellow]",
                border_style="yellow",
            )
        )

    console.print("\n[bold cyan]k-ai Doctor[/bold cyan]")
    console.print("[dim]" + "=" * 50 + "[/dim]\n")

    _check_config(cm, console)
    _check_registry_alignment(cm, memory, session_store, console)
    _check_providers(cm, console)
    _check_memory(cm, memory, console)
    _check_sessions(session_store, console)
    _check_qmd(cm, console)
    _check_tools(cm, console)
    _check_dependencies(console)
    _print_recovery_footer(cm, memory, session_store, console)
    console.print()


def _normalize_reset_targets(reset: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in reset:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        if raw not in _RESET_TARGETS:
            raise ValueError(
                f"Unknown doctor reset target '{item}'. "
                "Use one or more of: config, memory, sessions, all."
            )
        if raw == "all":
            return ["config", "memory", "sessions"]
        if raw not in normalized:
            normalized.append(raw)
    return normalized


def backup_runtime_state(cm: ConfigManager, memory: MemoryStore, session_store: SessionStore) -> Path:
    base_dir = Path(str(cm.get_nested("sessions", "directory", default="~/.k-ai/sessions"))).expanduser().parent
    backup_root = base_dir / "doctor-backups"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_dir = backup_root / f"doctor-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    config_path = cm.persist_target_path()
    if config_path.exists():
        shutil.copy2(config_path, backup_dir / config_path.name)

    if memory.path.exists():
        shutil.copy2(memory.path, backup_dir / memory.path.name)

    if session_store.directory.exists():
        shutil.copytree(session_store.directory, backup_dir / "sessions", dirs_exist_ok=True)

    return backup_dir


def _apply_resets(
    reset_targets: List[str],
    cm: ConfigManager,
    memory: MemoryStore,
    session_store: SessionStore,
) -> None:
    if "config" in reset_targets:
        config_path = cm.persist_target_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(ConfigManager.get_default_yaml(), encoding="utf-8")
        cm._load_and_merge()

    if "memory" in reset_targets:
        memory.entries = []
        memory._next_id = 1
        memory.save()
        memory.load()

    if "sessions" in reset_targets:
        if session_store.directory.exists():
            shutil.rmtree(session_store.directory)
        session_store.init()


def _icon(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]ERR[/red]"


def _warn_icon() -> str:
    return "[yellow]WARN[/yellow]"


def _check_config(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    table.add_row(_icon(True), f"Default config: {cm.default_config_path}")
    if cm.override_path:
        exists = cm.override_path.exists()
        table.add_row(_icon(exists), f"User config: {cm.override_path}")
    else:
        table.add_row(_icon(True), f"Persist path: {cm.persist_target_path()}")

    temp = cm.get("temperature", 0.7)
    if isinstance(temp, (int, float)) and temp > 1.5:
        table.add_row(_warn_icon(), f"temperature={temp} (unusually high)")
    else:
        table.add_row(_icon(True), f"temperature={temp}")

    coherence = cm.validate_runtime_coherence()
    for warning in coherence["warnings"]:
        table.add_row(_warn_icon(), warning)
    for error in coherence["errors"]:
        table.add_row(_icon(False), error)
    if not coherence["warnings"] and not coherence["errors"]:
        table.add_row(_icon(True), "Config coherence checks passed")

    console.print(Panel(table, title="[bold cyan]Config[/bold cyan]", border_style="cyan"))


def _check_registry_alignment(
    cm: ConfigManager,
    memory: MemoryStore,
    session_store: SessionStore,
    console: Console,
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    try:
        ctx = ToolContext(
            config=cm,
            memory=memory,
            session_store=session_store,
            console=console,
        )
        registry = create_registry(ctx)
        registry_names = set(registry.get_names())
        raw_catalog = cm.get_nested("tool_approval", "catalog", default={}) or {}
        catalog_names = {
            tool_name
            for group in raw_catalog.values()
            if isinstance(group, dict)
            for tool_name in group.keys()
        }
        missing = sorted(registry_names - catalog_names)
        extra = sorted(catalog_names - registry_names)
        if missing or extra:
            if missing:
                table.add_row(_icon(False), "Missing from tool_approval.catalog: " + ", ".join(missing))
            if extra:
                table.add_row(_icon(False), "Unknown tools in tool_approval.catalog: " + ", ".join(extra))
        else:
            table.add_row(_icon(True), f"Tool registry aligned with catalog ({len(registry_names)} tools)")
    except Exception as exc:
        table.add_row(_icon(False), f"Registry alignment check failed: {exc}")

    console.print(Panel(table, title="[bold cyan]Tool Registry[/bold cyan]", border_style="cyan"))


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


def _check_memory(cm: ConfigManager, memory: MemoryStore, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    ext_path = cm.get_nested("memory", "external_file", default="")
    if ext_path:
        content = load_external_memory(ext_path)
        if content:
            table.add_row(_icon(True), f"External: {ext_path} ({len(content)} bytes)")
        else:
            table.add_row(_warn_icon(), f"External: {ext_path} (not found or empty)")
    else:
        table.add_row("[dim]--[/dim]", "External: not configured")

    ok, msg = memory.validate()
    count = len(memory.entries)
    table.add_row(_icon(ok), f"Internal: {memory.path} ({count} entries) {msg}")

    console.print(Panel(table, title="[bold cyan]Memory[/bold cyan]", border_style="cyan"))


def _check_sessions(store: SessionStore, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    count = store.session_count()
    size = store.total_size_bytes()
    size_label = f"{size / 1024:.1f} KB" if size < 1048576 else f"{size / 1048576:.1f} MB"
    table.add_row(_icon(True), f"{store.directory} ({count} sessions, {size_label})")

    console.print(Panel(table, title="[bold cyan]Sessions[/bold cyan]", border_style="cyan"))


def _check_qmd(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    qmd_path = shutil.which("qmd")
    if qmd_path:
        table.add_row(_icon(True), f"qmd binary: {qmd_path}")
    else:
        table.add_row(_icon(False), "qmd binary: not found in PATH")

    table.add_row(
        _icon(bool(cm.get_nested("tools", "qmd", "enabled", default=True))),
        f"qmd capability: {'enabled' if cm.get_nested('tools', 'qmd', 'enabled', default=True) else 'disabled'}",
    )

    console.print(Panel(table, title="[bold cyan]QMD[/bold cyan]", border_style="cyan"))


def _check_tools(cm: ConfigManager, console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", width=6)
    table.add_column("item")

    py = shutil.which("python3")
    table.add_row(_icon(bool(py)), f"python3: {py or 'not found'}")
    for capability in ("python", "exa", "shell", "qmd"):
        cfg = cm.get_nested("tools", capability, default={})
        enabled = bool(cfg.get("enabled", True))
        label = capability
        if capability == "exa" and enabled:
            env_var = cfg.get("api_key_env_var", "EXA_API_KEY")
            key, _ = resolve_secret(env_var)
            table.add_row(_icon(bool(key)), f"exa auth: {env_var} {'found' if key else 'missing'}")
        table.add_row(_icon(enabled), f"{label}: {'enabled' if enabled else 'disabled'}")

    console.print(Panel(table, title="[bold cyan]Tools[/bold cyan]", border_style="cyan"))


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


def _print_recovery_footer(
    cm: ConfigManager,
    memory: MemoryStore,
    session_store: SessionStore,
    console: Console,
) -> None:
    config_path = cm.persist_target_path()
    base_dir = session_store.directory.parent
    backup_root = base_dir / "doctor-backups"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("field", style="dim", width=14)
    table.add_column("value")
    table.add_row("Config reset", "k-ai doctor --reset config")
    table.add_row("Memory reset", "k-ai doctor --reset memory")
    table.add_row("Sessions reset", "k-ai doctor --reset sessions")
    table.add_row("Full reset", "k-ai doctor --reset all")
    table.add_row("Config path", str(config_path))
    table.add_row("Memory path", str(memory.path))
    table.add_row("Sessions path", str(session_store.directory))
    table.add_row("Backup root", str(backup_root))
    table.add_row("Note", "Every doctor reset creates a timestamped backup before changing local state.")
    console.print(Panel(table, title="[bold cyan]Recovery Paths[/bold cyan]", border_style="cyan"))
