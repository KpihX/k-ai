"""Helpers for installing and resolving MCP servers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MCPInstallPlan:
    server_name: str
    transport: str
    package_name: str
    binary_name: str
    package_manager: str


def which_binary(name: str) -> str | None:
    resolved = shutil.which(str(name or "").strip())
    return resolved or None


def guess_binary_name(server_name: str, explicit_binary: str = "") -> str:
    if explicit_binary.strip():
        return explicit_binary.strip()
    normalized = str(server_name or "").strip().lower().replace("_", "-").replace(" ", "-")
    if normalized.startswith("mcp-server-"):
        return normalized
    return f"mcp-server-{normalized}"


def guess_official_package_name(server_name: str, explicit_package: str = "") -> str:
    if explicit_package.strip():
        return explicit_package.strip()
    normalized = str(server_name or "").strip().lower().replace("_", "-").replace(" ", "-")
    if normalized.startswith("@"):
        return normalized
    if normalized.startswith("server-"):
        normalized = normalized[len("server-") :]
    if normalized.startswith("mcp-server-"):
        normalized = normalized[len("mcp-server-") :]
    return f"@modelcontextprotocol/server-{normalized}"


def select_package_manager(preferred: str = "auto") -> str:
    normalized = str(preferred or "auto").strip().lower()
    if normalized in {"bun", "npm"}:
        if shutil.which(normalized):
            return normalized
        raise RuntimeError(f"Requested package manager '{normalized}' is not installed.")
    if shutil.which("bun"):
        return "bun"
    if shutil.which("npm"):
        return "npm"
    raise RuntimeError("Neither bun nor npm is available for MCP package installation.")


def install_npm_package(package_name: str, package_manager: str = "auto") -> dict[str, str]:
    manager = select_package_manager(package_manager)
    if manager == "bun":
        cmd = ["bun", "add", "--global", package_name]
    else:
        cmd = ["npm", "install", "-g", package_name]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(stderr or f"{manager} failed with exit code {proc.returncode}.")
    return {"package_manager": manager, "stdout": (proc.stdout or "").strip(), "stderr": (proc.stderr or "").strip()}


def npm_view_package(package_name: str) -> dict[str, object] | None:
    if not shutil.which("npm"):
        return None
    proc = subprocess.run(
        ["npm", "view", package_name, "version", "bin", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def build_install_plan(
    *,
    server_name: str,
    package_name: str = "",
    binary_name: str = "",
    package_manager: str = "auto",
) -> MCPInstallPlan:
    resolved_package = guess_official_package_name(server_name, explicit_package=package_name)
    resolved_binary = guess_binary_name(server_name, explicit_binary=binary_name)
    manager = select_package_manager(package_manager)
    return MCPInstallPlan(
        server_name=server_name,
        transport="stdio",
        package_name=resolved_package,
        binary_name=resolved_binary,
        package_manager=manager,
    )


def resolve_local_command(command_or_path: str) -> str | None:
    text = str(command_or_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute() or "/" in text:
        resolved = path.resolve()
        return str(resolved) if resolved.exists() else None
    return which_binary(text)
