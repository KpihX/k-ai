from __future__ import annotations

from typing import Any


TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "exa": {
        "label": "Exa web search",
        "tools": ["exa_search"],
        "mutable": True,
        "description": "Public web search through Exa.",
    },
    "python": {
        "label": "Python sandbox",
        "tools": [
            "python_exec",
            "python_sandbox_list_packages",
            "python_sandbox_install_packages",
            "python_sandbox_remove_packages",
        ],
        "mutable": True,
        "description": "Sandboxed Python execution and sandbox package management.",
    },
    "shell": {
        "label": "Shell sandbox",
        "tools": ["shell_exec"],
        "mutable": True,
        "description": "Sandboxed shell command execution.",
    },
    "qmd": {
        "label": "QMD knowledge tools",
        "tools": [
            "qmd_query",
            "qmd_search",
            "qmd_vsearch",
            "qmd_get",
            "qmd_multi_get",
            "qmd_ls",
            "qmd_collection_list",
            "qmd_collection_add",
            "qmd_collection_remove",
            "qmd_collection_show",
            "qmd_status",
            "qmd_update",
            "qmd_embed",
            "qmd_cleanup",
            "qmd_context_list",
            "qmd_context_add",
            "qmd_context_rm",
        ],
        "mutable": True,
        "description": "QMD-backed search, retrieval, collection, and maintenance tools.",
    },
    "mcp": {
        "label": "MCP server tools",
        "tools": [],
        "mutable": True,
        "description": "Dynamic tools imported from configured MCP servers.",
    },
}

TOOL_TO_CAPABILITY: dict[str, str] = {
    tool_name: capability
    for capability, spec in TOOL_CAPABILITIES.items()
    for tool_name in spec["tools"]
}


def list_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": spec["label"],
            "tools": list(spec["tools"]),
            "mutable": bool(spec.get("mutable", False)),
            "description": spec.get("description", ""),
        }
        for name, spec in TOOL_CAPABILITIES.items()
    ]


def normalize_capability_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized not in TOOL_CAPABILITIES:
        raise ValueError(
            "Unknown tool capability "
            f"'{name}'. Expected one of: {', '.join(sorted(TOOL_CAPABILITIES))}"
        )
    return normalized


def capability_for_tool(tool_name: str) -> str | None:
    if str(tool_name or "").startswith("mcp__"):
        return "mcp"
    return TOOL_TO_CAPABILITY.get(tool_name)


def capability_enabled(config: Any, capability: str) -> bool:
    name = normalize_capability_name(capability)
    return bool(config.get_nested("tools", name, "enabled", default=True))


def tool_enabled(config: Any, tool_name: str) -> bool:
    capability = capability_for_tool(tool_name)
    if capability is None:
        return True
    return capability_enabled(config, capability)
