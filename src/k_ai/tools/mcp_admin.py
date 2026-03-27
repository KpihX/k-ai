"""Administrative tools for the MCP runtime."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from rich.table import Table

from ..mcp.installers import (
    build_install_plan,
    guess_binary_name,
    guess_official_package_name,
    install_npm_package,
    npm_view_package,
    resolve_local_command,
    which_binary,
)
from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


def _server_cfg_from_args(arguments: Dict[str, Any]) -> dict[str, Any]:
    name = str(arguments.get("server_name", "") or "").strip()
    transport = str(arguments.get("transport", "stdio") or "stdio").strip().lower()
    enabled = bool(arguments.get("enabled", True))
    cfg: dict[str, Any] = {
        "enabled": enabled,
        "transport": transport,
        "roots": {
            "enabled": bool(arguments.get("roots_enabled", True)),
            "include_workspace_root": bool(arguments.get("include_workspace_root", True)),
            "additional_paths": list(arguments.get("additional_paths", []) or []),
        },
        "tools": {
            "include": list(arguments.get("include_tools", []) or []),
            "exclude": list(arguments.get("exclude_tools", []) or []),
        },
    }
    if transport == "stdio":
        cfg["command"] = str(arguments.get("command", "") or "").strip()
        cfg["args"] = list(arguments.get("args", []) or [])
        cwd = str(arguments.get("cwd", "") or "").strip()
        if cwd:
            cfg["cwd"] = cwd
        env = dict(arguments.get("env", {}) or {})
        if env:
            cfg["env"] = env
    else:
        cfg["url"] = str(arguments.get("url", "") or "").strip()
        headers = dict(arguments.get("headers", {}) or {})
        if headers:
            cfg["headers"] = headers
    if name:
        cfg.setdefault("meta", {})
    return cfg


class MCPServerListTool(InternalTool):
    name = "mcp_server_list"
    display_name = "List MCP Servers"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "List configured MCP servers plus discovered runtime state."
    parameters_schema = {"type": "object", "properties": {"force_refresh": {"type": "boolean"}}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.get_mcp_catalog is None:
            return ToolResult(success=False, message="MCP catalog is unavailable.")
        catalog = await ctx.get_mcp_catalog(force_refresh=bool(arguments.get("force_refresh", False)))
        lines = [f"servers={len(catalog.servers)} tools={len(catalog.tools)} resources={len(catalog.resources)} prompts={len(catalog.prompts)}"]
        for server in catalog.servers:
            lines.append(
                f"- {server.spec.name}: transport={server.spec.transport} tools={len(server.tools)} resources={len(server.resources)} prompts={len(server.prompts)}"
            )
        for issue in catalog.issues:
            lines.append(f"- issue: {issue.server_name}: {issue.message}")
        return ToolResult(success=True, message="\n".join(lines), data=catalog)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        catalog = result.data
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Server", style="cyan")
        table.add_column("Transport", style="magenta")
        table.add_column("Tools", justify="right")
        table.add_column("Resources", justify="right")
        table.add_column("Prompts", justify="right")
        for server in getattr(catalog, "servers", ()):
            table.add_row(server.spec.name, server.spec.transport, str(len(server.tools)), str(len(server.resources)), str(len(server.prompts)))
        return table


class MCPServerProbeTool(InternalTool):
    name = "mcp_server_probe"
    display_name = "Probe MCP Server"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "Probe local commands and likely npm package names for an MCP server."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string"},
            "command": {"type": "string"},
            "package_name": {"type": "string"},
        },
        "required": ["server_name"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        server_name = str(arguments.get("server_name", "") or "").strip()
        command = str(arguments.get("command", "") or "").strip()
        package_name = str(arguments.get("package_name", "") or "").strip()
        binary_name = guess_binary_name(server_name, explicit_binary=command)
        local_command = resolve_local_command(command or binary_name)
        guessed_package = guess_official_package_name(server_name, explicit_package=package_name)
        npm_info = npm_view_package(guessed_package)
        lines = [
            f"server_name={server_name}",
            f"binary_name={binary_name}",
            f"local_command={local_command or '(not found)'}",
            f"package_name={guessed_package}",
        ]
        if npm_info:
            lines.append(f"npm_version={npm_info.get('version', '')}")
            lines.append(f"npm_bin={npm_info.get('bin', {})}")
        return ToolResult(success=True, message="\n".join(lines), data={"local_command": local_command, "package_name": guessed_package, "npm": npm_info or {}})


class MCPServerUpsertTool(InternalTool):
    name = "mcp_server_upsert"
    display_name = "Upsert MCP Server"
    category = "mcp-admin"
    danger_level = "high"
    accent_color = "yellow"
    description = "Create or update one MCP server definition in the runtime config."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string"},
            "transport": {"type": "string", "enum": ["stdio", "streamable_http", "sse"]},
            "enabled": {"type": "boolean"},
            "command": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
            "url": {"type": "string"},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "roots_enabled": {"type": "boolean"},
            "include_workspace_root": {"type": "boolean"},
            "additional_paths": {"type": "array", "items": {"type": "string"}},
            "include_tools": {"type": "array", "items": {"type": "string"}},
            "exclude_tools": {"type": "array", "items": {"type": "string"}},
            "persist": {"type": "boolean"},
        },
        "required": ["server_name", "transport"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.apply_config_change is None or ctx.refresh_mcp_catalog is None:
            return ToolResult(success=False, message="MCP admin is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        if not server_name:
            return ToolResult(success=False, message="server_name is required.")
        cfg = _server_cfg_from_args(arguments)
        transport = cfg.get("transport", "stdio")
        if transport == "stdio" and not str(cfg.get("command", "")).strip():
            return ToolResult(success=False, message="command is required for stdio MCP servers.")
        if transport in {"streamable_http", "sse"} and not str(cfg.get("url", "")).strip():
            return ToolResult(success=False, message="url is required for HTTP/SSE MCP servers.")
        change = ctx.apply_config_change(f"mcp.servers.{server_name}", cfg, persist=bool(arguments.get("persist", True)))
        await ctx.refresh_mcp_catalog()
        return ToolResult(success=True, message=f"MCP server '{server_name}' upserted.", data=change)


class MCPServerRemoveTool(InternalTool):
    name = "mcp_server_remove"
    display_name = "Remove MCP Server"
    category = "mcp-admin"
    danger_level = "high"
    accent_color = "yellow"
    description = "Remove one MCP server definition from the runtime config."
    parameters_schema = {
        "type": "object",
        "properties": {"server_name": {"type": "string"}, "persist": {"type": "boolean"}},
        "required": ["server_name"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        server_name = str(arguments.get("server_name", "") or "").strip()
        if not server_name:
            return ToolResult(success=False, message="server_name is required.")
        raw_servers = dict(ctx.config.get_nested("mcp", "servers", default={}) or {})
        if server_name not in raw_servers:
            return ToolResult(success=False, message=f"Unknown MCP server '{server_name}'.")
        del raw_servers[server_name]
        if ctx.apply_config_change is None or ctx.refresh_mcp_catalog is None:
            return ToolResult(success=False, message="MCP admin is unavailable.")
        change = ctx.apply_config_change("mcp.servers", raw_servers, persist=bool(arguments.get("persist", True)))
        await ctx.refresh_mcp_catalog()
        return ToolResult(success=True, message=f"MCP server '{server_name}' removed.", data=change)


class MCPServerSetEnabledTool(InternalTool):
    name = "mcp_server_set_enabled"
    display_name = "Enable Or Disable MCP Server"
    category = "mcp-admin"
    danger_level = "medium"
    accent_color = "yellow"
    description = "Enable or disable one configured MCP server."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string"},
            "enabled": {"type": "boolean"},
            "persist": {"type": "boolean"},
        },
        "required": ["server_name", "enabled"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.apply_config_change is None or ctx.refresh_mcp_catalog is None:
            return ToolResult(success=False, message="MCP admin is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        enabled = bool(arguments.get("enabled", False))
        current = ctx.config.get_path(f"mcp.servers.{server_name}")
        if not isinstance(current, dict):
            return ToolResult(success=False, message=f"Unknown MCP server '{server_name}'.")
        updated = dict(current)
        updated["enabled"] = enabled
        change = ctx.apply_config_change(f"mcp.servers.{server_name}", updated, persist=bool(arguments.get("persist", True)))
        await ctx.refresh_mcp_catalog()
        state = "enabled" if enabled else "disabled"
        return ToolResult(success=True, message=f"MCP server '{server_name}' {state}.", data=change)


class MCPServerInstallTool(InternalTool):
    name = "mcp_server_install"
    display_name = "Install MCP Server"
    category = "mcp-admin"
    danger_level = "high"
    accent_color = "yellow"
    description = "Install a stdio MCP server package and register it in the runtime config."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string"},
            "package_name": {"type": "string"},
            "binary_name": {"type": "string"},
            "package_manager": {"type": "string", "enum": ["auto", "bun", "npm"]},
            "cwd": {"type": "string"},
            "persist": {"type": "boolean"},
        },
        "required": ["server_name"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.apply_config_change is None or ctx.refresh_mcp_catalog is None:
            return ToolResult(success=False, message="MCP admin is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        if not server_name:
            return ToolResult(success=False, message="server_name is required.")
        plan = build_install_plan(
            server_name=server_name,
            package_name=str(arguments.get("package_name", "") or ""),
            binary_name=str(arguments.get("binary_name", "") or ""),
            package_manager=str(arguments.get("package_manager", "auto") or "auto"),
        )
        install_result = install_npm_package(plan.package_name, package_manager=plan.package_manager)
        resolved_command = resolve_local_command(plan.binary_name)
        if not resolved_command:
            resolved_command = which_binary(plan.binary_name)
        if not resolved_command:
            return ToolResult(success=False, message=f"Package installed but binary '{plan.binary_name}' was not found on PATH.")
        cfg = {
            "enabled": True,
            "transport": "stdio",
            "command": resolved_command,
            "args": [],
            "cwd": str(arguments.get("cwd", "{workspace_root}") or "{workspace_root}"),
            "env": {},
            "roots": {
                "enabled": True,
                "include_workspace_root": True,
                "additional_paths": [],
            },
            "tools": {"include": [], "exclude": []},
        }
        change = ctx.apply_config_change(f"mcp.servers.{server_name}", cfg, persist=bool(arguments.get("persist", True)))
        await ctx.refresh_mcp_catalog()
        return ToolResult(
            success=True,
            message=f"Installed MCP server '{server_name}' from {plan.package_name} using {install_result['package_manager']}.",
            data={"config_change": change, "install": install_result, "resolved_command": resolved_command},
        )


class MCPResourceListTool(InternalTool):
    name = "mcp_resource_list"
    display_name = "List MCP Resources"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "List MCP resources advertised by one or all configured servers."
    parameters_schema = {
        "type": "object",
        "properties": {"server_name": {"type": "string"}, "force_refresh": {"type": "boolean"}},
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.get_mcp_catalog is None:
            return ToolResult(success=False, message="MCP catalog is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        catalog = await ctx.get_mcp_catalog(force_refresh=bool(arguments.get("force_refresh", False)))
        resources = [
            resource for resource in catalog.resources
            if not server_name or resource.server_name == server_name
        ]
        lines = [f"resources={len(resources)}"]
        for item in resources:
            lines.append(f"- {item.server_name}: {item.uri}")
        return ToolResult(success=True, message="\n".join(lines), data={"resources": resources})


class MCPResourceReadTool(InternalTool):
    name = "mcp_resource_read"
    display_name = "Read MCP Resource"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "Read one MCP resource by server name and URI."
    parameters_schema = {
        "type": "object",
        "properties": {"server_name": {"type": "string"}, "uri": {"type": "string"}},
        "required": ["server_name", "uri"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.mcp_read_resource is None:
            return ToolResult(success=False, message="MCP resource reads are unavailable.")
        result = await ctx.mcp_read_resource(
            str(arguments.get("server_name", "") or "").strip(),
            str(arguments.get("uri", "") or "").strip(),
        )
        return ToolResult(success=True, message=result.message, data=result.raw)


class MCPPromptListTool(InternalTool):
    name = "mcp_prompt_list"
    display_name = "List MCP Prompts"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "List MCP prompts advertised by one or all configured servers."
    parameters_schema = {
        "type": "object",
        "properties": {"server_name": {"type": "string"}, "force_refresh": {"type": "boolean"}},
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.get_mcp_catalog is None:
            return ToolResult(success=False, message="MCP catalog is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        catalog = await ctx.get_mcp_catalog(force_refresh=bool(arguments.get("force_refresh", False)))
        prompts = [prompt for prompt in catalog.prompts if not server_name or prompt.server_name == server_name]
        lines = [f"prompts={len(prompts)}"]
        for item in prompts:
            lines.append(f"- {item.server_name}: {item.name}")
        return ToolResult(success=True, message="\n".join(lines), data={"prompts": prompts})


class MCPResourceTemplateListTool(InternalTool):
    name = "mcp_resource_template_list"
    display_name = "List MCP Resource Templates"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "List MCP resource templates advertised by one or all configured servers."
    parameters_schema = {
        "type": "object",
        "properties": {"server_name": {"type": "string"}, "force_refresh": {"type": "boolean"}},
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.get_mcp_catalog is None:
            return ToolResult(success=False, message="MCP catalog is unavailable.")
        server_name = str(arguments.get("server_name", "") or "").strip()
        catalog = await ctx.get_mcp_catalog(force_refresh=bool(arguments.get("force_refresh", False)))
        templates = [
            template for template in catalog.resource_templates
            if not server_name or template.server_name == server_name
        ]
        lines = [f"resource_templates={len(templates)}"]
        for item in templates:
            lines.append(f"- {item.server_name}: {item.uri_template}")
        return ToolResult(success=True, message="\n".join(lines), data={"resource_templates": templates})


class MCPPromptGetTool(InternalTool):
    name = "mcp_prompt_get"
    display_name = "Get MCP Prompt"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "Load one MCP prompt by server name and prompt name."
    parameters_schema = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string"},
            "prompt_name": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["server_name", "prompt_name"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.mcp_get_prompt is None:
            return ToolResult(success=False, message="MCP prompt loading is unavailable.")
        result = await ctx.mcp_get_prompt(
            str(arguments.get("server_name", "") or "").strip(),
            str(arguments.get("prompt_name", "") or "").strip(),
            arguments={str(k): str(v) for k, v in dict(arguments.get("arguments", {}) or {}).items()},
        )
        message = result.message
        if result.description:
            message = f"{result.description}\n\n{message}".strip()
        return ToolResult(success=True, message=message, data=result.raw)


class MCPCatalogReloadTool(InternalTool):
    name = "mcp_catalog_reload"
    display_name = "Reload MCP Catalog"
    category = "mcp-admin"
    danger_level = "low"
    accent_color = "cyan"
    description = "Refresh MCP server discovery and imported MCP tools."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.refresh_mcp_catalog is None:
            return ToolResult(success=False, message="MCP reload is unavailable.")
        await ctx.refresh_mcp_catalog()
        return ToolResult(success=True, message="MCP catalog refreshed.")


def register_mcp_admin_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    registry.register(MCPServerListTool())
    registry.register(MCPServerProbeTool())
    registry.register(MCPServerUpsertTool())
    registry.register(MCPServerRemoveTool())
    registry.register(MCPServerSetEnabledTool())
    registry.register(MCPServerInstallTool())
    registry.register(MCPResourceListTool())
    registry.register(MCPResourceTemplateListTool())
    registry.register(MCPResourceReadTool())
    registry.register(MCPPromptListTool())
    registry.register(MCPPromptGetTool())
    registry.register(MCPCatalogReloadTool())
