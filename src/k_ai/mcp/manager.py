"""High-level MCP runtime orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..config import ConfigManager
from .client import MCPClient, MCPClientError, mcp_sdk_available
from .models import (
    MCPCatalog,
    MCPIssue,
    MCPPromptGetResult,
    MCPPromptSpec,
    MCPResourceReadResult,
    MCPResourceSpec,
    MCPRootSpec,
    MCPServerSnapshot,
    MCPServerSpec,
    MCPToolCallResult,
    MCPToolSpec,
)


class MCPManager:
    def __init__(self, config: ConfigManager, workspace_root: Path | None = None):
        self._config = config
        self._workspace_root = (workspace_root or Path.cwd()).expanduser().resolve()
        self._session_cwd = self._workspace_root
        self._catalog = MCPCatalog()
        self._client = MCPClient(
            protocol_version=str(self._config.get_nested("mcp", "protocol_version", default="2025-06-18") or "2025-06-18"),
            client_name=str(self._config.get_nested("mcp", "client", "name", default="k-ai") or "k-ai"),
            client_version=str(self._config.get_nested("mcp", "client", "version", default="0.2.0") or "0.2.0"),
            tool_prefix_template=str(self._config.get_nested("mcp", "tool_naming", "prefix_template", default="mcp__{server_name}__") or "mcp__{server_name}__"),
            read_timeout_seconds=int(self._config.get_nested("mcp", "read_timeout_seconds", default=30) or 30),
        )

    def enabled(self) -> bool:
        return bool(self._config.get_nested("mcp", "enabled", default=True)) and mcp_sdk_available()

    async def refresh(self) -> MCPCatalog:
        self._catalog = await self._discover_catalog()
        return self._catalog

    async def catalog(self, force_refresh: bool = False) -> MCPCatalog:
        if force_refresh or (not self._catalog.servers and not self._catalog.issues):
            self._catalog = await self._discover_catalog()
        return self._catalog

    def current_catalog(self) -> MCPCatalog:
        return self._catalog

    def set_workspace_root(self, workspace_root: Path | None) -> Path:
        resolved = (workspace_root or Path.cwd()).expanduser().resolve()
        self._workspace_root = resolved
        if self._session_cwd is None:
            self._session_cwd = resolved
        return resolved

    def set_session_cwd(self, cwd: Path | None) -> Path:
        resolved = (cwd or self._workspace_root).expanduser().resolve()
        self._session_cwd = resolved
        return resolved

    async def tools(self, force_refresh: bool = False) -> tuple[MCPToolSpec, ...]:
        return (await self.catalog(force_refresh=force_refresh)).tools

    async def resources(self, force_refresh: bool = False) -> tuple[MCPResourceSpec, ...]:
        return (await self.catalog(force_refresh=force_refresh)).resources

    async def prompts(self, force_refresh: bool = False) -> tuple[MCPPromptSpec, ...]:
        return (await self.catalog(force_refresh=force_refresh)).prompts

    async def runtime_summary(self, force_refresh: bool = False) -> str:
        catalog = await self.catalog(force_refresh=force_refresh)
        return self._runtime_summary_for(catalog)

    def runtime_summary_cached(self) -> str:
        return self._runtime_summary_for(self._catalog)

    def _runtime_summary_for(self, catalog: MCPCatalog) -> str:
        if bool(self._config.get_nested("mcp", "enabled", default=True)) and not mcp_sdk_available():
            return self.config_text(
                "mcp",
                "runtime",
                "runtime_missing_sdk_message",
                default="disabled (missing Python MCP SDK)",
            )
        if not catalog.servers and not catalog.issues:
            return self.config_text("mcp", "runtime", "runtime_none_message", default="none")
        return self.config_text(
            "mcp",
            "runtime",
            "summary_template",
            default="{server_count} server(s) | {tool_count} tool(s) | {resource_count} resource(s) | {prompt_count} prompt(s)",
            server_count=str(len(catalog.servers)),
            tool_count=str(len(catalog.tools)),
            resource_count=str(len(catalog.resources)),
            prompt_count=str(len(catalog.prompts)),
        )

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> MCPToolCallResult:
        tool = await self.get_tool(qualified_name)
        if tool is None:
            raise MCPClientError(f"Unknown MCP tool '{qualified_name}'.")
        spec = self._server_spec_by_name(tool.server_name)
        if spec is None:
            raise MCPClientError(f"MCP server '{tool.server_name}' is unavailable.")
        return await self._client.call_tool(
            spec=spec,
            tool_name=tool.name,
            qualified_name=tool.qualified_name,
            arguments=arguments or {},
        )

    async def get_tool(self, qualified_name: str) -> MCPToolSpec | None:
        for tool in await self.tools(force_refresh=False):
            if tool.qualified_name == qualified_name:
                return tool
        return None

    async def get_resource(self, server_name: str, uri: str) -> MCPResourceSpec | None:
        for item in await self.resources(force_refresh=False):
            if item.server_name == server_name and item.uri == uri:
                return item
        return None

    async def get_prompt(self, server_name: str, prompt_name: str) -> MCPPromptSpec | None:
        for item in await self.prompts(force_refresh=False):
            if item.server_name == server_name and item.name == prompt_name:
                return item
        return None

    async def read_resource(self, server_name: str, uri: str) -> MCPResourceReadResult:
        spec = self._server_spec_by_name(server_name)
        if spec is None:
            raise MCPClientError(f"MCP server '{server_name}' is unavailable.")
        return await self._client.read_resource(spec=spec, uri=uri)

    async def load_prompt(self, server_name: str, prompt_name: str, arguments: dict[str, str] | None = None) -> MCPPromptGetResult:
        spec = self._server_spec_by_name(server_name)
        if spec is None:
            raise MCPClientError(f"MCP server '{server_name}' is unavailable.")
        return await self._client.get_prompt(spec=spec, prompt_name=prompt_name, arguments=arguments or {})

    def config_text(self, *parts: str, default: str = "", **vars: str) -> str:
        raw = str(self._config.get_nested(*parts, default=default) or default)
        if not vars:
            return raw

        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        return raw.format_map(_SafeDict({key: str(value) for key, value in vars.items()}))

    async def render_catalog_section(self, force_refresh: bool = False) -> str:
        if not self.enabled():
            return ""
        if not bool(self._config.get_nested("mcp", "include_in_system_prompt", default=True)):
            return ""
        catalog = await self.catalog(force_refresh=force_refresh)
        return self._render_catalog(catalog)

    def render_catalog_section_cached(self) -> str:
        if not self.enabled():
            return ""
        if not bool(self._config.get_nested("mcp", "include_in_system_prompt", default=True)):
            return ""
        return self._render_catalog(self._catalog)

    def _render_catalog(self, catalog: MCPCatalog) -> str:
        if not catalog.servers and not catalog.issues:
            return ""
        lines: list[str] = []
        intro = str(self._config.get_nested("mcp", "prompts", "catalog_intro", default="") or "").strip()
        if intro:
            lines.append(intro)
        for server in catalog.servers:
            lines.append(
                self.config_text(
                    "mcp",
                    "runtime",
                    "server_line_template",
                    default="- {server_name}: transport={transport} tools={tool_count}",
                    server_name=server.server_title,
                    transport=server.spec.transport,
                    tool_count=str(len(server.tools)),
                )
            )
            for tool in server.tools:
                lines.append(
                    self.config_text(
                        "mcp",
                        "runtime",
                        "tool_line_template",
                        default="  - {tool_name}: {tool_description}",
                        tool_name=tool.qualified_name,
                        tool_description=tool.description or "(no description)",
                    )
                )
            if server.resources:
                lines.append(f"  - resources: {len(server.resources)}")
            if server.prompts:
                lines.append(f"  - prompts: {len(server.prompts)}")
        for issue in catalog.issues:
            lines.append(
                self.config_text(
                    "mcp",
                    "runtime",
                    "issues_template",
                    default="- issue: {issue_message}",
                    issue_message=f"{issue.server_name}: {issue.message}",
                )
            )
        return "\n".join(line for line in lines if line).strip()

    async def _discover_catalog(self) -> MCPCatalog:
        if not self.enabled():
            if bool(self._config.get_nested("mcp", "enabled", default=True)) and not mcp_sdk_available():
                return MCPCatalog(
                    issues=(
                        MCPIssue(
                            server_name="mcp",
                            message=self.config_text(
                                "mcp",
                                "runtime",
                                "runtime_missing_sdk_message",
                                default="disabled (missing Python MCP SDK)",
                            ),
                        ),
                    ),
                )
            return MCPCatalog()

        servers: list[MCPServerSnapshot] = []
        issues: list[MCPIssue] = []
        max_tools = int(self._config.get_nested("mcp", "max_tools_per_server", default=64) or 64)
        max_resources = int(self._config.get_nested("mcp", "max_resources_per_server", default=128) or 128)
        max_prompts = int(self._config.get_nested("mcp", "max_prompts_per_server", default=128) or 128)

        for spec in self._build_server_specs():
            if not spec.enabled:
                continue
            try:
                snapshot = await self._client.inspect_server(
                    spec,
                    max_tools=max_tools,
                    max_resources=max_resources,
                    max_prompts=max_prompts,
                )
            except Exception as exc:
                issues.append(MCPIssue(server_name=spec.name, message=str(exc)))
                continue
            servers.append(snapshot)

        return MCPCatalog(servers=tuple(servers), issues=tuple(issues))

    def _build_server_specs(self) -> tuple[MCPServerSpec, ...]:
        raw = self._config.get_nested("mcp", "servers", default={}) or {}
        if not isinstance(raw, dict):
            raise ValueError("mcp.servers must be a mapping.")
        servers: list[MCPServerSpec] = []
        for name, payload in raw.items():
            if not isinstance(payload, dict):
                raise ValueError(f"mcp.servers.{name} must be a mapping.")
            transport = str(payload.get("transport", "stdio") or "stdio").strip().lower()
            endpoint = str(payload.get("command", payload.get("url", "")) or "").strip()
            args = tuple(str(item) for item in (payload.get("args", []) or []))
            cwd_raw = str(payload.get("cwd", "") or "").strip()
            cwd = self._resolve_path_template(cwd_raw) if cwd_raw else None
            headers = {
                str(key): self._render_template(str(value))
                for key, value in ((payload.get("headers", {}) or {}) if transport in {"streamable_http", "sse"} else (payload.get("env", {}) or {})).items()
                if str(key).strip()
            }
            stderr_mode = str(
                payload.get(
                    "stderr",
                    self._config.get_nested("mcp", "runtime", "stdio_stderr_mode", default="quiet"),
                )
                or "quiet"
            ).strip().lower()
            roots = self._build_roots(payload.get("roots", {}) or {})
            effective_args = self._effective_server_args(
                server_name=str(name).strip(),
                command=self._render_template(endpoint),
                args=tuple(self._render_template(item) for item in args),
                roots=roots,
            )
            tools_cfg = payload.get("tools", {}) or {}
            servers.append(
                MCPServerSpec(
                    name=str(name).strip(),
                    enabled=bool(payload.get("enabled", True)),
                    transport=transport,
                    command=self._render_template(endpoint),
                    args=effective_args,
                    cwd=cwd,
                    env=headers,
                    stderr_mode=stderr_mode,
                    roots=roots,
                    include_tools=tuple(str(item).strip() for item in (tools_cfg.get("include", []) or []) if str(item).strip()),
                    exclude_tools=tuple(str(item).strip() for item in (tools_cfg.get("exclude", []) or []) if str(item).strip()),
                )
            )
        return tuple(servers)

    def _build_roots(self, payload: Mapping[str, Any]) -> tuple[MCPRootSpec, ...]:
        roots_enabled = bool(payload.get("enabled", self._config.get_nested("mcp", "roots", "enabled", default=True)))
        if not roots_enabled:
            return ()
        roots: list[MCPRootSpec] = []
        if bool(payload.get("include_workspace_root", self._config.get_nested("mcp", "roots", "include_workspace_root", default=True))):
            roots.append(MCPRootSpec(path=self._workspace_root, name=self._workspace_root.name or "workspace"))
        extra = list(payload.get("additional_paths", []) or []) or list(self._config.get_nested("mcp", "roots", "additional_paths", default=[]) or [])
        seen = {root.path for root in roots}
        for raw in extra:
            text = str(raw or "").strip()
            if not text:
                continue
            path = self._resolve_path_template(text)
            if path in seen:
                continue
            seen.add(path)
            roots.append(MCPRootSpec(path=path, name=path.name or str(path)))
        return tuple(roots)

    def _effective_server_args(
        self,
        *,
        server_name: str,
        command: str,
        args: tuple[str, ...],
        roots: tuple[MCPRootSpec, ...],
    ) -> tuple[str, ...]:
        if args:
            return args
        filesystem_server_name = str(
            self._config.get_nested("mcp", "admin", "install", "filesystem_server_name", default="filesystem")
            or "filesystem"
        ).strip()
        filesystem_binary = str(
            self._config.get_nested("mcp", "admin", "install", "filesystem_binary", default="mcp-server-filesystem")
            or "mcp-server-filesystem"
        ).strip()
        command_name = Path(command).name.strip()
        if server_name != filesystem_server_name and command_name != filesystem_binary:
            return args
        return tuple(str(root.path) for root in roots if str(root.path).strip())

    def _resolve_path_template(self, raw: str) -> Path:
        rendered = self._render_template(raw)
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = (self._session_cwd / path).resolve()
        else:
            path = path.resolve()
        return path

    def _render_template(self, text: str) -> str:
        rendered = str(text or "")
        rendered = rendered.replace("{workspace_root}", str(self._workspace_root))
        rendered = rendered.replace("{session_cwd}", str(self._session_cwd))
        return os.path.expandvars(rendered)

    def _server_by_name(self, name: str) -> MCPServerSnapshot | None:
        for server in self._catalog.servers:
            if server.spec.name == name:
                return server
        return None

    def _server_spec_by_name(self, name: str) -> MCPServerSpec | None:
        for spec in self._build_server_specs():
            if spec.name == name:
                return spec
        return None

    def configured_server_names(self) -> tuple[str, ...]:
        raw = self._config.get_nested("mcp", "servers", default={}) or {}
        if not isinstance(raw, dict):
            return ()
        return tuple(str(name).strip() for name in raw if str(name).strip())
