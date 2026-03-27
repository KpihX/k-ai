"""Core models for the MCP runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MCPRootSpec:
    path: Path
    name: str


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    enabled: bool
    transport: str
    command: str
    args: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str]
    stderr_mode: str
    roots: tuple[MCPRootSpec, ...]
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPToolSpec:
    server_name: str
    server_title: str
    name: str
    qualified_name: str
    description: str
    input_schema: Mapping[str, Any]
    annotations: Mapping[str, Any]
    title: str = ""


@dataclass(frozen=True)
class MCPResourceSpec:
    server_name: str
    server_title: str
    name: str
    uri: str
    description: str
    mime_type: str = ""
    size: int | None = None
    title: str = ""


@dataclass(frozen=True)
class MCPResourceTemplateSpec:
    server_name: str
    server_title: str
    name: str
    uri_template: str
    description: str
    mime_type: str = ""
    title: str = ""


@dataclass(frozen=True)
class MCPPromptSpec:
    server_name: str
    server_title: str
    name: str
    description: str
    arguments: tuple[Mapping[str, Any], ...] = ()
    title: str = ""


@dataclass(frozen=True)
class MCPIssue:
    server_name: str
    message: str


@dataclass(frozen=True)
class MCPServerSnapshot:
    spec: MCPServerSpec
    server_title: str
    protocol_version: str
    instructions: str
    capabilities: Mapping[str, Any]
    tools: tuple[MCPToolSpec, ...]
    resources: tuple[MCPResourceSpec, ...]
    resource_templates: tuple[MCPResourceTemplateSpec, ...]
    prompts: tuple[MCPPromptSpec, ...]
    issues: tuple[MCPIssue, ...] = ()


@dataclass(frozen=True)
class MCPCatalog:
    servers: tuple[MCPServerSnapshot, ...] = ()
    issues: tuple[MCPIssue, ...] = ()

    @property
    def tools(self) -> tuple[MCPToolSpec, ...]:
        return tuple(tool for server in self.servers for tool in server.tools)

    @property
    def resources(self) -> tuple[MCPResourceSpec, ...]:
        return tuple(item for server in self.servers for item in server.resources)

    @property
    def resource_templates(self) -> tuple[MCPResourceTemplateSpec, ...]:
        return tuple(item for server in self.servers for item in server.resource_templates)

    @property
    def prompts(self) -> tuple[MCPPromptSpec, ...]:
        return tuple(item for server in self.servers for item in server.prompts)


@dataclass(frozen=True)
class MCPToolCallResult:
    server_name: str
    tool_name: str
    qualified_name: str
    success: bool
    message: str
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPResourceReadResult:
    server_name: str
    uri: str
    message: str
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPPromptGetResult:
    server_name: str
    prompt_name: str
    description: str
    message: str
    raw: Mapping[str, Any] = field(default_factory=dict)
