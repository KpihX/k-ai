"""MCP client adapters built on the official Python SDK."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, Iterable, Sequence, TypeVar

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from .models import (
    MCPPromptGetResult,
    MCPPromptSpec,
    MCPResourceReadResult,
    MCPResourceSpec,
    MCPResourceTemplateSpec,
    MCPRootSpec,
    MCPServerSnapshot,
    MCPServerSpec,
    MCPToolCallResult,
    MCPToolSpec,
)


class MCPClientError(RuntimeError):
    """Raised when an MCP server cannot be contacted cleanly."""


T = TypeVar("T")


def _root_to_model(root: MCPRootSpec) -> types.Root:
    return types.Root(uri=root.path.resolve().as_uri(), name=root.name)


def _content_to_text(items: Sequence[Any]) -> str:
    lines: list[str] = []
    for item in items:
        if isinstance(item, types.TextContent):
            lines.append(item.text)
            continue
        if isinstance(item, types.ImageContent):
            mime = getattr(item, "mimeType", "image")
            lines.append(f"[image content: {mime}]")
            continue
        if isinstance(item, types.AudioContent):
            mime = getattr(item, "mimeType", "audio")
            lines.append(f"[audio content: {mime}]")
            continue
        if isinstance(item, types.EmbeddedResource):
            resource = item.resource
            uri = getattr(resource, "uri", "")
            text = getattr(resource, "text", None)
            if text:
                lines.append(text)
            elif uri:
                lines.append(f"[embedded resource: {uri}]")
            else:
                lines.append("[embedded resource]")
            continue
        dump = item.model_dump() if hasattr(item, "model_dump") else str(item)
        lines.append(str(dump))
    return "\n".join(line for line in lines if line).strip()


def _resource_contents_to_text(items: Sequence[Any]) -> str:
    lines: list[str] = []
    for item in items:
        if isinstance(item, types.TextResourceContents):
            lines.append(item.text)
            continue
        if isinstance(item, types.BlobResourceContents):
            mime = getattr(item, "mimeType", "application/octet-stream")
            lines.append(f"[blob resource: {mime} @ {item.uri}]")
            continue
        dump = item.model_dump() if hasattr(item, "model_dump") else str(item)
        lines.append(str(dump))
    return "\n".join(line for line in lines if line).strip()


def _prompt_messages_to_text(items: Sequence[Any]) -> str:
    chunks: list[str] = []
    for item in items:
        role = getattr(item, "role", "user")
        content = getattr(item, "content", None)
        text = _content_to_text([content]) if content is not None else ""
        chunks.append(f"[{role}] {text}".strip())
    return "\n".join(chunk for chunk in chunks if chunk).strip()


class MCPClient:
    """Transport-aware client around the official MCP SDK."""

    def __init__(
        self,
        *,
        protocol_version: str,
        client_name: str,
        client_version: str,
        tool_prefix_template: str = "mcp__{server_name}__",
        read_timeout_seconds: int = 30,
    ):
        self._protocol_version = str(protocol_version).strip()
        self._client_name = str(client_name).strip() or "k-ai"
        self._client_version = str(client_version).strip() or "0.0.0"
        self._tool_prefix_template = str(tool_prefix_template or "mcp__{server_name}__")
        self._read_timeout = timedelta(seconds=max(1, int(read_timeout_seconds)))

    async def inspect_server(
        self,
        spec: MCPServerSpec,
        *,
        max_tools: int = 64,
        max_resources: int = 128,
        max_prompts: int = 128,
    ) -> MCPServerSnapshot:
        async with self._session(spec) as session:
            init = await session.initialize()
            listed_tools = await self._optional_request(session.list_tools, default=types.ListToolsResult(tools=[]))
            listed_resources = await self._optional_request(session.list_resources, default=types.ListResourcesResult(resources=[]))
            listed_templates = await self._optional_request(
                session.list_resource_templates,
                default=types.ListResourceTemplatesResult(resourceTemplates=[]),
            )
            listed_prompts = await self._optional_request(session.list_prompts, default=types.ListPromptsResult(prompts=[]))
            server_title = init.serverInfo.name or spec.name
            return MCPServerSnapshot(
                spec=spec,
                server_title=server_title,
                protocol_version=init.protocolVersion,
                instructions=init.instructions or "",
                capabilities=init.capabilities.model_dump(exclude_none=True),
                tools=self._normalize_tools(spec=spec, server_title=server_title, raw_tools=listed_tools.tools, max_tools=max_tools),
                resources=self._normalize_resources(spec=spec, server_title=server_title, raw_resources=listed_resources.resources, max_resources=max_resources),
                resource_templates=self._normalize_resource_templates(spec=spec, server_title=server_title, raw_templates=listed_templates.resourceTemplates, max_resources=max_resources),
                prompts=self._normalize_prompts(spec=spec, server_title=server_title, raw_prompts=listed_prompts.prompts, max_prompts=max_prompts),
                issues=(),
            )

    @staticmethod
    async def _optional_request(call: Callable[..., Any], *, default: T) -> T:
        try:
            return await call()
        except McpError as exc:
            if exc.error.code == -32601:
                return default
            raise

    async def call_tool(
        self,
        *,
        spec: MCPServerSpec,
        tool_name: str,
        qualified_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolCallResult:
        async with self._session(spec) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments or {})
        message = _content_to_text(result.content)
        raw = result.model_dump(exclude_none=True)
        return MCPToolCallResult(
            server_name=spec.name,
            tool_name=tool_name,
            qualified_name=qualified_name,
            success=not bool(result.isError),
            message=message or ("MCP tool reported success." if not result.isError else "MCP tool reported an error."),
            raw=raw,
        )

    async def read_resource(self, *, spec: MCPServerSpec, uri: str) -> MCPResourceReadResult:
        async with self._session(spec) as session:
            await session.initialize()
            result = await session.read_resource(uri)
        message = _resource_contents_to_text(result.contents)
        return MCPResourceReadResult(
            server_name=spec.name,
            uri=uri,
            message=message or f"Read resource {uri}.",
            raw=result.model_dump(exclude_none=True),
        )

    async def get_prompt(
        self,
        *,
        spec: MCPServerSpec,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
    ) -> MCPPromptGetResult:
        async with self._session(spec) as session:
            await session.initialize()
            result = await session.get_prompt(prompt_name, arguments=arguments or {})
        message = _prompt_messages_to_text(result.messages)
        return MCPPromptGetResult(
            server_name=spec.name,
            prompt_name=prompt_name,
            description=result.description or "",
            message=message or f"Loaded prompt {prompt_name}.",
            raw=result.model_dump(exclude_none=True),
        )

    @asynccontextmanager
    async def _session(self, spec: MCPServerSpec) -> AsyncIterator[ClientSession]:
        async with self._transport_streams(spec) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self._read_timeout,
                list_roots_callback=self._list_roots_callback(spec.roots),
                client_info=types.Implementation(name=self._client_name, version=self._client_version),
            ) as session:
                yield session

    @asynccontextmanager
    async def _transport_streams(self, spec: MCPServerSpec):
        if spec.transport == "stdio":
            params = StdioServerParameters(
                command=spec.command,
                args=list(spec.args),
                env=dict(spec.env) if spec.env else None,
                cwd=spec.cwd,
            )
            async with stdio_client(params) as streams:
                yield streams
            return

        if spec.transport == "streamable_http":
            if not spec.command:
                raise MCPClientError(f"MCP server '{spec.name}' is missing its URL.")
            async with streamablehttp_client(
                spec.command,
                headers=dict(spec.env) if spec.env else None,
                timeout=self._read_timeout,
            ) as (read_stream, write_stream, _get_session_id):
                yield read_stream, write_stream
            return

        if spec.transport == "sse":
            if not spec.command:
                raise MCPClientError(f"MCP server '{spec.name}' is missing its URL.")
            async with sse_client(
                spec.command,
                headers=dict(spec.env) if spec.env else None,
                timeout=float(self._read_timeout.total_seconds()),
            ) as (read_stream, write_stream):
                yield read_stream, write_stream
            return

        raise MCPClientError(f"Unsupported MCP transport '{spec.transport}'.")

    def _normalize_tools(
        self,
        *,
        spec: MCPServerSpec,
        server_title: str,
        raw_tools: Iterable[types.Tool],
        max_tools: int,
    ) -> tuple[MCPToolSpec, ...]:
        include = set(spec.include_tools)
        exclude = set(spec.exclude_tools)
        prefix = self._tool_prefix_template.format(server_name=spec.name)
        items: list[MCPToolSpec] = []
        for tool in raw_tools:
            if include and tool.name not in include:
                continue
            if tool.name in exclude:
                continue
            items.append(
                MCPToolSpec(
                    server_name=spec.name,
                    server_title=server_title,
                    name=tool.name,
                    qualified_name=f"{prefix}{tool.name}",
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {"type": "object"},
                    annotations=tool.annotations.model_dump(exclude_none=True) if tool.annotations else {},
                    title=tool.title or "",
                )
            )
            if len(items) >= max_tools:
                break
        return tuple(items)

    def _normalize_resources(
        self,
        *,
        spec: MCPServerSpec,
        server_title: str,
        raw_resources: Iterable[types.Resource],
        max_resources: int,
    ) -> tuple[MCPResourceSpec, ...]:
        items: list[MCPResourceSpec] = []
        for resource in raw_resources:
            items.append(
                MCPResourceSpec(
                    server_name=spec.name,
                    server_title=server_title,
                    name=resource.name,
                    uri=str(resource.uri),
                    description=resource.description or "",
                    mime_type=resource.mimeType or "",
                    size=resource.size,
                    title=resource.title or "",
                )
            )
            if len(items) >= max_resources:
                break
        return tuple(items)

    def _normalize_resource_templates(
        self,
        *,
        spec: MCPServerSpec,
        server_title: str,
        raw_templates: Iterable[types.ResourceTemplate],
        max_resources: int,
    ) -> tuple[MCPResourceTemplateSpec, ...]:
        items: list[MCPResourceTemplateSpec] = []
        for resource in raw_templates:
            items.append(
                MCPResourceTemplateSpec(
                    server_name=spec.name,
                    server_title=server_title,
                    name=resource.name,
                    uri_template=resource.uriTemplate,
                    description=resource.description or "",
                    mime_type=resource.mimeType or "",
                    title=resource.title or "",
                )
            )
            if len(items) >= max_resources:
                break
        return tuple(items)

    def _normalize_prompts(
        self,
        *,
        spec: MCPServerSpec,
        server_title: str,
        raw_prompts: Iterable[types.Prompt],
        max_prompts: int,
    ) -> tuple[MCPPromptSpec, ...]:
        items: list[MCPPromptSpec] = []
        for prompt in raw_prompts:
            args = []
            for item in (prompt.arguments or []):
                args.append(item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item))
            items.append(
                MCPPromptSpec(
                    server_name=spec.name,
                    server_title=server_title,
                    name=prompt.name,
                    description=prompt.description or "",
                    arguments=tuple(args),
                    title=prompt.title or "",
                )
            )
            if len(items) >= max_prompts:
                break
        return tuple(items)

    @staticmethod
    def _list_roots_callback(roots: Sequence[MCPRootSpec]) -> Callable[[Any], Any]:
        async def _callback(_context: Any) -> types.ListRootsResult:
            return types.ListRootsResult(roots=[_root_to_model(root) for root in roots])

        return _callback
