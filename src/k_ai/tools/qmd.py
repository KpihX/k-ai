# src/k_ai/tools/qmd.py
"""
Complete QMD (Quick Markdown Search) integration.

Wraps all QMD CLI commands as internal tools:
  - Search: query (hybrid), search (BM25), vsearch (vector)
  - View: get (single doc), multi-get (batch), ls (list files)
  - Collections: collection list/add/remove/show
  - Maintenance: status, update, embed, cleanup
  - Context: context add/list/rm

All tools output JSON when possible for the LLM to process.
Session history files are automatically mapped to the QMD collection.
"""
import asyncio
import json
import re
import shutil
from typing import Any, Dict

from ..models import ToolResult
from ..tool_capabilities import capability_enabled
from .base import InternalTool, ToolContext, ToolRegistry


async def _run_qmd(*args: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a qmd CLI command and return (success, output)."""
    if not shutil.which("qmd"):
        return False, "qmd binary not found in PATH."
    try:
        proc = await asyncio.create_subprocess_exec(
            "qmd", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return False, err or out or f"Exit code {proc.returncode}"
        return True, out
    except asyncio.TimeoutError:
        return False, f"Timed out after {timeout}s."
    except Exception as e:
        return False, str(e)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else int(default)
    except (TypeError, ValueError):
        return int(default)


_SESSION_ID_RE = re.compile(r"\b([0-9a-f]{8,12})\b", re.IGNORECASE)


def _session_collection(ctx: ToolContext) -> str:
    return str(ctx.config.get_nested("tools", "qmd", "session_collection", default="k-ai"))


def _resolve_qmd_collection(
    collection: str | None,
    query: str,
    ctx: ToolContext,
) -> str | None:
    return _session_collection(ctx)


def _resolve_session_docid(raw: str, ctx: ToolContext) -> str | None:
    cleaned = raw.strip().strip("/")
    if not cleaned:
        return None
    base = cleaned.removesuffix(".jsonl")
    if not _SESSION_ID_RE.fullmatch(base):
        return None
    meta = ctx.session_store.get_session(base)
    if not meta:
        return None
    return f"{meta.id}.jsonl"


def _resolve_qmd_file(file: str, ctx: ToolContext) -> str:
    raw = file.strip()
    if not raw:
        return raw

    if raw.startswith("#"):
        return raw

    line_suffix = ""
    base = raw
    if ":" in raw and not raw.startswith("qmd://"):
        candidate_base, candidate_line = raw.rsplit(":", 1)
        if candidate_line.isdigit():
            base = candidate_base
            line_suffix = f":{candidate_line}"

    if base.startswith("qmd://"):
        prefix = f"qmd://{_session_collection(ctx)}/sessions/"
        if base.startswith(prefix):
            session_id = base.removeprefix(prefix).strip("/")
            meta = ctx.session_store.get_session(session_id)
            if meta:
                return f"qmd://{_session_collection(ctx)}/{meta.id}.jsonl{line_suffix}"
        collection_prefix = f"qmd://{_session_collection(ctx)}/"
        if base.startswith(collection_prefix):
            relative = base.removeprefix(collection_prefix)
            resolved_docid = _resolve_session_docid(relative, ctx)
            if resolved_docid:
                return f"{collection_prefix}{resolved_docid}{line_suffix}"
        if not base.startswith(f"qmd://{_session_collection(ctx)}/"):
            suffix = base.split("/", 3)[-1]
            return f"qmd://{_session_collection(ctx)}/{suffix}{line_suffix}"
        return raw

    if base.startswith("sessions/"):
        session_id = base.split("/", 1)[1]
        meta = ctx.session_store.get_session(session_id)
        if meta:
            return f"qmd://{_session_collection(ctx)}/{meta.id}.jsonl{line_suffix}"

    resolved_docid = _resolve_session_docid(base, ctx)
    if resolved_docid:
        return f"qmd://{_session_collection(ctx)}/{resolved_docid}{line_suffix}"

    return raw


def _render_qmd_hits(data: Any, fallback: str) -> Any:
    from rich.table import Table
    from rich.text import Text

    if not isinstance(data, list) or not data:
        return Text(fallback or "(no output)")

    table = Table(show_header=True, header_style="bold", border_style="cyan")
    table.add_column("Score", justify="right", width=7)
    table.add_column("Title", min_width=18)
    table.add_column("File", min_width=22)
    table.add_column("Snippet")
    for item in data[:8]:
        score = item.get("score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        snippet = item.get("snippet") or item.get("context") or item.get("body") or ""
        snippet = snippet.replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        table.add_row(
            score_text,
            str(item.get("title", item.get("docid", "(untitled)"))),
            str(item.get("file", "-")),
            snippet,
        )
    return table


def _qmd_disabled_result(ctx: ToolContext) -> ToolResult | None:
    if capability_enabled(ctx.config, "qmd"):
        return None
    return ToolResult(success=False, message="QMD capability is disabled in config.")


# ===================================================================
# Search tools
# ===================================================================

class QmdQueryTool(InternalTool):
    name = "qmd_query"
    display_name = "Hybrid Search"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = (
        "Hybrid semantic search: auto-expands the query, combines BM25 + vectors + reranking. "
        "Best for natural language questions about past conversations or knowledge."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "num_results": {"type": "integer", "description": "Max results."},
            "collection": {"type": "string", "description": "Filter by collection name."},
            "full": {"type": "boolean", "description": "Return full documents instead of snippets."},
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        cfg = ctx.config.get_nested("tools", "qmd", default={})
        query = arguments.get("query", "")
        default_n = _positive_int(cfg.get("limit", 5), 5)
        n = _positive_int(arguments.get("num_results", default_n), default_n)
        collection = _resolve_qmd_collection(arguments.get("collection"), query, ctx)
        full = arguments.get("full", False)
        timeout = int(cfg.get("query_timeout", 180))

        args = ["query", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])
        if full:
            args.append("--full")

        ok, out = await _run_qmd(*args, timeout=timeout)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                data = None
        return ToolResult(success=ok, message=out, data=data)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Query", str(arguments.get("query", "")))
        table.add_row("Collection", _session_collection(ctx))
        default_n = _positive_int(ctx.config.get_nested("tools", "qmd", "limit", default=5), 5)
        table.add_row("Results", str(_positive_int(arguments.get("num_results", default_n), default_n)))
        return [("Search Request", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        return _render_qmd_hits(result.data, fallback=result.message)


class QmdSearchTool(InternalTool):
    name = "qmd_search"
    display_name = "Keyword Search"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = "Full-text BM25 keyword search. Fast, no LLM needed. Good for exact terms."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search."},
            "num_results": {"type": "integer", "description": "Max results."},
            "collection": {"type": "string", "description": "Filter by collection."},
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        cfg = ctx.config.get_nested("tools", "qmd", default={})
        query = arguments.get("query", "")
        default_n = _positive_int(cfg.get("limit", 5), 5)
        n = _positive_int(arguments.get("num_results", default_n), default_n)
        collection = _resolve_qmd_collection(arguments.get("collection"), query, ctx)
        timeout = int(cfg.get("keyword_timeout", 30))

        args = ["search", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])

        ok, out = await _run_qmd(*args, timeout=timeout)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                data = None
        return ToolResult(success=ok, message=out, data=data)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        return QmdQueryTool.proposal_sections(self, arguments, ctx)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        return _render_qmd_hits(result.data, fallback=result.message)


class QmdVsearchTool(InternalTool):
    name = "qmd_vsearch"
    display_name = "Semantic Search"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = "Vector similarity search only. Good for semantic matching without keyword dependency."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic search query."},
            "num_results": {"type": "integer", "description": "Max results."},
            "collection": {"type": "string", "description": "Filter by collection."},
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        cfg = ctx.config.get_nested("tools", "qmd", default={})
        query = arguments.get("query", "")
        default_n = _positive_int(cfg.get("limit", 5), 5)
        n = _positive_int(arguments.get("num_results", default_n), default_n)
        collection = _resolve_qmd_collection(arguments.get("collection"), query, ctx)
        timeout = int(cfg.get("vector_timeout", 30))

        args = ["vsearch", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])

        ok, out = await _run_qmd(*args, timeout=timeout)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                data = None
        return ToolResult(success=ok, message=out, data=data)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        return QmdQueryTool.proposal_sections(self, arguments, ctx)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        return _render_qmd_hits(result.data, fallback=result.message)


# ===================================================================
# View tools
# ===================================================================

class QmdGetTool(InternalTool):
    name = "qmd_get"
    display_name = "Open Indexed Document"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = (
        "Show a single document or a line slice from the QMD index. "
        "Use file:line format and -l N for line count. "
        "Useful for viewing session history details."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "File path (qmd://collection/path or docid). Supports :line suffix."},
            "lines": {"type": "integer", "description": "Max lines to show."},
        },
        "required": ["file"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        file = _resolve_qmd_file(arguments.get("file", ""), ctx)
        lines = arguments.get("lines")
        args = ["get", file]
        if lines:
            args.extend(["-l", str(lines)])
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("File", _resolve_qmd_file(arguments.get("file", ""), ctx))
        if arguments.get("lines"):
            table.add_row("Lines", str(arguments["lines"]))
        return [("Document Request", table)]


class QmdMultiGetTool(InternalTool):
    name = "qmd_multi_get"
    display_name = "Open Multiple Documents"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = "Batch fetch multiple documents by glob pattern or comma-separated list."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern or comma-separated file list."},
            "lines": {"type": "integer", "description": "Max lines per file."},
        },
        "required": ["pattern"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        pattern = arguments.get("pattern", "")
        lines = arguments.get("lines")
        args = ["multi-get", pattern, "--json"]
        if lines:
            args.extend(["-l", str(lines)])
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


class QmdLsTool(InternalTool):
    name = "qmd_ls"
    display_name = "List Indexed Files"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = "List indexed files in a collection or all collections."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional: collection name or collection/path."},
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        path = arguments.get("path", "")
        args = ["ls"]
        args.append(path or _session_collection(ctx))
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


# ===================================================================
# Collection management
# ===================================================================

class QmdCollectionListTool(InternalTool):
    name = "qmd_collection_list"
    display_name = "List Collections"
    category = "knowledge"
    danger_level = "low"
    accent_color = "cyan"
    description = "List all QMD collections with their file counts and paths."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        ok, out = await _run_qmd("collection", "show", _session_collection(ctx))
        return ToolResult(success=ok, message=out)


class QmdCollectionAddTool(InternalTool):
    name = "qmd_collection_add"
    description = "Add a new folder as a QMD collection for indexing."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Collection name."},
            "path": {"type": "string", "description": "Folder path to index."},
            "pattern": {"type": "string", "description": "Glob pattern for files (e.g. '**/*.md')."},
        },
        "required": ["name", "path"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        name = arguments.get("name", "")
        path = arguments.get("path", "")
        pattern = arguments.get("pattern")
        args = ["collection", "add", name, path]
        if pattern:
            args.extend(["--pattern", pattern])
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


class QmdCollectionRemoveTool(InternalTool):
    name = "qmd_collection_remove"
    description = "Remove a QMD collection from the index."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Collection name to remove."},
        },
        "required": ["name"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        name = arguments.get("name", "")
        ok, out = await _run_qmd("collection", "remove", name)
        return ToolResult(success=ok, message=out)


class QmdCollectionShowTool(InternalTool):
    name = "qmd_collection_show"
    description = "Show details of a specific QMD collection."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Collection name."},
        },
        "required": ["name"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        name = arguments.get("name", "")
        ok, out = await _run_qmd("collection", "show", name)
        return ToolResult(success=ok, message=out)


# ===================================================================
# Maintenance
# ===================================================================

class QmdStatusTool(InternalTool):
    name = "qmd_status"
    description = "Show QMD index health: total docs, vectors, collections, last update."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        ok, out = await _run_qmd("status")
        return ToolResult(success=ok, message=out)


class QmdUpdateTool(InternalTool):
    name = "qmd_update"
    description = "Re-index all QMD collections. Optionally git pull first."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pull": {"type": "boolean", "description": "Git pull collections before re-indexing."},
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        pull = arguments.get("pull", False)
        args = ["update"]
        if pull:
            args.append("--pull")
        ok, out = await _run_qmd(*args, timeout=60)
        if not ok:
            return ToolResult(success=False, message=out)
        return ToolResult(success=True, message=f"Update done.\n{out}")


class QmdEmbedTool(InternalTool):
    name = "qmd_embed"
    description = "Generate or refresh vector embeddings for all indexed documents."
    parameters_schema = {
        "type": "object",
        "properties": {
            "force": {"type": "boolean", "description": "Force re-embed all (not just new)."},
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        force = arguments.get("force", False)
        args = ["embed"]
        if force:
            args.append("-f")
        ok, out = await _run_qmd(*args, timeout=120)
        return ToolResult(success=ok, message=out)


class QmdCleanupTool(InternalTool):
    name = "qmd_cleanup"
    description = "Clear QMD caches and vacuum the database."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        ok, out = await _run_qmd("cleanup")
        return ToolResult(success=ok, message=out)


# ===================================================================
# Context management
# ===================================================================

class QmdContextListTool(InternalTool):
    name = "qmd_context_list"
    description = "List human-written context summaries attached to collections."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        ok, out = await _run_qmd("context", "list")
        return ToolResult(success=ok, message=out)


class QmdContextAddTool(InternalTool):
    name = "qmd_context_add"
    description = "Attach a context summary to a QMD collection."
    parameters_schema = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "Collection name."},
            "text": {"type": "string", "description": "Context summary text."},
        },
        "required": ["collection", "text"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        collection = arguments.get("collection", "")
        text = arguments.get("text", "")
        ok, out = await _run_qmd("context", "add", collection, text)
        return ToolResult(success=ok, message=out)


class QmdContextRemoveTool(InternalTool):
    name = "qmd_context_rm"
    description = "Remove a context summary from a collection."
    parameters_schema = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "Collection name."},
        },
        "required": ["collection"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        disabled = _qmd_disabled_result(ctx)
        if disabled:
            return disabled
        collection = arguments.get("collection", "")
        ok, out = await _run_qmd("context", "rm", collection)
        return ToolResult(success=ok, message=out)


# ===================================================================
# Registration
# ===================================================================

def register_qmd_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Register all QMD tools."""
    for tool_cls in [
        # Search
        QmdQueryTool, QmdSearchTool, QmdVsearchTool,
        # View
        QmdGetTool, QmdMultiGetTool, QmdLsTool,
        # Collections
        QmdCollectionListTool, QmdCollectionAddTool,
        QmdCollectionRemoveTool, QmdCollectionShowTool,
        # Maintenance
        QmdStatusTool, QmdUpdateTool, QmdEmbedTool, QmdCleanupTool,
        # Context
        QmdContextListTool, QmdContextAddTool, QmdContextRemoveTool,
    ]:
        registry.register(tool_cls())
