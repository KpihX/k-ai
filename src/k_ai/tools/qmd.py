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
import shutil
from typing import Any, Dict

from ..models import ToolResult
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


# ===================================================================
# Search tools
# ===================================================================

class QmdQueryTool(InternalTool):
    name = "qmd_query"
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
        cfg = ctx.config.get_nested("tools", "qmd_search", default={})
        query = arguments.get("query", "")
        n = arguments.get("num_results", cfg.get("limit", 5))
        collection = arguments.get("collection")
        full = arguments.get("full", False)

        args = ["query", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])
        if full:
            args.append("--full")

        ok, out = await _run_qmd(*args, timeout=90)
        return ToolResult(success=ok, message=out)


class QmdSearchTool(InternalTool):
    name = "qmd_search"
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
        cfg = ctx.config.get_nested("tools", "qmd_search", default={})
        query = arguments.get("query", "")
        n = arguments.get("num_results", cfg.get("limit", 5))
        collection = arguments.get("collection")

        args = ["search", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])

        ok, out = await _run_qmd(*args, timeout=30)
        return ToolResult(success=ok, message=out)


class QmdVsearchTool(InternalTool):
    name = "qmd_vsearch"
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
        cfg = ctx.config.get_nested("tools", "qmd_search", default={})
        query = arguments.get("query", "")
        n = arguments.get("num_results", cfg.get("limit", 5))
        collection = arguments.get("collection")

        args = ["vsearch", query, "-n", str(n), "--json"]
        if collection:
            args.extend(["-c", collection])

        ok, out = await _run_qmd(*args, timeout=30)
        return ToolResult(success=ok, message=out)


# ===================================================================
# View tools
# ===================================================================

class QmdGetTool(InternalTool):
    name = "qmd_get"
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
        file = arguments.get("file", "")
        lines = arguments.get("lines")
        args = ["get", file]
        if lines:
            args.extend(["-l", str(lines)])
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


class QmdMultiGetTool(InternalTool):
    name = "qmd_multi_get"
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
        pattern = arguments.get("pattern", "")
        lines = arguments.get("lines")
        args = ["multi-get", pattern, "--json"]
        if lines:
            args.extend(["-l", str(lines)])
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


class QmdLsTool(InternalTool):
    name = "qmd_ls"
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
        path = arguments.get("path", "")
        args = ["ls"]
        if path:
            args.append(path)
        ok, out = await _run_qmd(*args)
        return ToolResult(success=ok, message=out)


# ===================================================================
# Collection management
# ===================================================================

class QmdCollectionListTool(InternalTool):
    name = "qmd_collection_list"
    description = "List all QMD collections with their file counts and paths."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ok, out = await _run_qmd("collection", "list")
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
