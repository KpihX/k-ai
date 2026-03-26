# src/k_ai/tools/external.py
"""
External tools: exa search, Python execution, shell execution.

All external tools with side effects require user approval.
"""
import asyncio
import subprocess
from typing import Any, Dict

from ..models import ToolResult
from ..secrets import resolve_secret
from .base import InternalTool, ToolContext, ToolRegistry


class ExaSearchTool(InternalTool):
    name = "exa_search"
    description = "Search the web using the Exa semantic search API."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default from config tools.exa_search.num_results).",
            },
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "exa_search", default={})
        if not tool_cfg.get("enabled", False):
            return ToolResult(success=False, message="exa_search is disabled in config.")

        api_key_var = tool_cfg.get("api_key_env_var", "EXA_API_KEY")
        api_key, _ = resolve_secret(api_key_var)
        if not api_key:
            return ToolResult(success=False, message=f"{api_key_var} not found.")

        query = arguments.get("query", "")
        num_results = arguments.get("num_results", tool_cfg.get("num_results", 5))

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json={
                        "query": query,
                        "num_results": num_results,
                        "use_autoprompt": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                lines = []
                for r in results:
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    lines.append(f"- {title}\n  {url}")
                return ToolResult(
                    success=True,
                    message="\n".join(lines) if lines else "No results found.",
                    data=results,
                )
        except Exception as e:
            return ToolResult(success=False, message=f"Exa search failed: {e}")


class PythonExecTool(InternalTool):
    name = "python_exec"
    description = (
        "Execute Python code in a sandboxed environment with scientific libraries "
        "(numpy, sympy, scipy, pandas, matplotlib, seaborn, scikit-learn). "
        "Use for calculations, data processing, plotting, or quick scripts."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }
    requires_approval = True

    # Default scientific packages for the sandbox venv
    _SANDBOX_PACKAGES = [
        "numpy", "sympy", "scipy", "pandas",
        "matplotlib", "seaborn", "scikit-learn",
    ]

    @staticmethod
    def _sandbox_dir(ctx: ToolContext) -> str:
        return str(ctx.config.get_nested(
            "tools", "python_exec", "sandbox_dir",
            default="~/.k-ai/sandbox",
        ))

    @staticmethod
    async def _ensure_sandbox(sandbox_dir: str) -> str:
        """Ensure the sandbox venv exists with scientific libs. Returns python path."""
        from pathlib import Path
        venv_path = Path(sandbox_dir).expanduser()
        python = venv_path / "bin" / "python"

        if python.exists():
            return str(python)

        # Create venv
        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "venv", str(venv_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

        # Install scientific packages
        pip = venv_path / "bin" / "pip"
        proc = await asyncio.create_subprocess_exec(
            str(pip), "install", "-q", *PythonExecTool._SANDBOX_PACKAGES,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)

        return str(python)

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "python_exec", default={})
        if not tool_cfg.get("enabled", True):
            return ToolResult(success=False, message="python_exec is disabled in config.")

        code = arguments.get("code", "")
        if not code.strip():
            return ToolResult(success=False, message="No code provided.")

        timeout = int(tool_cfg.get("timeout", 30))

        try:
            # Use sandbox venv with scientific libs
            sandbox_dir = self._sandbox_dir(ctx)
            python = await self._ensure_sandbox(sandbox_dir)

            proc = await asyncio.create_subprocess_exec(
                python, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")
            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    message=output if output else "(no output)",
                    data={"returncode": 0, "stdout": output, "stderr": errors},
                )
            return ToolResult(
                success=False,
                message=f"Exit code {proc.returncode}\n{errors}",
                data={"returncode": proc.returncode, "stdout": output, "stderr": errors},
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, message=f"Execution timed out after {timeout}s.")
        except Exception as e:
            return ToolResult(success=False, message=f"Execution failed: {e}")


class ShellExecTool(InternalTool):
    name = "shell_exec"
    description = "Execute a shell command and return stdout/stderr."
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "shell_exec", default={})
        if not tool_cfg.get("enabled", True):
            return ToolResult(success=False, message="shell_exec is disabled in config.")

        command = arguments.get("command", "")
        if not command.strip():
            return ToolResult(success=False, message="No command provided.")

        timeout = int(tool_cfg.get("timeout", 30))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")
            combined = output
            if errors:
                combined += f"\n--- stderr ---\n{errors}"
            return ToolResult(
                success=proc.returncode == 0,
                message=combined if combined.strip() else "(no output)",
                data={"returncode": proc.returncode, "stdout": output, "stderr": errors},
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, message=f"Command timed out after {timeout}s.")
        except Exception as e:
            return ToolResult(success=False, message=f"Command failed: {e}")


def register_external_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    for tool_cls in [ExaSearchTool, PythonExecTool, ShellExecTool]:
        registry.register(tool_cls())
