# src/k_ai/tools/external.py
"""
External tools: exa search, Python execution, shell execution.

All external tools with side effects require user approval.
"""
import asyncio
import ast
import json
import re
import shlex
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

from ..models import ToolResult
from ..secrets import resolve_secret
from ..tool_capabilities import capability_enabled
from ..ui_theme import resolve_syntax_theme
from .base import InternalTool, ToolContext, ToolRegistry


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else int(default)
    except (TypeError, ValueError):
        return int(default)


class ExaSearchTool(InternalTool):
    name = "exa_search"
    display_name = "Web Search"
    category = "web"
    danger_level = "low"
    accent_color = "cyan"
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
                "description": "Number of results (default from config tools.exa.num_results).",
            },
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "exa", default={})
        if not capability_enabled(ctx.config, "exa"):
            return ToolResult(success=False, message="exa_search is disabled in config.")

        api_key_var = tool_cfg.get("api_key_env_var", "EXA_API_KEY")
        api_key, _ = resolve_secret(api_key_var)
        if not api_key:
            return ToolResult(success=False, message=f"{api_key_var} not found.")

        query = arguments.get("query", "")
        default_num_results = _positive_int(tool_cfg.get("num_results", 5), 5)
        num_results = _positive_int(arguments.get("num_results", default_num_results), default_num_results)

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
    display_name = "Run Python"
    category = "execution"
    danger_level = "high"
    accent_color = "yellow"
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

    _FALLBACK_DEFAULT_PACKAGES = [
        "numpy", "sympy", "scipy", "pandas",
        "matplotlib", "seaborn", "scikit-learn",
    ]
    _PROTECTED_PACKAGES = {"pip", "setuptools", "wheel"}

    @staticmethod
    def _prepare_code(code: str) -> str:
        """
        Make python_exec behave more like a REPL:
        if the last statement is an expression, print its repr().
        """
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return code

        if not tree.body or not isinstance(tree.body[-1], ast.Expr):
            return code

        last_expr = tree.body[-1].value
        assign = ast.Assign(
            targets=[ast.Name(id="__k_ai_result", ctx=ast.Store())],
            value=last_expr,
        )
        print_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Call(
                    func=ast.Name(id="repr", ctx=ast.Load()),
                    args=[ast.Name(id="__k_ai_result", ctx=ast.Load())],
                    keywords=[],
                )],
                keywords=[],
            )
        )
        tree.body[-1] = assign
        tree.body.append(print_call)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.syntax import Syntax

        code = arguments.get("code", "")
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return [("Python Code", Syntax(code, "python", theme=syntax_theme, line_numbers=True, word_wrap=True))]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.syntax import Syntax

        msg = result.message
        if len(msg) > max_display_length:
            msg = msg[:max_display_length] + "\n...(truncated)"
        lexer = "text" if result.success else "pytb"
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return Syntax(msg or "(no output)", lexer, theme=syntax_theme, word_wrap=True)

    @staticmethod
    def _sandbox_dir(ctx: ToolContext) -> str:
        return str(ctx.config.get_nested(
            "tools", "python", "sandbox_dir",
            default="~/.k-ai/sandbox",
        ))

    @staticmethod
    def _cwd(ctx: ToolContext) -> str | None:
        if ctx.get_cwd:
            try:
                return ctx.get_cwd()
            except Exception:
                return None
        return None

    @classmethod
    def _default_packages(cls, ctx: ToolContext) -> List[str]:
        raw = ctx.config.get_nested("tools", "python", "default_packages", default=cls._FALLBACK_DEFAULT_PACKAGES)
        if not isinstance(raw, list):
            return list(cls._FALLBACK_DEFAULT_PACKAGES)
        packages: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in packages:
                packages.append(text)
        return packages or list(cls._FALLBACK_DEFAULT_PACKAGES)

    @staticmethod
    def _pip_path(sandbox_dir: str) -> str:
        return str(Path(sandbox_dir).expanduser() / "bin" / "pip")

    @staticmethod
    async def _wait_for_process(proc, timeout: int, ctx: ToolContext):
        task = asyncio.create_task(proc.communicate())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.1)
                if task in done:
                    return task.result()
                if ctx.is_interrupt_requested and ctx.is_interrupt_requested():
                    with suppress(ProcessLookupError):
                        proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        with suppress(ProcessLookupError):
                            proc.kill()
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
                    raise KeyboardInterrupt
                if loop.time() >= deadline:
                    with suppress(ProcessLookupError):
                        proc.terminate()
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
                    raise asyncio.TimeoutError
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    @staticmethod
    async def _ensure_sandbox(sandbox_dir: str, ctx: ToolContext) -> str:
        """Ensure the sandbox venv exists with scientific libs. Returns python path."""
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
        _, stderr = await PythonExecTool._wait_for_process(proc, timeout=30, ctx=ctx)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to create sandbox venv: {err or f'exit code {proc.returncode}'}")

        # Install scientific packages
        pip = venv_path / "bin" / "pip"
        proc = await asyncio.create_subprocess_exec(
            str(pip), "install", "-q", *PythonExecTool._default_packages(ctx),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await PythonExecTool._wait_for_process(proc, timeout=120, ctx=ctx)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to install sandbox packages: {err or f'exit code {proc.returncode}'}")

        return str(python)

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "python", default={})
        if not capability_enabled(ctx.config, "python"):
            return ToolResult(success=False, message="python_exec is disabled in config.")

        code = arguments.get("code", "")
        if not code.strip():
            return ToolResult(success=False, message="No code provided.")

        timeout = int(tool_cfg.get("timeout", 30))

        try:
            # Use sandbox venv with scientific libs
            sandbox_dir = self._sandbox_dir(ctx)
            python = await self._ensure_sandbox(sandbox_dir, ctx)

            prepared_code = self._prepare_code(code)
            proc = await asyncio.create_subprocess_exec(
                python, "-c", prepared_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd(ctx),
            )
            stdout, stderr = await self._wait_for_process(proc, timeout=timeout, ctx=ctx)
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


class PythonSandboxPackagesTool(InternalTool):
    category = "execution"
    accent_color = "yellow"

    @staticmethod
    async def _ensure_ready(ctx: ToolContext) -> tuple[str, str]:
        sandbox_dir = PythonExecTool._sandbox_dir(ctx)
        python = await PythonExecTool._ensure_sandbox(sandbox_dir, ctx)
        pip = PythonExecTool._pip_path(sandbox_dir)
        return python, pip

    @staticmethod
    async def _run_pip(ctx: ToolContext, args: List[str], timeout: int | None = None) -> tuple[int, str, str]:
        _, pip = await PythonSandboxPackagesTool._ensure_ready(ctx)
        effective_timeout = int(timeout or ctx.config.get_nested("tools", "python", "install_timeout", default=300))
        proc = await asyncio.create_subprocess_exec(
            pip, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PythonExecTool._cwd(ctx),
        )
        stdout, stderr = await PythonExecTool._wait_for_process(proc, timeout=effective_timeout, ctx=ctx)
        return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

    @staticmethod
    def _normalize_packages(arguments: Dict[str, Any]) -> List[str]:
        raw = arguments.get("packages", [])
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.split() if part.strip()]
        packages: List[str] = []
        for item in raw or []:
            name = str(item or "").strip()
            if name and name not in packages:
                packages.append(name)
        return packages


class PythonSandboxListPackagesTool(PythonSandboxPackagesTool):
    name = "python_sandbox_list_packages"
    display_name = "List Python Sandbox Packages"
    danger_level = "low"
    description = "List installed packages inside the dedicated python_exec sandbox."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "python", default={})
        if not capability_enabled(ctx.config, "python"):
            return ToolResult(success=False, message="python_exec is disabled in config.")
        try:
            code, stdout, stderr = await self._run_pip(ctx, ["list", "--format=json"])
            if code != 0:
                return ToolResult(success=False, message=stderr.strip() or stdout.strip() or f"pip list failed with code {code}")
            packages = json.loads(stdout or "[]")
            lines = [f"- {pkg['name']}=={pkg['version']}" for pkg in packages]
            return ToolResult(success=True, message="\n".join(lines) if lines else "No packages installed.", data=packages)
        except Exception as e:
            return ToolResult(success=False, message=f"Could not list sandbox packages: {e}")


class PythonSandboxInstallPackagesTool(PythonSandboxPackagesTool):
    name = "python_sandbox_install_packages"
    display_name = "Install Python Sandbox Packages"
    danger_level = "high"
    description = "Install one or more packages into the dedicated python_exec sandbox."
    parameters_schema = {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package specifiers to install into the sandbox.",
            },
        },
        "required": ["packages"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        packages = self._normalize_packages(arguments)
        if not packages:
            return ToolResult(success=False, message="No packages provided.")
        try:
            code, stdout, stderr = await self._run_pip(ctx, ["install", *packages])
            if code != 0:
                message = stderr.strip() or stdout.strip() or f"pip install failed with code {code}"
                return ToolResult(success=False, message=message)
            return ToolResult(success=True, message=f"Installed into sandbox: {', '.join(packages)}")
        except Exception as e:
            return ToolResult(success=False, message=f"Could not install sandbox packages: {e}")

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        packages = self._normalize_packages(arguments)
        return [("Packages", [("Install", ", ".join(packages) or "-")])]


class PythonSandboxRemovePackagesTool(PythonSandboxPackagesTool):
    name = "python_sandbox_remove_packages"
    display_name = "Remove Python Sandbox Packages"
    danger_level = "high"
    description = "Uninstall one or more non-core packages from the dedicated python_exec sandbox."
    parameters_schema = {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Installed package names to remove from the sandbox.",
            },
        },
        "required": ["packages"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        packages = self._normalize_packages(arguments)
        if not packages:
            return ToolResult(success=False, message="No packages provided.")
        forbidden = [pkg for pkg in packages if pkg.lower() in PythonExecTool._PROTECTED_PACKAGES]
        if forbidden:
            return ToolResult(success=False, message=f"Refusing to remove protected sandbox packages: {', '.join(forbidden)}")
        try:
            code, stdout, stderr = await self._run_pip(ctx, ["uninstall", "-y", *packages])
            if code != 0:
                message = stderr.strip() or stdout.strip() or f"pip uninstall failed with code {code}"
                return ToolResult(success=False, message=message)
            return ToolResult(success=True, message=f"Removed from sandbox: {', '.join(packages)}")
        except Exception as e:
            return ToolResult(success=False, message=f"Could not remove sandbox packages: {e}")

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        packages = self._normalize_packages(arguments)
        return [("Packages", [("Remove", ", ".join(packages) or "-")])]


class ShellExecTool(InternalTool):
    name = "shell_exec"
    display_name = "Run Shell Command"
    category = "execution"
    danger_level = "high"
    accent_color = "red"
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

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.syntax import Syntax

        command = arguments.get("command", "")
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return [("Shell Command", Syntax(command, "bash", theme=syntax_theme, line_numbers=False, word_wrap=True))]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.syntax import Syntax

        msg = result.message
        if len(msg) > max_display_length:
            msg = msg[:max_display_length] + "\n...(truncated)"
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return Syntax(msg or "(no output)", "text", theme=syntax_theme, word_wrap=True)

    @staticmethod
    def _window_output(text: str, *, max_lines: int = 80) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        lines = normalized.splitlines()
        if len(lines) <= max_lines:
            return normalized
        head_count = max_lines // 2
        tail_count = max_lines - head_count
        excerpt = lines[:head_count] + lines[-tail_count:]
        hidden = len(lines) - len(excerpt)
        return "\n".join(excerpt + [f"... {hidden} line(s) omitted ..."])

    @classmethod
    def _command_state_label(
        cls,
        *,
        success: bool,
        interrupted: bool,
        detached: bool,
    ) -> str:
        if interrupted:
            return "interrupted"
        if detached:
            return "detached"
        if success:
            return "completed"
        return "failed"

    @classmethod
    def _format_interactive_command_result(
        cls,
        *,
        command: str,
        stdout: str,
        cwd: Path | None,
        success: bool,
        interrupted: bool,
        detached: bool,
        returncode: int | None,
    ) -> str:
        state = cls._command_state_label(success=success, interrupted=interrupted, detached=detached)
        rows = [
            f"Interactive shell command finished with state={state}.",
            f"command={command}",
            f"cwd={cwd}" if cwd else "cwd=(unknown)",
            f"returncode={returncode}" if returncode is not None else "returncode=(unknown)",
            f"detached={detached}",
        ]
        excerpt = cls._window_output(stdout, max_lines=80)
        if excerpt:
            rows.extend(["", "--- output ---", excerpt])
        return "\n".join(rows).strip()

    @staticmethod
    def _cwd(ctx: ToolContext) -> str | None:
        if ctx.get_cwd:
            try:
                return ctx.get_cwd()
            except Exception:
                return None
        return None

    @staticmethod
    def _interactive_patterns(ctx: ToolContext) -> List[str]:
        raw = ctx.config.get_nested("tools", "shell", "interactive_patterns", default=[]) or []
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def _command_cwd(ctx: ToolContext) -> Path:
        if ctx.get_cwd:
            with suppress(Exception):
                return Path(ctx.get_cwd()).expanduser().resolve()
        return Path.cwd().resolve()

    @staticmethod
    def _contains_interactive_shell_primitives(text: str) -> bool:
        patterns = (
            r"(?i)\bread\b\s+-[A-Za-z]*p\b",
            r"(?i)\bread\b(?:\s+[^;&|]+)?\s+-[A-Za-z]*p\b",
            r"(?i)\bselect\b\s+\w+\s+in\b",
            r"(?i)\bwhiptail\b",
            r"(?i)\bdialog\b",
            r"(?i)\bfzf\b",
            r"(?i)\bgum\s+(confirm|input|choose)\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def _referenced_script_path(cls, command: str, ctx: ToolContext) -> Path | None:
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
        if not argv:
            return None
        cwd = cls._command_cwd(ctx)
        candidates: list[str] = []
        first = argv[0]
        if first in {"bash", "sh", "zsh"} and len(argv) >= 2:
            for arg in argv[1:]:
                if arg.startswith("-"):
                    continue
                candidates.append(arg)
                break
        elif first.endswith(".sh") or first.startswith("./") or first.startswith("../") or "/" in first:
            candidates.append(first)
        for raw in candidates:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = cwd / path
            with suppress(Exception):
                resolved = path.resolve()
                if resolved.is_file():
                    return resolved
        return None

    @classmethod
    def _script_looks_interactive(cls, command: str, ctx: ToolContext) -> bool:
        script_path = cls._referenced_script_path(command, ctx)
        if script_path is None:
            return False
        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")[:16384]
        except Exception:
            return False
        return cls._contains_interactive_shell_primitives(content)

    @classmethod
    def _looks_interactive(cls, command: str, ctx: ToolContext) -> bool:
        text = str(command or "").strip()
        if not text:
            return False
        if cls._contains_interactive_shell_primitives(text):
            return True
        for pattern in cls._interactive_patterns(ctx):
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                continue
        if cls._script_looks_interactive(text, ctx):
            return True
        return False

    @staticmethod
    def _interactive_command_message(command: str, ctx: ToolContext) -> str:
        template = str(
            ctx.config.get_nested(
                "tools",
                "shell",
                "interactive_command_message",
                default=(
                    "This command appears interactive or TTY-bound and should not run through shell_exec. "
                    "Use a local shell block instead, for example `!{command}`. "
                    "k-ai will hand control to you automatically if the command prompts for a password or further input."
                ),
            )
            or ""
        )
        try:
            return template.format(command=command.strip())
        except Exception:
            return template

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_cfg = ctx.config.get_nested("tools", "shell", default={})
        if not capability_enabled(ctx.config, "shell"):
            return ToolResult(success=False, message="shell_exec is disabled in config.")

        command = arguments.get("command", "")
        if not command.strip():
            return ToolResult(success=False, message="No command provided.")
        if self._looks_interactive(command, ctx):
            if ctx.run_local_interactive_shell:
                try:
                    result = await ctx.run_local_interactive_shell(command)
                    output = self._format_interactive_command_result(
                        command=command,
                        stdout=result.stdout or "",
                        cwd=result.cwd,
                        success=bool(result.success),
                        interrupted=bool(result.interrupted),
                        detached=bool((result.metadata or {}).get("detached")),
                        returncode=result.returncode,
                    )
                    return ToolResult(
                        success=bool(result.success),
                        message=output,
                        data={
                            "interactive_command": True,
                            "command": command,
                            "state": self._command_state_label(
                                success=bool(result.success),
                                interrupted=bool(result.interrupted),
                                detached=bool((result.metadata or {}).get("detached")),
                            ),
                            "stdout": result.stdout,
                            "cwd": str(result.cwd) if result.cwd else None,
                            "returncode": result.returncode,
                            "interrupted": bool(result.interrupted),
                            "detached": bool((result.metadata or {}).get("detached")),
                        },
                    )
                except Exception as e:
                    return ToolResult(success=False, message=f"Interactive local command failed: {e}")
            return ToolResult(
                success=False,
                message=self._interactive_command_message(command, ctx),
                data={"interactive_command": True, "command": command},
            )

        timeout = int(tool_cfg.get("timeout", 30))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd(ctx),
            )
            stdout, stderr = await PythonExecTool._wait_for_process(proc, timeout=timeout, ctx=ctx)
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
        except KeyboardInterrupt:
            return ToolResult(success=False, message="Command interrupted by user.", data={"interrupted": True})
        except Exception as e:
            return ToolResult(success=False, message=f"Command failed: {e}")


def register_external_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    for tool_cls in [
        ExaSearchTool,
        PythonExecTool,
        PythonSandboxListPackagesTool,
        PythonSandboxInstallPackagesTool,
        PythonSandboxRemovePackagesTool,
        ShellExecTool,
    ]:
        registry.register(tool_cls())
