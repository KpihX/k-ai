# k-ai

`k-ai` is a terminal-first LLM chat system with persistent sessions, runtime transparency, live config mutation, internal tools, and a Python package API.

It is designed around one principle: the chat loop, the slash commands, and the programmatic API should all act on the same session/config/runtime model.

## Core Model

```text
                 ┌─────────────────────────────────────┐
                 │           Built-in Defaults         │
                 │   src/k_ai/defaults/defaults.d/     │
                 └─────────────────┬───────────────────┘
                                   │ merge
                 ┌─────────────────▼───────────────────┐
                 │          ConfigManager               │
                 │  override file + live edits + CLI    │
                 └─────────────────┬───────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │                    ChatSession                       │
        │  prompt loop · tools · digest · compaction · UI     │
        └───────────────┬───────────────────────┬─────────────┘
                        │                       │
            ┌───────────▼──────────┐   ┌───────▼────────┐
            │ SessionStore          │   │ MemoryStore     │
            │ ~/.k-ai/sessions/*.jsonl│  │ ~/.k-ai/MEMORY │
            └──────────────────────┘   └────────────────┘
```

## Features

- Persistent chat sessions with `summary`, `themes`, and `session_type`.
- Rich runtime transparency: provider, model, auth mode, token source, context window, compaction threshold, limits.
- Human-in-the-loop tool approvals with per-tool governance.
- Full config management from chat, slash commands, or Python.
- Sandboxed Python and shell tools.
- QMD-backed history/document retrieval restricted to the `k-ai` session collection when appropriate.
- Robust interruption handling for prompt input, generation, and tool execution.
- Split default config fragments with cached loading for better maintainability and lower parse overhead.
- Native `SKILL.md` runtime with project/global discovery, prompt injection, and an internal `activate_skill` tool.
- Fast one-shot mode via `k-ai "..."` or `k-ai ask ...`, with no history or tool loop.
- Session-scoped working directories (`-C/--cwd`) shared by chat, local runners, and runtime tools.
- Mixed multiline input in chat: plain text to the LLM, `!` shell blocks, `>` Python blocks, and `/?` ephemeral contextual questions.
- Persistent PTY-backed local shell and Python runners, including secure focus mode for interactive prompts such as `sudo` passwords.
- Append-only long-response streaming with early visibility and no full-height Live panel clipping/flicker on long answers.
- Full-screen Textual TUI for `k-ai chat`: stable panes, modal tool approvals, runtime inspector, session sidebar, and a multiline composer.

## Problem-First Docs

Long-form architecture docs now live in the standalone docs site:

- Live docs site: [kpihx.github.io/k-ai-docs](https://kpihx.github.io/k-ai-docs/)
- Docs source repo: [github.com/KpihX/k-ai-docs](https://github.com/KpihX/k-ai-docs)
- Local docs entrypoint: [`docs/README.md`](docs/README.md)

They are written in the same spirit as `tutos_live`:

- problem first
- real examples
- ASCII diagrams
- request payload examples
- session / memory / tool-governance workflows

## Quick Start

```bash
git clone https://github.com/kpihx/k-ai.git
cd k-ai
make install
k-ai chat
```

Installation profiles:

- editable defaults: [`install/install.yaml`](install/install.yaml)
- installer docs: [`install/README.md`](install/README.md)

Installer behavior highlights:

- interactive by default, with explicit choices shown for each meaningful case
- prefers `uv` when available
- if `uv` is missing, proposes installing it
- if `uv` is declined, falls back to an isolated `k-ai` bootstrap virtualenv instead of polluting the system Python
- asks which live capability families should start enabled: `exa`, `python`, `shell`, `qmd`
- can skip verification entirely via `install/install.yaml` when used inside controlled test harnesses
- installs a managed runtime `.gitignore` in `~/.k-ai/`
- initializes a local git repo in `~/.k-ai/` and creates the first commit
- can auto-commit runtime state on each interactive chat exit using the session digest as the commit subject

You can keep the default interactive install, explicitly target the default
profile, or point to your own:

```bash
./scripts/install.sh
./scripts/install.sh -p
./scripts/install.sh -p defaults
./scripts/install.sh --path /path/to/my-install.yaml
```

Development:

```bash
uv sync --dev
uv run pytest -q
uv run k-ai chat
```

Chat UI backends:

- default: Textual full-screen TUI
- fallback: `k-ai chat --classic-ui`
- live config knob: `cli.ui.backend = textual|classic`

Skills runtime defaults:

- project overlays: `.k-ai/skills`, `.agents/skills`
- global skills: `~/.agents/skills`
- inspection commands: `/skills`, `/skills show <name>`, `/skills reload`, `/skills active`

Hooks runtime defaults:

- project overlays: `.k-ai/hooks`, `.agents/hooks`
- global hooks: `~/.agents/hooks`
- discovered config files per root: `hooks.yaml`, `hooks.yml`, `hooks.json`
- supported events: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PostToolUseFailure`, `Stop`
- inspection commands: `/hooks`, `/hooks reload`

MCP runtime defaults:

- config fragment: `src/k_ai/defaults/defaults.d/60-mcp.yaml`
- first bundled server: `filesystem` via `mcp-server-filesystem`
- transport foundation: official Python `mcp` SDK over `stdio`, `streamable_http`, and `sse`
- roots: workspace root exposed by default to MCP servers that request them
- protocol surfaces exposed: tools, resources, resource templates, prompts, roots
- inspection/admin commands:
  - `/mcp`
  - `/mcp tools`
  - `/mcp resources [server]`
  - `/mcp templates [server]`
  - `/mcp prompts [server]`
  - `/mcp probe <name> [command_or_package]`
  - `/mcp install <name> [package] [binary]`
  - `/mcp add-stdio <name> <command> [cwd]`
  - `/mcp add-http <name> <url> [streamable_http|sse]`
  - `/mcp enable|disable <name>`
  - `/mcp remove <name>`
  - `/mcp reload`

The staged architecture roadmap for `skills`, `hooks`, filesystem editing, and
MCP is tracked in [`VISION.md`](VISION.md).

Interaction runtime defaults:

- config fragment: `src/k_ai/defaults/defaults.d/70-interaction.yaml`
- one-shot entrypoints:
  - `k-ai "Explique ce repo"`
  - `k-ai ask "Quelle est la différence entre hooks et skills ?"`
- chat working directory override:
  - `k-ai chat -C ~/Work/AI/k_ai`
- multiline document syntax in chat:
  - plain text -> normal LLM turn
  - `!` -> persistent shell block
  - `>` -> persistent Python block
  - `/?` -> contextual but non-persistent, tool-less quick question
- default local block rendering:
  - buffered shell/Python result panels for clean UI
  - `/focus shell` or `/focus python` for truly interactive PTY control
- runner focus/admin commands:
  - `/cwd [path]`
  - `/focus shell`
  - `/focus python`

Published package identity:

- PyPI distribution name: `kpihx-ai`
- import module: `k_ai`
- installed CLI command: `k-ai`

If you install from PyPI instead of from source:

```bash
uv tool install kpihx-ai
# or
pipx install kpihx-ai
```

## Installation and Removal

Install:

```bash
make install
# or directly:
./scripts/install.sh
# or with an explicit install profile:
./scripts/install.sh -p defaults
./scripts/install.sh --path ./install/install.yaml
```

Purge runtime state:

```bash
make purge
# or directly:
./scripts/purge.sh --yes
# or for a custom runtime root:
./scripts/purge.sh --yes --runtime-dir /path/to/runtime
```

Make targets:

```bash
make install
make purge
make check
make test
make build
make publish
make push
make push-docs
make release
```

## Runtime Store Versioning

The local runtime store `~/.k-ai/` is now treated as a narrow git repo.

Tracked:

- `config.yaml`
- `MEMORY.json`
- `sessions/index.json`
- `sessions/*.jsonl`

Ignored:

- `sandbox/`
- any other runtime-heavy or ephemeral artifact outside the tracked list

The installer copies the managed template from [`install/.gitignore.runtime`](install/.gitignore.runtime), initializes `~/.k-ai/.git/`, runs the first `git add .`, and creates the initial commit.

During normal interactive use, `k-ai` can also auto-commit runtime changes on chat exit. The generated commit subject uses the session digest, for example:

```text
chat: Présentation détaillée de l'assistant k-ai
```

## CLI Usage

### Interactive chat

```bash
k-ai chat
k-ai chat -C ~/Work/AI/k_ai
k-ai chat --classic-ui
k-ai chat --provider mistral
k-ai chat --provider openai --model gpt-4o
k-ai chat --config ~/.k-ai/config.yaml
k-ai chat --temperature 0.2 --max-tokens 4096
```

### One-shot ask

```bash
k-ai "Explique ce dépôt"
k-ai ask "Que signifie MCP ?"
k-ai -C ~/Work/AI/k_ai "Quel est le rôle de session.py ?"
```

### Mixed multiline chat input

Inside `k-ai chat`, you can submit multiline documents that mix local execution
and model questions in one batch:

```text
!pwd
!git status --short
explique ce que tu vois
>import os
>os.getcwd()
/? rappelle-moi brièvement ce que signifie ce statut git
```

Rules:

- `!` lines are executed in the persistent session shell
- `>` lines are executed in the persistent session Python runner
- normal text is sent to the model
- `/?` asks a contextual quick question without polluting chat history

### Textual TUI

`k-ai chat` now opens a full-screen Textual application by default.

Main panes:

- top boot panel: recent sessions only at startup, then dismissible
- center: transcript + live streaming slot + multiline composer
- right: runtime inspector and activity log, hidden automatically on narrow terminals

Core bindings:

- `Ctrl+S` send the current composer buffer
- `F2` send the current composer buffer
- `Ctrl+Enter` send when the terminal forwards it distinctly
- `Ctrl+J` focus composer
- `Ctrl+B` focus the boot sessions table when visible
- `Ctrl+R` focus runtime
- `Ctrl+L` focus activity
- `Ctrl+Q` quit the TUI

Tool approvals now open as proper modal dialogs instead of being appended into
the same linear transcript flow.
- local runner outputs are injected into the next LLM block of the same batch

### Config CLI

Show the full built-in default template:

```bash
k-ai config show
```

List built-in config fragments:

```bash
k-ai config sections
```

Show only selected built-in fragments:

```bash
k-ai config show --section ui
k-ai config show --section models --section governance
```

Export the full default config:

```bash
k-ai config get -o my-config.yaml
```

Export only one or several sections to build a minimal override file:

```bash
k-ai config get -o prompts.yaml --section ui
k-ai config get -o providers-and-tools.yaml --section models --section governance
```

Open the active config or one built-in fragment in your editor:

```bash
k-ai config edit all
k-ai config edit ui
k-ai config edit governance
/config edit governance
```

Editor resolution order:

- `config.editor`
- `K_AI_EDITOR`
- `VISUAL`
- `EDITOR`
- `nano`

Tool proposal transparency:

- `cli.show_tool_rationale: true` keeps a justification panel visible before each tool.
- if the model emits no explanation, `k-ai` derives a fallback rationale from the tool description and main input.

OAuth note:

- `oauth.gemini` is implemented through a Google token JSON file.
- `token_path` should point to a persisted token containing at least `access_token`.
- If the token is expired, `refresh_token`, `client_id`, and `client_secret` are used to refresh it automatically.

Run diagnostics:

```bash
k-ai doctor
k-ai doctor --reset config
k-ai doctor --reset all
```

## Slash Commands

Session lifecycle:

- `/sessions [recent|oldest] [classic|meta]`
- `/load <id> [last_n]`
- `/extract <id> [offset] [limit]`
- `/digest [id]`
- `/compact`
- `/delete <id>`
- `/new [classic|meta]`

Runtime/config:

- `/status`
- `/tokens`
- `/settings [prefix]`
- `/set <key> <value>`
- `/model [name]`
- `/provider [name] [model]`
- `/tools capabilities`
- `/tools enable|disable <exa|python|shell|qmd|mcp>`
- `/mcp [list|tools|resources|templates|prompts|reload|probe|install|add-stdio|add-http|enable|disable|remove]`
- `/cwd [path]`
- `/focus <shell|python>`
- `/config show [key]`
- `/config show section:<name> [section:<name> ...]`
- `/config get [path] [section ...]`
- `/config save [path]`
- `/config sections`
- `/config edit [all|models|ui|sessions|governance|skills|hooks|mcp|interaction]`

Tools and memory:

- live capability switching only applies to mutable families (`exa`, `python`, `shell`, `qmd`, `mcp`)
- protected admin approval rules remain YAML-only by design
- `/tools show [ask|auto|default|session|global|protected]`
- `/tools ask|auto <target> [session|global] [tool|category|risk]`
- `/tools reset <target> [session|global] [tool|category|risk]`
- `/memory list|add|remove`
- `/qmd query|search|get|ls|status|update|embed|cleanup`

Everything above can also be triggered by the model through internal tools when appropriate.

## Config Layout

Built-in defaults are split into named fragments:

```text
src/k_ai/defaults/defaults.d/
├── 00-models.yaml
├── 10-ui-prompts.yaml
├── 20-sessions-memory.yaml
├── 30-runtime-governance.yaml
├── 40-skills.yaml
├── 50-hooks.yaml
├── 60-mcp.yaml
└── 70-interaction.yaml
```

Section names exposed in CLI:

- `models`
- `ui`
- `sessions`
- `governance`
- `skills`
- `hooks`
- `mcp`
- `interaction`

Recommended override strategy:

```text
1. Export only the sections you want to change.
2. Edit that smaller YAML file.
3. Pass it with --config or save it as ~/.k-ai/config.yaml.
4. Keep runtime-only experiments in chat via /set or the config tools.
```

## Package Usage

### Defaults only

```python
from k_ai import ConfigManager, ChatSession
import asyncio

cm = ConfigManager()
session = ChatSession(cm)
asyncio.run(session.send("Bonjour"))
```

### Fast one-shot ask

```python
from k_ai import ConfigManager, ChatSession
import asyncio

cm = ConfigManager()
session = ChatSession(cm, workspace_root="~/Work/AI/k_ai")
print(asyncio.run(session.ask("Explique le rôle de session.py")))
```

### Custom override file

```python
from k_ai import ConfigManager, ChatSession

cm = ConfigManager(override_path="~/.k-ai/config.yaml")
session = ChatSession(cm, provider="mistral")
```

You can also keep several smaller override files and choose one at startup:

```python
cm = ConfigManager(override_path="~/profiles/k-ai-prompts.yaml")
```

### Inline overrides

```python
cm = ConfigManager(
    override_path="~/.k-ai/config.yaml",
    temperature=0.2,
    max_tokens=4096,
)
```

### Export only one built-in section

```python
from k_ai import ConfigManager

yaml_text = ConfigManager.get_default_yaml(sections=["ui"])
print(yaml_text)
```

### List built-in sections

```python
from k_ai import ConfigManager

for section in ConfigManager.list_default_sections():
    print(section["name"], section["file"])
```

### Agentic programmatic call with tools

```python
import asyncio
from k_ai import ConfigManager, ChatSession, ToolCall

cm = ConfigManager()
session = ChatSession(cm)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

async def executor(tc: ToolCall) -> str:
    if tc.function_name == "get_weather":
        return f"22°C in {tc.arguments['location']}"
    raise ValueError(tc.function_name)

result = asyncio.run(session.send_with_tools("Weather in Paris?", tools, executor))
print(result)
```

## Runtime Transparency

The terminal runtime panel exposes:

- current provider / model / auth mode
- context usage and remaining capacity
- compaction threshold
- cumulative tokens
- token source: `provider` or `estimated`
- render mode
- tool result display/history limits
- config persistence path
- current session id / type

This is UI-only telemetry; it does not consume model tokens.

## Robustness Notes

- `Ctrl+C` at prompt: first press cancels input, second press exits.
- `Ctrl+C` during generation or tool execution: returns control to the prompt.
- Boot greeting failures do not create a session.
- Programmatic `send()` / `send_with_tools()` now rollback the whole turn on LLM failure instead of leaving partial persisted turns.
- Digest/compaction/exit summarization are best-effort; if the provider fails, the session remains usable and the main conversation state is preserved.
- Tool approval overrides are validated strictly against the built-in tool catalog, so malformed config fails fast instead of silently drifting.

## Runtime State on Disk

```text
~/.k-ai/
├── config.yaml
├── MEMORY.json
├── sandbox/
└── sessions/
    ├── index.json
    └── <session-id>.jsonl
```

## License

MIT
