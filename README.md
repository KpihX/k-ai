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

## Problem-First Docs

Long-form architecture docs now live in the standalone docs site:

- GitHub Pages: `https://kpihx.github.io/k-ai-docs/`
- Source repo: `https://github.com/KpihX/k-ai-docs`

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

Development:

```bash
uv sync --dev
uv run pytest -q
uv run k-ai chat
```

## Installation and Removal

Install:

```bash
make install
# or directly:
./scripts/install.sh
```

Purge runtime state:

```bash
make purge
# or directly:
./scripts/purge.sh
```

Make targets:

```bash
make install
make purge
make check
make test
make build
```

## CLI Usage

### Interactive chat

```bash
k-ai chat
k-ai chat --provider mistral
k-ai chat --provider openai --model gpt-4o
k-ai chat --config ~/.k-ai/config.yaml
k-ai chat --temperature 0.2 --max-tokens 4096
```

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
- `/config show [key]`
- `/config show section:<name> [section:<name> ...]`
- `/config get [path] [section ...]`
- `/config save [path]`
- `/config sections`

Tools and memory:

- `/tools show [ask|auto|default|session|global|protected]`
- `/tools ask|auto <target> [session|global] [tool|category|risk]`
- `/tools reset <target> [session|global] [tool|category|risk]`
- `/memory list|add|remove`
- `/qmd query|search|get|ls|status|update|embed|cleanup`

Everything above can also be triggered by the model through internal tools when appropriate.

## Config Layout

Built-in defaults are split into four fragments:

```text
src/k_ai/defaults/defaults.d/
├── 00-models.yaml
├── 10-ui-prompts.yaml
├── 20-sessions-memory.yaml
└── 30-runtime-governance.yaml
```

Section names exposed in CLI:

- `models`
- `ui`
- `sessions`
- `governance`

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
