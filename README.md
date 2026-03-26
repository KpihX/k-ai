# k-ai

`k-ai` is a terminal-first LLM chat CLI with persistent sessions, runtime transparency, live config management, internal tools, and session-aware retrieval.

## What It Does

- Rich interactive chat UI with streaming responses and tool proposals/results.
- Persistent session store in JSONL with digest generation: summary + themes.
- Full runtime transparency in the UI: provider, model, context window, compaction threshold, token stats, active limits, config persistence path.
- Live config management from chat: provider, model, `max_tokens`, UI limits, tool settings, and nested config keys.
- Internal tools for session lifecycle, config, memory, Python, shell, web search, and QMD-based history/document retrieval.
- QMD session retrieval restricted to the `k-ai` collection to avoid irrelevant cross-collection noise.
- Safer interactive flow with human validation for tool calls and interruption handling for prompt, streaming, and tool execution.

## Current State

- Boot flow shows recent sessions immediately.
- Boot assistant no longer re-asks for the session list; it may only suggest a direct resume.
- Session windows are explicit: load last `N`, extract `offset/limit`, refresh digest/themes on current or previous sessions.
- Runtime token stats fall back to estimates when the provider does not return usage counters.
- Test suite currently passes with `316` tests.

## Quick Start

```bash
git clone https://github.com/kpihx/k-ai.git
cd k-ai
./scripts/install.sh
k-ai chat
```

Development setup:

```bash
uv sync --dev
uv run pytest -q
uv run k-ai chat
```

## Chat Capabilities

Session management:

- `/sessions`
- `/load <id> [last_n]`
- `/extract <id> [offset] [limit]`
- `/digest [id]`
- `/delete <id>`
- `/compact`
- `/reset`

Runtime/config management:

- `/status`
- `/tokens`
- `/settings [prefix]`
- `/set <key> <value>`
- `/model [name]`
- `/provider [name] [model]`
- `/config show [key]`
- `/config save [path]`
- `/config get [path]`

Memory and search:

- `/memory list|add|remove`
- `/qmd query|search|get|ls|status|update|embed|cleanup`

All of these can also be triggered naturally by the model via internal tools.

## Runtime Transparency

The UI exposes:

- Active provider/model/auth mode.
- Context window usage and remaining capacity.
- Compaction threshold and current history depth.
- Token stats with source labeling:
  `provider` when usage is returned by the backend.
  `estimated` when usage must be inferred from message sizes.
- Tool display/history truncation limits.
- Config persistence target.

Relevant config keys:

```yaml
provider: "mistral"
model: ""
temperature: 0.7
max_tokens: 8192

cli:
  render_mode: "rich"
  show_runtime_stats: true
  runtime_stats_mode: "compact"
  tool_result_max_display: 500
  tool_result_max_history: 4000
  confirm_all_tools: true

config:
  persist_path: "~/.k-ai/config.yaml"
```

## Installation

The installer:

- syncs Python dependencies with `uv`
- installs the editable CLI entrypoint
- creates `~/.k-ai/`
- initializes memory/session storage
- creates the Python sandbox for `python_exec`
- installs/configures QMD when available
- configures the `k-ai` QMD collection
- runs `k-ai doctor`

Run:

```bash
./scripts/install.sh
```

To completely remove installed state:

```bash
./scripts/purge.sh
```

## Make Targets

```bash
make help
make install
make purge
make test
make check
make build
make push
```

## File Layout

```text
~/.k-ai/
├── config.yaml
├── MEMORY.json
├── sandbox/
└── sessions/
    ├── index.json
    └── <session>.jsonl
```

Repo layout:

```text
src/k_ai/
test/
scripts/
README.md
CHANGELOG.md
TODO.md
Makefile
```

## Notes

- `Ctrl+C` during prompt input returns control once, exits on the second press.
- `Ctrl+C` during generation/tool execution interrupts the current action and returns to the prompt.
- `Esc` is also handled during non-prompt interruption scopes, but terminal behavior is less uniform than `Ctrl+C`.
- Tool proposals remain human-validated by default.

## License

MIT
