# k-ai : The Conscious LLM CLI

```
  ╭──────────────────────────────────────────────────────────╮
  │  k-ai  │  Unified LLM Chat with Memory & Awareness      │
  │                                                          │
  │  Provider : mistral        Model : mistral-large-latest  │
  │  Temp     : 0.7            Tokens: 8192                  │
  ╰──────────────────────────────────────────────────────────╯
```

**k-ai** is a next-generation CLI for interacting with Large Language Models.
It combines a rich terminal UI with persistent sessions, memory, semantic search
(QMD), LaTeX math rendering, and a 32-tool internal system — all provider-agnostic
and 100% configurable.

## Architecture

```
                  ┌────────────────────────────────┐
                  │         k-ai CLI / API          │
                  └──────┬──────────┬──────────┬───┘
                         │          │          │
              ┌──────────┴──┐ ┌─────┴─────┐ ┌──┴─────────┐
              │ ChatSession │ │ 32 Tools  │ │  Doctor    │
              │ (agentic    │ │ Registry  │ │            │
              │  loop)      │ └─────┬─────┘ └────────────┘
              └──────┬──────┘       │
          ┌──────────┼─────────┐    │
    ┌─────┴──┐  ┌────┴───┐ ┌──┴────┴──────┐
    │LiteLLM │  │Session │ │    Memory     │
    │ Driver │  │ Store  │ │ ext + internal│
    └────────┘  └────────┘ └──────────────-┘
         │          │              │
    ┌────┴────┐  ┌──┴───┐    ┌────┴────┐
    │ 8 LLM   │  │ JSONL │   │MEMORY   │
    │providers │  │ files │   │ .json   │
    └─────────┘  └──────-┘   └────────-┘
```

## Quick Start

```bash
git clone https://github.com/kpihx/k-ai.git && cd k-ai
uv sync --dev

k-ai chat                                      # default provider
k-ai chat -p mistral -m mistral-large-latest    # specific
k-ai chat -t 0.2 -n 4096 -s "Expert Python."   # overrides
k-ai doctor                                     # diagnostics
```

## Features at a Glance

```
┌─ Sessions ──────────────────┐  ┌─ Memory ──────────────────┐
│ Auto-save every message     │  │ External (read-only):     │
│ Resume with /load <id>      │  │   ~/.agents/KERNEL.md     │
│ Auto-title after 1st exchange│  │ Internal (read-write):   │
│ Rich summary on exit        │  │   ~/.k-ai/MEMORY.json    │
│ History preview on load     │  │   /memory add|list|remove │
└─────────────────────────────┘  └───────────────────────────┘

┌─ 32 Internal Tools ────────────────────────────────────────┐
│ Session: new load exit rename list delete compact clear    │
│ Config:  set_config                                        │
│ Memory:  memory_add memory_list memory_remove              │
│ Search:  exa_search                                        │
│ Code:    python_exec shell_exec                            │
│ QMD:     query search vsearch get multi_get ls             │
│          collection_list/add/remove/show                   │
│          context_list/add/rm                               │
│          status update embed cleanup                       │
└────────────────────────────────────────────────────────────┘

┌─ Rendering ─────────────────┐  ┌─ Agentic Loop ───────────┐
│ 3 modes (config-driven):    │  │ LLM → tools → LLM → ...  │
│  raw      Plain text        │  │ Human-in-the-loop [y/n]   │
│  markdown Rich Markdown     │  │ Tool results in panels    │
│  rich     MD + Unicode math │  │ Max 10 rounds safety      │
│                             │  │ LangGraph-ready           │
│ LaTeX → Unicode: 200+ syms  │  └───────────────────────────┘
│ α β γ ∑ ∫ √ ℝ ℕ ⊂ ∈ ⟹ ...│
└─────────────────────────────┘
```

## Boot Flow (Consciousness)

```
1. Show welcome panel + recent sessions table
2. LLM greets proactively (based on sessions + memory)
3. User responds:
   ├─ Resume session → LLM proposes load_session → [y/n] → history loaded
   ├─ New topic → session created lazily on first real message
   └─ Close without responding → nothing saved
4. After 1st exchange → auto-generate session title
5. On exit → auto-generate title + rich summary
```

## Commands (all also available in natural language)

| Command | Description |
|---|---|
| `/help` | All commands |
| `/new` | New session |
| `/load <id>` | Resume session (shows recent history) |
| `/sessions` | List sessions |
| `/rename <title>` | Rename session |
| `/delete <id>` | Delete session |
| `/compact` | Compress history |
| `/clear` | Clear screen |
| `/reset` | Clear history |
| `/memory list\|add\|remove` | Manage memory |
| `/model [name]` | Switch model |
| `/provider [name]` | Switch provider |
| `/set <key> <value>` | Live config |
| `/doctor` | Full diagnostic |
| `/qmd query\|search\|get\|ls\|...` | Full QMD suite |
| `/exit` | Exit (auto-saves) |

## Configuration (`config.yaml`)

```yaml
provider: "ollama"          # or mistral, groq, gemini, anthropic, openai, xai, dashscope
temperature: 0.7
max_tokens: 8192
stream: true

cli:
  render_mode: "rich"       # raw | markdown | rich (with Unicode math)
  debug: false              # show raw prompts
  tool_result_max_display: 500

sessions:
  directory: "~/.k-ai/sessions"
  max_recent: 10

compaction:
  trigger_percent: 80       # auto-compact at 80% of context window
  keep_last_n: 10

memory:
  external_file: "~/.agents/KERNEL.md"
  internal_file: "~/.k-ai/MEMORY.json"

prompts:                    # all LLM instructions are configurable
  identity: "..."
  boot_with_sessions: "..."
  boot_no_sessions: "..."
  compact_summarize: "..."
  exit_title: "..."
  exit_summary: "..."

tools:
  exa_search: { enabled: true, api_key_env_var: "EXA_API_KEY" }
  python_exec: { enabled: true, timeout: 30 }
  shell_exec: { enabled: true, timeout: 30 }
  qmd_search: { enabled: true, limit: 5 }
```

## Providers

| Provider | Auth | Models |
|---|---|---|
| ollama | local | phi4-mini, llama3, qwen3.5, ... |
| mistral | API key | mistral-large, mistral-small, ... |
| groq | API key | qwen3-32b, llama-3.3-70b, ... |
| gemini | API key / OAuth | gemini-2.5-flash, gemini-2.5-pro |
| anthropic | API key | claude-sonnet-4-6, claude-opus-4-6 |
| openai | API key | gpt-4o, o1, o3-mini |
| xai | API key | grok-3-fast-beta, grok-4 |
| dashscope | API key | qwen-turbo, qwen-max |

## Library Usage

```python
import asyncio
from k_ai import ConfigManager, ChatSession, get_provider

cm = ConfigManager(provider="mistral", temperature=0.2)
session = ChatSession(cm)
response = asyncio.run(session.send("What is 2+2?"))

# Low-level streaming
provider = get_provider(cm)
async for chunk in provider.chat_stream(messages):
    print(chunk.delta_content, end="")
```

## File Layout

```
~/.k-ai/
├── sessions/
│   ├── index.json         # session metadata
│   ├── <uuid>.jsonl       # messages (one JSON per line)
│   └── ...
└── MEMORY.json            # internal memory
```

## Tests

```bash
uv run pytest test/ -q       # 283 tests
uv run pytest test/ -v       # verbose
uv run pytest test/ -k qmd   # filter
```

## License

MIT
