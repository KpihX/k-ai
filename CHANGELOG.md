# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- Full-screen Textual chat UI as the new default `k-ai chat` surface, with a session sidebar, runtime/activity inspector, multiline composer, dedicated streaming slot, and modal tool approvals.
- Fast one-shot ask mode through both `k-ai ask ...` and the root shorthand `k-ai "..."`, with a minimal no-tools prompt profile for low-latency questions.
- A dedicated interaction runtime config fragment (`70-interaction.yaml`) covering one-shot ask, `cwd`, multiline input syntax, and persistent local runners.
- Multiline mixed chat documents with four explicit block kinds:
  - plain LLM text
  - `!` persistent shell blocks
  - `>` persistent Python blocks
  - `/?` contextual but non-persistent tool-less questions
- Persistent PTY-backed user shell and Python runners, plus `/focus shell`, `/focus python`, and `/cwd` slash commands.
- CLI/root `-C/--cwd` support so chat, ask, shell blocks, Python blocks, and runtime tools all share the same explicit working directory.
- New tests for parser semantics, ask mode, cwd propagation, persistent runners, CLI root fallback, and local-output injection into mixed batches.
- Dedicated UI tests for append-only streaming, terminal-output sanitization, sentinel parsing, and focus enter/exit runner hooks.

### Changed

- `shell_exec` and `python_exec` now honor the session working directory instead of implicitly running in the process cwd.
- The interactive prompt is now multiline and routes submitted documents through a dedicated interaction parser instead of a single raw user string.
- `scripts/purge.sh` now detects installed uv tools more robustly before uninstalling them.
- Long assistant responses now stream append-only under a static assistant header instead of keeping one giant full-height Rich `Live` panel alive.
- Default `!` and `>` block rendering is now buffered into clean local-result panels, while explicit `/focus` remains the path for raw interactive PTY control.
- `k-ai chat` now defaults to the Textual TUI, while `--classic-ui` keeps the previous prompt-toolkit + Rich loop available as a fallback.

### Fixed

- Shell/Python local runner output now strips PTY control noise, sentinel-command leakage, and login-shell echo artifacts before those results reach the user-facing UI or the LLM batch context.
- The classic runtime still honors the previous `render_notice` / tool-approval hooks used by the existing test and compatibility surface while sharing the same core session engine as the new TUI.

## [0.2.0] - 2026-03-27

### Added

- Native MCP runtime foundation using the official Python `mcp` SDK, with `stdio` / `streamable_http` / `sse`, roots support, dynamic MCP tool import, MCP resources/prompts access, admin tools and `/mcp` commands, and the official `filesystem` server configured as the first bundled MCP.
- Native hooks runtime with Claude-style event names, directory discovery (`.k-ai/hooks`, `.agents/hooks`, `~/.agents/hooks`), strict config parsing, command execution with stdin JSON payloads, blocking pre-hooks, post-tool feedback, and `/hooks` inspection commands.
- Native `SKILL.md` runtime with `~/.agents/skills` default discovery, project overlays (`.k-ai/skills`, `.agents/skills`), lazy metadata/body loading, slash-command inspection, and the internal `activate_skill` tool.
- Managed runtime-store git tracking for `~/.k-ai/`, including a committed `.gitignore` template that keeps only `config.yaml`, `MEMORY.json`, and `sessions/*`.
- Automatic runtime-store commit support on interactive chat exit, with commit subjects derived from the session digest.
- Installation profile support for runtime git defaults (`runtime_git.*`) and installer-side initialization of the runtime git repo.
- Integration tests for `scripts/install.sh` and `scripts/purge.sh`, including custom runtime roots and purge edge cases.
- `VISION.md` at the project root to keep the staged `skills -> hooks -> filesystem -> MCP` roadmap explicit.

### Changed

- `scripts/install.sh` now offers installation of the official `@modelcontextprotocol/server-filesystem` package and persists MCP capability/runtime defaults into `config.yaml`.
- The system prompt now exposes a compact skill catalog and injects activated skill bodies into the current turn below internal prompts but above external memory.
- The installer now copies `install/.gitignore.runtime` into `~/.k-ai/.gitignore` instead of generating an overly broad ignore file inline.
- The installer now persists runtime-root paths and the QMD session collection coherently into runtime config, and can skip the verification phase entirely via `verification.enabled`.
- `scripts/purge.sh` now supports `--yes` and `--runtime-dir`, and resolves the uv package name dynamically from `pyproject.toml`.
- Explicit topic-shift requests now add a per-turn session-guidance hint so the model proposes `switch_session` / `new_session` more reliably before mixing unrelated topics.
- Session-shift guidance and interruption/session notices are now config-driven from the UI prompt YAML instead of being hardcoded in Python.

### Fixed

- MCP roots handshake now follows the official SDK callback contract, and MCP catalog discovery now tolerates optional protocol surfaces that a real server may not implement, which restores live compatibility with the official `filesystem` server.
- Tool catalog drift now covers the new `activate_skill` runtime tool, with test coverage for prompt injection and tool-loop continuation after skill activation.
- Runtime coherence checks now warn when `config.persist_path`, `memory.internal_file`, and `sessions.directory` do not share the same parent, which would make runtime git tracking ambiguous.
- Boolean config persistence bugs in `scripts/install.sh` that broke real installs.
- Missing local git identity in runtime-store repos, which could block the initial commit and later auto-commits on machines without global git identity.
- Non-interactive purge hanging on stdin instead of aborting safely.
- `new_session` now behaves like `switch_session` for carried user requests: after approval it can open the clean session and continue answering instead of stopping after the tool result.
- Stale config expectation in the test suite (`max_tokens`) is now aligned with the real default.
- Hardcoded regex topic-shift detection was removed in favor of prompt-driven per-turn guidance injected from config.
- Replayed carried messages after an approved session split can no longer loop on `new_session`; both session-split tools are now suppressed on the first carried turn inside the fresh session.

## [0.1.1] - 2026-03-26

### Added

- High-robustness config normalization for legacy tool paths (`tools.exa_search`, `tools.python_exec`, `tools.shell_exec`, `tools.qmd_search`) toward canonical capability families.
- Live capability switches for `exa`, `python`, `shell`, and `qmd`, shared by install, slash commands, LLM tools, and runtime filtering.
- Dedicated tool capability admin tools and slash commands:
  `tool_capability_list`
  `tool_capability_set`
  `/tools capabilities`
  `/tools enable|disable <exa|python|shell|qmd>`
- Doctor recovery workflows with backup-first resets:
  `k-ai doctor --reset config|memory|sessions|all`
  `/doctor reset ...`
- Isolated `uv` bootstrap fallback for installation when `uv` is absent or declined.
- Runtime transparency panel showing context usage, compaction threshold, active limits, config persistence path, and token source.
- Live config management from chat for nested config keys, provider/model changes, config inspection, config listing, and config persistence.
- Session digest generation with summary + themes, including manual refresh for current or past sessions.
- Explicit session window tools:
  `load_session(last_n=...)`
  `session_extract(offset, limit)`
- Session ordering support for `recent` and `oldest`.
- `scripts/purge.sh` for full uninstall/purge of runtime artifacts.
- Make targets for install, purge, test, check, and push.
- Split built-in config fragments under `src/k_ai/defaults/defaults.d/`.
- Partial config export/show for individual built-in sections.
- Google OAuth token loading with persisted token-file refresh support.

### Changed

- Boot flow now assumes the recent session table is already visible and no longer re-asks for it.
- Boot-time tool exposure is restricted to `load_session`.
- QMD session retrieval is restricted to the `k-ai` collection.
- Tool proposal/result rendering is now specialized per tool type.
- Runtime/token stats now fall back to estimated counts when provider usage is unavailable.
- Slash commands are aligned with the same runtime/config tool layer used by the LLM.
- Install script now prefers `make check` during verification and keeps setup idempotent.
- Programmatic `send()` and `send_with_tools()` now follow the same session lifecycle guarantees as the interactive CLI.
- CLI theming/spinner config is now actually consumed by the runtime UI.
- Tool availability is now modeled by capability family, not by per-tool duplicated `enabled` flags.
- Doctor now audits config coherence and tool/catalog alignment instead of only reporting shallow environment state.
- Release automation now includes `make push-docs`, `make publish-docs`, and `make release`.
- Package publication now uses a login-shell-aware `make publish` flow with `UV_PUBLISH_TOKEN` preflight, clean `dist/`, and duplicate-safe upload checks.
- PyPI distribution identity is now `kpihx-ai`, while the import module stays `k_ai` and the CLI stays `k-ai`.

### Fixed

- Silent drift between legacy user config keys and the canonical runtime tool capability config.
- Potential availability mismatches between tool families exposed to the LLM and tool execution at runtime.
- Empty/silent interactive turns are no longer accepted as successful turns; they retry briefly, then rollback cleanly instead of forcing the user to retype the same request.
- Duplicate tool calls within the same interactive turn now reuse the earlier result instead of repeatedly asking for the same validation.
- Session-switch carry-over no longer loops on repeated `switch_session` proposals when the transported user message explicitly mentioned switching.
- Internal memory now has explicit absolute precedence over internal prompts, which themselves take precedence over external memory when instructions conflict.
- Session lookup and QMD remapping for short IDs.
- Rollback after failed tool-follow-up turns to avoid mismatched tool call/result history.
- Duplicate validation UI around tool execution.
- Interrupt handling for prompt input, generation, and tool execution.
- Incorrect interpretation of “oldest sessions” as “most recent sessions”.
- Session token totals not being synchronized when providers omitted usage counters.
- Declared-but-unused config knobs around prompts/UI/tool approvals.
- OAuth provider declaration drift where Google was present in config but not implemented in code.

## [0.1.0] - 2026-03-25

### Added

- Initial project scaffolding with `uv`.
- Core architecture around `ConfigManager`, `LiteLLMDriver`, `ChatSession`, `CommandHandler`, and Rich UI rendering.
- Basic documentation and automation files.
