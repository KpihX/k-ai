# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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

### Fixed

- Silent drift between legacy user config keys and the canonical runtime tool capability config.
- Potential availability mismatches between tool families exposed to the LLM and tool execution at runtime.
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
