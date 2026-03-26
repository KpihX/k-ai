# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- Runtime transparency panel showing context usage, compaction threshold, active limits, config persistence path, and token source.
- Live config management from chat for nested config keys, provider/model changes, config inspection, config listing, and config persistence.
- Session digest generation with summary + themes, including manual refresh for current or past sessions.
- Explicit session window tools:
  `load_session(last_n=...)`
  `session_extract(offset, limit)`
- Session ordering support for `recent` and `oldest`.
- `scripts/purge.sh` for full uninstall/purge of runtime artifacts.
- Make targets for install, purge, test, check, and push.

### Changed

- Boot flow now assumes the recent session table is already visible and no longer re-asks for it.
- Boot-time tool exposure is restricted to `load_session`.
- QMD session retrieval is restricted to the `k-ai` collection.
- Tool proposal/result rendering is now specialized per tool type.
- Runtime/token stats now fall back to estimated counts when provider usage is unavailable.
- Slash commands are aligned with the same runtime/config tool layer used by the LLM.
- Install script now installs the editable CLI entrypoint and keeps setup idempotent.

### Fixed

- Session lookup and QMD remapping for short IDs.
- Rollback after failed tool-follow-up turns to avoid mismatched tool call/result history.
- Duplicate validation UI around tool execution.
- Interrupt handling for prompt input, generation, and tool execution.
- Incorrect interpretation of “oldest sessions” as “most recent sessions”.
- Session token totals not being synchronized when providers omitted usage counters.

## [0.1.0] - 2026-03-25

### Added

- Initial project scaffolding with `uv`.
- Core architecture around `ConfigManager`, `LiteLLMDriver`, `ChatSession`, `CommandHandler`, and Rich UI rendering.
- Basic documentation and automation files.
