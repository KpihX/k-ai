# TODO

## Done in this pass

- [x] Normalize legacy tool config paths into canonical capability families.
- [x] Add live capability toggles shared by install, slash commands, and LLM tools.
- [x] Harden doctor with coherence audits, tool/catalog alignment, and backup-first resets.
- [x] Add release automation targets for package publishing and docs pushing/publishing.
- [x] Make `make publish` robust with login-shell token resolution, clean builds, and duplicate-safe uploads.
- [x] Publish the package on PyPI under a unique distribution name while keeping `k_ai` / `k-ai` locally.
- [x] Unify tool proposal/result UI.
- [x] Restrict QMD session retrieval to the `k-ai` collection.
- [x] Add session digest generation with themes.
- [x] Add explicit session windows: load last `N`, extract `offset/limit`.
- [x] Unify config management through chat, tools, and slash commands.
- [x] Add runtime transparency panel.
- [x] Add interrupt handling for prompt, generation, and tool execution.
- [x] Add installation finalization and purge workflow.
- [x] Harden `install.sh` and `purge.sh` with real integration coverage, custom-runtime support, safe non-interactive purge behavior, and runtime-git identity fallback.
- [x] Make explicit topic-change requests reliably trigger session-tool proposals, and let approved `new_session` requests continue the carried conversation in the fresh session.
- [x] Refresh README, CHANGELOG, TODO, scripts, and Makefile.
- [x] Split built-in config into sectioned YAML fragments with cached loading.
- [x] Expose config sections from CLI and slash commands.
- [x] Remove or wire every previously orphaned config option.
- [x] Implement Google OAuth token loading/refresh for declared OAuth config.
- [x] Align programmatic API rollback/finalization with interactive session semantics.
- [x] Add a native `SKILL.md` runtime with project/global discovery, activation, inspection commands, prompt injection, and end-to-end tests.
- [x] Add the MCP foundation with dynamic stdio-backed tool import, roots support, `/mcp` inspection, and the official filesystem server as the first configured MCP.
- [x] Capture the staged roadmap in `VISION.md`.
- [x] Add a native interaction runtime for one-shot ask, explicit `cwd`, multiline mixed user input, and persistent local shell/Python runners.

## Next candidates

- [ ] Refactor long-response streaming away from a single full-height Rich `Live` panel to avoid bottom ellipsis/clipping, blinking redraws, and delayed visibility when content exceeds terminal height.
- [x] Implement the hooks layer after skills stabilization: structured lifecycle hooks with timeout policies, discovery roots, auditability, and Claude-style event naming.
- [ ] Deepen MCP runtime sophistication beyond the current foundation: persistent sessions where useful, richer roots refresh semantics, broader transport tuning, and higher-level server templates.
- [ ] Add a dedicated visual hint in the streaming UI for `Ctrl+C` / `Esc` interruption.
- [ ] Improve per-provider token estimation quality beyond character-based heuristics.
- [ ] Add snapshot/export command for runtime transparency as JSON/YAML.
- [ ] Add integration tests for interruption behavior in an interactive PTY.
- [ ] Reduce shell-side visual noise further during focused PTY interaction without weakening login-shell environment fidelity.
- [ ] Add a doctor sub-check that verifies Git remotes and publish prerequisites before `make release`.
