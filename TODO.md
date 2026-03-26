# TODO

## Done in this pass

- [x] Unify tool proposal/result UI.
- [x] Restrict QMD session retrieval to the `k-ai` collection.
- [x] Add session digest generation with themes.
- [x] Add explicit session windows: load last `N`, extract `offset/limit`.
- [x] Unify config management through chat, tools, and slash commands.
- [x] Add runtime transparency panel.
- [x] Add interrupt handling for prompt, generation, and tool execution.
- [x] Add installation finalization and purge workflow.
- [x] Refresh README, CHANGELOG, TODO, scripts, and Makefile.

## Next candidates

- [ ] Add a dedicated visual hint in the streaming UI for `Ctrl+C` / `Esc` interruption.
- [ ] Improve per-provider token estimation quality beyond character-based heuristics.
- [ ] Add richer session list columns for themes and persisted token totals.
- [ ] Add snapshot/export command for runtime transparency as JSON/YAML.
- [ ] Add integration tests for interruption behavior in an interactive PTY.
