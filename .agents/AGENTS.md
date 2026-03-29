# AGENTS.md — k-ai

> Project context for all AI agents working in this repository.
> Loaded automatically by all KπX agents when present at project root.

## KπX Mantras

**Exploration:** Problem First → Why before How → Visualization
**Architecture:** 0 Trust · 100% Control | 0 Magic · 100% Transparency | 0 Hardcoding · 100% Flexibility

## Project Overview

| Field | Value |
|-------|-------|
| Purpose | Terminal-first LLM chat system — persistent sessions, live config mutation, internal tools, programmatic API |
| Stack | Python (uv), LanceDB (vector store), LiteLLM (provider routing) |
| Status | 🟡 In progress |
| Docs site | `docs/` (Docsify) |

## Architecture Rules

- **Non-monolithic** — one package per concern; `src/k_ai/` layout
- **Single session model** — chat loop, slash commands, and programmatic API all act on the same session/config/runtime
- **Config-driven** — all values in `config.yaml`; no hardcoding
- **Debug flag** — `debug: false` in config; never `print()` in prod
- **lancedb/** is local vector store data — never commit (in `.gitignore`)

## Evolution Rules

- New feature → update `TODO.md` first, propose before acting
- Significant change → update `AGENTS.md` + `README.md` + relevant `docs/`
- Breaking change → bump version in `pyproject.toml` + entry in `CHANGELOG.md`
- Destructive / architectural → **stop and confirm with KπX first**
- **Makefile is the standard task runner** — `make push`, `make build`, `make release`

## Current Migration Notes

- External read-only agent memory is moving from `~/.agents/KERNEL.md` to `~/.agents/AGENTS.md`.
- Future system-prompt runtime context should always include date, time, timezone, and core machine facts.
- Resume UX should support both direct session-id resume and guided resume without forcing the user to type an id first.
