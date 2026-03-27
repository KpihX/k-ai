# k-ai Vision

This document captures the architecture direction for the next major capability
layers in `k-ai`. It exists to keep the implementation coherent while the work
lands incrementally.

## Principles

- Build one capability layer at a time, and fully stabilize it before opening
  the next one.
- Prefer portable standards and existing ecosystems over bespoke formats.
- **Hardcode nothing that belongs to runtime behaviour, UX wording, routing heuristics, paths, or policy. Put it in config unless there is a strong implementation-only reason not to. The target state is to evolve behaviour by editing config, not by reopening Python files.**
- Keep the core microlithic: cohesive packages, narrow interfaces, explicit
  ownership, and minimal cross-package coupling.
- Separate lifecycle/orchestration code from parsing, indexing, execution, and
  UI rendering.
- Make runtime behaviour inspectable. Every non-trivial subsystem should have
  observability surfaces, tests, and failure fallbacks.

## Planned Layers

1. Skills
2. Hooks
3. Filesystem editing primitives (`write`, `edit`, `diff`, preview)
4. Full MCP client support (`tools`, `resources`, `prompts`, `roots`, transport)
5. High-fidelity interaction runtime (`ask`, mixed input parsing, persistent local runners, focus)

## Skills Direction

`k-ai` will implement a native skills runtime aligned with the emerging
`SKILL.md` ecosystem.

Targets:

- Default global skills directory: `~/.agents/skills`
- Optional project-local overlays: `.k-ai/skills`, `.agents/skills`
- Recursive discovery of `SKILL.md`
- Progressive disclosure:
  - metadata indexed at startup/refresh
  - full skill body loaded only on activation
  - referenced resources loaded on demand later
- Deterministic activation:
  - explicit mention wins
  - local automatic activation stays minimal and deterministic
  - broader semantic routing is delegated to the model through the visible skill catalog
  - bounded number of activated skills per turn
- Strong precedence:
  - internal memory
  - internal prompts/instructions
  - activated skills
  - external memory

Implementation shape:

- `k_ai.skills.models`
- `k_ai.skills.parser`
- `k_ai.skills.registry`
- `k_ai.skills.selector`
- `k_ai.skills.manager`

The session layer should only orchestrate skill resolution and prompt
injection. Parsing, caching, and activation logic must remain isolated.

## Hooks Direction

Hooks come after skills are stable.
Status: implemented.

Targets:

- Structured pre/post lifecycle hooks
- Strict JSON contract
- Timeout + fail mode policies
- Auditable execution
- No uncontrolled prompt mutation

Expected events:

- `PreModelCall`
- `PostModelCall`
- `PreToolUse`
- `PermissionRequest`
- `PostToolUse`
- `PostToolUseFailure`
- `PreSessionSwitch`
- `PostSessionSwitch`
- `Exit`

## Filesystem Editing Direction

Do not reinvent advanced file mutation semantics if an existing standard server
already provides them better.

Direction:

- Add a robust abstraction layer in `k-ai`
- Prefer reusing MCP filesystem semantics for editing behaviours
- Keep approval, auditing, and root-boundary enforcement in `k-ai`

## MCP Direction

MCP support is now started with a foundation based on the official Python SDK
and the official filesystem server as the first configured MCP.
Status: tools/resources/prompts/roots/admin surfaces implemented; deeper
transport/runtime sophistication can still grow incrementally.

Targets:

- Transport abstraction (`stdio` first, HTTP later)
- Capability negotiation
- Dynamic registry for MCP tools
- Support for `resources`, `prompts`, and `roots`
- Root boundary enforcement and safe caching

## Quality Bar

Each layer is considered complete only when it has:

- a coherent package structure
- explicit runtime configuration
- a command or inspection surface when applicable
- exhaustive unit/integration tests
- real execution validation in a sandbox
- updated docs/changelog/TODO where relevant

## Interaction Runtime Direction

The chat input surface must evolve without turning `session.py` into an
implicit parser or a terminal spaghetti stack.

Targets:

- Root one-shot mode via `k-ai "..."` plus explicit `k-ai ask ...`
- Explicit session working directory (`-C/--cwd`) shared across:
  - chat
  - one-shot ask
  - local shell runner
  - local Python runner
  - runtime tools such as `shell_exec` / `python_exec`
- Deterministic multiline document parsing with explicit block prefixes
- Persistent user-side runners per session, not per block
- PTY-backed focus mode for interactive shell/Python prompts
- No persistence of focused keystrokes or ephemeral quick questions

Implementation shape:

- `k_ai.interaction.cwd`
- `k_ai.interaction.models`
- `k_ai.interaction.parser`
- `k_ai.interaction.runners`

The session layer should orchestrate these subsystems, not absorb their
terminal-control, parsing, or execution details.
