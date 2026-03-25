# k-ai Project TODO

## Phase 1: Core Backend Consolidation

- [x] **ConfigManager**:
    - [x] Create `config.py`.
    - [x] Implement `ConfigManager` class.
    - [x] Logic to load default `config.yaml` from package data.
    - [x] Logic to load and merge an external `config.yaml`.
- [x] **LLM Core**:
    - [x] Create `llm_core.py`.
    - [x] Move `LiteLLMDriver` into it.
    - [x] Refactor `get_provider` to use `ConfigManager`.
    - [x] Ensure API keys and provider settings are read from config.

## Phase 2: UI & Session Logic

- [x] **StreamingRenderer**:
    - [x] Create `ui.py`.
    - [x] Move/Refine `StreamingRenderer` class.
    - [x] Add config options for UI themes and elements.
- [x] **ChatSession**:
    - [x] Create `session.py`.
    - [x] Implement `ChatSession` class to encapsulate all session state.
    - [x] `ChatSession` should be initialized with a `ConfigManager` instance.
- [x] **CommandHandler**:
    - [x] Create `commands.py`.
    - [x] Implement `CommandHandler` for `/slash` commands.
    - [x] Implement `/config get` command.

## Phase 3: Packaging & Entry Points

- [x] **CLI Entry Point**:
    - [x] Create `main.py` using Typer.
    - [x] `main.py` should parse CLI args (`--config-path`).
    - [x] It should instantiate `ConfigManager` and `ChatSession`.
- [x] **Library Entry Point**:
    - [x] Create `src/k_ai/__init__.py`.
    - [x] Expose `ChatSession`, `ConfigManager`, `get_provider` for library usage.

## Phase 4: Git & Automation

- [x] **Makefile**:
    - [x] Create `Makefile`.
    - [x] Add `push` target for all remotes.
- [x] **Git Repository**:
    - [x] Initialize Git repository.
    - [x] Create repositories on GitHub and GitLab.
    - [x] Add remotes.
