# k-ai Project TODO

## Phase 1: Core Backend Consolidation

- [ ] **ConfigManager**:
    - [ ] Create `config.py`.
    - [ ] Implement `ConfigManager` class.
    - [ ] Logic to load default `config.yaml` from package data.
    - [ ] Logic to load and merge an external `config.yaml`.
- [ ] **LLM Core**:
    - [ ] Create `llm_core.py`.
    - [ ] Move `LiteLLMDriver` into it.
    - [ ] Refactor `get_provider` to use `ConfigManager`.
    - [ ] Ensure API keys and provider settings are read from config.

## Phase 2: UI & Session Logic

- [ ] **StreamingRenderer**:
    - [ ] Create `ui.py`.
    - [ ] Move/Refine `StreamingRenderer` class.
    - [ ] Add config options for UI themes and elements.
- [ ] **ChatSession**:
    - [ ] Create `session.py`.
    - [ ] Implement `ChatSession` class to encapsulate all session state.
    - [ ] `ChatSession` should be initialized with a `ConfigManager` instance.
- [ ] **CommandHandler**:
    - [ ] Create `commands.py`.
    - [ ] Implement `CommandHandler` for `/slash` commands.
    - [ ] Implement `/config get` command.

## Phase 3: Packaging & Entry Points

- [ ] **CLI Entry Point**:
    - [ ] Create `main.py` using Typer.
    - [ ] `main.py` should parse CLI args (`--config-path`).
    - [ ] It should instantiate `ConfigManager` and `ChatSession`.
- [ ] **Library Entry Point**:
    - [ ] Create `src/k_ai/__init__.py`.
    - [ ] Expose `ChatSession`, `ConfigManager`, `get_provider` for library usage.

## Phase 4: Git & Automation

- [ ] **Makefile**:
    - [ ] Create `Makefile`.
    - [x] Add `push` target for all remotes.
- [ ] **Git Repository**:
    - [x] Initialize Git repository.
    - [x] Create repositories on GitHub and GitLab.
    - [x] Add remotes.
