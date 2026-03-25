# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-03-25

### Added
- Initial project scaffolding with `uv`.
- Core application structure with a modular design:
  - `config.py`: `ConfigManager` for hierarchical configuration.
  - `llm_core.py`: `LiteLLMDriver` for provider-agnostic LLM interaction.
  - `models.py`: Pydantic models for data consistency.
  - `ui.py`: `StreamingRenderer` for a rich, interactive UI.
  - `session.py`: `ChatSession` to encapsulate chat logic.
  - `commands.py`: `CommandHandler` for `/slash` commands.
  - `main.py`: CLI entry point using Typer.
- `README.md`, `CHANGELOG.md`, `TODO.md`, and `Makefile` created.
- Git repository initialized and remotes set up for GitHub and GitLab.
