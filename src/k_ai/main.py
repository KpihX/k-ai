# src/k_ai/main.py
"""
Main CLI entry point for k-ai.
"""
import asyncio
import typer
from typing import Optional

from .config import ConfigManager
from .session import ChatSession

app = typer.Typer(
    name="k-ai",
    help="k-ai: The Unified LLM CLI.",
    no_args_is_help=True,
)

@app.command()
def chat(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the default provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the default model."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to a custom config.yaml file."),
):
    """
    Starts an interactive chat session.
    """
    try:
        config_manager = ConfigManager(override_path=config_path)
        chat_session = ChatSession(config_manager, provider=provider, model=model)
        asyncio.run(chat_session.start())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
