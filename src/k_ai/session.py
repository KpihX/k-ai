# src/k_ai/session.py
"""
Manages an interactive chat session.
"""
import asyncio
from typing import List
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from .config import ConfigManager
from .llm_core import get_provider, LLMProvider
from .models import Message, MessageRole
from .ui import StreamingRenderer
from .commands import CommandHandler

class ChatSession:
    def __init__(self, config_manager: ConfigManager, provider: str = None, model: str = None):
        self.cm = config_manager
        self.console = Console()
        self.llm: LLMProvider = get_provider(self.cm, provider=provider, model=model)
        self.history: List[Message] = []
        self.command_handler = CommandHandler(self)

    async def start(self):
        """Starts the interactive chat session."""
        self._print_welcome_panel()
        session = PromptSession(history=InMemoryHistory())

        while True:
            try:
                user_input = await session.prompt_async("You: ")
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if await self.command_handler.handle(user_input):
                        continue # Command handled, loop to next prompt
                    else:
                        break # Exit command was issued
                
                self.history.append(Message(role=MessageRole.USER, content=user_input))
                
                with StreamingRenderer(self.console, self.llm.model_name) as renderer:
                    async for chunk in self.llm.chat_stream(self.history):
                        renderer.update(chunk)
                
                self.history.append(Message(role=MessageRole.ASSISTANT, content=renderer.full_content))

            except (EOFError, KeyboardInterrupt):
                break
        
        self.console.print("\n[bold green]Goodbye![/bold green]")

    def _print_welcome_panel(self):
        from rich.panel import Panel
        from rich.text import Text

        title = Text("k-ai Unified Chat", style="bold cyan", justify="center")
        grid = Text.from_markup(
            f"[dim]Provider:[/dim] [bold white]{self.llm.provider_name}[/bold white]\n"
            f"[dim]Model:   [/dim] [bold white]{self.llm.model_name}[/bold white]"
        )
        self.console.print(Panel(grid, title=title, border_style="cyan", padding=(1, 2)))
        self.console.print("Type /help for a list of commands.", style="dim")
