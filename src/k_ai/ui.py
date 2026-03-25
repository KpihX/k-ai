# src/k_ai/ui.py
"""
User Interface logic for k-ai.
"""
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from typing import Optional

from .models import CompletionChunk

class StreamingRenderer:
    def __init__(self, console: Console, model_name: str):
        self.console = console
        self.model_name = model_name
        self.full_content = ""
        self.full_thought = ""
        self.thinking_committed = False
        self.live: Optional[Live] = None

    def __enter__(self):
        self.live = Live(Spinner("dots", text="Thinking..."), console=self.console, refresh_per_second=12, transient=True)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()
        # Print final static output
        if self.full_thought:
            self.console.print(Panel(Markdown(self.full_thought), title="Thinking", border_style="yellow"))
        if self.full_content:
            self.console.print(Panel(Markdown(self.full_content), title=f"🤖 {self.model_name}", border_style="green"))

    def update(self, chunk: CompletionChunk):
        if not self.live:
            return

        if chunk.delta_thought:
            self.full_thought += chunk.delta_thought
        
        if chunk.delta_content:
            self.full_content += chunk.delta_content
        
        # Commit thinking panel as soon as first content token arrives
        if chunk.delta_content and self.full_thought and not self.thinking_committed:
            self.live.stop() # Stop the spinner live display
            self.console.print(Panel(Markdown(self.full_thought), title="Thinking", border_style="yellow"))
            self.thinking_committed = True
            # Start a new live display for the content
            self.live = Live(Panel(Markdown(self.full_content), title=f"🤖 {self.model_name}", border_style="green"), console=self.console, refresh_per_second=12)
            self.live.start()

        # Update live display
        if self.thinking_committed:
            self.live.update(Panel(Markdown(self.full_content), title=f"🤖 {self.model_name}", border_style="green"))
        elif self.full_thought:
            self.live.update(Panel(Markdown(self.full_thought), title="Thinking", border_style="yellow", subtitle="[dim]Assistant is thinking...[/dim]"))
