# src/k_ai/commands.py
"""
Handles /slash commands for the interactive chat.
"""
import os
from typing import TYPE_CHECKING
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from .session import ChatSession

class CommandHandler:
    def __init__(self, session: "ChatSession"):
        self.session = session
        self.console = session.console

    async def handle(self, text: str) -> bool:
        """
        Handles a slash command. Returns False if the session should exit.
        """
        command, *args = text.lstrip("/").split()
        
        if command == "help":
            self.show_help()
        elif command == "exit":
            return False
        elif command == "config" and args and args[0] == "get":
            self.get_config()
        else:
            self.console.print(f"[bold red]Unknown command:[/bold red] {command}")
        
        return True

    def show_help(self):
        """Displays the help message."""
        table = Table(title="Available Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        table.add_row("/help", "Show this help message.")
        table.add_row("/exit", "Exit the chat session.")
        table.add_row("/config get", "Copy the default config file to the current directory.")
        self.console.print(table)

    def get_config(self):
        """Copies the default config file to the current directory."""
        try:
            default_path = self.session.cm.default_config_path
            target_path = "config.yaml"
            
            if os.path.exists(target_path):
                self.console.print(f"[yellow]Warning:[/] '{target_path}' already exists. Overwrite? (y/n)")
                # In a real CLI, we would handle user input here.
                # For now, we just print the warning.
                if input().lower() != 'y':
                    self.console.print("Aborted.")
                    return

            with open(default_path, 'r') as f_in, open(target_path, 'w') as f_out:
                f_out.write(f_in.read())
            
            self.console.print(f"[green]Success:[/] Default configuration copied to [bold]'{target_path}'[/bold].")

        except Exception as e:
            self.console.print(f"[bold red]Error:[/] Could not copy config file: {e}")
