"""Console script for abm_initialization_tools."""
import abm_initialization_tools

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for abm_initialization_tools."""
    console.print("Replace this message by putting your code into "
               "abm_initialization_tools.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
