from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()

def print_epoch_header(epoch: int, epochs: int, lr: float) -> None:
    console.rule(f"[bold]Epoch {epoch}/{epochs}[/bold]  lr={lr:.2e}")

def print_metrics_table(title: str, metrics: Dict[str, Any]) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.6f}")
        else:
            table.add_row(k, str(v))
    console.print(table)
