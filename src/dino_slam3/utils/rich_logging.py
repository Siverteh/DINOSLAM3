from __future__ import annotations

from typing import Dict, Any

from rich.console import Console
from rich.table import Table


_console = Console()


def print_epoch_header(epoch: int, epochs: int, lr: float) -> None:
    _console.rule(f"[bold]Epoch {epoch}/{epochs}  lr={lr:.2e}[/bold]")


def print_metrics_table(title: str, metrics: Dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    # stable ordering
    preferred = [
        "loss_total",
        "loss_desc",
        "loss_det",
        "loss_sparse",
        "loss_offset",
        "loss_rel",
        "valid_ratio",
    ]
    keys = [k for k in preferred if k in metrics] + [k for k in metrics.keys() if k not in preferred]

    for k in keys:
        v = metrics[k]
        if isinstance(v, float):
            table.add_row(k, f"{v:.6f}")
        else:
            table.add_row(k, str(v))
    _console.print(table)


def print_save_notice(path: str, reason: str) -> None:
    _console.print(f"[green]Saved checkpoint ({reason}):[/green] {path}")


def print_match_table(title: str, diag: Dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Match/Geom Metric", justify="left")
    table.add_column("Value", justify="right")

    preferred = [
        "kpts1",
        "kpts2",
        "matches",
        "valid_match_ratio",
        "inlier_rate@3px",
        "mean_reproj_err",
        "mean_reproj_err_inliers",
    ]
    keys = [k for k in preferred if k in diag] + [k for k in diag.keys() if k not in preferred]

    for k in keys:
        v = diag.get(k, 0.0)
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    _console.print(table)
