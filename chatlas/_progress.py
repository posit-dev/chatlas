"""Progress tracking utilities for chatlas operations."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

__all__ = ("ProgressTracker",)


class ProgressTracker:
    """Context manager for tracking progress with customizable task messages."""

    def __init__(self, task_description: str, total: int):
        """
        Initialize progress tracker.

        Parameters
        ----------
        task_description
            Description of the task being tracked
        total
            Total number of units to track
        """
        self.task_description = task_description
        self.total = total
        self.progress = None
        self.task_id = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            self.task_description,
            total=self.total,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def advance(self, amount: int = 1):
        """Advance the progress by the specified amount."""
        if self.progress and self.task_id is not None:
            self.progress.advance(self.task_id, amount)

    def update_description(self, description: str):
        """Update the task description."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description)
