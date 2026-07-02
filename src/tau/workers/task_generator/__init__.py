"""Task-generator worker: mine commits, turn them into tasks, store the candidates."""

from __future__ import annotations

from .config import GeneratorConfig
from .dummy import DummyTaskClient
from .main import main
from .pipeline import run
from .workqueue import FetchedCommit, FetchRequest, WorkQueue

__all__ = [
    "DummyTaskClient",
    "FetchRequest",
    "FetchedCommit",
    "GeneratorConfig",
    "WorkQueue",
    "main",
    "run",
]
