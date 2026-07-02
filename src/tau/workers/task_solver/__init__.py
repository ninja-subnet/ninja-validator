"""Task-solver worker: qualify CANDIDATE tasks (king viability) and produce
challenger solutions for active challenges, each in a locked-down sandbox."""

from __future__ import annotations

from .config import SolverConfig
from .loop import run
from .main import main

__all__ = ["SolverConfig", "main", "run"]
