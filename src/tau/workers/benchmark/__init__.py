"""Benchmark worker: watch the ``kings`` table and benchmark each new king's
submitted agent against SWE-bench Pro, saving results to a dedicated folder."""

from __future__ import annotations

from .config import BenchmarkConfig
from .loop import run
from .main import main

__all__ = ["BenchmarkConfig", "main", "run"]
