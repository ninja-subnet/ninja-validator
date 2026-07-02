"""The weight-setter worker: distribute emission across the rolling king window."""

from __future__ import annotations

from .config import WeightSetterConfig
from .loop import StepResult, run, step
from .main import main

__all__ = ["WeightSetterConfig", "StepResult", "main", "run", "step"]
