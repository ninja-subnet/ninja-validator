"""The duel-resolver worker: drives the challenge/king lifecycle off the DB seam."""

from __future__ import annotations

from .config import DuelResolverConfig
from .main import main
from .pipeline import run_duel_resolver

__all__ = ["DuelResolverConfig", "main", "run_duel_resolver"]
