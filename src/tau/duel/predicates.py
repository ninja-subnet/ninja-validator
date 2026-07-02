"""Pure functions that score a duel from the challenger's perspective."""

from __future__ import annotations


def challenger_wins(wins: int, losses: int, margin: int) -> bool:
    """True when the challenger has more wins than losses, beyond `margin`."""
    return wins > losses + margin


def challenger_wins_by_mean_score(
    *, score_mean_delta: float, score_mean_rounds: int, margin: float
) -> bool:
    """True when the challenger's raw mean score clears the configured margin."""
    return score_mean_rounds > 0 and score_mean_delta >= margin


def challenger_is_unbeatable(
    wins: int, losses: int, remaining_rounds: int, margin: int
) -> bool:
    """True if the challenger wins even if every unresolved round goes to the king."""
    return challenger_wins(wins, losses + max(0, remaining_rounds), margin)


def challenger_cannot_catch(
    wins: int, losses: int, remaining_rounds: int, margin: int
) -> bool:
    """True if the challenger loses even if every unresolved round goes to the challenger."""
    return not challenger_wins(wins + max(0, remaining_rounds), losses, margin)
