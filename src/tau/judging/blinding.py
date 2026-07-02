from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

from .parsing import RawVerdict

Winner: TypeAlias = Literal["king", "challenger", "tie"]


@dataclass(frozen=True, slots=True)
class Blinded:
    candidate_a_patch: str
    candidate_b_patch: str
    a_is_king: bool  # True: candidate_a == king; False: candidate_a == challenger


class Unblinded(NamedTuple):
    winner: Winner
    king_score: float
    challenger_score: float


def blind(seed: str, king_patch: str, challenger_patch: str) -> Blinded:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    a_is_king = digest[0] % 2 == 0
    if a_is_king:
        return Blinded(
            candidate_a_patch=king_patch,
            candidate_b_patch=challenger_patch,
            a_is_king=True,
        )
    return Blinded(
        candidate_a_patch=challenger_patch,
        candidate_b_patch=king_patch,
        a_is_king=False,
    )


def unblind(blinded: Blinded, verdict: RawVerdict) -> Unblinded:
    if blinded.a_is_king:
        king_score, challenger_score = verdict.score_a, verdict.score_b
        winner = _candidate_to_role(verdict.winner, a="king", b="challenger")
    else:
        king_score, challenger_score = verdict.score_b, verdict.score_a
        winner = _candidate_to_role(verdict.winner, a="challenger", b="king")
    return Unblinded(winner=winner, king_score=king_score, challenger_score=challenger_score)


def _candidate_to_role(candidate_winner: str, *, a: Winner, b: Winner) -> Winner:
    if candidate_winner == "candidate_a":
        return a
    if candidate_winner == "candidate_b":
        return b
    return "tie"
