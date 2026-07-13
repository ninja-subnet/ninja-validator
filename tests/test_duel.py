"""Unit tests for the pure duel logic (predicates + decide)."""

from __future__ import annotations

from tau.db.status import PoolType
from tau.duel import (
    ActiveChallenge,
    AdvancePool,
    ChallengeSnapshot,
    CloseChallenge,
    CloseReason,
    DEFAULT_MEAN_SCORE_MARGIN,
    DuelScoringMethod,
    Nothing,
    OpenChallenge,
    Promote,
    Tally,
    WaitReason,
    challenger_cannot_catch,
    challenger_is_unbeatable,
    challenger_wins,
    decide,
)

P1 = PoolType.POOL_ONE
P2 = PoolType.POOL_TWO


def _active(
    pool: PoolType = P1,
    *,
    wins: int = 0,
    losses: int = 0,
    ties: int = 0,
    king_score_mean: float = 0.0,
    challenger_score_mean: float = 0.0,
    score_mean_delta: float = 0.0,
    score_mean_rounds: int = 0,
    king_token_boost: float = 0.0,
    challenger_token_boost: float = 0.0,
    target: int = 50,
    registered: bool = True,
) -> ActiveChallenge:
    return ActiveChallenge(
        challenger_submission_id="chal",
        king_submission_id="king",
        pool=pool,
        pool_target=target,
        tally=Tally(
            wins,
            losses,
            ties,
            king_score_mean=king_score_mean,
            challenger_score_mean=challenger_score_mean,
            score_mean_delta=score_mean_delta,
            score_mean_rounds=score_mean_rounds,
            king_token_boost=king_token_boost,
            challenger_token_boost=challenger_token_boost,
        ),
        challenger_registered=registered,
    )


def _snapshot(
    active: ActiveChallenge | None = None,
    *,
    king: str | None = "king",
    next_challenger: str | None = None,
    task_pools_ready: bool = True,
) -> ChallengeSnapshot:
    return ChallengeSnapshot(
        reigning_king_submission_id=king,
        active_challenge=active,
        next_challenger_submission_id=next_challenger,
        task_pools_ready=task_pools_ready,
    )


# -- predicates ---------------------------------------------------------------------


def test_challenger_wins_needs_strictly_more_than_losses_plus_margin() -> None:
    assert challenger_wins(2, 1, 0) is True
    assert challenger_wins(2, 2, 0) is False  # a tie on the tally is not a win
    assert challenger_wins(2, 1, 1) is False  # margin not cleared
    assert challenger_wins(3, 1, 1) is True


def test_unbeatable_assumes_all_remaining_go_to_king() -> None:
    assert challenger_is_unbeatable(3, 0, 2, 0) is True  # 3 > 0 + 2
    assert challenger_is_unbeatable(2, 0, 2, 0) is False  # 2 > 2 is false


def test_cannot_catch_assumes_all_remaining_go_to_challenger() -> None:
    assert challenger_cannot_catch(0, 3, 2, 0) is True  # 0 + 2 <= 3
    assert challenger_cannot_catch(2, 1, 3, 0) is False  # 2 + 3 > 1


# -- derived helpers ----------------------------------------------------------------


def test_tally_judged_and_remaining() -> None:
    assert Tally(2, 3, 1).judged == 6
    assert Tally(2, 3, 1, score_mean_rounds=4).judged == 6
    assert _active(wins=10, losses=5, ties=5, target=50).remaining == 30
    assert (
        _active(wins=30, losses=30, target=50).remaining == 0
    )  # over-count floors at 0


# -- decide: king presence / opening ------------------------------------------------


def test_no_king_waits_even_with_a_challenger_queued() -> None:
    assert decide(_snapshot(king=None, next_challenger="chal")) == Nothing(
        WaitReason.NO_KING
    )


def test_king_no_challenge_no_challenger_waits() -> None:
    assert decide(_snapshot(king="k", next_challenger=None)) == Nothing(
        WaitReason.NO_CHALLENGER
    )


def test_king_waits_for_both_task_pools_before_opening_challenge() -> None:
    assert decide(
        _snapshot(
            king="k",
            next_challenger="chal",
            task_pools_ready=False,
        )
    ) == Nothing(WaitReason.POOLS_NOT_READY)


def test_paused_resolver_does_not_open_a_new_challenge() -> None:
    assert decide(
        _snapshot(king="k", next_challenger="chal"),
        new_challenges_paused=True,
    ) == Nothing(WaitReason.NEW_CHALLENGES_PAUSED)


def test_paused_resolver_still_finishes_the_active_duel() -> None:
    active = _active(P2, wins=2, target=2)
    assert decide(
        _snapshot(active),
        new_challenges_paused=True,
    ) == Promote(active)


def test_king_no_challenge_opens_for_next_challenger() -> None:
    assert decide(_snapshot(king="k", next_challenger="chal")) == OpenChallenge(
        "k", "chal"
    )


# -- decide: resolving an active challenge ------------------------------------------


def test_pool_one_unbeatable_advances() -> None:
    active = _active(P1, wins=26, losses=0, target=50)
    assert decide(_snapshot(active)) == AdvancePool(active)


def test_pool_two_unbeatable_promotes() -> None:
    active = _active(P2, wins=26, losses=0, target=50)
    assert decide(_snapshot(active)) == Promote(active)


def test_cannot_catch_closes_king_defended() -> None:
    active = _active(P1, wins=0, losses=26, target=50)
    assert decide(_snapshot(active)) == CloseChallenge(
        active, CloseReason.KING_DEFENDED
    )


def test_one_nil_does_not_decide_prematurely() -> None:
    assert decide(_snapshot(_active(P2, wins=1, losses=0, target=50))) == Nothing(
        WaitReason.DUEL_IN_PROGRESS
    )


# -- decide: final tally (remaining == 0) -------------------------------------------


def test_final_win_in_pool_one_advances() -> None:
    active = _active(P1, wins=2, losses=0, target=2)
    assert decide(_snapshot(active)) == AdvancePool(active)


def test_final_tie_on_tally_king_holds() -> None:
    active = _active(P1, wins=1, losses=1, target=2)
    assert decide(_snapshot(active)) == CloseChallenge(
        active, CloseReason.KING_DEFENDED
    )


def test_all_ties_king_holds() -> None:
    # Failure-ties fill the pool but never help the challenger.
    active = _active(P1, wins=0, losses=0, ties=2, target=2)
    assert decide(_snapshot(active)) == CloseChallenge(
        active, CloseReason.KING_DEFENDED
    )


# -- decide: deregistration short-circuit -------------------------------------------


def test_deregistered_challenger_closes_even_when_winning() -> None:
    # A tally that would otherwise promote, but the challenger left the chain.
    active = _active(P2, wins=50, losses=0, target=50, registered=False)
    assert decide(_snapshot(active)) == CloseChallenge(
        active, CloseReason.CHALLENGER_DEREGISTERED
    )


# -- decide: win margin -------------------------------------------------------------


def test_win_margin_can_flip_a_narrow_final_result() -> None:
    active = _active(P1, wins=2, losses=1, target=3)  # remaining == 0
    assert decide(_snapshot(active)) == AdvancePool(active)  # margin 0: 2 > 1
    assert decide(_snapshot(active), round_win_margin=1) == CloseChallenge(
        active, CloseReason.KING_DEFENDED
    )  # margin 1: 2 > 2 is false


# -- decide: mean-score mode --------------------------------------------------------


def test_mean_scoring_waits_until_all_rounds_are_judged() -> None:
    active = _active(
        P1,
        wins=1,
        losses=0,
        target=3,
        king_score_mean=0.2,
        challenger_score_mean=0.8,
        score_mean_delta=0.6,
        score_mean_rounds=1,
    )
    assert decide(_snapshot(active), scoring_method=DuelScoringMethod.MEAN) == Nothing(
        WaitReason.DUEL_IN_PROGRESS
    )


def test_mean_scoring_advances_pool_one_when_margin_clears() -> None:
    active = _active(
        P1,
        wins=0,
        losses=2,
        target=2,
        king_score_mean=0.40,
        challenger_score_mean=0.46,
        score_mean_delta=0.06,
        score_mean_rounds=2,
    )
    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
        mean_score_margin=0.05,
    ) == AdvancePool(active)


def test_mean_scoring_promotes_pool_two_when_margin_clears() -> None:
    active = _active(
        P2,
        wins=0,
        losses=1,
        ties=1,
        target=2,
        king_score_mean=0.50,
        challenger_score_mean=0.55,
        score_mean_delta=0.05,
        score_mean_rounds=2,
    )
    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
        mean_score_margin=0.05,
    ) == Promote(active)


def test_mean_scoring_king_holds_when_margin_does_not_clear() -> None:
    active = _active(
        P1,
        wins=2,
        losses=0,
        target=2,
        king_score_mean=0.50,
        challenger_score_mean=0.54,
        score_mean_delta=0.04,
        score_mean_rounds=2,
    )
    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
        mean_score_margin=0.05,
    ) == CloseChallenge(active, CloseReason.KING_DEFENDED)


def test_mean_scoring_token_boost_can_clear_the_same_single_gate() -> None:
    active = _active(
        P1,
        ties=2,
        target=2,
        king_score_mean=0.50,
        challenger_score_mean=0.58,
        score_mean_delta=0.08,
        score_mean_rounds=2,
        challenger_token_boost=0.03,
    )

    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
        mean_score_margin=0.10,
    ) == AdvancePool(active)


def test_mean_scoring_king_token_boost_can_prevent_a_pass() -> None:
    active = _active(
        P1,
        ties=2,
        target=2,
        king_score_mean=0.50,
        challenger_score_mean=0.61,
        score_mean_delta=0.11,
        score_mean_rounds=2,
        king_token_boost=0.02,
    )

    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
        mean_score_margin=0.10,
    ) == CloseChallenge(active, CloseReason.KING_DEFENDED)


def test_mean_scoring_default_margin_is_010() -> None:
    active = _active(
        P1,
        wins=0,
        losses=0,
        ties=2,
        target=2,
        king_score_mean=0.50,
        challenger_score_mean=0.54,
        score_mean_delta=0.04,
        score_mean_rounds=2,
    )
    assert DEFAULT_MEAN_SCORE_MARGIN == 0.10
    assert decide(
        _snapshot(active),
        scoring_method=DuelScoringMethod.MEAN,
    ) == CloseChallenge(active, CloseReason.KING_DEFENDED)


def test_mean_scoring_king_holds_without_scored_rounds() -> None:
    active = _active(P1, wins=0, losses=0, ties=2, target=2)
    assert decide(
        _snapshot(active), scoring_method=DuelScoringMethod.MEAN
    ) == CloseChallenge(active, CloseReason.KING_DEFENDED)
