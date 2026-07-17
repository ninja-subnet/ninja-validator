"""Integration tests for the duel-resolver DB seam (`DuelResolverDb.snapshot`).

Same Postgres requirement as test_db_generator: set ``TAU_TEST_DATABASE_URL`` to a
throwaway database. Skipped unless that server is reachable.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import make_url, select, text

from tau.db import DuelResolverDb
from tau.db.engine import (
    async_session_scope,
    create_async_db_engine,
    create_db_engine,
)
from tau.db.models import (
    Base,
    Challenge,
    DuelResolution,
    DuelTaskSolution,
    Judgement,
    King,
    KingArchive,
    Registration,
    Rollout,
    Submission,
    Task,
    TaskScreening,
    TaskSolution,
)
from tau.db.status import (
    ChallengeStatus,
    DuelOutcome,
    PoolType,
    SubmissionStatus,
    TaskStatus,
)
from tau.duel import ActiveChallenge, DuelScoringMethod, Tally, TokenEfficiencyConfig
from tau.pools import PoolTargets
from tau.rollouts import rollout_id

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_TEST_URL = os.environ.get("TAU_TEST_DATABASE_URL")
_BLOCK_DATE = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
_ROUND_RULE = {
    "scoring_method": DuelScoringMethod.ROUND_WINS,
    "round_win_margin": 0,
    "mean_score_margin": 0.05,
}


def _maintenance_url(url: str) -> str:
    return make_url(url).set(database="postgres").render_as_string(hide_password=False)


def _server_reachable(url: str | None) -> bool:
    if not url:
        return False
    try:
        engine = create_db_engine(_maintenance_url(url))
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _server_reachable(_TEST_URL),
    reason="TAU_TEST_DATABASE_URL unset or its Postgres server unreachable — skipping DB tests",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    assert _TEST_URL is not None
    db_name = make_url(_TEST_URL).database
    engine = create_db_engine(_maintenance_url(_TEST_URL))
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            already = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not already:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        engine.dispose()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[DuelResolverDb]:
    engine = create_async_db_engine(_TEST_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    resolver = DuelResolverDb(_TEST_URL)
    try:
        yield resolver
    finally:
        await resolver.aclose()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# -- seed helpers -------------------------------------------------------------------


def _submission(
    session,
    sid: str,
    *,
    hotkey: str,
    block: int,
    status: SubmissionStatus | None = SubmissionStatus.ELIGIBLE,
) -> None:
    session.add(
        Submission(
            submission_id=sid,
            block=block,
            hotkey=hotkey,
            status_id=None if status is None else int(status),
        )
    )


def _registration(session, *, uid: int, hotkey: str, block: int) -> None:
    session.add(
        Registration(
            uid=uid,
            ss58_hot=hotkey,
            ss58_cold=f"cold-{uid}",
            block=block,
            block_date=_BLOCK_DATE,
        )
    )


def _task_with_solutions(
    session, *, task_id: str, king: str, challenger: str, pool: PoolType
) -> None:
    """A QUALIFIED task with the king's and challenger's solutions."""
    session.add(
        Task(
            task_id=task_id,
            king_id=king,
            pool_type=int(pool),
            problem_statement="p",
            status_id=int(TaskStatus.QUALIFIED),
            repo_clone_url="u",
            parent_sha="p",
            commit_sha="c",
            reference_patch="r",
            content_fingerprint=f"fp-{task_id}",
        )
    )
    session.add(TaskSolution(task_id=task_id, submission_id=king, solution="ksol"))
    session.add(
        TaskSolution(task_id=task_id, submission_id=challenger, solution="csol")
    )


def _duel_solution(
    session,
    *,
    task_id: str,
    challenger: str,
    submission: str,
    total_tokens: int | None,
) -> None:
    session.add(
        DuelTaskSolution(
            task_id=task_id,
            challenger_submission_id=challenger,
            submission_id=submission,
            solution="patch",
            usage_summary=(
                None if total_tokens is None else {"total_tokens": total_tokens}
            ),
        )
    )


def _qualified_task(session, *, task_id: str, king: str, pool: PoolType) -> None:
    session.add(
        Task(
            task_id=task_id,
            king_id=king,
            pool_type=int(pool),
            problem_statement="p",
            status_id=int(TaskStatus.QUALIFIED),
            repo_clone_url="u",
            parent_sha="p",
            commit_sha="c",
            reference_patch="r",
            content_fingerprint=f"fp-{task_id}",
        )
    )


def _ac(
    challenger: str,
    king: str,
    pool: PoolType,
    *,
    tally: Tally | None = None,
    pool_target: int = 1,
) -> ActiveChallenge:
    """An ActiveChallenge for the write methods: challenger id + pool drive the guard;
    tally/pool_target are what the duel_resolutions row records."""
    return ActiveChallenge(
        challenger_submission_id=challenger,
        king_submission_id=king,
        pool=pool,
        pool_target=pool_target,
        tally=tally if tally is not None else Tally(0, 0, 0),
        challenger_registered=True,
    )


def _score_pair(winner: str) -> tuple[float, float]:
    if winner == "king":
        return 1.0, 0.0
    if winner == "challenger":
        return 0.0, 1.0
    return 0.5, 0.5


def _judgement(
    session,
    *,
    task_id: str,
    king: str,
    challenger: str,
    winner: str,
    king_score: float | None = None,
    challenger_score: float | None = None,
    error: str | None = None,
) -> None:
    # Keep solution rows available for old-path compatibility; the resolver itself
    # only tallies saved judgements.
    default_king_score, default_challenger_score = _score_pair(winner)
    session.add(
        Judgement(
            task_id=task_id,
            king_submission_id=king,
            challenger_submission_id=challenger,
            llm_winner=winner,
            king_score=default_king_score if king_score is None else king_score,
            challenger_score=(
                default_challenger_score
                if challenger_score is None
                else challenger_score
            ),
            error=error,
        )
    )


# -- tests --------------------------------------------------------------------------


async def test_snapshot_is_empty_without_a_king(db: DuelResolverDb) -> None:
    snap = await db.snapshot(PoolTargets())
    assert snap.reigning_king_submission_id is None
    assert snap.active_challenge is None
    assert snap.next_challenger_submission_id is None


async def test_snapshot_opens_for_the_oldest_eligible_registered_submission(
    db: DuelResolverDb,
) -> None:
    targets = PoolTargets(pool_one=1, pool_two=1)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        # The winner: oldest by block among eligible + registered + not king/challenger.
        _submission(session, "c-old", hotkey="hk-old", block=10)
        _submission(session, "c-new", hotkey="hk-new", block=20)
        _registration(session, uid=1, hotkey="hk-old", block=10)
        _registration(session, uid=2, hotkey="hk-new", block=20)
        # Excluded: ineligible status.
        _submission(session, "c-inelig", hotkey="hk-inelig", block=2, status=None)
        _registration(session, uid=3, hotkey="hk-inelig", block=2)
        # Excluded: hotkey superseded on its uid -> not in the current metagraph.
        _submission(session, "c-unreg", hotkey="hk-unreg", block=3)
        _registration(session, uid=4, hotkey="hk-unreg", block=3)
        _registration(session, uid=4, hotkey="hk-took-over", block=99)
        # Excluded: already challenged (a closed challenge row exists).
        _submission(session, "c-past", hotkey="hk-past", block=4)
        _registration(session, uid=5, hotkey="hk-past", block=4)
        session.add(
            Challenge(
                challenger_submission_id="c-past",
                king_id="k",
                status=int(ChallengeStatus.CLOSED),
            )
        )
        _qualified_task(session, task_id="p1", king="k", pool=PoolType.POOL_ONE)
        _qualified_task(session, task_id="p2", king="k", pool=PoolType.POOL_TWO)

    snap = await db.snapshot(targets)
    assert snap.reigning_king_submission_id == "k"
    assert snap.active_challenge is None
    assert snap.next_challenger_submission_id == "c-old"
    assert snap.task_pools_ready is True


async def test_snapshot_hides_challenger_until_both_task_pools_are_ready(
    db: DuelResolverDb,
) -> None:
    targets = PoolTargets(pool_one=1, pool_two=1)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        _registration(session, uid=1, hotkey="hk-c", block=2)
        _qualified_task(session, task_id="p1", king="k", pool=PoolType.POOL_ONE)

    snap = await db.snapshot(targets)

    assert snap.active_challenge is None
    assert snap.next_challenger_submission_id is None
    assert snap.task_pools_ready is False


async def test_snapshot_reports_active_challenge_tally(db: DuelResolverDb) -> None:
    targets = PoolTargets(pool_one=8, pool_two=8)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        _registration(session, uid=1, hotkey="hk-c", block=2)
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_ONE),
            )
        )
        # Pool one: 2 challenger wins, 1 king win, 1 tie. The pool-two round must NOT
        # leak into the pool-one tally.
        rounds = [
            ("t1", PoolType.POOL_ONE, "challenger"),
            ("t2", PoolType.POOL_ONE, "challenger"),
            ("t3", PoolType.POOL_ONE, "king"),
            ("t4", PoolType.POOL_ONE, "tie"),
            ("t5", PoolType.POOL_TWO, "challenger"),
        ]
        for task_id, pool, _winner in rounds:
            _task_with_solutions(
                session, task_id=task_id, king="k", challenger="c", pool=pool
            )
        await session.flush()  # solutions must exist before their judgements
        for task_id, _pool, winner in rounds:
            _judgement(
                session, task_id=task_id, king="k", challenger="c", winner=winner
            )

    snap = await db.snapshot(targets)
    active = snap.active_challenge
    assert active is not None
    assert active.challenger_submission_id == "c"
    assert active.king_submission_id == "k"
    assert active.pool is PoolType.POOL_ONE
    assert active.pool_target == 8
    assert active.tally == Tally(
        wins=2,
        losses=1,
        ties=1,
        king_score_mean=0.375,
        challenger_score_mean=0.625,
        score_mean_delta=0.25,
        score_mean_rounds=4,
    )
    assert active.challenger_registered is True
    # decide ignores it while a duel runs, so the seam skips computing it.
    assert snap.next_challenger_submission_id is None


async def test_snapshot_calculates_symmetric_per_task_token_boosts(
    db: DuelResolverDb,
) -> None:
    targets = PoolTargets(pool_one=4, pool_two=4)
    rounds = [
        # Challenger uses half: challenger saving 0.50.
        ("t1", 0.50, 0.50, 100, 50),
        # King uses one quarter: king saving 0.75.
        ("t2", 0.50, 0.50, 50, 200),
        # A neutral judge-error fallback must not earn the apparent saving.
        ("t3", 0.50, 0.50, 100, 50),
        # Missing challenger usage contributes no saving; king tokens still total.
        ("t4", 0.50, 0.50, 75, None),
    ]
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        _registration(session, uid=1, hotkey="hk-c", block=2)
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_ONE),
            )
        )
        for (
            task_id,
            king_score,
            challenger_score,
            king_tokens,
            challenger_tokens,
        ) in rounds:
            _task_with_solutions(
                session,
                task_id=task_id,
                king="k",
                challenger="c",
                pool=PoolType.POOL_ONE,
            )
            _duel_solution(
                session,
                task_id=task_id,
                challenger="c",
                submission="k",
                total_tokens=king_tokens,
            )
            _duel_solution(
                session,
                task_id=task_id,
                challenger="c",
                submission="c",
                total_tokens=challenger_tokens,
            )
        await session.flush()
        for (
            task_id,
            king_score,
            challenger_score,
            _king_tokens,
            _challenger_tokens,
        ) in rounds:
            _judgement(
                session,
                task_id=task_id,
                king="k",
                challenger="c",
                winner="tie",
                king_score=king_score,
                challenger_score=challenger_score,
                error="judge failed" if task_id == "t3" else None,
            )

    snap = await db.snapshot(
        targets,
        token_efficiency=TokenEfficiencyConfig(enabled=True),
    )
    assert snap.active_challenge is not None
    tally = snap.active_challenge.tally
    assert tally.king_score_mean == pytest.approx(0.50)
    assert tally.challenger_score_mean == pytest.approx(0.50)
    assert tally.score_mean_delta == 0
    assert tally.king_total_tokens == 325
    assert tally.challenger_total_tokens is None
    assert tally.token_comparison_rounds == 3
    # Savings use the full four-task denominator, not three comparable tasks.
    assert tally.king_token_savings_mean == pytest.approx(0.75 / 4)
    assert tally.challenger_token_savings_mean == pytest.approx(0.50 / 4)
    assert tally.king_token_boost == pytest.approx(0.028125)
    assert tally.challenger_token_boost == pytest.approx(0.01875)
    assert tally.combined_score_delta == pytest.approx(-0.009375)


async def test_snapshot_token_totals_include_solved_rows_not_yet_judged(
    db: DuelResolverDb,
) -> None:
    targets = PoolTargets(pool_one=2, pool_two=2)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        _registration(session, uid=1, hotkey="hk-c", block=2)
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_ONE),
            )
        )
        for task_id, king_tokens, challenger_tokens in (
            ("judged", 100, 50),
            ("solved-only", 200, 80),
        ):
            _task_with_solutions(
                session,
                task_id=task_id,
                king="k",
                challenger="c",
                pool=PoolType.POOL_ONE,
            )
            _duel_solution(
                session,
                task_id=task_id,
                challenger="c",
                submission="k",
                total_tokens=king_tokens,
            )
            _duel_solution(
                session,
                task_id=task_id,
                challenger="c",
                submission="c",
                total_tokens=challenger_tokens,
            )
        await session.flush()
        _judgement(
            session,
            task_id="judged",
            king="k",
            challenger="c",
            winner="tie",
        )

    snap = await db.snapshot(
        targets,
        token_efficiency=TokenEfficiencyConfig(enabled=True),
    )

    assert snap.active_challenge is not None
    tally = snap.active_challenge.tally
    assert tally.king_total_tokens == 300
    assert tally.challenger_total_tokens == 130
    assert tally.token_comparison_rounds == 1
    assert tally.challenger_token_savings_mean == pytest.approx(0.25)
    assert tally.challenger_token_boost == pytest.approx(0.0375)


async def test_snapshot_marks_a_deregistered_challenger(db: DuelResolverDb) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        # uid 1 held hk-c, then was taken over by another hotkey at a higher block.
        _registration(session, uid=1, hotkey="hk-c", block=2)
        _registration(session, uid=1, hotkey="hk-other", block=50)
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_TWO),
            )
        )

    snap = await db.snapshot(PoolTargets())
    assert snap.active_challenge is not None
    assert snap.active_challenge.pool is PoolType.POOL_TWO
    assert snap.active_challenge.challenger_registered is False


async def test_snapshot_stale_when_submission_predates_current_registration(
    db: DuelResolverDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=10)  # made under the first reg
        session.add(King(king_id="k"))
        _registration(session, uid=0, hotkey="hk-k", block=1)
        # hk-c registered (uid 1, block 5), lapsed (uid 1 taken over at block 50), then
        # re-registered on a new uid at block 100 -- so it is current, but at block 100.
        _registration(session, uid=1, hotkey="hk-c", block=5)
        _registration(session, uid=1, hotkey="hk-other", block=50)
        _registration(session, uid=2, hotkey="hk-c", block=100)
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_ONE),
            )
        )

    snap = await db.snapshot(PoolTargets())
    assert snap.active_challenge is not None
    # hk-c is in the metagraph, but the submission (block 10) predates its current
    # registration (block 100) -> a prior, dead identity -> stale.
    assert snap.active_challenge.challenger_registered is False


# -- writes -------------------------------------------------------------------------


async def _seed_king_and_challenger(db: DuelResolverDb) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))


async def _challenge_status(db: DuelResolverDb) -> int | None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        challenge = await session.get(Challenge, "c")
        return None if challenge is None else challenge.status


async def _resolutions(
    db: DuelResolverDb, challenger: str = "c"
) -> list[DuelResolution]:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        rows = await session.scalars(
            select(DuelResolution)
            .where(DuelResolution.challenger_submission_id == challenger)
            .order_by(DuelResolution.pool_type)
        )
        return list(rows)


async def test_open_challenge_inserts_pool_one_and_is_idempotent(
    db: DuelResolverDb,
) -> None:
    await _seed_king_and_challenger(db)
    assert await db.open_challenge("k", "c") is True
    assert await db.open_challenge("k", "c") is False  # conflict -> no-op
    assert await _challenge_status(db) == int(ChallengeStatus.POOL_ONE)


async def test_advance_pool_records_the_win_and_is_guarded(db: DuelResolverDb) -> None:
    await _seed_king_and_challenger(db)
    await db.open_challenge("k", "c")
    challenge = _ac(
        "c",
        "k",
        PoolType.POOL_ONE,
        tally=Tally(
            3,
            1,
            0,
            king_score_mean=0.25,
            challenger_score_mean=0.75,
            score_mean_delta=0.5,
            score_mean_rounds=4,
            king_total_tokens=1200,
            challenger_total_tokens=900,
            token_comparison_rounds=4,
            king_token_savings_mean=0.10,
            challenger_token_savings_mean=0.20,
            king_token_boost=0.01,
            challenger_token_boost=0.02,
        ),
        pool_target=5,
    )
    token_config = TokenEfficiencyConfig(
        enabled=True,
        score_tolerance=0.04,
        min_score=0.25,
        bonus_multiplier=0.10,
    )
    assert (
        await db.advance_pool(
            challenge,
            scoring_method=DuelScoringMethod.ROUND_WINS,
            round_win_margin=2,
            mean_score_margin=0.05,
            token_efficiency=token_config,
        )
        is True
    )
    assert await _challenge_status(db) == int(ChallengeStatus.POOL_TWO)
    rows = await _resolutions(db)
    assert len(rows) == 1
    assert rows[0].pool_type == int(PoolType.POOL_ONE)
    assert rows[0].outcome == int(DuelOutcome.CHALLENGER_WON)
    assert (rows[0].challenger_wins, rows[0].challenger_losses, rows[0].ties) == (
        3,
        1,
        0,
    )
    assert rows[0].best_of == 5
    assert rows[0].scoring_method == "round_wins"
    assert rows[0].round_win_margin == 2
    assert rows[0].mean_score_margin == 0.05
    assert rows[0].king_score_mean == 0.25
    assert rows[0].challenger_score_mean == 0.75
    assert rows[0].score_mean_delta == 0.5
    assert rows[0].score_mean_rounds == 4
    # Round-win mode never applies or advertises a token modifier, even if a caller
    # accidentally passes an enabled token config and a tally containing boosts.
    assert rows[0].token_bonus_enabled is False
    assert rows[0].token_score_tolerance == 0.04
    assert rows[0].token_min_score == 0.25
    assert rows[0].token_bonus_multiplier == 0.10
    assert rows[0].king_total_tokens == 1200
    assert rows[0].challenger_total_tokens == 900
    assert rows[0].token_comparison_rounds == 4
    assert rows[0].king_token_savings_mean == 0
    assert rows[0].challenger_token_savings_mean == 0
    assert rows[0].king_token_boost == 0
    assert rows[0].challenger_token_boost == 0
    assert rows[0].king_combined_score == 0.25
    assert rows[0].challenger_combined_score == 0.75
    assert rows[0].combined_score_delta == 0.50
    # Stale retry (still expecting POOL_ONE) is a no-op now that it is POOL_TWO,
    # and records nothing more.
    assert (
        await db.advance_pool(
            challenge,
            scoring_method=DuelScoringMethod.ROUND_WINS,
            round_win_margin=2,
            mean_score_margin=0.05,
        )
        is False
    )
    assert len(await _resolutions(db)) == 1


async def test_close_challenge_sets_closed_and_records_outcome(
    db: DuelResolverDb,
) -> None:
    await _seed_king_and_challenger(db)
    await db.open_challenge("k", "c")
    challenge = _ac(
        "c",
        "k",
        PoolType.POOL_ONE,
        tally=Tally(
            0,
            3,
            1,
            king_score_mean=0.50,
            challenger_score_mean=0.54,
            score_mean_delta=0.04,
            score_mean_rounds=4,
            king_total_tokens=1000,
            challenger_total_tokens=800,
            token_comparison_rounds=4,
            king_token_savings_mean=0.10,
            challenger_token_savings_mean=0.20,
            king_token_boost=0.01,
            challenger_token_boost=0.02,
        ),
        pool_target=4,
    )
    assert (
        await db.close_challenge(
            challenge,
            DuelOutcome.KING_WON,
            scoring_method=DuelScoringMethod.MEAN,
            round_win_margin=1,
            mean_score_margin=0.075,
            token_efficiency=TokenEfficiencyConfig(
                enabled=True,
                score_tolerance=0.04,
                min_score=0.25,
                bonus_multiplier=0.10,
            ),
        )
        is True
    )
    assert await _challenge_status(db) == int(ChallengeStatus.CLOSED)
    rows = await _resolutions(db)
    assert len(rows) == 1
    assert rows[0].pool_type == int(PoolType.POOL_ONE)
    assert rows[0].outcome == int(DuelOutcome.KING_WON)
    assert (rows[0].challenger_wins, rows[0].challenger_losses, rows[0].ties) == (
        0,
        3,
        1,
    )
    assert rows[0].best_of == 4
    assert rows[0].scoring_method == "mean"
    assert rows[0].round_win_margin == 1
    assert rows[0].mean_score_margin == 0.075
    assert rows[0].token_bonus_enabled is True
    assert rows[0].king_total_tokens == 1000
    assert rows[0].challenger_total_tokens == 800
    assert rows[0].king_token_boost == 0.01
    assert rows[0].challenger_token_boost == 0.02
    assert rows[0].king_combined_score == 0.51
    assert rows[0].challenger_combined_score == 0.56
    assert rows[0].combined_score_delta == pytest.approx(0.05)


async def test_promote_crowns_the_challenger_and_records(db: DuelResolverDb) -> None:
    await _seed_king_and_challenger(db)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_TWO),
            )
        )
    challenge = _ac("c", "k", PoolType.POOL_TWO, tally=Tally(4, 0, 1), pool_target=5)
    assert await db.promote(challenge, **_ROUND_RULE) is True
    assert await _challenge_status(db) == int(ChallengeStatus.CLOSED)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        assert await session.get(King, "c") is not None  # the challenger now reigns
        archive = await session.get(KingArchive, "k")
        assert archive is not None
        assert archive.promoted_to == "c"
        assert archive.status == "pending"
    rows = await _resolutions(db)
    assert len(rows) == 1
    assert rows[0].pool_type == int(PoolType.POOL_TWO)
    assert rows[0].outcome == int(DuelOutcome.CHALLENGER_WON)
    # The new king is the reigning one (latest king_from).
    assert (await db.snapshot(PoolTargets())).reigning_king_submission_id == "c"


async def test_promote_is_a_noop_when_not_in_pool_two(db: DuelResolverDb) -> None:
    await _seed_king_and_challenger(db)
    await db.open_challenge("k", "c")  # POOL_ONE, not POOL_TWO
    # Stale promote (snapshot thought POOL_TWO) crowns nobody, leaves the challenge,
    # and records no resolution.
    assert await db.promote(_ac("c", "k", PoolType.POOL_TWO), **_ROUND_RULE) is False
    assert await _challenge_status(db) == int(ChallengeStatus.POOL_ONE)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        assert await session.get(King, "c") is None
    assert await _resolutions(db) == []


async def test_export_king_dataset_normalizes_new_and_legacy_rollouts(
    db: DuelResolverDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(session, "k", hotkey="hk-k", block=1)
        _submission(session, "c", hotkey="hk-c", block=2)
        session.add(King(king_id="k"))
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.CLOSED),
            )
        )
        session.add(
            Task(
                task_id="t1",
                king_id="k",
                pool_type=int(PoolType.POOL_ONE),
                problem_statement="fix it",
                status_id=int(TaskStatus.QUALIFIED),
                repo_clone_url="https://github.com/octo/repo.git",
                parent_sha="p",
                commit_sha="c",
                reference_patch="reference",
                content_fingerprint="fp-t1",
            )
        )
        session.add(
            TaskScreening(
                task_id="t1",
                king_submission_id="k",
                qualification_solution="qualification patch",
                king_score=0.4,
                max_score=0.8,
            )
        )
        await session.flush()
        session.add(
            Rollout(
                rollout_id=rollout_id(
                    phase="qualification", task_id="t1", submission_id="k"
                ),
                phase="qualification",
                task_id="t1",
                submission_id="k",
                success=True,
                solution_diff="qualification patch",
                exit_reason="completed",
                duration_seconds=1.0,
                usage_summary={"total_tokens": 10},
                events=[{"index": 0, "type": "llm_call"}],
            )
        )
        session.add(
            DuelTaskSolution(
                task_id="t1",
                challenger_submission_id="c",
                submission_id="c",
                solution="legacy duel patch",
                duration=2.0,
                exit_reason="completed",
                usage_summary={"total_tokens": 20},
            )
        )

    tasks = await db.export_king_tasks("k")
    rollouts = [
        row
        async for batch in db.stream_king_rollouts("k", batch_size=1)
        for row in batch
    ]

    assert tasks[0]["task_id"] == "t1"
    assert tasks[0]["screening"]["king_score"] == 0.4
    assert len(rollouts) == 2
    qualification = next(row for row in rollouts if row["phase"] == "qualification")
    legacy_duel = next(row for row in rollouts if row["phase"] == "duel")
    assert qualification["capture_available"] is True
    assert qualification["events"] == [{"index": 0, "type": "llm_call"}]
    assert legacy_duel["capture_available"] is False
    assert legacy_duel["solution_diff"] == "legacy duel patch"


async def test_archive_job_claim_retry_and_completion(db: DuelResolverDb) -> None:
    await _seed_king_and_challenger(db)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Challenge(
                challenger_submission_id="c",
                king_id="k",
                status=int(ChallengeStatus.POOL_TWO),
            )
        )
    assert await db.promote(_ac("c", "k", PoolType.POOL_TWO), **_ROUND_RULE)

    first = await db.claim_king_archive(lease_seconds=60)
    assert first is not None
    assert (first.king_id, first.promoted_to, first.attempt) == ("k", "c", 1)
    assert await db.claim_king_archive(lease_seconds=60) is None
    assert await db.retry_king_archive("k", error="temporary", delay_seconds=0)

    second = await db.claim_king_archive(lease_seconds=60)
    assert second is not None and second.attempt == 2
    assert await db.complete_king_archive("k")
    assert await db.claim_king_archive(lease_seconds=60) is None
