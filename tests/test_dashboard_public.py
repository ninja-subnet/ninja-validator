import datetime as dt
import json
from unittest.mock import Mock

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from tau.db.status import DuelOutcome, PoolType, SubmissionStatus, TaskStatus
from tau.pools import PoolTargets
from tau.dashboard.public import (
    DashboardConfig,
    PublicDashboard,
    _assemble_payload,
    _active_rounds,
    _duel_score_round_rows,
    _freshness_from_timestamp,
    _king_pool_summaries,
    _public_duel_id,
    _public_task_id,
    _public_submission,
    _public_repo,
    _public_round_winner,
    _queue,
    _recent_kings,
    _resolution_score_fields,
    _scoring_config,
    _task_token_score_fields,
)


def test_public_duel_id_matches_static_duel_explorer_mapping() -> None:
    assert _public_duel_id("5FCArHjuMfddGuyh-19ade1d3d1ec9f14") == 995206


def test_public_task_id_is_stable_and_opaque() -> None:
    public_id = _public_task_id("secret-task-id")

    assert public_id == _public_task_id("secret-task-id")
    assert public_id != _public_task_id("another-task-id")
    assert public_id.startswith("task-")
    assert "secret-task-id" not in public_id


def test_public_repo_exposes_github_and_hides_local_paths() -> None:
    assert _public_repo("https://github.com/ninja/repo.git", "sub-1") == (
        "ninja/repo",
        "https://github.com/ninja/repo",
    )
    assert _public_repo("ninja/repo@abcdef", "sub-2") == (
        "ninja/repo",
        "https://github.com/ninja/repo",
    )
    assert _public_repo("/srv/tau/submissions/private-id", "private-id") == (
        "private-submission/private-id",
        None,
    )


def test_public_repo_rewrites_unarbos_links_to_ninja_subnet() -> None:
    assert _public_repo("https://github.com/unarbos/ninja.git", "sub-1") == (
        "ninja-subnet/ninja",
        "https://github.com/ninja-subnet/ninja",
    )
    assert _public_repo("unarbos/tau@abcdef", "sub-2") == (
        "ninja-subnet/ninja-validator",
        "https://github.com/ninja-subnet/ninja-validator",
    )


def test_public_round_winner_marks_error_even_without_llm_winner() -> None:
    assert _public_round_winner(None, "timeout") == "error"
    assert _public_round_winner(None, None) == "pending"
    assert _public_round_winner("challenger", None) == "challenger"
    assert _public_round_winner("unexpected", None) == "tie"


def test_recent_kings_exposes_real_defenses_reign_duration_and_shares() -> None:
    now = dt.datetime(2026, 7, 13, 12, 0, tzinfo=dt.UTC)
    current_from = now - dt.timedelta(hours=2)
    prior_from = current_from - dt.timedelta(hours=3)
    result = Mock()
    result.mappings.return_value = [
        {
            "submission_id": "current",
            "king_from": current_from,
            "king_until": None,
            "hotkey": "current-hotkey",
            "source": "/private/current",
            "uid": 1,
            "king_duels_defended": 174,
        },
        {
            "submission_id": "prior",
            "king_from": prior_from,
            "king_until": current_from,
            "hotkey": "prior-hotkey",
            "source": "/private/prior",
            "uid": 2,
            "king_duels_defended": 42,
        },
    ]
    session = Mock()
    session.execute.return_value = result

    kings = _recent_kings(session, limit=5, now=now)

    assert [king["submission_id"] for king in kings] == ["current", "prior"]
    assert [king["king_duels_defended"] for king in kings] == [174, 42]
    assert [king["hold_seconds"] for king in kings] == [7200, 10800]
    assert [king["share"] for king in kings] == [0.60, 0.40]
    assert session.execute.call_args.args[1] == {
        "limit": 5,
        "king_won": int(DuelOutcome.KING_WON),
    }


def test_resolution_score_fields_expose_raw_boost_final_and_token_totals() -> None:
    fields = _resolution_score_fields(
        {
            "king_score_mean": 0.50,
            "challenger_score_mean": 0.58,
            "score_mean_delta": 0.08,
            "token_bonus_enabled": True,
            "token_score_tolerance": 0.05,
            "token_min_score": 0.20,
            "token_bonus_multiplier": 0.15,
            "king_total_tokens": 120000,
            "challenger_total_tokens": 60000,
            "token_comparison_rounds": 50,
            "king_token_savings_mean": 0.01,
            "challenger_token_savings_mean": 0.20,
            "king_token_boost": 0.0015,
            "challenger_token_boost": 0.03,
            "king_combined_score": 0.5015,
            "challenger_combined_score": 0.61,
            "combined_score_delta": 0.1085,
        }
    )

    assert fields["king_score_mean"] == 0.50
    assert fields["challenger_score_mean"] == 0.58
    assert fields["score_mean_delta"] == 0.08
    assert fields["king_token_boost"] == 0.0015
    assert fields["challenger_token_boost"] == 0.03
    assert fields["king_combined_score"] == 0.5015
    assert fields["challenger_combined_score"] == 0.61
    assert fields["combined_score_delta"] == 0.1085
    assert fields["final_mean_delta"] == 0.1085
    assert fields["king_total_tokens"] == 120000
    assert fields["challenger_total_tokens"] == 60000


def test_resolution_score_fields_keep_legacy_raw_score_and_unknown_tokens() -> None:
    fields = _resolution_score_fields(
        {
            "king_score_mean": 0.50,
            "challenger_score_mean": 0.58,
            "score_mean_delta": 0.08,
        }
    )

    assert fields["king_combined_score"] == 0.50
    assert fields["challenger_combined_score"] == 0.58
    assert fields["combined_score_delta"] == 0.08
    assert fields["final_mean_delta"] == 0.08
    assert fields["king_token_boost"] is None
    assert fields["challenger_token_boost"] is None
    assert fields["king_total_tokens"] is None
    assert fields["challenger_total_tokens"] is None


def test_task_token_score_fields_show_saving_and_exact_pool_contribution() -> None:
    fields = _task_token_score_fields(
        {
            "king_score": 0.50,
            "challenger_score": 0.50,
            "king_usage_summary": {"total_tokens": 100},
            "challenger_usage_summary": {"total_tokens": 50},
            "token_bonus_enabled": True,
            "token_score_tolerance": 0.05,
            "token_min_score": 0.20,
            "token_bonus_multiplier": 0.15,
            "token_pool_target": 50,
            "judge_error": False,
        }
    )

    assert fields["king_tokens"] == 100
    assert fields["challenger_tokens"] == 50
    assert fields["king_token_saving"] == 0
    assert fields["challenger_token_saving"] == 0.5
    assert fields["king_token_contribution"] == 0
    assert fields["challenger_token_contribution"] == 0.0015
    assert fields["challenger_token_reason"] == "Eligible"


def test_task_token_score_fields_explain_ineligible_quality() -> None:
    fields = _task_token_score_fields(
        {
            "king_score": 0.50,
            "challenger_score": 0.44,
            "king_usage_summary": json.dumps({"total_tokens": 100}),
            "challenger_usage_summary": json.dumps({"total_tokens": 50}),
            "token_bonus_enabled": True,
            "token_score_tolerance": 0.05,
            "token_min_score": 0.20,
            "token_bonus_multiplier": 0.15,
            "token_pool_target": 50,
            "judge_error": False,
        }
    )

    assert fields["challenger_token_eligible"] is False
    assert fields["challenger_token_saving"] == 0
    assert fields["challenger_token_contribution"] == 0
    assert fields["challenger_token_reason"] == "More than 0.05 behind"


def test_task_token_score_fields_never_rewards_judge_errors() -> None:
    fields = _task_token_score_fields(
        {
            "king_score": 0.50,
            "challenger_score": 0.50,
            "king_usage_summary": {"total_tokens": 100},
            "challenger_usage_summary": {"total_tokens": 50},
            "token_bonus_enabled": True,
            "token_score_tolerance": 0.05,
            "token_min_score": 0.20,
            "token_bonus_multiplier": 0.15,
            "token_pool_target": 50,
            "judge_error": True,
        }
    )

    assert fields["challenger_token_contribution"] == 0
    assert fields["challenger_token_reason"] == "Judge error"


def test_public_submission_includes_public_accepted_timestamp() -> None:
    accepted_at = dt.datetime(2026, 7, 3, 15, 32, 19, tzinfo=dt.UTC)

    item = _public_submission(
        {
            "submission_id": "sub-1",
            "uid": 66,
            "hotkey": "hot",
            "source": "/private/submission",
            "accepted_at": accepted_at,
        }
    )

    assert item["accepted_at"] == "2026-07-03T15:32:19Z"
    assert item["repo"] == "private-submission/sub-1"


def test_freshness_statuses_are_public_and_timestamp_based() -> None:
    now = dt.datetime(2026, 7, 3, 12, 0, tzinfo=dt.UTC)

    live = _freshness_from_timestamp(
        now - dt.timedelta(seconds=30), now=now, fresh_seconds=60
    )
    quiet = _freshness_from_timestamp(
        now - dt.timedelta(seconds=90), now=now, fresh_seconds=60
    )
    delayed = _freshness_from_timestamp(
        now - dt.timedelta(seconds=10), now=now, fresh_seconds=60, delayed=True
    )
    unknown = _freshness_from_timestamp(None, now=now, fresh_seconds=60)

    assert live["status"] == "live"
    assert quiet["status"] == "quiet"
    assert delayed["status"] == "delayed"
    assert unknown == {
        "status": "unknown",
        "last_seen_at": None,
        "age_seconds": None,
        "detail": "",
    }


def test_deployed_shell_dashboard_aliases_return_live_payload(monkeypatch) -> None:
    dashboard = object.__new__(PublicDashboard)
    monkeypatch.setattr(dashboard, "payload", lambda: {"source": "postgres"})

    for path in (
        "/dashboard-home.json",
        "/dashboard-summary.json",
        "/api/dashboard",
        "/api/dashboard/current",
        "/api/dashboard/home",
        "/api/dashboard/summary",
    ):
        status, payload = dashboard.response_for(path)
        assert status == 200
        assert payload == {"source": "postgres"}


def test_duels_endpoint_includes_scoring_and_current_king(monkeypatch) -> None:
    dashboard = object.__new__(PublicDashboard)
    current_king = {"submission_id": "king-submission"}
    scoring = {"method": "mean", "mean_score_margin": 0.10}
    monkeypatch.setattr(
        dashboard,
        "payload",
        lambda: {
            "updated_at": "2026-07-13T12:00:00Z",
            "duels": [],
            "duels_total": 0,
            "current_king": current_king,
            "status": {"scoring": scoring},
        },
    )

    status, payload = dashboard.response_for("/api/dashboard/duels")

    assert status == 200
    assert payload["scoring"] == scoring
    assert payload["current_king"] == current_king


def test_dashboard_scoring_default_margin_is_point_ten() -> None:
    assert _scoring_config({})["mean_score_margin"] == 0.10


def test_active_rounds_render_in_judgement_arrival_order() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        session.execute(
            text(
                """
                CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY,
                    king_id TEXT NOT NULL,
                    pool_type INTEGER NOT NULL,
                    status_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE judgements (
                    task_id TEXT NOT NULL,
                    king_submission_id TEXT NOT NULL,
                    challenger_submission_id TEXT NOT NULL,
                    llm_winner TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO tasks VALUES
                    ('task-1', 'king', 1, 1, '2026-07-03T00:00:01Z'),
                    ('task-2', 'king', 1, 1, '2026-07-03T00:00:02Z'),
                    ('task-3', 'king', 1, 1, '2026-07-03T00:00:03Z')
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO judgements VALUES
                    ('task-3', 'king', 'challenger', 'challenger', NULL, '2026-07-03T00:00:10Z'),
                    ('task-1', 'king', 'challenger', 'king', NULL, '2026-07-03T00:00:11Z')
                """
            )
        )
        active = {
            "king_submission_id": "king",
            "challenger_submission_id": "challenger",
            "active_pool": int(PoolType.POOL_ONE),
        }

        rounds = _active_rounds(session, active, limit=3)

    assert rounds == [
        {"round": 1, "task_name": "result 01", "winner": "challenger"},
        {"round": 2, "task_name": "result 02", "winner": "king"},
        {"round": 3, "task_name": "result 03", "winner": "pending"},
    ]


def test_duel_score_round_rows_use_historical_task_order_after_pool_replacement() -> (
    None
):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        session.execute(
            text(
                """
                CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY,
                    king_id TEXT NOT NULL,
                    pool_type INTEGER NOT NULL,
                    status_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE judgements (
                    task_id TEXT NOT NULL,
                    king_submission_id TEXT NOT NULL,
                    challenger_submission_id TEXT NOT NULL,
                    llm_winner TEXT,
                    king_score REAL,
                    challenger_score REAL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE duel_task_solutions (
                    task_id TEXT NOT NULL,
                    challenger_submission_id TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    usage_summary TEXT
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE duel_resolutions (
                    challenger_submission_id TEXT NOT NULL,
                    pool_type INTEGER NOT NULL,
                    token_bonus_enabled INTEGER NOT NULL,
                    token_score_tolerance REAL,
                    token_min_score REAL,
                    token_bonus_multiplier REAL,
                    best_of INTEGER NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO tasks VALUES
                    ('task-1', 'king', 1, :disqualified, '2026-07-03T00:00:01Z'),
                    ('task-2', 'king', 1, :disqualified, '2026-07-03T00:00:02Z'),
                    ('task-3', 'king', 1, :disqualified, '2026-07-03T00:00:03Z'),
                    ('task-4', 'king', 2, :disqualified, '2026-07-03T00:00:04Z'),
                    ('replacement-1', 'king', 1, :qualified, '2026-07-04T00:00:01Z'),
                    ('replacement-2', 'king', 2, :qualified, '2026-07-04T00:00:02Z')
                """
            ),
            {
                "qualified": int(TaskStatus.QUALIFIED),
                "disqualified": int(TaskStatus.DISQUALIFIED),
            },
        )
        session.execute(
            text(
                """
                INSERT INTO judgements VALUES
                    ('task-3', 'king', 'challenger', 'tie', 0.5, 0.5, NULL, '2026-07-03T00:00:10Z'),
                    ('task-1', 'king', 'challenger', 'king', 0.9, 0.1, NULL, '2026-07-03T00:00:11Z'),
                    ('task-4', 'king', 'challenger', 'challenger', 0.1, 0.9, NULL, '2026-07-03T00:00:12Z'),
                    ('task-2', 'king', 'challenger', 'challenger', 0.2, 0.8, NULL, '2026-07-03T00:00:13Z')
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO duel_resolutions VALUES
                    ('challenger', 1, 1, 0.05, 0.20, 0.15, 3),
                    ('challenger', 2, 1, 0.05, 0.20, 0.15, 1)
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO duel_task_solutions VALUES
                    ('task-3', 'challenger', 'king', '{"total_tokens": 100}'),
                    ('task-3', 'challenger', 'challenger', '{"total_tokens": 50}')
                """
            )
        )

        rows = _duel_score_round_rows(
            session,
            king_id="king",
            challenger_id="challenger",
            targets=PoolTargets(pool_one=3, pool_two=1),
        )

    assert [
        (
            row["public_round"],
            row["pool_round"],
            row["llm_winner"],
            row["king_score"],
            row["challenger_score"],
        )
        for row in rows
    ] == [
        (1, 1, "king", 0.9, 0.1),
        (2, 2, "challenger", 0.2, 0.8),
        (3, 3, "tie", 0.5, 0.5),
        (4, 1, "challenger", 0.1, 0.9),
    ]
    token_fields = _task_token_score_fields(rows[2])
    assert token_fields["king_tokens"] == 100
    assert token_fields["challenger_tokens"] == 50
    assert token_fields["challenger_token_saving"] == 0.5
    assert abs(token_fields["challenger_token_contribution"] - 0.025) < 1e-12


def test_duel_solution_artifact_exposes_metadata_without_task_secrets(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TAU_POOL_ONE_TARGET", "1")
    monkeypatch.setenv("TAU_POOL_TWO_TARGET", "1")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        session.execute(
            text(
                """
                CREATE TABLE challenges (
                    challenger_submission_id TEXT PRIMARY KEY,
                    king_id TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY,
                    king_id TEXT NOT NULL,
                    pool_type INTEGER NOT NULL,
                    status_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE duel_task_solutions (
                    task_id TEXT NOT NULL,
                    challenger_submission_id TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    solution TEXT,
                    duration REAL,
                    exit_reason TEXT,
                    usage_summary TEXT
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE judgements (
                    task_id TEXT NOT NULL,
                    king_submission_id TEXT NOT NULL,
                    challenger_submission_id TEXT NOT NULL,
                    llm_winner TEXT,
                    king_score REAL,
                    challenger_score REAL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO challenges VALUES
                    ('challenger-submission-secret', 'king-submission-secret', 2, '2026-07-03T00:00:00Z')
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO tasks VALUES
                    ('secret-task-one', 'king-submission-secret', 1, 1, '2026-07-03T00:00:01Z'),
                    ('secret-task-two', 'king-submission-secret', 2, 2, '2026-07-03T00:00:02Z'),
                    ('replacement-task-two', 'king-submission-secret', 2, 1, '2026-07-04T00:00:02Z')
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO duel_task_solutions VALUES
                    ('secret-task-two', 'challenger-submission-secret', 'king-submission-secret', '', 9.0, 'completed', NULL),
                    (
                        'secret-task-two',
                        'challenger-submission-secret',
                        'challenger-submission-secret',
                        'diff --git a/secret b/secret\n--- a/secret\n+++ b/secret\n@@\n-SECRET_TASK_CONTEXT\n+SECRET_TASK_CONTEXT_FIXED\n',
                        12.5,
                        'completed',
                        :usage_summary
                    )
                """
            ),
            {
                "usage_summary": json.dumps(
                    {
                        "request_count": 2,
                        "success_count": 1,
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "cost": 0.0,
                        "last_upstream_error": "internal-upstream.invalid do-not-publish marker",
                        "requests": [
                            {
                                "method": "POST",
                                "path": "http://internal-upstream.invalid/v1/chat/completions",
                                "status_code": 200,
                                "latency_ms": 1234,
                                "first_token_latency_ms": 120,
                                "prompt_tokens": 11,
                                "completion_tokens": 7,
                                "total_tokens": 18,
                                "cost": None,
                                "error": "do-not-publish marker from internal-upstream.invalid",
                            }
                        ],
                    }
                )
            },
        )
        session.execute(
            text(
                """
                INSERT INTO judgements VALUES
                    ('secret-task-two', 'king-submission-secret', 'challenger-submission-secret', 'challenger', 0.25, 0.75, NULL, '2026-07-03T00:00:10Z')
                """
            )
        )
        session.commit()

    dashboard = PublicDashboard(engine, DashboardConfig())
    try:
        public_duel_id = _public_duel_id("challenger-submission-secret")
        status, payload = dashboard.response_for(
            f"/api/duels/{public_duel_id}/rounds/2/solutions/challenger.solve.json"
        )
    finally:
        dashboard.close()

    assert status == 200
    assert payload["stage"] == "solve"
    assert payload["solution_name"] == "challenger"
    assert payload["round"] == 2
    assert payload["pool"] == "POOL_TWO"
    assert payload["pool_round"] == 1
    assert payload["task_name"] == "set 02 result 01"
    assert payload["public_task_id"] == _public_task_id("secret-task-two")
    assert payload["result"] == {
        "available": True,
        "success": True,
        "success_inferred": True,
        "exit_reason": "completed",
        "elapsed_seconds": 12.5,
        "diff_available": True,
        "nonempty_diff": True,
        "changed_lines": 2,
        "usage_summary": {
            "request_count": 2,
            "success_count": 1,
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "requests": [
                {
                    "index": 0,
                    "method": "POST",
                    "status_code": 200,
                    "latency_ms": 1234,
                    "first_token_latency_ms": 120,
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            ],
        },
    }
    assert payload["solutions"]["king"]["changed_lines"] == 0
    assert payload["judgement"]["winner"] == "challenger"
    assert payload["judgement"]["score_delta"] == 0.5

    encoded = json.dumps(payload)
    assert "secret-task-two" not in encoded
    assert "SECRET_TASK_CONTEXT" not in encoded
    assert "diff --git" not in encoded
    assert "king-submission-secret" not in encoded
    assert "challenger-submission-secret" not in encoded
    assert "internal-upstream.invalid" not in encoded
    assert "do-not-publish marker" not in encoded
    assert "chat/completions" not in encoded


def test_king_pool_summaries_render_without_active_duel() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        session.execute(
            text(
                """
                CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY,
                    king_id TEXT NOT NULL,
                    pool_type INTEGER NOT NULL,
                    status_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO tasks VALUES
                    ('task-1', 'king', 1, 1, '2026-07-03T00:00:01Z'),
                    ('task-2', 'king', 1, 0, '2026-07-03T00:00:02Z'),
                    ('task-3', 'king', 2, 1, '2026-07-03T00:00:03Z'),
                    ('task-4', 'king', 2, 2, '2026-07-03T00:00:04Z')
                """
            )
        )

        pools = _king_pool_summaries(
            session,
            {"submission_id": "king"},
            PoolTargets(pool_one=3, pool_two=2),
        )

    assert pools[1]["label"] == "Pool 1"
    assert pools[1]["task_count"] == 2
    assert pools[1]["qualified_count"] == 1
    assert pools[1]["candidate_count"] == 1
    assert pools[1]["remaining_rounds"] == 1
    assert pools[2]["label"] == "Pool 2"
    assert pools[2]["task_count"] == 1
    assert pools[2]["disqualified_count"] == 1


def test_queue_mirrors_live_worker_queue() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        session.execute(
            text(
                """
                CREATE TABLE submissions (
                    submission_id TEXT PRIMARY KEY,
                    block INTEGER NOT NULL,
                    hotkey TEXT NOT NULL,
                    source TEXT,
                    status_id INTEGER
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE kings (
                    king_id TEXT PRIMARY KEY
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE challenges (
                    challenger_submission_id TEXT PRIMARY KEY
                )
                """
            )
        )
        session.execute(
            text(
                """
                CREATE TABLE registrations (
                    uid INTEGER NOT NULL,
                    ss58_hot TEXT NOT NULL,
                    block INTEGER NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO submissions VALUES
                    ('eligible-sub', 10, 'eligible-hotkey', NULL, :eligible),
                    ('unverified-sub', 11, 'unverified-hotkey', NULL, :unverified),
                    ('disqualified-sub', 12, 'disqualified-hotkey', NULL, :disqualified),
                    ('needs-review-sub', 13, 'needs-review-hotkey', NULL, :needs_review),
                    ('deregistered-sub', 14, 'deregistered-hotkey', NULL, :eligible),
                    ('stale-sub', 15, 'stale-hotkey', NULL, :unverified)
                """
            ),
            {
                "eligible": int(SubmissionStatus.ELIGIBLE),
                "unverified": int(SubmissionStatus.UNVERIFIED),
                "disqualified": int(SubmissionStatus.DISQUALIFIED),
                "needs_review": int(SubmissionStatus.NEEDS_REVIEW),
            },
        )
        session.execute(
            text(
                """
                INSERT INTO registrations VALUES
                    (1, 'eligible-hotkey', 10),
                    (3, 'unverified-hotkey', 11),
                    (2, 'disqualified-hotkey', 12),
                    (4, 'needs-review-hotkey', 13),
                    (5, 'stale-hotkey', 16)
                """
            )
        )

        queue = _queue(
            session,
            limit=10,
            now=dt.datetime(2026, 7, 3, 12, 0, tzinfo=dt.UTC),
        )

    assert [item["submission_id"] for item in queue] == [
        "eligible-sub",
        "unverified-sub",
    ]


def test_assembled_payload_keeps_static_shell_contract_and_mean_gate() -> None:
    now = dt.datetime(2026, 7, 3, 12, 0, tzinfo=dt.UTC)
    started_at = now - dt.timedelta(hours=2)
    scoring = {
        "method": "mean",
        "win_margin": 0,
        "mean_score_margin": 0.05,
        "duel_rounds": 50,
        "pool_one_target": 50,
        "pool_two_target": 50,
    }
    current_king = {
        "submission_id": "king-submission",
        "uid": 1,
        "hotkey": "king-hot",
        "repo": "private-submission/king-submission",
        "repo_full_name": "private-submission/king-submission",
        "repo_url": None,
    }
    active = {
        "duel_id": 7,
        "challenger_submission_id": "challenger-submission",
        "king_submission_id": "king-submission",
        "active_pool": int(PoolType.POOL_ONE),
        "active_pool_name": "POOL_ONE",
        "created_at": now,
        "king_hotkey": "king-hot",
        "king_uid": 1,
        "king_repo": "private-submission/king-submission",
        "king_repo_url": None,
        "challenger_hotkey": "challenger-hot",
        "challenger_uid": 2,
        "challenger_repo": "ninja/challenger",
        "challenger_repo_url": "https://github.com/ninja/challenger",
    }
    progress = {
        "pool": "POOL_ONE",
        "target": 50,
        "active": True,
        "task_count": 50,
        "candidate_count": 0,
        "qualified_count": 50,
        "disqualified_count": 0,
        "king_solved_count": 50,
        "challenger_solved_count": 50,
        "judged_rounds": 6,
        "remaining_rounds": 44,
        "wins": 4,
        "losses": 1,
        "ties": 1,
        "errors": 0,
        "mean": {
            "king_score": 0.52,
            "challenger_score": 0.59,
            "delta": 0.07,
            "rounds": 6,
        },
    }

    payload = _assemble_payload(
        now=now,
        validator_started_at=started_at,
        config=DashboardConfig(netuid=66),
        current_king=current_king,
        active=active,
        active_progress=progress,
        active_rounds=[{"round": 1, "winner": "challenger"}],
        pool_summaries={int(PoolType.POOL_ONE): progress},
        scoring=scoring,
        recent_duels=[],
        duels_total=0,
        recent_kings=[current_king],
        leaderboard=[current_king],
        queue=[],
        workers={"judging": {"status": "live"}},
    )

    active_duel = payload["status"]["active_duel"]
    assert payload["source"] == "postgres"
    assert payload["updated_at"] == "2026-07-03T12:00:00Z"
    assert payload["status"]["validator_started_at"] == "2026-07-03T10:00:00Z"
    assert payload["current_king"]["repo_url"] is None
    assert active_duel["duel_rounds"] == 50
    assert active_duel["active_pool"] == 1
    assert active_duel["active_pool_name"] == "POOL_ONE"
    assert active_duel["pool_id"] == 1
    assert active_duel["pool_name"] == "POOL_ONE"
    assert active_duel["pool_label"] == "Pool 1"
    assert payload["status"]["pools"]["1"]["pool_id"] == 1
    assert payload["status"]["pools"]["1"]["pool_name"] == "POOL_ONE"
    assert payload["status"]["pools"]["1"]["pool_label"] == "Pool 1"
    assert active_duel["threshold"] == 2
    assert active_duel["scoring_method"] == "mean"
    assert active_duel["king_score_mean"] == 0.52
    assert active_duel["challenger_score_mean"] == 0.59
    assert active_duel["score_mean_delta"] == 0.07
    assert active_duel["score_mean_rounds"] == 6
    assert active_duel["mean_score_delta"] == 0.07
    assert active_duel["mean_score_threshold"] == 0.05
    assert active_duel["mean_score_gate_met"] is True
    assert "source" not in payload["current_king"]


def test_pools_endpoint_exposes_active_pool_aliases(monkeypatch) -> None:
    dashboard = object.__new__(PublicDashboard)
    monkeypatch.setattr(
        dashboard,
        "payload",
        lambda: {
            "updated_at": "2026-07-03T12:00:00Z",
            "status": {
                "active_duel": {
                    "pool": "POOL_TWO",
                    "active_pool": 2,
                    "active_pool_name": "POOL_TWO",
                    "pool_label": "Pool 2",
                },
                "pools": {
                    "2": {
                        "pool": "POOL_TWO",
                        "pool_id": 2,
                        "pool_name": "POOL_TWO",
                        "pool_label": "Pool 2",
                    }
                },
            },
        },
    )

    status, payload = dashboard.response_for("/api/dashboard/pools")

    assert status == 200
    assert payload["active_pool"] == "POOL_TWO"
    assert payload["active_pool_id"] == 2
    assert payload["active_pool_name"] == "POOL_TWO"
    assert payload["active_pool_label"] == "Pool 2"
