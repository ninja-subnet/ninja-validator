"""DB-backed public dashboard payloads for ninja66.ai.

The public site should show live validator state without exposing task prompts,
solution diffs, judge rationales, local filesystem paths, or raw submission
bundles. This module builds a small read model from Postgres and keeps the old
static dashboard JSON routes alive while the frontend migrates to `/api/dashboard/*`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from tau.bittensor.types import BLOCK_SECONDS
from tau.db.engine import create_db_engine, session_factory, session_scope
from tau.db.status import (
    ChallengeStatus,
    DuelOutcome,
    PoolType,
    SubmissionStatus,
    TaskStatus,
)
from tau.pools import PoolTargets
from tau.utils.env import env_float, env_int, env_str
from tau.utils.logging import configure_logging

log = logging.getLogger(__name__)

_GITHUB_RE = re.compile(r"github\.com[:/](?P<repo>[^/\s]+/[^/@\s#]+)", re.I)
_OWNER_REPO_RE = re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@|$)")
_NINJA_SUBNET_OWNER = "ninja-subnet"
_NINJA_SUBNET_REPO_ALIASES = {
    "tau": "ninja-validator",
}
_PUBLIC_ROUTES = {
    "/dashboard-home.json",
    "/dashboard-summary.json",
    "/api/dashboard",
    "/api/dashboard/current",
    "/api/dashboard/home",
    "/api/dashboard/summary",
}
_DUEL_NOTES_RE = re.compile(r"^/api/dashboard/duels/(?P<duel_id>\d{1,6})/notes$")
_DUEL_SOLUTION_RE = re.compile(
    r"^/api/duels/(?P<duel_id>\d{1,6})/rounds/(?P<round>\d{1,3})"
    r"/solutions/(?P<solution>king|challenger)\.solve\.json$"
)


def _public_duel_id(challenger_submission_id: object) -> int:
    return 100000 + (zlib.crc32(str(challenger_submission_id).encode("utf-8")) % 900000)


def _public_task_id(task_id: object) -> str:
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:12]
    return f"task-{digest}"


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8066
    recent_duels: int = 40
    fresh_seconds: float = 600.0
    netuid: int = 66

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DashboardConfig:
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            host=env_str(env, "TAU_DASHBOARD_HOST", d.host),
            port=env_int(env, "TAU_DASHBOARD_PORT", d.port),
            recent_duels=max(
                1, env_int(env, "TAU_DASHBOARD_RECENT_DUELS", d.recent_duels)
            ),
            fresh_seconds=max(
                1.0, env_float(env, "TAU_DASHBOARD_FRESH_SECONDS", d.fresh_seconds)
            ),
            netuid=env_int(env, "TAU_NETUID", d.netuid),
        )


class PublicDashboard:
    def __init__(self, engine: Engine, config: DashboardConfig) -> None:
        self._engine = engine
        self._sessions = session_factory(engine)
        self._config = config
        self._started_at = _utcnow()

    @classmethod
    def create(
        cls, *, config: DashboardConfig | None = None, db_url: str | None = None
    ) -> PublicDashboard:
        return cls(create_db_engine(db_url), config or DashboardConfig.from_env())

    def close(self) -> None:
        self._engine.dispose()

    def payload(self) -> dict[str, Any]:
        now = _utcnow()
        with session_scope(self._sessions) as session:
            current_king = _current_king(session)
            active = _active_challenge(session, current_king)
            scoring = _scoring_config(os.environ)
            pool_targets = PoolTargets.from_env(os.environ)
            pool_summaries = (
                _pool_summaries(session, active, pool_targets)
                if active
                else _king_pool_summaries(session, current_king, pool_targets)
            )
            active_progress = (
                pool_summaries.get(active["active_pool"]) if active else None
            )
            active_rounds = _active_rounds(
                session,
                active,
                limit=(active_progress or {}).get("target", 0),
            )
            duels_total = _completed_duel_count(session)
            recent_duels = _recent_completed_duels(
                session, limit=self._config.recent_duels
            )
            recent_kings = _recent_kings(session, limit=12)
            leaderboard = _leaderboard(session, current_king, limit=20)
            queue = _queue(session, limit=500, now=now)
            workers = _worker_freshness(
                session,
                now=now,
                fresh_seconds=self._config.fresh_seconds,
                active=active,
                active_progress=active_progress,
            )

        return _assemble_payload(
            now=now,
            validator_started_at=self._started_at,
            config=self._config,
            current_king=current_king,
            active=active,
            active_progress=active_progress,
            active_rounds=active_rounds,
            pool_summaries=pool_summaries,
            scoring=scoring,
            recent_duels=recent_duels,
            duels_total=duels_total,
            recent_kings=recent_kings,
            leaderboard=leaderboard,
            queue=queue,
            workers=workers,
        )

    def submissions(self) -> dict[str, Any]:
        with session_scope(self._sessions) as session:
            rows = session.execute(
                text(
                    """
                    SELECT
                        s.submission_id,
                        s.hotkey,
                        s.status_id,
                        s.source,
                        s.block,
                        (
                            SELECT r.uid
                            FROM registrations r
                            WHERE r.ss58_hot = s.hotkey
                            ORDER BY r.block DESC
                            LIMIT 1
                        ) AS uid
                    FROM submissions s
                    ORDER BY s.block DESC, s.submission_id
                    LIMIT 500
                    """
                )
            ).mappings()
            rows = list(rows)
            anchor_block = _max_block(rows)
            now = _utcnow()
            submissions = []
            for row in rows:
                item = _public_submission(
                    row,
                    submitted_at=_submission_submitted_at(
                        row, anchor_block=anchor_block, now=now
                    ),
                )
                item["status"] = _submission_status_name(row["status_id"])
                submissions.append(item)
        return {"updated_at": _iso(_utcnow()), "submissions": submissions}

    def response_for(self, path: str) -> tuple[int, dict[str, Any]]:
        if path in _PUBLIC_ROUTES:
            return HTTPStatus.OK, self.payload()
        solution_match = _DUEL_SOLUTION_RE.match(path)
        if solution_match:
            with session_scope(self._sessions) as session:
                artifact = _duel_solution_artifact(
                    session,
                    duel_id=int(solution_match.group("duel_id")),
                    round_number=int(solution_match.group("round")),
                    solution_name=solution_match.group("solution"),
                    targets=PoolTargets.from_env(os.environ),
                )
            if artifact is None:
                return HTTPStatus.NOT_FOUND, {"error": "duel round solution not found"}
            return HTTPStatus.OK, artifact
        notes_match = _DUEL_NOTES_RE.match(path)
        if notes_match:
            with session_scope(self._sessions) as session:
                detail = _duel_score_notes(
                    session, duel_id=int(notes_match.group("duel_id"))
                )
            if detail is None:
                return HTTPStatus.NOT_FOUND, {"error": "duel not found"}
            return HTTPStatus.OK, detail
        if path == "/api/dashboard/duels":
            payload = self.payload()
            return HTTPStatus.OK, {
                "updated_at": payload["updated_at"],
                "duels": payload["duels"],
                "duels_total": payload["duels_total"],
                "scoring": payload["status"].get("scoring", {}),
                "current_king": payload.get("current_king"),
            }
        if path == "/api/dashboard/pools":
            payload = self.payload()
            active_duel = payload["status"].get("active_duel")
            active_pool_id = active_duel.get("active_pool") if active_duel else None
            active_pool_name = (
                active_duel.get("active_pool_name") if active_duel else None
            )
            return HTTPStatus.OK, {
                "updated_at": payload["updated_at"],
                "active_pool": active_duel.get("pool") if active_duel else None,
                "active_pool_id": active_pool_id,
                "active_pool_name": active_pool_name,
                "active_pool_label": active_duel.get("pool_label")
                if active_duel
                else None,
                "active_duel": active_duel,
                "pools": payload["status"].get("pools", {}),
            }
        if path == "/api/dashboard/health":
            payload = self.payload()
            return HTTPStatus.OK, {
                "updated_at": payload["updated_at"],
                "workers": payload["status"].get("workers", {}),
            }
        if path == "/api/submissions":
            return HTTPStatus.OK, self.submissions()
        if path == "/health":
            return HTTPStatus.OK, {"ok": True}
        return HTTPStatus.NOT_FOUND, {"error": "not found"}


def serve() -> None:
    configure_logging()
    config = DashboardConfig.from_env()
    dashboard = PublicDashboard.create(config=config)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            self._send_json(write_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name
            self._send_json(write_body=False)

        def _send_json(self, *, write_body: bool) -> None:
            try:
                status, payload = dashboard.response_for(urlparse(self.path).path)
            except Exception:  # noqa: BLE001 - keep public endpoint returning JSON.
                log.exception("dashboard request failed")
                status, payload = (
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "dashboard unavailable"},
                )
            body = json.dumps(
                payload, default=_json_default, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if write_body:
                self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook name
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    try:
        server = ThreadingHTTPServer((config.host, config.port), Handler)
        server.serve_forever()
    finally:
        dashboard.close()


def _assemble_payload(
    *,
    now: dt.datetime,
    validator_started_at: dt.datetime,
    config: DashboardConfig,
    current_king: dict[str, Any] | None,
    active: dict[str, Any] | None,
    active_progress: dict[str, Any] | None,
    active_rounds: list[dict[str, Any]],
    pool_summaries: dict[int, dict[str, Any]],
    scoring: dict[str, Any],
    recent_duels: list[dict[str, Any]],
    duels_total: int,
    recent_kings: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    workers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    active_duel = _legacy_active_duel(active, active_progress, active_rounds, scoring)
    total_rounds = sum(duel.get("round_count", 0) for duel in recent_duels)
    total_rounds += (active_progress or {}).get("judged_rounds", 0)
    status = {
        "netuid": config.netuid,
        "validator_started_at": _iso(validator_started_at),
        "active_duel": active_duel,
        "pools": {
            str(pool): _pool_summary_payload(pool, summary)
            for pool, summary in sorted(pool_summaries.items())
        },
        "scoring": scoring,
        "workers": workers,
        "recent_kings": recent_kings,
        "leaderboard": leaderboard,
        "queue": queue,
        "miners_seen": _unique_miners(recent_kings, leaderboard, queue, current_king),
        "total_rounds": total_rounds,
        "links": {
            "dashboard_home": "/dashboard-home.json",
            "dashboard_summary": "/dashboard-summary.json",
            "duels_html": "/duels.html",
            "duels_index": "/duels/index.json",
        },
    }
    return {
        "updated_at": _iso(now),
        "source": "postgres",
        "current_king": current_king,
        "status": status,
        "duels": recent_duels,
        "duels_total": duels_total,
        "leaderboard": leaderboard,
    }


def _legacy_active_duel(
    active: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    rounds: list[dict[str, Any]],
    scoring: dict[str, Any],
) -> dict[str, Any] | None:
    if active is None or progress is None:
        return None
    target = progress["target"]
    wins = progress["wins"]
    losses = progress["losses"]
    margin = int(scoring.get("win_margin", 0) or 0)
    round_threshold = losses + margin + 1
    mean_score_margin = float(scoring.get("mean_score_margin", 0.0) or 0.0)
    mean_score_delta = progress["mean"]["delta"]
    return {
        "duel_id": active["duel_id"],
        "public_duel_id": active.get("public_duel_id", active["duel_id"]),
        "db_duel_id": active.get("db_duel_id"),
        "challenge_id": active["challenger_submission_id"],
        "scoring_method": scoring.get("method"),
        "task_set_phase": "confirmation_retest"
        if active["active_pool"] == int(PoolType.POOL_TWO)
        else "primary",
        "duel_rounds": target,
        "active_pool": active["active_pool"],
        "active_pool_name": active["active_pool_name"],
        "pool_id": active["active_pool"],
        "pool_name": active["active_pool_name"],
        "pool_label": _pool_label(active["active_pool"]),
        "pool": active["active_pool_name"],
        "pool_target": target,
        "rounds": rounds,
        "wins": wins,
        "losses": losses,
        "ties": progress["ties"],
        "errors": progress["errors"],
        "threshold": round_threshold,
        "round_win_threshold": round_threshold,
        "mean_score_delta": mean_score_delta,
        "score_mean_delta": mean_score_delta,
        "mean_score_threshold": mean_score_margin,
        "mean_score_margin": mean_score_margin,
        "king_score_mean": progress["mean"]["king_score"],
        "challenger_score_mean": progress["mean"]["challenger_score"],
        "score_mean_rounds": progress["mean"]["rounds"],
        "mean_score_gate_met": (
            mean_score_delta is not None and mean_score_delta >= mean_score_margin
        ),
        "judged_rounds": progress["judged_rounds"],
        "remaining_rounds": progress["remaining_rounds"],
        "mean": progress["mean"],
        "king_uid": active["king_uid"],
        "king_hotkey": active["king_hotkey"],
        "king_repo": active["king_repo"],
        "king_repo_url": active["king_repo_url"],
        "challenger_uid": active["challenger_uid"],
        "challenger_hotkey": active["challenger_hotkey"],
        "challenger_repo": active["challenger_repo"],
        "challenger_repo_url": active["challenger_repo_url"],
    }


def _pool_summary_payload(pool: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_id": pool,
        "pool_name": _pool_name(pool),
        "pool_label": _pool_label(pool),
        "label": _pool_label(pool),
        **summary,
    }


def _current_king(session: Session) -> dict[str, Any] | None:
    row = (
        session.execute(
            text(
                """
            SELECT
                k.king_id AS submission_id,
                k.king_from,
                s.hotkey,
                s.source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = s.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS uid
            FROM kings k
            JOIN submissions s ON s.submission_id = k.king_id
            ORDER BY k.king_from DESC
            LIMIT 1
            """
            )
        )
        .mappings()
        .first()
    )
    return _public_submission(row, crowned_at=row["king_from"]) if row else None


def _active_challenge(
    session: Session, current_king: dict[str, Any] | None
) -> dict[str, Any] | None:
    if current_king is None:
        return None
    row = (
        session.execute(
            text(
                """
            WITH numbered AS (
                SELECT
                    c.*,
                    row_number() OVER (ORDER BY c.created_at, c.challenger_submission_id) AS duel_id
                FROM challenges c
            )
            SELECT
                c.duel_id,
                c.challenger_submission_id,
                c.king_id,
                c.status AS active_pool,
                c.created_at,
                ks.hotkey AS king_hotkey,
                ks.source AS king_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = ks.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS king_uid,
                cs.hotkey AS challenger_hotkey,
                cs.source AS challenger_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = cs.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS challenger_uid
            FROM numbered c
            JOIN submissions ks ON ks.submission_id = c.king_id
            JOIN submissions cs ON cs.submission_id = c.challenger_submission_id
            WHERE c.king_id = :king_id
              AND c.status IN (:pool_one, :pool_two)
            ORDER BY c.created_at, c.challenger_submission_id
            LIMIT 1
            """
            ),
            {
                "king_id": current_king["submission_id"],
                "pool_one": int(ChallengeStatus.POOL_ONE),
                "pool_two": int(ChallengeStatus.POOL_TWO),
            },
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    king_repo, king_url = _public_repo(row["king_source"], row["king_id"])
    challenger_repo, challenger_url = _public_repo(
        row["challenger_source"], row["challenger_submission_id"]
    )
    active_pool = int(row["active_pool"])
    public_duel_id = _public_duel_id(row["challenger_submission_id"])
    return {
        "duel_id": public_duel_id,
        "public_duel_id": public_duel_id,
        "db_duel_id": int(row["duel_id"]),
        "challenger_submission_id": row["challenger_submission_id"],
        "king_submission_id": row["king_id"],
        "active_pool": active_pool,
        "active_pool_name": _pool_name(active_pool),
        "created_at": row["created_at"],
        "king_hotkey": row["king_hotkey"],
        "king_uid": row["king_uid"],
        "king_repo": king_repo,
        "king_repo_url": king_url,
        "challenger_hotkey": row["challenger_hotkey"],
        "challenger_uid": row["challenger_uid"],
        "challenger_repo": challenger_repo,
        "challenger_repo_url": challenger_url,
    }


def _pool_summaries(
    session: Session, active: dict[str, Any] | None, targets: PoolTargets
) -> dict[int, dict[str, Any]]:
    if active is None:
        return {}
    rows = session.execute(
        text(
            """
            WITH ranked_tasks AS (
                SELECT
                    t.*,
                    row_number() OVER (
                        PARTITION BY t.pool_type
                        ORDER BY t.created_at, t.task_id
                    ) AS public_n
                FROM tasks t
                WHERE t.king_id = :king_id
                  AND t.pool_type IN (:pool_one, :pool_two)
                  AND t.status_id <> :disqualified
            ),
            public_tasks AS (
                SELECT *
                FROM ranked_tasks
                WHERE (
                    pool_type = :pool_one
                    AND public_n <= :pool_one_target
                ) OR (
                    pool_type = :pool_two
                    AND public_n <= :pool_two_target
                )
            ),
            disqualified AS (
                SELECT t.pool_type, count(*) AS disqualified_count
                FROM tasks t
                WHERE t.king_id = :king_id
                  AND t.pool_type IN (:pool_one, :pool_two)
                  AND t.status_id = :disqualified
                GROUP BY t.pool_type
            )
            SELECT
                t.pool_type,
                count(*) AS task_count,
                count(*) FILTER (WHERE t.status_id = :candidate) AS candidate_count,
                count(*) FILTER (WHERE t.status_id = :qualified) AS qualified_count,
                max(coalesce(d.disqualified_count, 0)) AS disqualified_count,
                count(ks.task_id) AS king_solved_count,
                count(cs.task_id) AS challenger_solved_count,
                count(j.task_id) AS judged_rounds,
                count(j.task_id) FILTER (WHERE j.llm_winner = 'challenger') AS wins,
                count(j.task_id) FILTER (WHERE j.llm_winner = 'king') AS losses,
                count(j.task_id) FILTER (
                    WHERE j.task_id IS NOT NULL
                      AND coalesce(j.llm_winner, '') NOT IN ('king', 'challenger')
                      AND j.error IS NULL
                ) AS ties,
                count(j.task_id) FILTER (WHERE j.error IS NOT NULL) AS errors,
                count(j.task_id) FILTER (
                    WHERE j.king_score IS NOT NULL
                      AND j.challenger_score IS NOT NULL
                ) AS score_mean_rounds,
                avg(j.king_score) FILTER (
                    WHERE j.king_score IS NOT NULL
                      AND j.challenger_score IS NOT NULL
                ) AS king_score_mean,
                avg(j.challenger_score) FILTER (
                    WHERE j.king_score IS NOT NULL
                      AND j.challenger_score IS NOT NULL
                ) AS challenger_score_mean
            FROM public_tasks t
            LEFT JOIN disqualified d
              ON d.pool_type = t.pool_type
            LEFT JOIN duel_task_solutions ks
              ON ks.task_id = t.task_id
             AND ks.challenger_submission_id = :challenger_id
             AND ks.submission_id = :king_id
            LEFT JOIN duel_task_solutions cs
              ON cs.task_id = t.task_id
             AND cs.challenger_submission_id = :challenger_id
             AND cs.submission_id = :challenger_id
            LEFT JOIN judgements j
              ON j.task_id = t.task_id
             AND j.king_submission_id = :king_id
             AND j.challenger_submission_id = :challenger_id
            GROUP BY t.pool_type
            """
        ),
        {
            "king_id": active["king_submission_id"],
            "challenger_id": active["challenger_submission_id"],
            "candidate": int(TaskStatus.CANDIDATE),
            "qualified": int(TaskStatus.QUALIFIED),
            "disqualified": int(TaskStatus.DISQUALIFIED),
            "pool_one": int(PoolType.POOL_ONE),
            "pool_two": int(PoolType.POOL_TWO),
            "pool_one_target": targets.pool_one,
            "pool_two_target": targets.pool_two,
        },
    ).mappings()
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        pool = int(row["pool_type"])
        target = targets.target(PoolType(pool))
        king_mean = _float_or_none(row["king_score_mean"])
        challenger_mean = _float_or_none(row["challenger_score_mean"])
        delta = (
            challenger_mean - king_mean
            if challenger_mean is not None and king_mean is not None
            else None
        )
        judged = int(row["judged_rounds"])
        output[pool] = {
            "pool_id": pool,
            "pool_name": _pool_name(pool),
            "pool_label": _pool_label(pool),
            "label": _pool_label(pool),
            "pool": _pool_name(pool),
            "target": target,
            "active": pool == active["active_pool"],
            "task_count": int(row["task_count"]),
            "candidate_count": int(row["candidate_count"]),
            "qualified_count": int(row["qualified_count"]),
            "disqualified_count": int(row["disqualified_count"]),
            "king_solved_count": int(row["king_solved_count"]),
            "challenger_solved_count": int(row["challenger_solved_count"]),
            "judged_rounds": judged,
            "remaining_rounds": max(0, target - judged),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "ties": int(row["ties"]),
            "errors": int(row["errors"]),
            "mean": {
                "king_score": king_mean,
                "challenger_score": challenger_mean,
                "delta": delta,
                "rounds": int(row["score_mean_rounds"]),
            },
            "mean_score_delta": delta,
        }
    for pool_type in PoolType:
        output.setdefault(
            int(pool_type),
            {
                "pool_id": int(pool_type),
                "pool_name": _pool_name(int(pool_type)),
                "pool_label": _pool_label(int(pool_type)),
                "label": _pool_label(int(pool_type)),
                "pool": _pool_name(int(pool_type)),
                "target": targets.target(pool_type),
                "active": int(pool_type) == active["active_pool"],
                "task_count": 0,
                "candidate_count": 0,
                "qualified_count": 0,
                "disqualified_count": 0,
                "king_solved_count": 0,
                "challenger_solved_count": 0,
                "judged_rounds": 0,
                "remaining_rounds": targets.target(pool_type),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "errors": 0,
                "mean": {
                    "king_score": None,
                    "challenger_score": None,
                    "delta": None,
                    "rounds": 0,
                },
            },
        )
    return output


def _king_pool_summaries(
    session: Session, current_king: dict[str, Any] | None, targets: PoolTargets
) -> dict[int, dict[str, Any]]:
    if current_king is None:
        return {}
    rows = session.execute(
        text(
            """
            WITH ranked_tasks AS (
                SELECT
                    t.*,
                    row_number() OVER (
                        PARTITION BY t.pool_type
                        ORDER BY t.created_at, t.task_id
                    ) AS public_n
                FROM tasks t
                WHERE t.king_id = :king_id
                  AND t.pool_type IN (:pool_one, :pool_two)
                  AND t.status_id <> :disqualified
            ),
            public_tasks AS (
                SELECT *
                FROM ranked_tasks
                WHERE (
                    pool_type = :pool_one
                    AND public_n <= :pool_one_target
                ) OR (
                    pool_type = :pool_two
                    AND public_n <= :pool_two_target
                )
            ),
            disqualified AS (
                SELECT t.pool_type, count(*) AS disqualified_count
                FROM tasks t
                WHERE t.king_id = :king_id
                  AND t.pool_type IN (:pool_one, :pool_two)
                  AND t.status_id = :disqualified
                GROUP BY t.pool_type
            )
            SELECT
                t.pool_type,
                count(*) AS task_count,
                count(*) FILTER (WHERE t.status_id = :candidate) AS candidate_count,
                count(*) FILTER (WHERE t.status_id = :qualified) AS qualified_count,
                max(coalesce(d.disqualified_count, 0)) AS disqualified_count
            FROM public_tasks t
            LEFT JOIN disqualified d
              ON d.pool_type = t.pool_type
            GROUP BY t.pool_type
            """
        ),
        {
            "king_id": current_king["submission_id"],
            "candidate": int(TaskStatus.CANDIDATE),
            "qualified": int(TaskStatus.QUALIFIED),
            "disqualified": int(TaskStatus.DISQUALIFIED),
            "pool_one": int(PoolType.POOL_ONE),
            "pool_two": int(PoolType.POOL_TWO),
            "pool_one_target": targets.pool_one,
            "pool_two_target": targets.pool_two,
        },
    ).mappings()
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        pool = int(row["pool_type"])
        target = targets.target(PoolType(pool))
        task_count = int(row["task_count"])
        output[pool] = {
            "pool_id": pool,
            "pool_name": _pool_name(pool),
            "pool_label": _pool_label(pool),
            "label": _pool_label(pool),
            "pool": _pool_name(pool),
            "target": target,
            "active": False,
            "task_count": task_count,
            "candidate_count": int(row["candidate_count"]),
            "qualified_count": int(row["qualified_count"]),
            "disqualified_count": int(row["disqualified_count"]),
            "king_solved_count": 0,
            "challenger_solved_count": 0,
            "judged_rounds": 0,
            "remaining_rounds": max(0, target - task_count),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "errors": 0,
            "mean": {
                "king_score": None,
                "challenger_score": None,
                "delta": None,
                "rounds": 0,
            },
            "mean_score_delta": None,
        }
    for pool_type in PoolType:
        output.setdefault(
            int(pool_type),
            {
                "pool_id": int(pool_type),
                "pool_name": _pool_name(int(pool_type)),
                "pool_label": _pool_label(int(pool_type)),
                "label": _pool_label(int(pool_type)),
                "pool": _pool_name(int(pool_type)),
                "target": targets.target(pool_type),
                "active": False,
                "task_count": 0,
                "candidate_count": 0,
                "qualified_count": 0,
                "disqualified_count": 0,
                "king_solved_count": 0,
                "challenger_solved_count": 0,
                "judged_rounds": 0,
                "remaining_rounds": targets.target(pool_type),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "errors": 0,
                "mean": {
                    "king_score": None,
                    "challenger_score": None,
                    "delta": None,
                    "rounds": 0,
                },
                "mean_score_delta": None,
            },
        )
    return output


def _active_rounds(
    session: Session, active: dict[str, Any] | None, *, limit: int
) -> list[dict[str, Any]]:
    if active is None or limit <= 0:
        return []
    rows = session.execute(
        text(
            """
            WITH ranked_tasks AS (
                SELECT
                    t.task_id,
                    row_number() OVER (ORDER BY t.created_at, t.task_id) AS task_n
                FROM tasks t
                WHERE t.king_id = :king_id
                  AND t.pool_type = :pool_type
                  AND t.status_id <> :disqualified
                ORDER BY t.created_at, t.task_id
                LIMIT :limit
            )
            SELECT
                row_number() OVER (ORDER BY j.created_at, t.task_n, t.task_id) AS n,
                j.llm_winner,
                j.error
            FROM ranked_tasks t
            JOIN judgements j
              ON j.task_id = t.task_id
             AND j.king_submission_id = :king_id
             AND j.challenger_submission_id = :challenger_id
            ORDER BY j.created_at, t.task_n, t.task_id
            LIMIT :limit
            """
        ),
        {
            "king_id": active["king_submission_id"],
            "challenger_id": active["challenger_submission_id"],
            "pool_type": active["active_pool"],
            "disqualified": int(TaskStatus.DISQUALIFIED),
            "limit": limit,
        },
    ).mappings()
    rounds = [
        {
            "round": int(row["n"]),
            "task_name": f"result {int(row['n']):02d}",
            "winner": _public_round_winner(row["llm_winner"], row["error"]),
        }
        for row in rows
    ]
    rounds.extend(
        {
            "round": n,
            "task_name": f"result {n:02d}",
            "winner": "pending",
        }
        for n in range(len(rounds) + 1, limit + 1)
    )
    return rounds


def _completed_duel_count(session: Session) -> int:
    return int(
        session.execute(
            text(
                """
                SELECT count(*)
                FROM challenges c
                WHERE c.status = :closed
                  AND EXISTS (
                    SELECT 1
                    FROM duel_resolutions dr
                    WHERE dr.challenger_submission_id = c.challenger_submission_id
                  )
                """
            ),
            {"closed": int(ChallengeStatus.CLOSED)},
        ).scalar_one()
        or 0
    )


def _recent_completed_duels(session: Session, *, limit: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            WITH completed AS (
                SELECT
                    c.challenger_submission_id,
                    c.king_id,
                    min(c.created_at) AS started_at,
                    max(dr.created_at) AS finished_at,
                    sum(dr.challenger_wins) AS wins,
                    sum(dr.challenger_losses) AS losses,
                    sum(dr.ties) AS ties,
                    sum(dr.challenger_wins + dr.challenger_losses + dr.ties) AS round_count,
                    bool_or(dr.pool_type = :pool_two AND dr.outcome = :challenger_won) AS king_replaced,
                    (array_agg(dr.scoring_method ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS scoring_method,
                    (array_agg(dr.mean_score_margin ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS mean_score_margin,
                    (array_agg(dr.king_score_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_score_mean,
                    (array_agg(dr.challenger_score_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_score_mean,
                    (array_agg(dr.score_mean_delta ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS score_mean_delta,
                    (array_agg(dr.score_mean_rounds ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS score_mean_rounds,
                    (array_agg(dr.token_bonus_enabled ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_bonus_enabled,
                    (array_agg(dr.token_score_tolerance ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_score_tolerance,
                    (array_agg(dr.token_min_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_min_score,
                    (array_agg(dr.token_bonus_multiplier ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_bonus_multiplier,
                    CASE
                        WHEN count(*) = count(dr.king_total_tokens)
                        THEN sum(dr.king_total_tokens)
                    END AS king_total_tokens,
                    CASE
                        WHEN count(*) = count(dr.challenger_total_tokens)
                        THEN sum(dr.challenger_total_tokens)
                    END AS challenger_total_tokens,
                    sum(dr.token_comparison_rounds) AS token_comparison_rounds,
                    (array_agg(dr.king_token_savings_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_token_savings_mean,
                    (array_agg(dr.challenger_token_savings_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_token_savings_mean,
                    (array_agg(dr.king_token_boost ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_token_boost,
                    (array_agg(dr.challenger_token_boost ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_token_boost,
                    (array_agg(dr.king_combined_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_combined_score,
                    (array_agg(dr.challenger_combined_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_combined_score,
                    (array_agg(dr.combined_score_delta ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS combined_score_delta
                FROM challenges c
                JOIN duel_resolutions dr
                  ON dr.challenger_submission_id = c.challenger_submission_id
                WHERE c.status = :closed
                GROUP BY c.challenger_submission_id, c.king_id
            ),
            numbered AS (
                SELECT
                    row_number() OVER (ORDER BY finished_at, challenger_submission_id) AS duel_id,
                    completed.*
                FROM completed
            ),
            selected AS (
                SELECT *
                FROM numbered
                ORDER BY finished_at DESC, challenger_submission_id DESC
                LIMIT :limit
            )
            SELECT
                selected.*,
                ks.hotkey AS king_hotkey,
                ks.source AS king_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = ks.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS king_uid,
                cs.hotkey AS challenger_hotkey,
                cs.source AS challenger_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = cs.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS challenger_uid
            FROM selected
            JOIN submissions ks ON ks.submission_id = selected.king_id
            JOIN submissions cs ON cs.submission_id = selected.challenger_submission_id
            ORDER BY selected.finished_at, selected.challenger_submission_id
            """
        ),
        {
            "closed": int(ChallengeStatus.CLOSED),
            "pool_two": int(PoolType.POOL_TWO),
            "challenger_won": int(DuelOutcome.CHALLENGER_WON),
            "limit": limit,
        },
    ).mappings()
    duels = []
    for row in rows:
        king_repo, king_url = _public_repo(row["king_source"], row["king_id"])
        challenger_repo, challenger_url = _public_repo(
            row["challenger_source"], row["challenger_submission_id"]
        )
        public_duel_id = _public_duel_id(row["challenger_submission_id"])
        duels.append(
            {
                "duel_id": public_duel_id,
                "public_duel_id": public_duel_id,
                "db_duel_id": int(row["duel_id"]),
                "challenge_id": row["challenger_submission_id"],
                "started_at": _iso(row["started_at"]),
                "finished_at": _iso(row["finished_at"]),
                "wins": int(row["wins"] or 0),
                "losses": int(row["losses"] or 0),
                "ties": int(row["ties"] or 0),
                "round_count": int(row["round_count"] or 0),
                "king_replaced": bool(row["king_replaced"]),
                "scoring_method": row["scoring_method"],
                "mean_score_margin": _float_or_none(row["mean_score_margin"]),
                "score_mean_rounds": int(row["score_mean_rounds"] or 0),
                **_resolution_score_fields(row),
                "score_notes_url": (f"/api/dashboard/duels/{public_duel_id:06d}/notes"),
                "king_uid": row["king_uid"],
                "king_hotkey": row["king_hotkey"],
                "king_repo": king_repo,
                "king_repo_url": king_url,
                "challenger_uid": row["challenger_uid"],
                "challenger_hotkey": row["challenger_hotkey"],
                "hotkey": row["challenger_hotkey"],
                "challenger_repo": challenger_repo,
                "challenger_repo_url": challenger_url,
            }
        )
    return duels


def _duel_score_notes(session: Session, *, duel_id: int) -> dict[str, Any] | None:
    if duel_id >= 100000:
        rows = session.execute(
            text(
                """
                WITH completed AS (
                    SELECT
                        c.challenger_submission_id,
                        c.king_id,
                        max(dr.created_at) AS finished_at
                    FROM challenges c
                    JOIN duel_resolutions dr
                      ON dr.challenger_submission_id = c.challenger_submission_id
                    WHERE c.status = :closed
                    GROUP BY c.challenger_submission_id, c.king_id
                ),
                numbered AS (
                    SELECT
                        row_number() OVER (ORDER BY finished_at, challenger_submission_id) AS duel_id,
                        completed.challenger_submission_id
                    FROM completed
                )
                SELECT duel_id, challenger_submission_id
                FROM numbered
                """
            ),
            {"closed": int(ChallengeStatus.CLOSED)},
        ).mappings()
        match = next(
            (
                row
                for row in rows
                if _public_duel_id(row["challenger_submission_id"]) == duel_id
            ),
            None,
        )
        if match is None:
            return None
        duel_id = int(match["duel_id"])

    row = (
        session.execute(
            text(
                """
            WITH completed AS (
                SELECT
                    c.challenger_submission_id,
                    c.king_id,
                    min(c.created_at) AS started_at,
                    max(dr.created_at) AS finished_at,
                    sum(dr.challenger_wins) AS wins,
                    sum(dr.challenger_losses) AS losses,
                    sum(dr.ties) AS ties,
                    sum(dr.challenger_wins + dr.challenger_losses + dr.ties) AS round_count,
                    bool_or(dr.pool_type = :pool_two AND dr.outcome = :challenger_won) AS king_replaced,
                    (array_agg(dr.scoring_method ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS scoring_method,
                    (array_agg(dr.mean_score_margin ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS mean_score_margin,
                    (array_agg(dr.king_score_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_score_mean,
                    (array_agg(dr.challenger_score_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_score_mean,
                    (array_agg(dr.score_mean_delta ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS score_mean_delta,
                    (array_agg(dr.score_mean_rounds ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS score_mean_rounds,
                    (array_agg(dr.token_bonus_enabled ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_bonus_enabled,
                    (array_agg(dr.token_score_tolerance ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_score_tolerance,
                    (array_agg(dr.token_min_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_min_score,
                    (array_agg(dr.token_bonus_multiplier ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS token_bonus_multiplier,
                    CASE
                        WHEN count(*) = count(dr.king_total_tokens)
                        THEN sum(dr.king_total_tokens)
                    END AS king_total_tokens,
                    CASE
                        WHEN count(*) = count(dr.challenger_total_tokens)
                        THEN sum(dr.challenger_total_tokens)
                    END AS challenger_total_tokens,
                    sum(dr.token_comparison_rounds) AS token_comparison_rounds,
                    (array_agg(dr.king_token_savings_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_token_savings_mean,
                    (array_agg(dr.challenger_token_savings_mean ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_token_savings_mean,
                    (array_agg(dr.king_token_boost ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_token_boost,
                    (array_agg(dr.challenger_token_boost ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_token_boost,
                    (array_agg(dr.king_combined_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS king_combined_score,
                    (array_agg(dr.challenger_combined_score ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS challenger_combined_score,
                    (array_agg(dr.combined_score_delta ORDER BY dr.pool_type DESC, dr.created_at DESC))[1] AS combined_score_delta
                FROM challenges c
                JOIN duel_resolutions dr
                  ON dr.challenger_submission_id = c.challenger_submission_id
                WHERE c.status = :closed
                GROUP BY c.challenger_submission_id, c.king_id
            ),
            numbered AS (
                SELECT
                    row_number() OVER (ORDER BY finished_at, challenger_submission_id) AS duel_id,
                    completed.*
                FROM completed
            )
            SELECT
                numbered.*,
                ks.hotkey AS king_hotkey,
                ks.source AS king_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = ks.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS king_uid,
                cs.hotkey AS challenger_hotkey,
                cs.source AS challenger_source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = cs.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS challenger_uid
            FROM numbered
            JOIN submissions ks ON ks.submission_id = numbered.king_id
            JOIN submissions cs ON cs.submission_id = numbered.challenger_submission_id
            WHERE numbered.duel_id = :duel_id
            LIMIT 1
            """
            ),
            {
                "closed": int(ChallengeStatus.CLOSED),
                "pool_two": int(PoolType.POOL_TWO),
                "challenger_won": int(DuelOutcome.CHALLENGER_WON),
                "duel_id": duel_id,
            },
        )
        .mappings()
        .first()
    )
    if row is None:
        return None

    round_rows = _duel_score_round_rows(
        session,
        king_id=row["king_id"],
        challenger_id=row["challenger_submission_id"],
        targets=PoolTargets.from_env(os.environ),
    )

    rounds: list[dict[str, Any]] = []
    for round_row in round_rows:
        pool = int(round_row["pool_type"])
        king_score = _float_or_none(round_row["king_score"])
        challenger_score = _float_or_none(round_row["challenger_score"])
        delta = (
            challenger_score - king_score
            if king_score is not None and challenger_score is not None
            else None
        )
        pool_round = int(round_row["pool_round"])
        rounds.append(
            {
                "round": int(round_row["public_round"]),
                "pool_id": pool,
                "pool_name": _pool_name(pool),
                "pool_label": _pool_label(pool),
                "pool": _pool_name(pool),
                "pool_round": pool_round,
                "task_name": f"set {pool:02d} result {pool_round:02d}",
                "public_task_id": _public_task_id(round_row["task_id"]),
                "winner": _public_round_winner(
                    round_row["llm_winner"], round_row["judge_error"]
                ),
                "king_score": king_score,
                "challenger_score": challenger_score,
                "score_delta": delta,
                **_task_token_score_fields(round_row),
                "scored_at": _iso(round_row["created_at"]),
            }
        )

    king_repo, king_url = _public_repo(row["king_source"], row["king_id"])
    challenger_repo, challenger_url = _public_repo(
        row["challenger_source"], row["challenger_submission_id"]
    )
    return {
        "updated_at": _iso(_utcnow()),
        "public_note": "Score and token-efficiency duel notes. Task prompts, patches, solutions, judge rationales, model names, and raw task ids are intentionally omitted. Stable opaque public task ids are included for cross-duel comparison.",
        "duel_id": _public_duel_id(row["challenger_submission_id"]),
        "public_duel_id": _public_duel_id(row["challenger_submission_id"]),
        "db_duel_id": int(row["duel_id"]),
        "challenge_id": row["challenger_submission_id"],
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
        "wins": int(row["wins"] or 0),
        "losses": int(row["losses"] or 0),
        "ties": int(row["ties"] or 0),
        "round_count": int(row["round_count"] or 0),
        "king_replaced": bool(row["king_replaced"]),
        "scoring_method": row["scoring_method"],
        "mean_score_margin": _float_or_none(row["mean_score_margin"]),
        "score_mean_rounds": int(row["score_mean_rounds"] or 0),
        **_resolution_score_fields(row),
        "king_uid": row["king_uid"],
        "king_hotkey": row["king_hotkey"],
        "king_repo": king_repo,
        "king_repo_url": king_url,
        "challenger_uid": row["challenger_uid"],
        "challenger_hotkey": row["challenger_hotkey"],
        "hotkey": row["challenger_hotkey"],
        "challenger_repo": challenger_repo,
        "challenger_repo_url": challenger_url,
        "rounds": rounds,
    }


def _duel_score_round_rows(
    session: Session,
    *,
    king_id: str,
    challenger_id: str,
    targets: PoolTargets,
) -> list[Mapping[str, Any]]:
    return list(
        session.execute(
            text(
                """
                WITH ranked_duel_tasks AS (
                    SELECT
                        t.task_id,
                        t.pool_type,
                        j.llm_winner,
                        j.king_score,
                        j.challenger_score,
                        j.error AS judge_error,
                        j.created_at AS judged_at,
                        king_solution.usage_summary AS king_usage_summary,
                        challenger_solution.usage_summary AS challenger_usage_summary,
                        dr.token_bonus_enabled,
                        dr.token_score_tolerance,
                        dr.token_min_score,
                        dr.token_bonus_multiplier,
                        dr.best_of AS token_pool_target,
                        row_number() OVER (
                            PARTITION BY t.pool_type
                            ORDER BY t.created_at, t.task_id
                        ) AS pool_round
                    FROM tasks t
                    JOIN judgements j
                      ON j.task_id = t.task_id
                     AND j.king_submission_id = :king_id
                     AND j.challenger_submission_id = :challenger_id
                    LEFT JOIN duel_task_solutions king_solution
                      ON king_solution.task_id = t.task_id
                     AND king_solution.challenger_submission_id = :challenger_id
                     AND king_solution.submission_id = :king_id
                    LEFT JOIN duel_task_solutions challenger_solution
                      ON challenger_solution.task_id = t.task_id
                     AND challenger_solution.challenger_submission_id = :challenger_id
                     AND challenger_solution.submission_id = :challenger_id
                    LEFT JOIN duel_resolutions dr
                      ON dr.challenger_submission_id = :challenger_id
                     AND dr.pool_type = t.pool_type
                    WHERE t.king_id = :king_id
                      AND t.pool_type IN (:pool_one, :pool_two)
                ),
                public_tasks AS (
                    SELECT
                        task_id,
                        pool_type,
                        pool_round,
                        llm_winner,
                        king_score,
                        challenger_score,
                        judge_error,
                        judged_at,
                        king_usage_summary,
                        challenger_usage_summary,
                        token_bonus_enabled,
                        token_score_tolerance,
                        token_min_score,
                        token_bonus_multiplier,
                        token_pool_target,
                        CASE
                            WHEN pool_type = :pool_one THEN pool_round
                            ELSE :pool_one_target + pool_round
                        END AS public_round
                    FROM ranked_duel_tasks
                    WHERE (
                        pool_type = :pool_one
                        AND pool_round <= :pool_one_target
                    ) OR (
                        pool_type = :pool_two
                        AND pool_round <= :pool_two_target
                    )
                )
                SELECT
                    t.task_id,
                    t.public_round,
                    t.pool_type,
                    t.pool_round,
                    t.llm_winner,
                    t.king_score,
                    t.challenger_score,
                    t.judge_error IS NOT NULL AS judge_error,
                    t.king_usage_summary,
                    t.challenger_usage_summary,
                    t.token_bonus_enabled,
                    t.token_score_tolerance,
                    t.token_min_score,
                    t.token_bonus_multiplier,
                    t.token_pool_target,
                    t.judged_at AS created_at
                FROM public_tasks t
                ORDER BY t.public_round
                """
            ),
            {
                "king_id": king_id,
                "challenger_id": challenger_id,
                "pool_one": int(PoolType.POOL_ONE),
                "pool_two": int(PoolType.POOL_TWO),
                "pool_one_target": targets.pool_one,
                "pool_two_target": targets.pool_two,
            },
        ).mappings()
    )


def _duel_solution_artifact(
    session: Session,
    *,
    duel_id: int,
    round_number: int,
    solution_name: str,
    targets: PoolTargets,
) -> dict[str, Any] | None:
    if round_number < 1 or solution_name not in {"king", "challenger"}:
        return None
    challenge = _duel_challenge_identity(session, duel_id=duel_id)
    if challenge is None:
        return None

    row = (
        session.execute(
            text(
                """
            WITH ranked_duel_tasks AS (
                SELECT
                    t.task_id,
                    t.pool_type,
                    row_number() OVER (
                        PARTITION BY t.pool_type
                        ORDER BY t.created_at, t.task_id
                    ) AS pool_round
                FROM tasks t
                JOIN judgements j
                  ON j.task_id = t.task_id
                 AND j.king_submission_id = :king_id
                 AND j.challenger_submission_id = :challenger_id
                WHERE t.king_id = :king_id
                  AND t.pool_type IN (:pool_one, :pool_two)
            ),
            public_tasks AS (
                SELECT
                    task_id,
                    pool_type,
                    pool_round,
                    CASE
                        WHEN pool_type = :pool_one THEN pool_round
                        ELSE :pool_one_target + pool_round
                    END AS public_round
                FROM ranked_duel_tasks
                WHERE (
                    pool_type = :pool_one
                    AND pool_round <= :pool_one_target
                ) OR (
                    pool_type = :pool_two
                    AND pool_round <= :pool_two_target
                )
            )
            SELECT
                t.task_id,
                t.public_round,
                t.pool_type,
                t.pool_round,
                ks.solution AS king_solution,
                ks.duration AS king_duration,
                ks.exit_reason AS king_exit_reason,
                ks.usage_summary AS king_usage_summary,
                cs.solution AS challenger_solution,
                cs.duration AS challenger_duration,
                cs.exit_reason AS challenger_exit_reason,
                cs.usage_summary AS challenger_usage_summary,
                j.llm_winner,
                j.king_score,
                j.challenger_score,
                j.error AS judge_error,
                j.created_at AS judged_at
            FROM public_tasks t
            LEFT JOIN duel_task_solutions ks
              ON ks.task_id = t.task_id
             AND ks.challenger_submission_id = :challenger_id
             AND ks.submission_id = :king_id
            LEFT JOIN duel_task_solutions cs
              ON cs.task_id = t.task_id
             AND cs.challenger_submission_id = :challenger_id
             AND cs.submission_id = :challenger_id
            LEFT JOIN judgements j
              ON j.task_id = t.task_id
             AND j.king_submission_id = :king_id
             AND j.challenger_submission_id = :challenger_id
            WHERE t.public_round = :round_number
            LIMIT 1
            """
            ),
            {
                "king_id": challenge["king_id"],
                "challenger_id": challenge["challenger_submission_id"],
                "pool_one": int(PoolType.POOL_ONE),
                "pool_two": int(PoolType.POOL_TWO),
                "pool_one_target": targets.pool_one,
                "pool_two_target": targets.pool_two,
                "round_number": round_number,
            },
        )
        .mappings()
        .first()
    )
    if row is None:
        return None

    pool = int(row["pool_type"])
    pool_round = int(row["pool_round"])
    public_duel_id = _public_duel_id(challenge["challenger_submission_id"])
    solutions = {
        "king": _public_solution_summary(
            solution=row["king_solution"],
            duration=row["king_duration"],
            exit_reason=row["king_exit_reason"],
            usage_summary=row["king_usage_summary"],
        ),
        "challenger": _public_solution_summary(
            solution=row["challenger_solution"],
            duration=row["challenger_duration"],
            exit_reason=row["challenger_exit_reason"],
            usage_summary=row["challenger_usage_summary"],
        ),
    }
    selected = solutions[solution_name]
    king_score = _float_or_none(row["king_score"])
    challenger_score = _float_or_none(row["challenger_score"])
    score_delta = (
        challenger_score - king_score
        if king_score is not None and challenger_score is not None
        else None
    )
    judged_at = _iso(row["judged_at"])
    return {
        "stage": "solve",
        "solution_name": solution_name,
        "duel_id": public_duel_id,
        "public_duel_id": public_duel_id,
        "db_duel_id": int(challenge["db_duel_id"]),
        "round": int(row["public_round"]),
        "pool_id": pool,
        "pool": _pool_name(pool),
        "pool_label": _pool_label(pool),
        "pool_round": pool_round,
        "task_name": f"set {pool:02d} result {pool_round:02d}",
        "public_task_id": _public_task_id(row["task_id"]),
        "public_db_metadata": True,
        "privacy": {
            "omitted": [
                "task_id",
                "problem_statement",
                "repo_clone_url",
                "commit_sha",
                "reference_patch",
                "raw_solution_diff",
                "judge_rationale",
                "judge_model",
                "agent_source",
                "api_key",
                "upstream_url",
                "server_ip",
                "raw_rollout_bodies",
            ],
            "note": "Task content and raw patches are intentionally omitted.",
        },
        "result": selected,
        "solutions": solutions,
        "judgement": {
            "available": judged_at is not None,
            "winner": _public_round_winner(row["llm_winner"], row["judge_error"]),
            "king_score": king_score,
            "challenger_score": challenger_score,
            "score_delta": score_delta,
            "judge_error": bool(row["judge_error"]),
            "scored_at": judged_at,
        },
        "created_at": judged_at,
    }


def _duel_challenge_identity(
    session: Session, *, duel_id: int
) -> dict[str, Any] | None:
    rows = list(
        session.execute(
            text(
                """
                WITH numbered AS (
                    SELECT
                        c.challenger_submission_id,
                        c.king_id,
                        c.status,
                        c.created_at,
                        row_number() OVER (
                            ORDER BY c.created_at, c.challenger_submission_id
                        ) AS db_duel_id
                    FROM challenges c
                )
                SELECT *
                FROM numbered
                ORDER BY db_duel_id
                """
            )
        ).mappings()
    )
    if duel_id >= 100000:
        return next(
            (
                dict(row)
                for row in rows
                if _public_duel_id(row["challenger_submission_id"]) == duel_id
            ),
            None,
        )
    return next(
        (dict(row) for row in rows if int(row["db_duel_id"]) == duel_id),
        None,
    )


def _public_solution_summary(
    *,
    solution: object,
    duration: object,
    exit_reason: object,
    usage_summary: object,
) -> dict[str, Any]:
    diff = "" if solution is None else str(solution)
    changed_lines = _public_changed_lines(diff)
    available = solution is not None or duration is not None or exit_reason is not None
    return {
        "available": available,
        "success": bool(exit_reason == "completed" and changed_lines > 0)
        if available
        else None,
        "success_inferred": available,
        "exit_reason": exit_reason,
        "elapsed_seconds": _float_or_none(duration),
        "diff_available": bool(diff),
        "nonempty_diff": bool(diff.strip()),
        "changed_lines": changed_lines,
        "usage_summary": _public_usage_summary(usage_summary),
    }


def _public_usage_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    output: dict[str, Any] = {}
    for key in (
        "request_count",
        "rejected_request_count",
        "first_token_count",
        "success_count",
        "error_count",
        "upstream_error_count",
        "upstream_timeout_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        int_value = _int_or_none(key, value)
        if int_value is not None:
            output[key] = int_value
    requests = value.get("requests")
    observed_request_cost = (
        any(
            isinstance(raw, dict) and _float_or_none(raw.get("cost")) is not None
            for raw in requests
        )
        if isinstance(requests, list)
        else False
    )
    cost = _float_or_none(value.get("cost"))
    if cost is not None and (cost != 0.0 or observed_request_cost):
        output["cost"] = cost
    budget_reason = value.get("budget_exceeded_reason")
    if budget_reason is not None:
        output["budget_exceeded_reason"] = str(budget_reason)[:120]
    if isinstance(requests, list):
        output["requests"] = [
            request
            for index, raw in enumerate(requests)
            if (request := _public_usage_request(raw, index=index)) is not None
        ]
    return output or None


def _public_usage_request(raw: object, *, index: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    output: dict[str, Any] = {"index": index}
    method = raw.get("method")
    if method is not None:
        method_text = str(method).upper()
        if method_text in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            output["method"] = method_text
    for key in (
        "status_code",
        "latency_ms",
        "first_token_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        int_value = _int_or_none(key, raw)
        if int_value is not None:
            output[key] = int_value
    cost = _float_or_none(raw.get("cost"))
    if cost is not None:
        output["cost"] = cost
    rejected = raw.get("rejected")
    if rejected is not None:
        output["rejected"] = bool(rejected)
    return output


def _public_changed_lines(diff: str) -> int:
    count = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1
    return count


def _recent_kings(session: Session, *, limit: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT
                k.king_id AS submission_id,
                k.king_from,
                s.hotkey,
                s.source,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = s.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS uid
            FROM kings k
            JOIN submissions s ON s.submission_id = k.king_id
            ORDER BY k.king_from DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).mappings()
    kings = []
    for index, row in enumerate(rows):
        item = _public_submission(row, crowned_at=row["king_from"])
        item["share"] = 1.0 if index == 0 else None
        kings.append(item)
    return kings


def _leaderboard(
    session: Session, current_king: dict[str, Any] | None, *, limit: int
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            WITH participant_ids AS (
                SELECT king_id AS submission_id FROM kings
                UNION
                SELECT king_id FROM challenges
                UNION
                SELECT challenger_submission_id FROM challenges
            ),
            stats AS (
                SELECT
                    p.submission_id,
                    count(dr.challenger_submission_id) FILTER (
                        WHERE c.king_id = p.submission_id
                          AND dr.outcome = :king_won
                    ) AS defenses,
                    count(dr.challenger_submission_id) FILTER (
                        WHERE c.challenger_submission_id = p.submission_id
                          AND dr.pool_type = :pool_two
                          AND dr.outcome = :challenger_won
                    ) AS promotions,
                    count(DISTINCT c.challenger_submission_id) FILTER (
                        WHERE c.king_id = p.submission_id
                           OR c.challenger_submission_id = p.submission_id
                    ) AS duels
                FROM participant_ids p
                LEFT JOIN challenges c
                  ON c.king_id = p.submission_id
                  OR c.challenger_submission_id = p.submission_id
                LEFT JOIN duel_resolutions dr
                  ON dr.challenger_submission_id = c.challenger_submission_id
                GROUP BY p.submission_id
            )
            SELECT
                s.submission_id,
                s.hotkey,
                s.source,
                stats.defenses,
                stats.promotions,
                stats.duels,
                (
                    SELECT r.uid
                    FROM registrations r
                    WHERE r.ss58_hot = s.hotkey
                    ORDER BY r.block DESC
                    LIMIT 1
                ) AS uid
            FROM stats
            JOIN submissions s ON s.submission_id = stats.submission_id
            ORDER BY
                CASE WHEN s.submission_id = :current_king THEN 0 ELSE 1 END,
                stats.promotions DESC,
                stats.defenses DESC,
                stats.duels DESC,
                s.submission_id
            LIMIT :limit
            """
        ),
        {
            "current_king": current_king["submission_id"] if current_king else "",
            "king_won": int(DuelOutcome.KING_WON),
            "challenger_won": int(DuelOutcome.CHALLENGER_WON),
            "pool_two": int(PoolType.POOL_TWO),
            "limit": limit,
        },
    ).mappings()
    output = []
    for row in rows:
        item = _public_submission(row)
        item.update(
            {
                "defenses": int(row["defenses"] or 0),
                "promotions": int(row["promotions"] or 0),
                "duels": int(row["duels"] or 0),
                "is_current_king": bool(
                    current_king
                    and row["submission_id"] == current_king["submission_id"]
                ),
            }
        )
        output.append(item)
    return output


def _queue(session: Session, *, limit: int, now: dt.datetime) -> list[dict[str, Any]]:
    rows = list(
        session.execute(
            text(
                """
                WITH anchor AS (
                    SELECT max(block) AS block
                    FROM submissions
                ),
                current_meta AS (
                    SELECT r.uid, r.ss58_hot, r.block
                    FROM registrations r
                    JOIN (
                        SELECT uid, max(block) AS block
                        FROM registrations
                        GROUP BY uid
                    ) latest
                      ON latest.uid = r.uid
                     AND latest.block = r.block
                )
                SELECT
                    s.submission_id,
                    s.hotkey,
                    s.source,
                    s.block,
                    anchor.block AS anchor_block,
                    current_meta.uid AS uid
                FROM submissions s
                CROSS JOIN anchor
                JOIN current_meta
                  ON current_meta.ss58_hot = s.hotkey
                WHERE s.status_id IN (:unverified, :eligible)
                  AND current_meta.block <= s.block
                  AND s.submission_id NOT IN (SELECT king_id FROM kings)
                  AND s.submission_id NOT IN (SELECT challenger_submission_id FROM challenges)
                ORDER BY s.block, s.submission_id
                LIMIT :limit
                """
            ),
            {
                "unverified": int(SubmissionStatus.UNVERIFIED),
                "eligible": int(SubmissionStatus.ELIGIBLE),
                "limit": limit,
            },
        ).mappings()
    )
    return [
        _public_submission(
            row,
            submitted_at=_submission_submitted_at(
                row, anchor_block=row.get("anchor_block"), now=now
            ),
        )
        for row in rows
    ]


def _worker_freshness(
    session: Session,
    *,
    now: dt.datetime,
    fresh_seconds: float,
    active: dict[str, Any] | None,
    active_progress: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    latest_task = session.execute(
        text("SELECT max(created_at) FROM tasks")
    ).scalar_one()
    latest_registration = session.execute(
        text("SELECT max(inserted_at) FROM registrations")
    ).scalar_one()
    latest_qualification = session.execute(
        text("SELECT max(updated_at) FROM submission_qualifications")
    ).scalar_one()
    latest_judgement = session.execute(
        text("SELECT max(created_at) FROM judgements")
    ).scalar_one()
    latest_resolution = session.execute(
        text("SELECT max(created_at) FROM duel_resolutions")
    ).scalar_one()
    latest_challenge = session.execute(
        text("SELECT max(created_at) FROM challenges")
    ).scalar_one()
    judged = (active_progress or {}).get("judged_rounds", 0)
    target = (active_progress or {}).get("target", 0)
    king_solved = (active_progress or {}).get("king_solved_count", 0)
    challenger_solved = (active_progress or {}).get("challenger_solved_count", 0)
    qualified = (active_progress or {}).get("qualified_count", 0)
    needed_rounds = min(qualified, target)
    paired_solved = min(king_solved, challenger_solved)
    solution_backlog = max(0, needed_rounds * 2 - king_solved - challenger_solved)
    judge_backlog = max(0, paired_solved - judged)
    resolver_ready = bool(active and target and judged >= target)
    return {
        "subnet": _freshness_from_timestamp(
            latest_registration, now=now, fresh_seconds=fresh_seconds
        ),
        "qualification": _freshness_from_timestamp(
            latest_qualification, now=now, fresh_seconds=fresh_seconds
        ),
        "task_pool": _freshness_from_timestamp(
            latest_task, now=now, fresh_seconds=fresh_seconds
        ),
        "solutions": _freshness_from_timestamp(
            latest_task,
            now=now,
            fresh_seconds=fresh_seconds,
            delayed=solution_backlog > 0,
            detail=f"{solution_backlog} pending solution(s)"
            if solution_backlog
            else "",
        ),
        "judging": _freshness_from_timestamp(
            latest_judgement,
            now=now,
            fresh_seconds=fresh_seconds,
            delayed=judge_backlog > 0,
            detail=f"{judge_backlog} pending comparison(s)" if judge_backlog else "",
        ),
        "duel_state": _freshness_from_timestamp(
            latest_resolution or latest_challenge,
            now=now,
            fresh_seconds=fresh_seconds,
            delayed=resolver_ready,
            detail="pool target reached; transition pending" if resolver_ready else "",
        ),
    }


def _freshness_from_timestamp(
    value: dt.datetime | None,
    *,
    now: dt.datetime,
    fresh_seconds: float,
    delayed: bool = False,
    detail: str = "",
) -> dict[str, Any]:
    if value is None:
        return {
            "status": "unknown",
            "last_seen_at": None,
            "age_seconds": None,
            "detail": detail,
        }
    age = max(0.0, (now - _as_utc(value)).total_seconds())
    if delayed:
        status = "delayed"
    elif age <= fresh_seconds:
        status = "live"
    else:
        status = "quiet"
    return {
        "status": status,
        "last_seen_at": _iso(value),
        "age_seconds": round(age, 1),
        "detail": detail,
    }


def _scoring_config(env: Mapping[str, str]) -> dict[str, Any]:
    method = env_str(env, "TAU_DUEL_SCORING_METHOD", "round_wins").strip("\"'").lower()
    if method not in {"round_wins", "mean"}:
        method = "round_wins"
    margin = env_float(env, "TAU_DUEL_MEAN_SCORE_MARGIN", 0.10)
    win_margin = env_int(env, "TAU_DUEL_ROUND_WIN_MARGIN", 0)
    targets = PoolTargets.from_env(env)
    return {
        "method": method,
        "win_margin": win_margin,
        "mean_score_margin": margin,
        "duel_rounds": targets.pool_one,
        "pool_one_target": targets.pool_one,
        "pool_two_target": targets.pool_two,
    }


def _public_submission(
    row: Mapping[str, Any],
    *,
    crowned_at: dt.datetime | None = None,
    submitted_at: dt.datetime | None = None,
) -> dict[str, Any]:
    repo, repo_url = _public_repo(row.get("source"), row["submission_id"])
    item = {
        "submission_id": row["submission_id"],
        "uid": row.get("uid"),
        "hotkey": row.get("hotkey"),
        "repo": repo,
        "repo_full_name": repo,
        "repo_url": repo_url,
    }
    if crowned_at is not None:
        item["king_since"] = _iso(crowned_at)
        item["crowned_at"] = _iso(crowned_at)
    block = row.get("block")
    if block is not None:
        item["submitted_block"] = int(block)
    if submitted_at is not None:
        item["submitted_at"] = _iso(submitted_at)
    accepted_at = row.get("accepted_at")
    if accepted_at is not None:
        item["accepted_at"] = _iso(accepted_at)
    return item


def _max_block(rows: list[Mapping[str, Any]]) -> int | None:
    blocks = [int(row["block"]) for row in rows if row.get("block") is not None]
    return max(blocks) if blocks else None


def _submission_submitted_at(
    row: Mapping[str, Any], *, anchor_block: object, now: dt.datetime
) -> dt.datetime | None:
    return _submission_bundle_mtime(row.get("submission_id")) or _estimated_block_time(
        row.get("block"), anchor_block, now=now
    )


def _submission_bundle_mtime(submission_id: object) -> dt.datetime | None:
    if not submission_id:
        return None
    path = Path(env_str(os.environ, "TAU_SUBMISSIONS_DIR", "submissions")) / str(
        submission_id
    )
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
    except OSError:
        return None


def _estimated_block_time(
    block: object, anchor_block: object, *, now: dt.datetime
) -> dt.datetime | None:
    if block is None or anchor_block is None:
        return None
    try:
        block_number = int(block)
        anchor_number = int(anchor_block)
    except (TypeError, ValueError):
        return None
    blocks_ago = max(0, anchor_number - block_number)
    return now - dt.timedelta(seconds=blocks_ago * BLOCK_SECONDS)


def _public_repo(source: object, submission_id: object) -> tuple[str, str | None]:
    raw = str(source or "").strip()
    match = _GITHUB_RE.search(raw)
    if match:
        repo = _ninja_subnet_repo(match.group("repo").removesuffix(".git"))
        return repo, f"https://github.com/{repo}"
    match = _OWNER_REPO_RE.match(raw)
    if match and not raw.startswith(("/", ".")):
        repo = _ninja_subnet_repo(match.group("repo").removesuffix(".git"))
        return repo, f"https://github.com/{repo}"
    short_id = str(submission_id or "unknown")[:16]
    return f"private-submission/{short_id}", None


def _ninja_subnet_repo(repo: str) -> str:
    owner, sep, name = repo.partition("/")
    if sep and owner.lower() == "unarbos":
        name = _NINJA_SUBNET_REPO_ALIASES.get(name.lower(), name)
        return f"{_NINJA_SUBNET_OWNER}/{name}"
    return repo


def _public_round_winner(winner: object, error: object) -> str:
    if error:
        return "error"
    if winner is None:
        return "pending"
    text_value = str(winner)
    if text_value in {"king", "challenger", "tie"}:
        return text_value
    return "tie"


def _pool_name(pool: int) -> str:
    try:
        return PoolType(pool).name
    except ValueError:
        return f"POOL_{pool}"


def _pool_label(pool: int) -> str:
    labels = {
        int(PoolType.POOL_ONE): "Pool 1",
        int(PoolType.POOL_TWO): "Pool 2",
    }
    return labels.get(pool, f"Pool {pool}")


def _submission_status_name(status_id: object) -> str:
    if status_id is None:
        return "unknown"
    try:
        return SubmissionStatus(int(status_id)).name.lower()
    except (TypeError, ValueError):
        return "unknown"


def _unique_miners(*groups: Any) -> int:
    seen: set[str] = set()
    for group in groups:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if not item:
                continue
            key = item.get("hotkey") or item.get("submission_id")
            if key:
                seen.add(str(key))
    return len(seen)


def _int_or_none(key: str, row: Mapping[str, Any]) -> int | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolution_score_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Public raw quality, token modifier, merged score, and duel token totals."""
    raw_delta = _float_or_none(row.get("score_mean_delta"))
    combined_delta = _float_or_none(row.get("combined_score_delta"))
    if combined_delta is None:
        combined_delta = raw_delta
    king_quality = _float_or_none(row.get("king_score_mean"))
    challenger_quality = _float_or_none(row.get("challenger_score_mean"))
    king_boost = _float_or_none(row.get("king_token_boost"))
    challenger_boost = _float_or_none(row.get("challenger_token_boost"))
    king_combined = _float_or_none(row.get("king_combined_score"))
    challenger_combined = _float_or_none(row.get("challenger_combined_score"))
    if king_combined is None:
        king_combined = king_quality
    if challenger_combined is None:
        challenger_combined = challenger_quality
    return {
        "king_score_mean": king_quality,
        "challenger_score_mean": challenger_quality,
        "score_mean_delta": raw_delta,
        "mean_score_delta": raw_delta,
        "token_bonus_enabled": bool(row.get("token_bonus_enabled", False)),
        "token_score_tolerance": _float_or_none(row.get("token_score_tolerance")),
        "token_min_score": _float_or_none(row.get("token_min_score")),
        "token_bonus_multiplier": _float_or_none(row.get("token_bonus_multiplier")),
        "king_total_tokens": _int_or_none("king_total_tokens", row),
        "challenger_total_tokens": _int_or_none("challenger_total_tokens", row),
        "token_comparison_rounds": _int_or_none("token_comparison_rounds", row),
        "king_token_savings_mean": _float_or_none(row.get("king_token_savings_mean")),
        "challenger_token_savings_mean": _float_or_none(
            row.get("challenger_token_savings_mean")
        ),
        "king_token_boost": king_boost,
        "challenger_token_boost": challenger_boost,
        "king_combined_score": king_combined,
        "challenger_combined_score": challenger_combined,
        "combined_score_delta": combined_delta,
        # Kept for older duel-page clients; this alias now correctly means final.
        "final_mean_delta": combined_delta,
    }


def _task_token_score_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Public token saving and exact score contribution for one judged task."""
    king_tokens = _usage_total_tokens(row.get("king_usage_summary"))
    challenger_tokens = _usage_total_tokens(row.get("challenger_usage_summary"))
    enabled = bool(row.get("token_bonus_enabled", False))
    tolerance = _float_or_none(row.get("token_score_tolerance"))
    min_score = _float_or_none(row.get("token_min_score"))
    multiplier = _float_or_none(row.get("token_bonus_multiplier"))
    pool_target = _int_or_none("token_pool_target", row)
    king_score = _float_or_none(row.get("king_score"))
    challenger_score = _float_or_none(row.get("challenger_score"))
    judge_error = bool(row.get("judge_error"))

    def side(
        *,
        own_score: float | None,
        opponent_score: float | None,
        own_tokens: int | None,
        opponent_tokens: int | None,
    ) -> tuple[bool, float | None, float, str]:
        if not enabled:
            return False, 0.0, 0.0, "Token bonus disabled"
        if judge_error:
            return False, 0.0, 0.0, "Judge error"
        if (
            tolerance is None
            or min_score is None
            or multiplier is None
            or pool_target is None
            or pool_target <= 0
        ):
            return False, None, 0.0, "Token settings unavailable"
        if own_score is None or opponent_score is None:
            return False, None, 0.0, "Score unavailable"
        if own_tokens is None or opponent_tokens is None:
            return False, None, 0.0, "Token usage unavailable"
        if own_score < min_score:
            return False, 0.0, 0.0, f"Score below {min_score:.2f}"
        if own_score < opponent_score - tolerance:
            return False, 0.0, 0.0, f"More than {tolerance:.2f} behind"
        if opponent_tokens <= 0:
            return False, 0.0, 0.0, "Opponent used zero tokens"
        saving = max(0.0, 1.0 - own_tokens / opponent_tokens)
        contribution = saving / pool_target * multiplier
        reason = "Eligible" if saving > 0 else "Eligible; no token saving"
        return True, saving, contribution, reason

    king_eligible, king_saving, king_contribution, king_reason = side(
        own_score=king_score,
        opponent_score=challenger_score,
        own_tokens=king_tokens,
        opponent_tokens=challenger_tokens,
    )
    (
        challenger_eligible,
        challenger_saving,
        challenger_contribution,
        challenger_reason,
    ) = side(
        own_score=challenger_score,
        opponent_score=king_score,
        own_tokens=challenger_tokens,
        opponent_tokens=king_tokens,
    )
    return {
        "king_tokens": king_tokens,
        "challenger_tokens": challenger_tokens,
        "king_token_eligible": king_eligible,
        "challenger_token_eligible": challenger_eligible,
        "king_token_saving": king_saving,
        "challenger_token_saving": challenger_saving,
        "king_token_contribution": king_contribution,
        "challenger_token_contribution": challenger_contribution,
        "king_token_reason": king_reason,
        "challenger_token_reason": challenger_reason,
        "token_pool_target": pool_target,
        "token_bonus_multiplier": multiplier,
    }


def _usage_total_tokens(value: object) -> int | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    total = value.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        return None
    return total


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dt.datetime):
        return str(value)
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _json_default(value: object) -> str:
    if isinstance(value, dt.datetime):
        return _iso(value) or ""
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
