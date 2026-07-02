"""Exceptions for the GitHub commit source.

Three distinct failure modes, kept separate so the sampler's control flow stays
legible:

- ``GitHubRequestError`` — a request failed at the transport level (HTTP error
  after the rotator/fallback gave up, or a ``gh`` CLI failure). Raised by
  :class:`tau.github.client.GitHubClient`.
- ``CommitRejected`` — a fetched commit is not usable as a task source
  (structural reason like a merge commit, or a quality-gate reason). Caught
  inside the sampling loop; never escapes ``sample_commit``.
- ``CommitSampleError`` — no usable commit was found within the attempt budget.
  Terminal: raised out of ``sample_commit``. Never raised directly; the sampler
  raises one of its two subtypes so the caller dispatches on type alone:
  ``NoCommitMetCriteria`` (benign quality churn — retry promptly) or
  ``CommitSourceUnavailable`` (barren / rate-limited source — back off).
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum


class RejectReason(StrEnum):
    """Why a sampled commit was discarded (sampling-loop telemetry)."""

    DUPLICATE = "duplicate"  # already screened out this run (reject cache)
    STRUCTURAL = "structural"  # no parent / merge / no changed files
    QUALITY = "quality"  # fails the quality gate (too small / not code)
    FETCH_ERROR = "fetch_error"  # could not be fetched or parsed (transport)


class GitHubRequestError(RuntimeError):
    """A GitHub request failed at the transport level."""


class CommitRejected(Exception):
    """A fetched commit cannot be used as a task source."""

    def __init__(self, message: str, *, reason: RejectReason) -> None:
        super().__init__(message)
        self.reason = reason


class CommitSampleError(RuntimeError):
    """No usable commit could be sampled within the attempt budget.

    Base type; the sampler raises one of the two subtypes below. ``rejections`` is
    the per-reason tally for the exhausted round (observability).
    """

    def __init__(
        self, message: str, *, rejections: Counter[RejectReason] | None = None
    ) -> None:
        super().__init__(message)
        self.rejections: Counter[RejectReason] = rejections if rejections is not None else Counter()


class NoCommitMetCriteria(CommitSampleError):
    """Every candidate this round was screened out on quality/structure.

    Benign churn given a strict quality bar -- the caller should retry promptly,
    not back off.
    """


class CommitSourceUnavailable(CommitSampleError):
    """The commit source was empty or rate-limited (or fetches kept erroring).

    The caller should back off before retrying so it does not hammer GitHub.
    """
