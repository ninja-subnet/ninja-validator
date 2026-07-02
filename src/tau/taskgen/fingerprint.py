"""Stable content identity for a mined commit — the task dedup key.

``content_fingerprint`` is a sha256 over the commit sha, parent sha, and a hash
of the reference patch, so the same upstream commit yields the same fingerprint
regardless of when, by which worker, or — unlike the monolith's
``task_content_fingerprint`` — from which repo it was mined. Dropping the repo
collapses one commit mined from many forks into a single task. Stored in
``tasks.content_fingerprint``; drives the ``ON CONFLICT DO NOTHING`` dedup in
:meth:`tau.db.generator.GeneratorDb.insert_task_candidate`.
"""

from __future__ import annotations

import hashlib
import json

from tau.github import CommitCandidate


def content_fingerprint(candidate: CommitCandidate) -> str:
    """Return the stable dedup fingerprint for *candidate*.

    Independent of ``repo_full_name``: a git commit sha is content-addressed, so
    the same sha across forks is the same change and must dedupe to one task. The
    patch component hashes ``combined_patch`` (the stored ``reference_patch``).
    """
    patch_sha = hashlib.sha256(candidate.combined_patch.encode("utf-8")).hexdigest()
    payload = {
        "commit_sha": candidate.commit_sha.strip().lower(),
        "parent_sha": candidate.parent_sha.strip().lower(),
        "patch_sha256": patch_sha,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
