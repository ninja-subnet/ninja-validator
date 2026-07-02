"""Generate a task description from a mined commit via an LLM.

Thin async orchestration: render the prompt, call the injected ``LLMClient``
(reusing ``tau.openrouter``), time it, and parse the result. All prompt building
and parsing live in ``prompt.py``, so this module stays a one-call wrapper. The
legacy ``run_claude`` subprocess path is dropped: the redesign only talks to
OpenRouter through the injected client.
"""

from __future__ import annotations

import logging
import time

from tau.github import CommitCandidate
from tau.openrouter import LLMClient, TextPrompt

from .prompt import SYSTEM_PROMPT, build_generation_prompt, parse_generated_task
from .types import GeneratedTask

log = logging.getLogger(__name__)


async def generate_task_description(
    *,
    candidate: CommitCandidate,
    client: LLMClient,
) -> GeneratedTask:
    """Ask *client* to turn *candidate* into a task.

    Raises ``TaskGenerationError`` when the model output isn't a usable task (no
    JSON / empty description) and lets client transport errors propagate — in both
    cases the worker decides whether to retry the LLM or skip to another commit.
    """
    prompt = TextPrompt(build_generation_prompt(candidate), system=SYSTEM_PROMPT)
    start = time.monotonic()
    raw_output = await client.complete_text(prompt)
    elapsed = time.monotonic() - start
    task = parse_generated_task(
        candidate=candidate, raw_output=raw_output, elapsed_seconds=elapsed
    )
    log.debug(
        "Generated task %r from %s@%s in %.2fs",
        task.title,
        candidate.repo_full_name,
        candidate.short_sha,
        elapsed,
    )
    return task
