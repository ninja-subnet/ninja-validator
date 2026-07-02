"""Exceptions for task generation."""

from __future__ import annotations


class TaskGenerationError(Exception):
    """The model did not return a usable task (no JSON object, or empty description).

    Generation runs over an abundance of mined commits, so a failed call should be
    a *skip*: the worker retries the LLM or moves to the next commit rather than
    storing a content-free task that would pollute the pool and the duel signal.
    """
