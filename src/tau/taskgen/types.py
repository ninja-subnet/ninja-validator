"""The generated task value object.

A ``GeneratedTask`` is the LLM's natural-language rendering of a mined commit:
title + description + acceptance criteria. ``prompt_text`` is the single string
the task-generator stores as ``tasks.problem_statement`` and the solver shows the
agent. Clean-room port of the dataclass in the monolith's ``task_generation.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GeneratedTask:
    title: str
    description: str
    acceptance_criteria: list[str]
    # Provenance / diagnostics — the raw LLM output and how long the call took.
    raw_output: str = ""
    elapsed_seconds: float = 0.0

    @property
    def prompt_text(self) -> str:
        """The task as a single problem statement (what the solver agent sees)."""
        criteria = "\n".join(f"- {item}" for item in self.acceptance_criteria)
        return (
            f"{self.title}\n\n"
            f"{self.description.strip()}\n\n"
            f"Acceptance criteria:\n{criteria}"
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": list(self.acceptance_criteria),
            "raw_output": self.raw_output,
            "elapsed_seconds": self.elapsed_seconds,
            "prompt_text": self.prompt_text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedTask:
        return cls(
            title=str(payload.get("title") or ""),
            description=str(payload.get("description") or ""),
            acceptance_criteria=[
                str(item).strip()
                for item in (payload.get("acceptance_criteria") or [])
                if str(item).strip()
            ],
            raw_output=str(payload.get("raw_output") or ""),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
        )
