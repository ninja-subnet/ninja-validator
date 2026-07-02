"""Token-free dummy LLM for the task-generator: fabricates task-description JSON."""

from __future__ import annotations

import json
import re

from tau.openrouter.dummy import DummyLLMClient, DummyLLMConfig

# Lift the commit subject out of build_generation_prompt's output for a readable
# fabricated title; best-effort, falls back to a counter. `.+` stops at the newline
# (re.MULTILINE), so a multi-line message yields just its first line.
_COMMIT_MESSAGE_RE = re.compile(r"^Commit message: (.+)$", re.MULTILINE)


class DummyTaskClient(DummyLLMClient):
    """Fabricates task-description JSON matching `tau.taskgen`'s parser contract."""

    def __init__(
        self,
        *,
        model: str = "dummy/echo",
        timeout: float = 120.0,
        config: DummyLLMConfig | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, config=config)
        self._counter = 0

    def _fabricate(self, prompt_text: str) -> str:
        self._counter += 1
        match = _COMMIT_MESSAGE_RE.search(prompt_text)
        subject = match.group(1).strip() if match else ""
        title = subject[:80] or f"Dummy task {self._counter}"
        payload = {
            "title": title,
            "description": (
                f"Reproduce the behavior described by this change: {title}. "
                "Implement it so the affected area works as intended. "
                "(Fabricated by DummyTaskClient for testing; no real model was called.)"
            ),
            "acceptance_criteria": [
                "The intended behavior change is implemented.",
                "Behavior in unaffected areas is preserved.",
            ],
        }
        return json.dumps(payload)
