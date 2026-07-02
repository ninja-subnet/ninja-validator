"""Token-free dummy LLM for the judge: fabricates judge-verdict JSON."""

from __future__ import annotations

import json

from tau.openrouter.dummy import DummyLLMClient

_TIE_MARGIN = 2


class DummyJudgeClient(DummyLLMClient):
    """Fabricates verdict JSON (random winner + two 0-100 scores) for `parse_verdict`."""

    def _fabricate(self, prompt_text: str) -> str:

        score_a = self._rng.randint(0, 100)
        score_b = self._rng.randint(0, 100)

        if score_a > score_b + _TIE_MARGIN:
            winner = "candidate_a"
        elif score_b > score_a + _TIE_MARGIN:
            winner = "candidate_b"
        else:
            winner = "tie"
        payload = {
            "winner": winner,
            "candidate_a_score": score_a,
            "candidate_b_score": score_b,
            "rationale": "Fabricated by DummyJudgeClient for testing; no real model was called.",
        }
        return json.dumps(payload)
