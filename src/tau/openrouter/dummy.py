"""Token-free `LLMClient` stand-ins for testing the workers without spending tokens.

`DummyLLMClient` models latency + failures; a subclass supplies the output
(`DummyTaskClient` -> task JSON, `DummyJudgeClient` -> verdict JSON).
"""

from __future__ import annotations

import abc
import asyncio
import math
import random
from dataclasses import dataclass

from .client import RenderablePrompt

# Output with no JSON object -> the domain parser raises (TaskGenerationError /
# ValueError), exercising the worker's retry / failure path.
_UNUSABLE_OUTPUT = "Sorry, I can't help with that."


@dataclass(frozen=True, slots=True)
class DummyLLMConfig:
    """Behaviour knobs for a :class:`DummyLLMClient`.

    Latency = a log-normal body plus a slow mixture:
    - body: median ``avg_latency_seconds``, log-space spread ``latency_sigma``
      (strictly positive, right-skewed -- the expected range).
    - tail: with probability ``slow_rate`` a call is drawn in
      ``[timeout, timeout * outlier_factor]`` (the caller's per-attempt timeout
      then trips it -- the timeout-busting outliers).

    ``failure_rate`` of calls return unusable output instead of a result. ``seed``
    makes the draws reproducible (None = nondeterministic).
    """

    avg_latency_seconds: float = 2.0
    latency_sigma: float = 0.4
    slow_rate: float = 0.0
    outlier_factor: float = 2.0
    failure_rate: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.avg_latency_seconds < 0:
            raise ValueError("avg_latency_seconds must be non-negative")
        if self.latency_sigma < 0:
            raise ValueError("latency_sigma must be non-negative")
        if not 0.0 <= self.slow_rate <= 1.0:
            raise ValueError("slow_rate must be in [0, 1]")
        if self.outlier_factor < 1.0:
            raise ValueError("outlier_factor must be >= 1")
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0, 1]")


class DummyLLMClient(abc.ABC):
    """Abstract token-free ``LLMClient``: models latency + failure, subclass the output.

    Satisfies the ``LLMClient`` protocol structurally. ``complete_text`` sleeps a
    modelled latency, then with probability ``failure_rate`` returns unusable
    output, else delegates to ``_fabricate`` for the domain-specific result.
    """

    def __init__(
        self,
        *,
        model: str = "dummy/echo",
        timeout: float = 120.0,
        config: DummyLLMConfig | None = None,
    ) -> None:
        self._model = model
        # The caller's per-attempt timeout: slow outliers target it so they
        # deterministically blow past it and exercise the timeout path.
        self._timeout = timeout
        self._config = config if config is not None else DummyLLMConfig()
        self._rng = random.Random(self._config.seed)

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> DummyLLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return None

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        """Sleep a modelled latency, then return a fabricated result (or garbage)."""
        await asyncio.sleep(self._next_latency())
        if self._rng.random() < self._config.failure_rate:
            return self._unusable_output()
        return self._fabricate(prompt.as_text())

    def _next_latency(self) -> float:
        """Body latency (log-normal), or a timeout-busting outlier with prob slow_rate."""
        cfg = self._config
        if self._rng.random() < cfg.slow_rate:
            return self._rng.uniform(self._timeout, self._timeout * cfg.outlier_factor)
        if cfg.avg_latency_seconds <= 0:
            return 0.0
        return self._rng.lognormvariate(math.log(cfg.avg_latency_seconds), cfg.latency_sigma)

    @abc.abstractmethod
    def _fabricate(self, prompt_text: str) -> str:
        """Return a successful, well-formed model output for this domain."""

    def _unusable_output(self) -> str:
        """Output that fails this domain's parser (default: a no-JSON apology)."""
        return _UNUSABLE_OUTPUT
