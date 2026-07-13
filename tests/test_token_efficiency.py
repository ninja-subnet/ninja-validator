"""Pure tests for the symmetric per-task token-efficiency modifier."""

from __future__ import annotations

import pytest

from tau.duel import (
    TokenEfficiencyConfig,
    TokenEfficiencyRound,
    calculate_token_efficiency,
)


def _stats(
    *rounds: TokenEfficiencyRound,
    target: int | None = None,
    config: TokenEfficiencyConfig | None = None,
):
    return calculate_token_efficiency(
        rounds,
        pool_target=len(rounds) if target is None else target,
        config=config or TokenEfficiencyConfig(enabled=True),
    )


def test_half_the_tokens_adds_half_a_task_saving() -> None:
    stats = _stats(TokenEfficiencyRound(0.5, 0.5, 100, 50))

    assert stats.king_savings_mean == 0
    assert stats.challenger_savings_mean == 0.5
    assert stats.king_boost == 0
    assert stats.challenger_boost == pytest.approx(0.075)
    assert stats.king_total_tokens == 100
    assert stats.challenger_total_tokens == 50


def test_savings_are_divided_by_full_pool_not_eligible_rounds() -> None:
    stats = _stats(
        TokenEfficiencyRound(0.5, 0.5, 100, 50),
        target=2,
    )

    assert stats.challenger_savings_mean == 0.25
    assert stats.challenger_boost == pytest.approx(0.0375)


@pytest.mark.parametrize(
    ("own_score", "opponent_score", "expected_saving"),
    [
        (0.20, 0.25, 0.5),  # exact quality floor and tolerance are eligible
        (0.20, 0.250001, 0.0),
        (0.199999, 0.20, 0.0),
        (0.80, 0.20, 0.5),  # the higher-quality side may earn a bonus too
    ],
)
def test_quality_eligibility_boundaries(
    own_score: float, opponent_score: float, expected_saving: float
) -> None:
    stats = _stats(TokenEfficiencyRound(opponent_score, own_score, 100, 50))
    assert stats.challenger_savings_mean == expected_saving


@pytest.mark.parametrize(
    ("own_tokens", "opponent_tokens", "expected_saving"),
    [
        (100, 100, 0.0),
        (150, 100, 0.0),
        (0, 100, 1.0),
        (50, 0, 0.0),
        (None, 100, 0.0),
    ],
)
def test_token_edge_cases(
    own_tokens: int | None,
    opponent_tokens: int | None,
    expected_saving: float,
) -> None:
    stats = _stats(TokenEfficiencyRound(0.5, 0.5, opponent_tokens, own_tokens))
    assert stats.challenger_savings_mean == expected_saving


def test_both_sides_are_calculated_independently_across_tasks() -> None:
    stats = _stats(
        TokenEfficiencyRound(0.6, 0.6, 50, 100),
        TokenEfficiencyRound(0.6, 0.6, 100, 50),
    )

    assert stats.king_savings_mean == 0.25
    assert stats.challenger_savings_mean == 0.25
    assert stats.king_boost == pytest.approx(0.0375)
    assert stats.challenger_boost == pytest.approx(0.0375)
    assert stats.token_comparison_rounds == 2


def test_disabled_modifier_keeps_totals_but_zeroes_savings_and_boosts() -> None:
    stats = _stats(
        TokenEfficiencyRound(0.5, 0.5, 100, 50),
        config=TokenEfficiencyConfig(enabled=False),
    )

    assert (stats.king_total_tokens, stats.challenger_total_tokens) == (100, 50)
    assert stats.king_savings_mean == 0
    assert stats.challenger_savings_mean == 0
    assert stats.king_boost == 0
    assert stats.challenger_boost == 0


def test_missing_scores_still_count_tokens_but_cannot_add_savings() -> None:
    stats = _stats(TokenEfficiencyRound(None, None, 100, 50))

    assert (stats.king_total_tokens, stats.challenger_total_tokens) == (100, 50)
    assert stats.token_comparison_rounds == 1
    assert stats.challenger_savings_mean == 0


def test_missing_usage_makes_only_that_sides_total_unavailable() -> None:
    stats = _stats(
        TokenEfficiencyRound(0.5, 0.5, 100, 50),
        TokenEfficiencyRound(0.5, 0.5, 75, None),
    )

    assert stats.king_total_tokens == 175
    assert stats.challenger_total_tokens is None


def test_judge_error_fallback_cannot_create_a_bonus() -> None:
    stats = _stats(
        TokenEfficiencyRound(
            0.5,
            0.5,
            100,
            50,
            judgement_valid=False,
        )
    )

    assert stats.challenger_savings_mean == 0
    assert stats.challenger_boost == 0
    assert (stats.king_total_tokens, stats.challenger_total_tokens) == (100, 50)


@pytest.mark.parametrize(
    "config",
    [
        TokenEfficiencyConfig(score_tolerance=0),
        TokenEfficiencyConfig(min_score=0),
        TokenEfficiencyConfig(bonus_multiplier=0),
    ],
)
def test_valid_zero_config_values(config: TokenEfficiencyConfig) -> None:
    assert config is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_tolerance": -0.01},
        {"score_tolerance": 1.01},
        {"min_score": -0.01},
        {"min_score": 1.01},
        {"bonus_multiplier": -0.01},
        {"bonus_multiplier": float("inf")},
        {"bonus_multiplier": float("nan")},
    ],
)
def test_invalid_config_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TokenEfficiencyConfig(**kwargs)


def test_enabled_must_be_a_boolean() -> None:
    with pytest.raises(ValueError):
        TokenEfficiencyConfig(enabled="true")  # type: ignore[arg-type]
