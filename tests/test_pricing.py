"""Pure cost/latency economics (ADR-0003): token pricing + percentile interpolation."""

from __future__ import annotations

import pytest

from pl_jobs_lora.eval.pricing import cost_usd, percentile


def test_cost_usd_prices_tokens_per_million():
    # 1M input @ $1 + 1M output @ $5 = $6
    assert cost_usd(1_000_000, 1_000_000, input_usd_per_mtok=1.0, output_usd_per_mtok=5.0) == 6.0
    assert cost_usd(0, 0, input_usd_per_mtok=1.0, output_usd_per_mtok=5.0) == 0.0


def test_cost_usd_scales_linearly():
    a = cost_usd(500, 200, input_usd_per_mtok=1.0, output_usd_per_mtok=5.0)
    b = cost_usd(1000, 400, input_usd_per_mtok=1.0, output_usd_per_mtok=5.0)
    assert b == pytest.approx(2 * a)


def test_percentile_empty_and_singleton():
    assert percentile([], 50) is None
    assert percentile([4.2], 95) == 4.2


def test_percentile_interpolates():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 0.0
    assert percentile(xs, 50) == 2.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 95) == pytest.approx(3.8)  # rank = 0.95*4 = 3.8 -> 3 + 0.8*(4-3)
