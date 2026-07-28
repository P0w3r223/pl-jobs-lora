"""Pure cost/latency economics for the comparison report (ADR-0003).

No model, no network — just arithmetic over the token counts and per-call latencies the
baseline runner records. API variants price out at ``tokens x per-MTok list price``; local
variants (the base/LoRA/GGUF runs) carry no token counts and report no marginal cost.
"""

from __future__ import annotations


def cost_usd(
    input_tokens: int, output_tokens: int, *,
    input_usd_per_mtok: float, output_usd_per_mtok: float,
) -> float:
    """USD for one API call: input/output tokens each priced per 1M tokens."""
    return (
        input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok
    ) / 1_000_000


def percentile(values: list[float], p: float) -> float | None:
    """The ``p``-th percentile (0..100) by linear interpolation; ``None`` on empty input."""
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100) * (len(xs) - 1)
    lo = int(rank)
    if lo + 1 >= len(xs):
        return xs[-1]
    return xs[lo] + (rank - lo) * (xs[lo + 1] - xs[lo])
