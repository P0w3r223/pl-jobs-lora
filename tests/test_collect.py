"""Collector parsing (pure): prose extraction, leakage guard, platform-gold normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from pl_jobs_lora.config import load_config
from pl_jobs_lora.dataset.collect import (
    _collect_line,
    _fetch_examples,
    _spread_sample,
    build_gold,
    build_prose,
    extract_next_data,
    parse_offer_page,
)
from pl_jobs_lora.normalize import load_tech_aliases

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "offer_sample.html"


@pytest.fixture
def offer() -> dict:
    data = extract_next_data(_FIXTURE.read_text(encoding="utf-8"))
    return data["props"]["pageProps"]["offer"]


class _Resp:
    """The two things `_fetch_examples` asks of a response, and nothing else."""

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Session:
    """A session that answers each URL from a script: HTML, or an exception to raise."""

    def __init__(self, answers: dict):
        self.answers = answers

    def get(self, url, timeout=None):
        answer = self.answers[url]
        if isinstance(answer, Exception):
            raise answer
        return _Resp(answer)


def test_the_sampler_spreads_and_never_divides_by_zero():
    """`_spread_sample`'s two branches, which are the arithmetic a sampler gets wrong.

    Never called by the suite until this test: the whole network half of the collector is
    untouched by `tests/test_collect.py`, which enters through `parse_offer_page` over the
    synthetic fixture. This one is pure and needs nothing.
    """
    urls = [f"u{i}" for i in range(10)]
    assert _spread_sample(urls, 20) == urls          # n >= len -> everything, unsampled
    assert _spread_sample(urls, 10) == urls
    assert len(_spread_sample(urls, 3)) == 3         # spread, not the first three
    assert _spread_sample(urls, 3) == ["u0", "u3", "u6"]
    assert _spread_sample(urls, 9)[:2] == ["u0", "u1"]   # step floors to 1, then truncates


def test_a_dead_url_and_an_unusable_page_are_counted_apart(monkeypatch):
    """The collector's drops, named. `good-practices.md` §3: never swallow without recording.

    The handler is right to drop a bad URL — a run of 800 cannot stop for one — but the run
    used to emit a single number covering a failed request, a page the parser refused, and
    nothing else. Two different facts about the collection, folded into one, and the README's
    `800 -> 710` line then attributed the whole difference to deduplication.
    """
    monkeypatch.setattr("pl_jobs_lora.dataset.collect.time.sleep", lambda _s: None)
    good = _FIXTURE.read_text(encoding="utf-8")
    session = _Session({
        "ok": good,
        "dead": requests.ConnectionError("no route"),
        "empty": "<html><body>no next data</body></html>",
    })
    cfg = load_config()
    out, stats = _fetch_examples(cfg, session, ["ok", "dead", "empty"], {})

    assert len(out) == 1
    assert stats == {"requested": 3, "usable": 1, "fetch_failed": 1, "unusable_page": 1}
    assert "1 fetch failed" in _collect_line(stats)
    assert "1 unusable page" in _collect_line(stats)


def test_prose_has_titled_sections(offer):
    prose = build_prose(offer)
    for title in ("O projekcie", "Twój zakres obowiązków", "Nasze wymagania"):
        assert title in prose
    assert "pipeline" in prose      # jsonSections.bullets
    assert "3+ lata" in prose       # jsonSections.bullets


def test_prose_excludes_tech_widget_and_textsections(offer):
    prose = build_prose(offer)
    # "Technologies we use" jsonSection + the leaky textSection echo are both dropped.
    assert "PostgreSQL" not in prose
    assert "Expected" not in prose
    assert "redundant flattened mirror" not in prose


def test_gold_is_normalized(offer):
    gold = build_gold(offer, load_tech_aliases())
    assert gold["title"] == "Senior Python Developer"
    assert gold["seniority"] == ["senior", "mid"]        # regular -> mid
    assert gold["work_mode"] == ["remote", "hybrid"]     # home-office -> remote
    assert gold["tech_expected"] == ["python", "postgresql"]
    assert gold["tech_optional"] == ["react"]            # ReactJS -> react
    assert gold["salary"] == {
        "kind": "b2b", "currency": "PLN",
        "amount_from": 100, "amount_to": 140, "period": "hour",
    }


def test_parse_offer_page_end_to_end(offer):
    ex = parse_offer_page(_FIXTURE.read_text(encoding="utf-8"), load_tech_aliases())
    assert ex.offer_id == "fixture-001"
    assert ex.pub_date == "2026-07-13T11:32:56Z"
    assert ex.gold["tech_expected"] == ["python", "postgresql"]


def test_missing_prose_is_skipped():
    html = '<script id="__NEXT_DATA__">{"props":{"pageProps":{"offer":{"id":"x"}}}}</script>'
    assert parse_offer_page(html, {}) is None
