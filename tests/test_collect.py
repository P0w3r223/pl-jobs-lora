"""Collector parsing (pure): prose extraction, leakage guard, platform-gold normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from pl_jobs_lora.dataset.collect import (
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
