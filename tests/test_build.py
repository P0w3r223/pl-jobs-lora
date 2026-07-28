"""Dataset record building: usability filters + repost dedupe (pure, no network)."""

from __future__ import annotations

from pl_jobs_lora.dataset.build import build_dataset, build_record, dedupe
from pl_jobs_lora.dataset.collect import DevExample

_LONG = "Szukamy programisty. " * 20  # comfortably over the 200-char floor


def _ex(offer_id: str, *, prose: str | None = None, pub_date: str | None = "2026-07-01T00:00:00Z",
        gold: dict | None = None) -> DevExample:
    # Default prose is unique per offer id so distinct offers aren't dedup'd by prose hash.
    return DevExample(
        offer_id=offer_id, url=f"https://x/{offer_id}",
        pub_date=pub_date, prose=prose if prose is not None else f"{_LONG}{offer_id}",
        gold=gold if gold is not None else {"title": "Dev"},
    )


def test_build_record_keeps_usable_offer():
    rec = build_record(_ex("a"), min_prose_chars=200)
    assert rec == {
        "offer_id": "a", "url": "https://x/a", "pub_date": "2026-07-01T00:00:00Z",
        "prose": f"{_LONG}a", "gold": {"title": "Dev"},
    }


def test_build_record_drops_short_prose():
    assert build_record(_ex("a", prose="too short"), min_prose_chars=200) is None


def test_build_record_drops_missing_pub_date():
    assert build_record(_ex("a", pub_date=None), min_prose_chars=200) is None


def test_build_record_drops_label_empty_offer():
    # Parsed fine but every gold field is empty -> no learnable signal.
    empty = {"title": None, "seniority": [], "work_mode": [], "tech_expected": [],
             "tech_optional": [], "salary": None}
    assert build_record(_ex("a", gold=empty), min_prose_chars=200) is None


def test_build_record_keeps_offer_with_only_tech_signal():
    gold = {"title": None, "seniority": [], "tech_expected": ["python"], "salary": None}
    assert build_record(_ex("a", gold=gold), min_prose_chars=200) is not None


def test_build_record_folds_unicode_line_separators():
    # U+2028 / U+2029 / U+0085 embedded in prose become \n so the frozen text stays clean.
    dirty = _LONG + "\u2028end\u2029x\u0085y"
    rec = build_record(_ex("a", prose=dirty), min_prose_chars=200)
    assert not any(sep in rec["prose"] for sep in ("\u2028", "\u2029", "\u0085"))
    assert rec["prose"].endswith("\nend\nx\ny")


def test_dedupe_by_offer_id():
    recs = [build_record(_ex("a"), min_prose_chars=200) for _ in range(2)]
    assert [r["offer_id"] for r in dedupe(recs)] == ["a"]


def test_dedupe_by_prose_hash_across_ids():
    # Same posting reposted under a new id (whitespace/case differ) -> one record.
    a = build_record(_ex("a", prose=_LONG), min_prose_chars=200)
    b = build_record(_ex("b", prose="  " + _LONG.upper() + "  "), min_prose_chars=200)
    assert len(dedupe([a, b])) == 1


def test_build_dataset_stats():
    examples = [
        _ex("a"), _ex("a"),                       # duplicate id
        _ex("b", prose="short"),                  # filtered (prose)
        _ex("c", pub_date=None),                  # filtered (no date)
        _ex("d"),
    ]
    records, stats = build_dataset(examples, min_prose_chars=200)
    assert {r["offer_id"] for r in records} == {"a", "d"}
    assert stats == {
        "collected": 5, "passed_filters": 3, "dropped_filters": 2,
        "dropped_duplicates": 1, "records": 2,
    }
