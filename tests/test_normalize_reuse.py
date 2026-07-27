"""The vendored normalization behaves identically to it-job-radar's — the property the
scorer relies on for a fair comparison (ADR-0002/0003)."""

from pl_jobs_lora.normalize import (
    load_tech_aliases,
    normalize_currency,
    normalize_seniority,
    normalize_technology,
    normalize_work_mode,
)


def test_tech_aliases_load_and_alias_hit():
    idx = load_tech_aliases()
    assert idx  # non-empty
    # An alias maps to its canonical form.
    assert normalize_technology("ReactJS", idx) == "react"
    assert normalize_technology("react.js", idx) == "react"


def test_tech_fuzzy_and_unknown():
    idx = load_tech_aliases()
    # Fuzzy: a near-miss still lands on the canonical name.
    assert normalize_technology("pythonn", idx) == "python"
    # Unknown tech is kept, lowercased (not dropped).
    assert normalize_technology("Elixir", idx) == "elixir"


def test_seniority_polish_quirk():
    assert normalize_seniority("regular") == "mid"
    assert normalize_seniority("Starszy") == "senior"
    assert normalize_seniority(None) is None


def test_work_mode_and_currency():
    assert normalize_work_mode("home-office") == "remote"
    assert normalize_work_mode("stationary") == "office"
    assert normalize_currency("zł") == "PLN"
    assert normalize_currency("EUR") == "EUR"
