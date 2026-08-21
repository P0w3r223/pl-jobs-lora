"""Data ceiling (ADR-0003): is a gold label present in the prose the model is given at all?

Pure and offline — synthetic postings, no model, no network.
"""

from __future__ import annotations

from pl_jobs_lora.eval.ceiling import (
    CEILING_FIELDS,
    _compile,
    _surface_forms,
    answerable_ceilings,
    render_markdown,
    term_in_prose,
)

_ALIASES = {"js": "javascript", "javascript": "javascript", "reactjs": "react", "react": "react"}


def _rec(prose, expected=(), optional=()):
    return {"prose": prose, "gold": {"tech_expected": list(expected),
                                     "tech_optional": list(optional)}}


def test_a_term_counts_when_any_known_spelling_is_in_the_prose():
    patterns = _compile(_surface_forms(_ALIASES))
    assert term_in_prose("javascript", "Znajomość JS mile widziana", patterns)
    assert term_in_prose("react", "Pracujemy w ReactJS", patterns)
    assert not term_in_prose("react", "Pracujemy w Angularze", patterns)


def test_matching_respects_word_boundaries():
    """Substring matching would count 'js' inside 'jsonschema' and inflate every ceiling."""
    patterns = _compile(_surface_forms(_ALIASES))
    assert not term_in_prose("javascript", "Piszemy jsonschema", patterns)


def test_an_unmapped_term_falls_back_to_its_canonical_form():
    patterns = _compile(_surface_forms(_ALIASES))
    assert term_in_prose("kubernetes", "Wdrożenia na Kubernetes", patterns)
    assert not term_in_prose("kubernetes", "Wdrożenia na k8s", patterns), (
        "no alias entry means an unusual spelling reads as absent — conservative by design"
    )


def test_ceiling_is_the_share_of_gold_terms_present_in_prose():
    records = [
        _rec("Szukamy kogoś do React i JS", expected=["react", "javascript"]),
        _rec("Szukamy kogoś do React", expected=["react", "kubernetes"]),
    ]
    c = answerable_ceilings(records, _ALIASES)["tech_expected"]
    assert c.terms == 4 and c.present == 3
    assert c.share == 0.75
    assert c.records_with_gold == 2 and c.records_unanswerable == 0


def test_a_posting_whose_gold_is_nowhere_in_its_prose_is_unanswerable():
    records = [
        _rec("Opis stanowiska bez technologii", expected=["react"]),
        _rec("Pracujemy w React", expected=["react"]),
    ]
    c = answerable_ceilings(records, _ALIASES)["tech_expected"]
    assert c.records_unanswerable == 1 and c.unanswerable_share == 0.5


def test_a_field_with_no_gold_anywhere_is_unmeasurable_not_zero():
    c = answerable_ceilings([_rec("Cokolwiek", expected=["react"])], _ALIASES)["tech_optional"]
    assert c.terms == 0 and c.share is None and c.unanswerable_share is None


def test_only_open_vocabulary_fields_get_a_ceiling():
    """seniority/work_mode are expressed in free Polish; a token search there measures nothing."""
    assert CEILING_FIELDS == ("tech_expected", "tech_optional")
    ceilings = answerable_ceilings([_rec("x", expected=["react"])], _ALIASES)
    assert set(ceilings) == set(CEILING_FIELDS)


def test_render_reports_achieved_recall_against_the_ceiling():
    records = [_rec("React i JS", expected=["react", "javascript"]),
               _rec("Nic", expected=["kubernetes"])]
    ceilings = answerable_ceilings(records, _ALIASES)
    md = render_markdown(ceilings, {"tech_expected": 0.33, "tech_optional": None})
    assert "## Data ceiling (model-free)" in md
    assert "tech_expected" in md
    assert "% of what the input makes recoverable" in md


def test_render_omits_the_achieved_line_when_no_recall_was_measured():
    ceilings = answerable_ceilings([_rec("React", expected=["react"])], _ALIASES)
    md = render_markdown(ceilings, {"tech_expected": None, "tech_optional": None})
    assert "Best recall achieved" not in md
