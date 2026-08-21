"""Pure agreement layer: pairwise, kappa chance-correction, triangulation buckets."""

from __future__ import annotations

from pl_jobs_lora.dataset.agreement import (
    BUCKETS,
    AgreementReport,
    cohen_kappa,
    disagreeing_ids,
    field_disagreements,
    pairwise,
    triangulate,
)

_TOL = 0.05


def _rec(offer_id, **fields):
    base = {
        "offer_id": offer_id, "title": "", "seniority": [], "work_mode": [],
        "tech_expected": [], "tech_optional": [], "salary": None,
    }
    return {**base, **fields}


def test_pairwise_perfect_agreement_is_one():
    a = [_rec("x", seniority=["mid"], work_mode=["remote"], tech_expected=["python"], title="Dev")]
    b = [dict(a[0])]
    r = pairwise(a, b, salary_rel_tolerance=_TOL).as_dict()
    assert r["json_validity"] == 1.0
    assert r["title_agreement"] == 1.0
    for f in ("seniority", "work_mode", "tech_expected"):
        assert r["per_field"][f]["agreement"] == 1.0
        assert r["per_field"][f]["f1"] == 1.0
    assert r["per_field"]["seniority"]["kappa"] == 1.0


def test_pairwise_partial_disagreement():
    a = [_rec("1", seniority=["mid"]), _rec("2", seniority=["senior"])]
    b = [_rec("1", seniority=["mid"]), _rec("2", seniority=["junior"])]  # id 2 differs
    r = pairwise(a, b, salary_rel_tolerance=_TOL).as_dict()
    assert r["per_field"]["seniority"]["agreement"] == 0.5
    assert r["per_field"]["seniority"]["f1"] < 1.0
    assert r["per_field"]["seniority"]["kappa"] < 1.0


def test_cohen_kappa_chance_correction():
    # 75% raw agreement, but one category dominates → chance-corrected kappa is 0.
    a = ["A", "A", "B", "A"]
    b = ["A", "A", "A", "A"]
    assert abs(cohen_kappa(a, b)) < 1e-9


def test_cohen_kappa_perfect_is_one():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0


def test_triangulate_buckets():
    # seniority is engineered to land one record in each bucket; other fields are empty == empty.
    llm = [_rec("agree", seniority=["mid"]), _rec("gap", seniority=["senior"]),
           _rec("llmerr", seniority=["junior"]), _rec("differ", seniority=["junior"])]
    platform = [_rec("agree", seniority=["mid"]), _rec("gap", seniority=["mid"]),
                _rec("llmerr", seniority=["mid"]), _rec("differ", seniority=["mid"])]
    human = [_rec("agree", seniority=["mid"]), _rec("gap", seniority=["senior"]),
             _rec("llmerr", seniority=["mid"]), _rec("differ", seniority=["senior"])]
    t = triangulate(llm, platform, human, salary_rel_tolerance=_TOL).as_dict()
    assert t["n"] == 4
    sen = t["per_field"]["seniority"]
    assert sen == {"all_agree": 1, "platform_gap_caught": 1, "llm_error": 1, "all_differ": 1}
    # empty-on-all-sides fields are vacuous agreement, so totals still reflect seniority.
    assert t["totals"]["platform_gap_caught"] == 1
    assert t["totals"]["llm_error"] == 1
    assert t["totals"]["all_differ"] == 1


def test_triangulate_only_over_human_ids():
    llm = [_rec("a", seniority=["mid"]), _rec("b", seniority=["mid"])]
    platform = [_rec("a", seniority=["mid"]), _rec("b", seniority=["mid"])]
    human = [_rec("a", seniority=["mid"])]  # only "a" was human-checked
    t = triangulate(llm, platform, human, salary_rel_tolerance=_TOL)
    assert t.n == 1  # "b" is excluded — no human reference


def test_field_disagreements_flags_only_differing_fields():
    a = _rec("1", seniority=["mid"], work_mode=["remote"], title="Dev")
    b = _rec("1", seniority=["senior"], work_mode=["remote"], title="Dev")
    flags = field_disagreements(a, b, salary_rel_tolerance=_TOL)
    assert flags["seniority"] is True
    assert flags["work_mode"] is False
    assert flags["title"] is False


def test_disagreeing_ids_matched_only():
    a = [_rec("1", seniority=["mid"]), _rec("2", seniority=["mid"]), _rec("3", seniority=["mid"])]
    b = [_rec("1", seniority=["mid"]), _rec("2", seniority=["senior"])]  # id 3 absent from b
    ids = disagreeing_ids(a, b, salary_rel_tolerance=_TOL)
    assert ids == ["2"]  # id 1 agrees, id 3 has no gold to compare


def test_report_shapes_and_render():
    a = [_rec("x", seniority=["mid"], title="Dev")]
    b = [_rec("x", seniority=["senior"], title="Dev")]
    report = AgreementReport(
        metadata={"proposer": "bielik-1.5b", "arbiter": "claude-opus-4-8"},
        llm_vs_platform=pairwise(a, b, salary_rel_tolerance=_TOL),
        llm_vs_human=pairwise(a, a, salary_rel_tolerance=_TOL),
        platform_vs_human=pairwise(b, a, salary_rel_tolerance=_TOL),
        triangulation=triangulate(a, b, a, salary_rel_tolerance=_TOL),
    )
    d = report.as_dict()
    assert set(d) == {
        "metadata", "llm_vs_platform", "llm_vs_human", "platform_vs_human", "triangulation",
    }
    md = report.render_markdown()
    assert "LLM ↔ platform" in md and "Triangulation" in md
    for b_name in BUCKETS:
        assert b_name in md


def test_render_survives_a_pair_with_no_title():
    """`title_exact` is None when neither leg carries a title — the renderer must not `round()` it.

    The regression: widening ``title_exact`` to ``float | None`` guarded ``as_dict`` but not the
    markdown renderer, so any title-less pair raised ``TypeError``. The existing tests all pass
    ``title="Dev"``, which is exactly why this went unnoticed.
    """
    a = [_rec("x", seniority=["mid"])]  # title defaults to "" -> no support
    r = pairwise(a, [dict(a[0])], salary_rel_tolerance=_TOL)
    assert r.title_exact is None
    report = AgreementReport(metadata={}, llm_vs_platform=r)
    md = report.render_markdown()          # must not raise
    assert "exact = -" in md


def test_unmeasurable_metrics_render_as_dash_never_as_none():
    """`-` means "unmeasurable here"; a literal `None` in a published table reads as a bug."""
    a = [_rec("x", seniority=["mid"])]     # no tech, no salary, no title on either leg
    pair = pairwise(a, [dict(a[0])], salary_rel_tolerance=_TOL)
    md = AgreementReport(metadata={}, llm_vs_platform=pair).render_markdown()
    assert "None" not in md
    assert "salary (detect; cur/kind/amt)" in md
