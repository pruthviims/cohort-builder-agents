"""IQVIA LAAD-style claims profile: semantic views, claims logic, capability checks, cross-dataset runs."""
from __future__ import annotations

import copy
from collections import defaultdict
from datetime import date

import pytest
from fake_llm import LAAD_SGLT2_QUERY, T2DM_QUERY, FakeLLM

from cohort_builder.agents.validator import validate
from cohort_builder.config import REPO_ROOT
from cohort_builder.evaluation import run_eval
from cohort_builder.ir import CohortDefinition
from cohort_builder.orchestrator import CohortBuilder

GOLD = REPO_ROOT / "eval" / "gold"


def gold(name: str) -> CohortDefinition:
    return CohortDefinition.model_validate_json((GOLD / f"{name}.json").read_text())


# ---- semantic layer ---------------------------------------------------------------
def test_semantic_views_map_codes_and_statuses(laad_builder):
    q = lambda sql: laad_builder.con.execute(sql).fetchall()  # noqa: E731
    statuses = dict(q("SELECT claim_status, count(*) FROM sem.drug_exposure GROUP BY 1"))
    assert set(statuses) == {"paid", "rejected", "reversed"}
    # reversal transactions are not exposures themselves
    n_rev_rows = q("SELECT count(*) FROM laad.rx_claims WHERE txn_type = 'REVERSAL'")[0][0]
    assert n_rev_rows > 0 and statuses["reversed"] == n_rev_rows
    # dot-less ICD-10-CM codes map to standard SNOMED concepts
    assert q("SELECT DISTINCT condition_concept_id FROM sem.condition_occurrence WHERE source_value='E119'") \
        == [(201826,)]
    # NDCs map to RxNorm products, which roll up to ingredients
    assert q("SELECT DISTINCT drug_concept_id FROM sem.drug_exposure WHERE source_value='99999020101'") \
        == [(2000005004,)]
    # unpivoted diagnosis positions
    assert {r[0] for r in q("SELECT DISTINCT dx_position FROM sem.condition_occurrence")} >= {1, 2}


def test_activity_based_observation_splits_on_gaps(laad_builder):
    q = lambda sql: laad_builder.con.execute(sql).fetchall()  # noqa: E731
    multi = q("SELECT person_id FROM sem.observation_period GROUP BY 1 HAVING count(*) > 1 LIMIT 1")
    assert multi
    periods = q(f"SELECT observation_period_start_date, observation_period_end_date FROM sem.observation_period "
                f"WHERE person_id = {multi[0][0]} ORDER BY 1")
    for (s1, e1), (s2, _) in zip(periods, periods[1:]):
        assert (s2 - e1).days > 365  # split only on gaps longer than max_activity_gap_days


# ---- independent reference implementation from the RAW laad tables -----------------
def _reference_sglt2_t2d(con) -> set[int]:
    sglt2_ndcs = {"99999020101", "99999020201", "99999020301", "99999010301"}  # incl. empagliflozin/metformin
    t2d_codes = {"E119", "E1122"}
    rx = con.execute("SELECT claim_id, patient_id, svc_dt, ndc, txn_type, orig_claim_id FROM laad.rx_claims").fetchall()
    reversed_ids = {r[5] for r in rx if r[4] == "REVERSAL"}
    activity = defaultdict(set)
    for _, pid, d, *_ in rx:
        activity[pid].add(d)
    for pid, d in con.execute("SELECT patient_id, svc_dt FROM laad.dx_claims UNION ALL "
                              "SELECT patient_id, svc_dt FROM laad.px_claims").fetchall():
        activity[pid].add(d)

    def periods(pid):
        ds = sorted(activity[pid])
        out, start = [], ds[0]
        for prev, cur in zip(ds, ds[1:]):
            if (cur - prev).days > 365:
                out.append((start, prev))
                start = cur
        out.append((start, ds[-1]))
        return out

    first_paid: dict[int, date] = {}
    for cid, pid, d, ndc, txn, _ in rx:
        if ndc in sglt2_ndcs and txn == "PAID" and cid not in reversed_ids:
            first_paid[pid] = min(d, first_paid.get(pid, d))
    t2d = defaultdict(list)
    for pid, d, *codes in con.execute("SELECT patient_id, svc_dt, dx1, dx2, dx3, dx4 FROM laad.dx_claims").fetchall():
        t2d[pid] += [d for c in codes if c in t2d_codes]
    out = set()
    for pid, idx in first_paid.items():
        per = next(((s, e) for s, e in periods(pid) if s <= idx <= e), None)
        if per is None or (idx - per[0]).days < 365:
            continue
        hits = [d for d in t2d[pid] if 0 <= (idx - d).days <= 365]
        if len(hits) >= 2 and (max(hits) - min(hits)).days >= 30:
            out.add(pid)
    return out


def test_claims_cohort_matches_reference_implementation(laad_builder):
    got = laad_builder.executor.person_ids(laad_builder.compiler.compile(gold("laad_sglt2_t2d")))
    expected = _reference_sglt2_t2d(laad_builder.con)
    assert len(expected) > 10
    assert got == expected


def test_claim_status_and_span_change_results(laad_builder):
    base = gold("laad_sglt2_t2d")
    n = lambda ir: len(laad_builder.executor.person_ids(laad_builder.compiler.compile(ir)))  # noqa: E731
    any_status = base.model_copy(deep=True)
    any_status.index_event.claim_status = ["paid", "rejected", "reversed"]
    no_span = base.model_copy(deep=True)
    no_span.inclusion[0].min_span_days = None
    assert n(no_span) > n(base)                 # spacing rule removes people
    assert n(any_status) != n(base)             # counting rejected/reversed claims changes the cohort
    assert base.semantic_hash() != any_status.semantic_hash()


# ---- capability checks ----------------------------------------------------------------
def test_lab_criteria_are_rejected_on_laad(laad_builder, example_ir_path):
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())  # uses HbA1c values
    issues, _ = validate(ir, laad_builder.ontology, laad_builder.vocab, laad_builder.executor)
    msgs = [i.message for i in issues if i.severity == "error" and i.stage == "dataset"]
    assert any("Measurement" in m and "iqvia_laad" in m for m in msgs)


def test_claims_attributes_rejected_on_omop(builder):
    issues, _ = validate(gold("laad_sglt2_t2d"), builder.ontology, builder.vocab, builder.executor)
    assert any(i.stage == "dataset" and "claim status" in i.message for i in issues)


def test_unsupported_request_stops_without_retries(laad_builder, laad_llm):
    r = laad_builder.ask(T2DM_QUERY)  # asks for HbA1c > 8%, which LAAD cannot answer
    assert r.status == "needs_review"
    assert any(i["stage"] == "dataset" for i in r.issues)
    steps = [s["agent_name"] for s in laad_builder.store.get_run(r.run_id)["steps"]]
    assert steps.count("intent_parser") == 1 and "critic" not in steps


# ---- agents on LAAD ---------------------------------------------------------------------
def test_agents_build_gold_cohort_on_laad(laad_builder):
    r = laad_builder.ask(LAAD_SGLT2_QUERY)
    assert r.status == "draft", r.issues
    assert r.ir.index_event.claim_status == ["paid"]          # dataset default made explicit
    assert r.ir.semantic_hash() == gold("laad_sglt2_t2d").semantic_hash()
    assert r.manifest["dataset"]["name"] == "iqvia_laad"
    assert "LAAD-style" in r.manifest["data_snapshot"]
    assert "Data source:** iqvia_laad" in r.explanation
    assert laad_builder.store.get_definition(r.cohort_definition_id)["dataset"] == "iqvia_laad"


def test_laad_golden_eval(laad_builder):
    report = run_eval(laad_builder, REPO_ROOT / "eval" / "golden_cases_laad.yaml", repeats=2)
    assert report["passed"], report["summary"]
    assert report["summary"]["patient_jaccard"] == 1.0


def test_same_definition_runs_on_both_datasets(settings, fake_llm):
    """Logic built on OMOP can execute on LAAD if LAAD can answer it; otherwise it is refused."""
    omop = CohortBuilder(settings, backend=fake_llm)
    portable, _ = omop.submit_ir(gold("sertraline_depression"), "alice")       # drug + diagnosis only
    labs, _ = omop.submit_ir(gold("t2dm_metformin_hba1c"), "alice")            # needs HbA1c values
    for d in (portable, labs):
        omop.review(d, "dr_reviewer", "approved")
    omop_n = omop.execute(portable, "alice")["person_count"]

    laad = CohortBuilder(settings.__class__(**{**settings.__dict__, "dataset": "iqvia_laad"}),
                         backend=fake_llm, con=omop.con)
    out = laad.execute(portable, "alice")
    assert out["dataset"] == "iqvia_laad" and out["person_count"] > 0 and omop_n > 0
    with pytest.raises(ValueError, match="cannot run on dataset 'iqvia_laad'"):
        laad.execute(labs, "alice")
    gens = omop.con.execute("SELECT dataset FROM meta.cohort_generation WHERE cohort_definition_id = ? "
                            "ORDER BY executed_at", [portable]).fetchall()
    assert [g[0] for g in gens] == ["omop_demo", "iqvia_laad"]


def test_rejection_cohort_explanation(laad_builder):
    from cohort_builder.agents.explainer import Explainer

    text = Explainer(laad_builder.ontology, laad_builder.vocab).explain(gold("laad_sglt2_rejected_then_paid"))
    assert "pharmacy claim for" in text and "[rejected claims only]" in text and "[paid claims only]" in text


def test_retry_feedback_still_works_on_laad(settings):
    from fake_llm import LAAD_SGLT2_INTENT

    broken = copy.deepcopy(LAAD_SGLT2_INTENT)
    broken["criteria"][0]["mention_key"] = "unknown"  # structural error -> feedback -> retry
    llm = FakeLLM({LAAD_SGLT2_QUERY: [broken, LAAD_SGLT2_INTENT]})
    b = CohortBuilder(settings.__class__(**{**settings.__dict__, "dataset": "iqvia_laad"}), backend=llm)
    r = b.ask(LAAD_SGLT2_QUERY)
    assert r.status == "draft"
    assert [s["agent_name"] for s in b.store.get_run(r.run_id)["steps"]].count("intent_parser") == 2
