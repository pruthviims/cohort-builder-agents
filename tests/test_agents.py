"""Agent graph, grounding, retries, review workflow, caching, replay, API and eval (fake LLM)."""
from __future__ import annotations

import copy

import pytest
from fake_llm import T2DM_INTENT, T2DM_QUERY, FakeLLM

from cohort_builder.config import REPO_ROOT
from cohort_builder.evaluation import run_eval
from cohort_builder.llm import ReplayMiss
from cohort_builder.orchestrator import CohortBuilder


def test_ask_builds_valid_draft(builder):
    r = builder.ask(T2DM_QUERY, "alice")
    assert r.status == "draft", r.issues
    ir = r.ir
    assert ir.concept_set(ir.index_event.concept_set_id).items[0].concept_id == 1503297
    sources = {cs.name: cs.source for cs in ir.concept_sets}
    assert "curated:t2dm@v1" in sources.values() and "curated:type1_diabetes@v1" in sources.values()
    assert ir.inclusion[1].value_filter.unit_concept_id == 8554
    m = r.manifest
    for key in ("ir_semantic_hash", "prompts", "ontology", "vocabulary_version", "compiler_version", "model",
                "data_snapshot"):
        assert m[key]
    assert m["tool_calls"] > 0 and m["llm_calls"]["total"] >= 6
    assert "HbA1c" in r.explanation or "Hemoglobin A1c" in r.explanation


def test_agent_matches_gold_patients(builder, example_ir_path):
    from cohort_builder.ir import CohortDefinition

    gold = CohortDefinition.model_validate_json(example_ir_path.read_text())
    r = builder.ask(T2DM_QUERY)
    assert r.ir.semantic_hash() == gold.semantic_hash()


def test_review_gate_and_execution(builder):
    r = builder.ask(T2DM_QUERY)
    with pytest.raises(PermissionError):
        builder.execute(r.cohort_definition_id, "bob")
    builder.review(r.cohort_definition_id, "dr_reviewer", "approved", "looks right")
    out = builder.execute(r.cohort_definition_id, "bob")
    n = builder.con.execute("SELECT count(*) FROM results.cohort WHERE generation_id=?",
                            [out["generation_id"]]).fetchone()[0]
    assert n == out["person_count"] > 0
    # executing again yields identical people (replay reproducibility)
    out2 = builder.execute(r.cohort_definition_id, "bob")
    a = builder.con.execute("SELECT list(subject_id ORDER BY subject_id) FROM results.cohort WHERE generation_id=?",
                            [out["generation_id"]]).fetchone()[0]
    b = builder.con.execute("SELECT list(subject_id ORDER BY subject_id) FROM results.cohort WHERE generation_id=?",
                            [out2["generation_id"]]).fetchone()[0]
    assert a == b and out["sql_hash"] == out2["sql_hash"]


def test_resolver_cannot_submit_ungrounded_ids(settings):
    llm = FakeLLM({T2DM_QUERY: [T2DM_INTENT]},
                  resolver_scripts={"metformin": [("submit_ids", [1503297])]})  # guesses before searching
    b = CohortBuilder(settings, backend=llm)
    r = b.ask(T2DM_QUERY)
    assert r.status == "draft"
    rejected = [req for req in llm.requests
                if any("not returned by any tool call" in str(m.get("content")) for m in req["messages"])]
    assert rejected, "the ungrounded submission should have been rejected and retried"


def test_validation_error_triggers_retry_with_feedback(settings):
    bad = copy.deepcopy(T2DM_INTENT)
    bad["criteria"][1]["value"]["unit_text"] = "furlongs"
    llm = FakeLLM({T2DM_QUERY: [bad, T2DM_INTENT]})
    b = CohortBuilder(settings, backend=llm)
    r = b.ask(T2DM_QUERY)
    assert r.status == "draft"
    steps = b.store.get_run(r.run_id)["steps"]
    assert [s["agent_name"] for s in steps].count("intent_parser") == 2
    # concept resolutions are reused across the retry, not recomputed
    assert [s["agent_name"] for s in steps].count("concept_resolver") == 4


def _critic_revise_first(stage):
    def critic(user, n):
        if n == 1:
            return {"verdict": "revise", "issues": [{"stage": stage, "message": "fix it"}]}
        return {"verdict": "pass", "issues": [], "notes": "ok"}
    return critic


def test_critic_intent_revision_then_pass(settings):
    first = copy.deepcopy(T2DM_INTENT)
    first["age_min"] = 21  # valid but not what was asked
    llm = FakeLLM({T2DM_QUERY: [first, T2DM_INTENT]}, critic=_critic_revise_first("intent"))
    b = CohortBuilder(settings, backend=llm)
    r = b.ask(T2DM_QUERY)
    assert r.status == "draft" and llm.critic_calls == 2
    assert r.ir.demographics.age_min == 18


def test_concept_revision_reresolves_and_stops_when_unchanged(settings):
    llm = FakeLLM({T2DM_QUERY: [T2DM_INTENT]}, critic=_critic_revise_first("concepts"))
    b = CohortBuilder(settings, backend=llm)
    r = b.ask(T2DM_QUERY)
    steps = [s["agent_name"] for s in b.store.get_run(r.run_id)["steps"]]
    assert steps.count("concept_resolver") == 8          # concepts re-resolved with feedback
    assert llm.critic_calls == 1                         # same logic is not sent to the critic again
    assert r.status == "needs_review"
    assert any("did not change" in i["message"] for i in r.issues)


def test_persistent_critic_rejection_needs_review(settings):
    llm = FakeLLM({T2DM_QUERY: [T2DM_INTENT]},
                  critic=lambda u, n: {"verdict": "revise", "issues": [{"stage": "intent", "message": "wrong"}]})
    b = CohortBuilder(settings, backend=llm)
    r = b.ask(T2DM_QUERY)
    assert r.status == "needs_review" and r.cohort_definition_id is not None
    with pytest.raises(ValueError):
        b.review(r.cohort_definition_id, "x", "approved")


def test_cache_and_replay_reproduce_run(builder, fake_llm):
    first = builder.ask(T2DM_QUERY)
    calls_after_first = fake_llm.calls
    second = builder.ask(T2DM_QUERY)  # cached mode: identical requests hit the cache
    assert fake_llm.calls == calls_after_first
    assert second.manifest["llm_calls"]["cache_hits"] == second.manifest["llm_calls"]["total"]
    rep = builder.replay(first.run_id)
    assert rep["identical"] and rep["replay_status"] == "draft"


def test_replay_fails_loudly_on_changed_inputs(builder):
    first = builder.ask(T2DM_QUERY)
    builder.store.con.execute("DELETE FROM meta.llm_cache")
    rep = builder.replay(first.run_id)
    assert rep["replay_status"] == "failed"
    assert any("no recorded response" in i["message"] for i in rep["issues"])
    assert ReplayMiss  # exported for callers


# API tests (with authentication) live in tests/test_api_security.py


def test_eval_harness_on_golden_cases(builder):
    report = run_eval(builder, REPO_ROOT / "eval" / "golden_cases.yaml", repeats=2)
    assert report["passed"], report["summary"]
    assert report["summary"]["patient_jaccard"] == 1.0
    assert report["summary"]["consistency"] == 1.0
    assert {c["therapeutic_area"] for c in report["cases"]} >= {"cardiometabolic", "renal", "mental_health"}
