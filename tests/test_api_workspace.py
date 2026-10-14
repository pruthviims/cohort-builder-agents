"""API additions behind the analytics workspace: conversational refinement through the existing pipeline,
version lineage, review history, run listing, executions, comparison, concept details and system status.
SYNTHETIC DATA ONLY; a scripted fake LLM."""

from __future__ import annotations

import copy

import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.api import create_app
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import SecurityConfig, issue_token
from fake_llm import GOLDEN_INTENTS, T2DM_INTENT, T2DM_QUERY, FakeLLM, _crit

REVISION = "Change the HbA1c threshold to 9% and make the lookback period six months."
REFINED_QUERY = f"{T2DM_QUERY}\n\nRevision: {REVISION}"
REFINED_INTENT = copy.deepcopy(T2DM_INTENT)
REFINED_INTENT["criteria"][1] = _crit(
    "inclusion", "HbA1c > 9% in prior 6 months", "hba1c", -180, 0, {"op": ">", "value": 9, "unit_text": "%"}
)


@pytest.fixture
def ws(settings, tmp_path):
    llm = FakeLLM({**GOLDEN_INTENTS, REFINED_QUERY: [REFINED_INTENT]})
    b = CohortBuilder(settings, backend=llm)
    entries, heads = [], {}
    for name, roles, tenant in [
        ("alice", ["author"], "default"),
        ("rita", ["reviewer"], "default"),
        ("eddie", ["executor"], "default"),
        ("olga", ["author", "reviewer", "executor"], "other"),
        ("ada", ["admin"], "default"),
    ]:
        token, entry = issue_token(name, roles, tenant, expires_days=1)
        entries.append(entry)
        heads[name] = {"Authorization": f"Bearer {token}"}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    c = TestClient(create_app(b, SecurityConfig(tokens_file=tmp_path / "t.yaml")), raise_server_exceptions=False)
    yield c, heads
    b.con.close()


def test_refinement_creates_a_child_version_with_an_exact_diff(ws):
    c, h = ws
    first = c.post("/cohorts/ask", json={"query": T2DM_QUERY}, headers=h["alice"]).json()
    assert first["status"] == "draft"
    parent = first["cohort_definition_id"]
    r = c.post(f"/cohorts/{parent}/refine", json={"instruction": REVISION}, headers=h["alice"])
    assert r.status_code == 200, r.text
    out = r.json()
    child = out["cohort_definition_id"]
    assert child != parent and out["parent_definition_id"] == parent
    assert out["refinement_basis"] == "original_request"
    changed = {x["path"]: (x["before"], x["after"]) for x in out["changes"]}
    # exactly the requested threshold, window and the criterion's name changed; everything else preserved
    assert set(changed) == {
        "inclusion[inc_2].name",
        "inclusion[inc_2].value_filter.value",
        "inclusion[inc_2].window.start_days",
    }
    assert changed["inclusion[inc_2].value_filter.value"] == (8.0, 9.0)
    assert changed["inclusion[inc_2].window.start_days"] == (-365, -180)
    assert not any(p.startswith(("concept_sets", "exclusion", "index_event", "demographics")) for p in changed)
    # the new version is its own draft; the parent is untouched
    assert c.get(f"/cohorts/{child}", headers=h["alice"]).json()["parent_definition_id"] == parent
    versions = c.get(f"/cohorts/{child}/versions", headers=h["rita"]).json()
    assert [v["cohort_definition_id"] for v in versions] == [parent, child]
    # other tenants cannot see or refine it
    assert c.get(f"/cohorts/{child}/versions", headers=h["olga"]).status_code == 404
    assert c.post(f"/cohorts/{child}/refine", json={"instruction": "x"}, headers=h["olga"]).status_code == 404
    assert c.post(f"/cohorts/{child}/refine", json={"instruction": "x"}, headers=h["rita"]).status_code == 403
    assert c.post(f"/cohorts/{child}/refine", json={"instruction": "   "}, headers=h["alice"]).status_code == 422


def test_refining_an_approved_version_never_inherits_the_approval(ws):
    c, h = ws
    parent = c.post("/cohorts/ask", json={"query": T2DM_QUERY}, headers=h["alice"]).json()["cohort_definition_id"]
    assert c.post(f"/cohorts/{parent}/review", json={"decision": "approved"}, headers=h["rita"]).status_code == 200
    child = c.post(f"/cohorts/{parent}/refine", json={"instruction": REVISION}, headers=h["alice"]).json()
    assert c.get(f"/cohorts/{child['cohort_definition_id']}", headers=h["alice"]).json()["status"] == "draft"
    blocked = c.post(f"/cohorts/{child['cohort_definition_id']}/execute", headers=h["eddie"])
    assert blocked.status_code == 403 and "approve" in blocked.json()["detail"]
    assert c.get(f"/cohorts/{parent}/reviews", headers=h["eddie"]).json()[0]["decision"] == "approved"


def test_refinement_without_an_llm_answer_reports_failure_honestly(ws):
    c, h = ws
    parent = c.post("/cohorts/ask", json={"query": T2DM_QUERY}, headers=h["alice"]).json()["cohort_definition_id"]
    before = len(c.get(f"/cohorts/{parent}/versions", headers=h["alice"]).json())
    # the scripted model has no answer for this revision (like an unavailable or failing model)
    r = c.post(f"/cohorts/{parent}/refine", json={"instruction": "something never scripted"}, headers=h["alice"])
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "failed" and out["cohort_definition_id"] is None and out["changes"] == []
    assert any(i["severity"] == "error" for i in out["issues"])
    assert len(c.get(f"/cohorts/{parent}/versions", headers=h["alice"]).json()) == before  # nothing saved


def test_runs_generations_compare_concepts_and_status(ws):
    c, h = ws
    d = c.post("/cohorts/ask", json={"query": T2DM_QUERY}, headers=h["alice"]).json()
    def_id = d["cohort_definition_id"]
    # run listing: authors see their own, reviewers the tenant, other tenants nothing
    assert [r["run_id"] for r in c.get("/runs", headers=h["alice"]).json()] == [d["run_id"]]
    assert "user_query" not in c.get("/runs", headers=h["rita"]).json()[0]
    assert c.get("/runs", headers=h["olga"]).json() == []
    assert c.get("/runs", headers=h["eddie"]).status_code == 403
    c.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h["rita"])
    g1 = c.post(f"/cohorts/{def_id}/execute", headers=h["eddie"]).json()["generation_id"]
    g2 = c.post(f"/cohorts/{def_id}/execute", headers=h["eddie"]).json()["generation_id"]
    gens = c.get(f"/cohorts/{def_id}/generations", headers=h["eddie"]).json()
    assert {g["generation_id"] for g in gens} == {g1, g2} and "summary" not in gens[0]
    assert c.get(f"/cohorts/{def_id}/generations", headers=h["alice"]).status_code == 403
    cmp = c.post("/cohorts/compare", json={"generation_ids": [g1, g2]}, headers=h["rita"]).json()
    assert cmp["overlaps"][0]["first"] == g1 and cmp["warnings"] == []
    assert c.post("/cohorts/compare", json={"generation_ids": [g1, g2]}, headers=h["olga"]).status_code == 404
    assert c.post("/cohorts/compare", json={"generation_ids": [g1]}, headers=h["rita"]).status_code == 422
    concept = c.get("/concepts/201826", headers=h["alice"]).json()
    assert concept["concept_id"] == 201826 and concept["domain_id"] == "Condition"
    assert c.get("/concepts/987654321", headers=h["alice"]).status_code == 404
    st = c.get("/status", headers=h["eddie"]).json()
    assert st["checks"]["database"]["ok"] and st["checks"]["dataset"]["ok"]
    assert st["checks"]["llm"]["api_key_configured"] in (True, False)
    assert "ANTHROPIC_API_KEY" not in str(st) and "sk-" not in str(st)
    assert st["data"] == "synthetic" and st["governance"]["allow_self_approval"] is False
    assert c.get("/status").status_code == 401
