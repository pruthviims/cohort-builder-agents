"""Proxy cohort workflow: versioning, governance, privacy, reference validation, comparison,
natural-language drafting, API and MCP authorization. Synthetic data and placeholder concepts only."""

from __future__ import annotations

import asyncio
import glob
import json

import pytest
import yaml
from fastapi.testclient import TestClient
from mcp import Client

from cohort_builder.api import create_app
from cohort_builder.compiler import Compiler
from cohort_builder.config import REPO_ROOT
from cohort_builder.executor import suppress_partition, suppress_with_total
from cohort_builder.ir import CohortDefinition
from cohort_builder.mcp_server import create_server
from cohort_builder.ontology import Ontology
from cohort_builder.proxy import ProxyDefinition
from cohort_builder.proxy_service import VersionConflict, suppress_overlap
from cohort_builder.security import GovernanceError, GovernancePolicy, SecurityConfig, issue_token
from fake_llm import FakeLLM
from proxy_fixtures import CHEMO_A, DX, EXAMPLE, EXPECTED, SQUAMOUS, make_builder

YAML = EXAMPLE.read_text()


@pytest.fixture
def pb(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    yield b
    b.con.close()


def example(**over) -> ProxyDefinition:
    data = ProxyDefinition.from_file(EXAMPLE).to_dict()
    data.update(over)
    return ProxyDefinition.model_validate(data)


def approved(b, p=None, author="alice", reviewer="bob", tenant="default") -> int:
    r = b.submit_proxy(p or example(), author, tenant)
    assert r["status"] == "draft", r["issues"]
    b.review(r["cohort_definition_id"], reviewer, "approved")
    return r["cohort_definition_id"]


# ---- existing cohort definitions are unaffected -------------------------------------------------------
COHORT_SQL_HASHES = {  # recorded from the commit before proxy cohorts were added
    ("omop_demo", "eval/gold/ckd_egfr.json"): "sha256:b89a702a48c02cf370363adddf0fd02ddc13235fd10498b482cc75f8d97ad2c5",
    (
        "omop_demo",
        "eval/gold/hfref_sglt2.json",
    ): "sha256:785e424af36d7ca121e1c436d62d6c8b52377133a4667be45b1d28958c1f68d0",
    (
        "omop_demo",
        "eval/gold/sertraline_depression.json",
    ): "sha256:5de03ff7b2f40bf5a36b1503bb060c1e4208a6e2d0e9747413639296b8762415",
    (
        "omop_demo",
        "eval/gold/t2dm_metformin_hba1c.json",
    ): "sha256:7bb4824842540e56548ce6290a0809e33596039ceefa8dc310eabc3107a6de3c",
    (
        "iqvia_laad",
        "eval/gold/laad_sertraline_primary_depression.json",
    ): "sha256:4d8d8b6f9fe75cc91852e98eb09bd56276d8009bb2ce53186ecddb0a8b5e1ac1",
    (
        "iqvia_laad",
        "eval/gold/laad_sglt2_rejected_then_paid.json",
    ): "sha256:b7752f5726bd4f06a879502429ea917409d9fe907476d30028b6dbd94ef8d93b",
    (
        "iqvia_laad",
        "eval/gold/laad_sglt2_t2d.json",
    ): "sha256:a8deb3828b1d128a9cccb764e4722414db46d72dca6c92076f1f417648adc7da",
    (
        "iqvia_laad",
        "eval/gold/sertraline_depression.json",
    ): "sha256:035ecb8fa2c62055efcbae7461673f8711232a74018b7fe295416e77df096e04",
}


def test_existing_cohort_sql_is_byte_identical():
    for (ds, path), expected in COHORT_SQL_HASHES.items():
        ir = CohortDefinition.model_validate_json((REPO_ROOT / path).read_text())
        assert Compiler(Ontology.load(REPO_ROOT / "ontology", ds)).compile(ir).sql_hash == expected, (ds, path)
    assert len(glob.glob(str(REPO_ROOT / "eval/gold/*.json"))) == 7


# ---- versioning and immutability -------------------------------------------------------------------------
def test_versions_are_immutable(pb):
    first = pb.submit_proxy(example(), "alice")
    again = pb.submit_proxy(example(), "alice")  # identical content: idempotent
    assert again["existing"] and again["cohort_definition_id"] == first["cohort_definition_id"]
    changed = example(post_observation_days=200)
    with pytest.raises(VersionConflict, match="create|new version"):
        pb.submit_proxy(changed, "alice")
    v2 = pb.submit_proxy(example(version="1.1", post_observation_days=200), "alice")
    row = pb.store.get_definition(v2["cohort_definition_id"])
    assert row["parent_definition_id"] == first["cohort_definition_id"]  # lineage
    assert row["algorithm_version"] == "1.1" and row["kind"] == "proxy" and row["created_by"] == "alice"
    versions = pb.proxy_versions(first["cohort_definition_id"])
    assert [v["algorithm_version"] for v in versions] == ["1.1", "1.0"]
    assert pb.next_version("default", "escc_proxy_example") == "1.2"
    stored = pb.store.get_definition(first["cohort_definition_id"])
    assert ProxyDefinition.model_validate(stored["ir"]).content_hash() == example().content_hash()


def test_execution_is_reproducible(pb):
    did = approved(pb)
    a, b = pb.execute(did, "carol"), pb.execute(did, "carol")
    assert a["sql_hash"] == b["sql_hash"] and a["attrition"] == b["attrition"]
    assert a["evidence_summary"] == b["evidence_summary"]
    members = pb.con.execute(
        "SELECT generation_id, list(subject_id ORDER BY subject_id) FROM results.cohort GROUP BY 1"
    ).fetchall()
    assert len(members) == 2 and members[0][1] == members[1][1] == sorted(EXPECTED)


# ---- governance: review before execution ----------------------------------------------------------------
def test_draft_and_self_approved_definitions_do_not_run(pb):
    did = pb.submit_proxy(example(), "alice")["cohort_definition_id"]
    with pytest.raises(GovernanceError, match="approve it before execution"):
        pb.execute(did, "carol")
    with pytest.raises(GovernanceError, match="self-approval"):
        pb.review(did, "alice", "approved")
    pb.review(did, "bob", "approved")
    out = pb.execute(did, "carol")
    assert out["person_count"] == 4 and out["algorithm"]["classification"] == "exploratory"
    actions = {(e["action"], e["outcome"]) for e in pb.store.audit_events()}
    assert {
        ("definition.execute", "denied"),
        ("definition.review", "denied"),
        ("proxy.submit", "success"),
        ("definition.execute", "success"),
    } <= actions


def test_definition_with_errors_cannot_be_approved(tmp_path):
    b = make_builder(tmp_path, pathology=False, min_cell=1)
    r = b.submit_proxy(example(), "alice")
    assert r["status"] == "needs_review"
    assert any("Proxy rule requires pathology evidence" in i["message"] for i in r["issues"])
    with pytest.raises(ValueError, match="unresolved errors"):
        b.review(r["cohort_definition_id"], "bob", "approved")


def test_review_packet_contents(pb):
    did = pb.submit_proxy(example(), "alice")["cohort_definition_id"]
    packet = pb.proxy_review_packet(did)
    for key in (
        "target",
        "evidence",
        "logic",
        "temporal_rules",
        "dataset",
        "expected_tiers",
        "versions",
        "governance",
        "reviewer_checklist",
        "dry_run_attrition",
    ):
        assert packet[key] is not None, key
    assert packet["algorithm"]["created_by"] == "alice" and packet["algorithm"]["version"] == "1.0"
    assert set(packet["versions"]["concept_sets"]) == {cs.id for cs in example().concept_sets}
    text = json.dumps(packet, default=str).lower()
    assert "true esophageal" not in text and "identifies true" not in text
    assert "not a probability" in text and "does not establish" in text
    sql = pb.compile_proxy(did)
    assert set(sql["statements"]) == {"cohort", "assignment", "evidence", "attrition", "summary"}
    assert sql["sql_hash"] == packet["algorithm"]["sql_hash"]


# ---- privacy -------------------------------------------------------------------------------------------
def test_aggregates_are_suppressed_with_default_threshold(tmp_path):
    b = make_builder(tmp_path)  # min_cell from the ontology (10): every count here is small
    did = approved(b)
    out = b.execute(did, "carol")
    assert out["person_count"] == "<10"
    summary = out["evidence_summary"]
    shown = [e["patients"] for e in summary["evidence"]] + [t["patients"] for t in summary["tiers"]]
    assert all(v in (0, "<10", "suppressed") for v in shown)
    assert not any(isinstance(v, int) and 0 < v < 10 for v in json_numbers(out))
    results = b.proxy_results(did)
    assert results["evidence_summary"] == summary and results["person_count"] == "<10"


def json_numbers(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("sequence", "cohort_definition_id", "min_cell_count", "evidence_score"):
                yield from json_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_numbers(v)
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield obj


def test_suppression_helpers():
    assert suppress_with_total(495, 500, 10) == "suppressed" and suppress_with_total(5, 500, 10) == "<10"
    assert suppress_partition({"high": 300, "moderate": 5, "exploratory": 120, "none": 75}, 10) == {
        "high": 300,
        "moderate": "<10",
        "exploratory": 120,
        "none": "suppressed",
    }
    assert suppress_overlap(100, 100, 95, 10) == {"both": "suppressed", "only_first": "<10", "only_second": "<10"}
    assert suppress_overlap(100, 200, 50, 10) == {"both": 50, "only_first": 50, "only_second": 150}
    assert suppress_overlap(100, 200, 95, 10) == {
        "both": "suppressed",
        "only_first": "<10",
        "only_second": "suppressed",
    }


def test_patient_level_output_requires_admin_and_flag(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    did = approved(b)
    b.execute(did, "carol")
    with pytest.raises(GovernanceError):
        b.patient_explanation(did, 5, "root", is_admin=True)  # flag off (default)
    b.policy = GovernancePolicy(allow_patient_level=True)
    with pytest.raises(GovernanceError):
        b.patient_explanation(did, 5, "carol", is_admin=False)
    out = b.patient_explanation(did, 5, "root", is_admin=True)
    assert out["tier"] == "high" and out["in_cohort"] and out["score_note"].startswith("The evidence score")
    present = {e["id"]: e["present"] for e in out["evidence"]}
    assert present["squamous_path"] and present["chemo"] and not present["repeated_dx"]
    assert {c["name"]: c["present"] for c in out["conflicts"]} == {"adeno_histology": True}
    assert next(t for t in out["tiers"] if t["name"] == "high")["rule_holds"] is True
    assert b.patient_explanation(did, 6, "root", is_admin=True)["in_cohort"] is False
    events = [e for e in b.store.audit_events() if e["action"] == "proxy.patient_explanation"]
    assert {e["outcome"] for e in events} == {"denied", "success"}
    assert all("subject_id" not in e["details"] and "5" != e["details"].get("subject_ref") for e in events)


# ---- reference-standard validation ------------------------------------------------------------------------
LABELS = [(1, True), (2, True), (5, True), (3, False), (4, False), (6, False), (7, False)]


def test_reference_validation_metrics(pb):
    did = approved(pb)
    pb.execute(did, "carol")
    with pytest.raises(KeyError):  # no reference standard -> no metrics, ever
        pb.validate_against_reference(did, "chart_review", "bob")
    pb.load_reference("chart_review", LABELS, "synthetic chart review (test fixture)", "root")
    with pytest.raises(ValueError, match="immutable"):
        pb.load_reference("chart_review", LABELS, "again", "root")
    r = pb.validate_against_reference(did, "chart_review", "bob")
    assert r["confusion_matrix"] == {"TP": 3, "FP": 1, "FN": 0, "TN": 3}
    m = r["metrics"]
    assert m["reported"] and (m["sensitivity"], m["specificity"], m["ppv"], m["npv"], m["f1"]) == (
        1.0,
        0.75,
        0.75,
        1.0,
        0.8571,
    )
    high = pb.validate_against_reference(did, "chart_review", "bob", positive_tiers=["high"])
    assert high["confusion_matrix"] == {"TP": 2, "FP": 0, "FN": 1, "TN": 4}
    assert (high["metrics"]["sensitivity"], high["metrics"]["ppv"]) == (0.6667, 1.0)


def test_reference_metrics_withheld_when_small_or_one_class(tmp_path):
    b = make_builder(tmp_path)  # min cell 10
    did = approved(b)
    b.execute(did, "carol")
    b.load_reference("small", LABELS, "synthetic", "root")
    r = b.validate_against_reference(did, "small", "bob")
    assert not r["metrics"]["reported"] and "sensitivity" not in r["metrics"]
    assert set(r["confusion_matrix"].values()) <= {0, "<10", "suppressed"}
    b.executor.min_cell = 1
    b.load_reference("cases_only", [(1, True), (2, True)], "synthetic", "root")
    one = b.validate_against_reference(did, "cases_only", "bob")
    assert not one["metrics"]["reported"] and "both cases and non-cases" in one["metrics"]["withheld_reasons"][0]


def test_clinically_validated_needs_matching_validation(pb):
    did = approved(pb)
    pb.execute(did, "carol")
    with pytest.raises(GovernanceError, match="not a recorded reference validation"):
        pb.submit_proxy(
            example(version="2.0", classification="clinically_validated", validation_reference="x"), "alice"
        )
    pb.load_reference("chart_review", LABELS, "synthetic chart review", "root")
    vid = pb.validate_against_reference(did, "chart_review", "bob")["validation_id"]
    other_logic = example(
        version="2.1", classification="clinically_validated", validation_reference=vid, prior_observation_days=30
    )
    with pytest.raises(GovernanceError, match="different algorithm logic"):
        pb.submit_proxy(other_logic, "alice")
    ok = pb.submit_proxy(
        example(version="2.0", classification="clinically_validated", validation_reference=vid), "alice"
    )
    assert ok["status"] == "draft"
    assert (
        "validated against a reference standard"
        in pb.proxy_review_packet(ok["cohort_definition_id"])["algorithm"]["classification_label"]
    )


# ---- comparison ---------------------------------------------------------------------------------------------
def test_compare_generations(pb):
    a = pb.execute(approved(pb), "carol")["generation_id"]
    strict = example(version="1.1", tiers=[t for t in example().to_dict()["tiers"] if t["name"] == "high"])
    b = pb.execute(approved(pb, strict), "carol")["generation_id"]
    out = pb.compare_generations([a, b], "bob")
    assert out["overlaps"][0] | {} == {"first": a, "second": b, "both": 2, "only_first": 2, "only_second": 0}
    assert [g["person_count"] for g in out["generations"]] == [4, 2]
    pb.executor.min_cell = 10
    hidden = pb.compare_generations([a, b], "bob")["overlaps"][0]
    assert {hidden["both"], hidden["only_first"]} <= {"<10", "suppressed"}
    with pytest.raises(ValueError):
        pb.compare_generations([a], "bob")
    with pytest.raises(KeyError):
        pb.compare_generations([a, b], "bob", tenant="other")


# ---- natural language -> draft (never executed) --------------------------------------------------------------
NL_QUERY = "Find likely ESCC patients: esophageal cancer diagnosis plus squamous pathology and chemotherapy."
NL_DRAFT = {
    "algorithm_name": "nl_escc_proxy",
    "target_name": "ESCC-like target (synthetic)",
    "mentions": [
        {"key": "esoph", "text": "esophageal cancer", "entity": "ConditionOccurrence"},
        {"key": "chemo", "text": "escc chemotherapy", "entity": "DrugExposure"},
        {"key": "squamous", "text": "squamous pathology", "entity": "Measurement"},
    ],
    "index_mention_key": "esoph",
    "evidence": [
        {
            "id": "dx",
            "name": "Diagnosis",
            "category": "supporting",
            "mention_key": "esoph",
            "window_start_days": 0,
            "window_end_days": 0,
        },
        {
            "id": "chemo",
            "name": "Chemotherapy",
            "category": "treatment",
            "mention_key": "chemo",
            "window_start_days": 0,
            "window_end_days": 180,
        },
        {
            "id": "path",
            "name": "Squamous pathology",
            "category": "pathology",
            "mention_key": "squamous",
            "window_start_days": -30,
            "window_end_days": 90,
        },
    ],
    "tiers": [
        {"name": "high", "rule": {"all": [{"evidence": "dx"}, {"evidence": "path"}, {"evidence": "chemo"}]}},
        {"name": "low", "rule": {"evidence": "dx"}},
    ],
}


def test_natural_language_creates_reviewable_draft_only(tmp_path):
    llm = FakeLLM({}, proxies={NL_QUERY: [NL_DRAFT]})
    b = make_builder(tmp_path, backend=llm, min_cell=1)
    r = b.ask_proxy(NL_QUERY, "alice")
    assert r.status == "draft", r.issues
    assert r.ir.classification == "exploratory" and r.ir.version == "1.0"
    ids = {cs.id: [i.concept_id for i in cs.items] for cs in r.ir.concept_sets}
    assert ids == {"cs_esoph": [DX], "cs_chemo": [CHEMO_A], "cs_squamous": [SQUAMOUS]}  # grounded by the resolver
    assert b.con.execute("SELECT count(*) FROM meta.cohort_generation").fetchone()[0] == 0  # nothing executed
    assert b.con.execute("SELECT count(*) FROM results.cohort").fetchone()[0] == 0
    with pytest.raises(GovernanceError):
        b.execute(r.cohort_definition_id, "alice")
    assert r.manifest["kind"] == "proxy"
    replay = b.replay(r.run_id)
    assert replay["identical"]
    # the replay saved its own draft (1.1); a further request gets the next version, never an overwrite
    assert b.ask_proxy(NL_QUERY, "alice").ir.version == "1.2"


# ---- HTTP API --------------------------------------------------------------------------------------------------
USERS = {
    "alice": (["author"], "research"),
    "bob": (["reviewer"], "research"),
    "carol": (["executor"], "research"),
    "victor": (["viewer"], "research"),
    "eve": (["author", "reviewer", "executor"], "other"),
    "root": (["admin"], "research"),
}


@pytest.fixture
def api(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    entries, toks = [], {}
    for name, (roles, tenant) in USERS.items():
        token, entry = issue_token(name, roles, tenant, expires_days=30)
        entries.append(entry)
        toks[name] = {"Authorization": f"Bearer {token}"}
    path = tmp_path / "tokens.yaml"
    path.write_text(yaml.safe_dump({"tokens": entries}))

    def client(**cfg):
        return TestClient(create_app(b, SecurityConfig(tokens_file=path, **cfg)), raise_server_exceptions=False)

    yield client, toks, b
    b.con.close()


def test_api_proxy_workflow_and_roles(api):
    mk, t, _ = api
    c = mk()
    assert c.post("/proxy-cohorts", json={"yaml": YAML}).status_code == 401
    assert c.post("/proxy-cohorts", json={"yaml": YAML}, headers=t["victor"]).status_code == 403
    assert c.post("/proxy-cohorts/validate", json={"yaml": YAML}, headers=t["alice"]).json()["valid"]
    bad = c.post("/proxy-cohorts/validate", json={"definition": {"algorithm_name": "x"}}, headers=t["alice"])
    assert bad.status_code == 422
    r = c.post("/proxy-cohorts", json={"yaml": YAML}, headers=t["alice"])
    assert r.status_code == 200, r.text
    did = r.json()["cohort_definition_id"]
    conflict = c.post(
        "/proxy-cohorts",
        json={"yaml": YAML.replace("prior_observation_days: 365", "prior_observation_days: 300")},
        headers=t["alice"],
    )
    assert conflict.status_code == 409 and "new version" in conflict.json()["detail"]
    assert c.get(f"/proxy-cohorts/{did}/review-packet", headers=t["victor"]).status_code == 200
    assert c.post(f"/proxy-cohorts/{did}/compile", headers=t["victor"]).json()["statements"]["cohort"]
    assert c.post(f"/proxy-cohorts/{did}/execute", headers=t["carol"]).status_code == 403  # draft
    assert (
        c.post(f"/proxy-cohorts/{did}/review", json={"decision": "approved"}, headers=t["alice"]).status_code == 403
    )  # authors cannot review
    assert c.post(f"/proxy-cohorts/{did}/review", json={"decision": "approved"}, headers=t["bob"]).status_code == 200
    assert c.post(f"/proxy-cohorts/{did}/execute", headers=t["alice"]).status_code == 403  # not an executor
    ex = c.post(f"/proxy-cohorts/{did}/execute", headers=t["carol"])
    assert ex.status_code == 200 and ex.json()["person_count"] == 4
    assert c.get(f"/proxy-cohorts/{did}/results", headers=t["victor"]).status_code == 403
    res = c.get(f"/proxy-cohorts/{did}/evidence-summary", headers=t["bob"]).json()
    assert {x["name"]: x["patients"] for x in res["evidence_summary"]["tiers"]}["high"] == 2
    assert [v["algorithm_version"] for v in c.get(f"/proxy-cohorts/{did}/versions", headers=t["victor"]).json()] == [
        "1.0"
    ]
    # other tenants see nothing
    assert c.get(f"/proxy-cohorts/{did}", headers=t["eve"]).status_code == 404
    assert c.post(f"/proxy-cohorts/{did}/execute", headers=t["eve"]).status_code == 404


def test_api_patient_level_and_references(api):
    mk, t, b = api
    c = mk()
    did = approved(b, tenant="research")
    b.execute(did, "carol")
    url = f"/proxy-cohorts/{did}/patients/5/explanation"
    assert c.get(url, headers=t["bob"]).status_code == 403  # not admin
    assert c.get(url, headers=t["root"]).status_code == 403  # admin, but the policy flag is off
    labels = [{"person_id": p, "is_case": v} for p, v in LABELS]
    assert (
        c.post(
            "/proxy-references", json={"name": "cr", "source": "synthetic", "labels": labels}, headers=t["bob"]
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/proxy-references", json={"name": "cr", "source": "synthetic", "labels": labels}, headers=t["root"]
        ).status_code
        == 200
    )
    v = c.post(f"/proxy-cohorts/{did}/reference-validation", json={"reference_name": "cr"}, headers=t["bob"])
    assert v.status_code == 200 and v.json()["metrics"]["sensitivity"] == 1.0
    flagged = mk(allow_patient_level=True)
    out = flagged.get(url, headers=t["root"])
    assert out.status_code == 200 and out.json()["tier"] == "high"
    assert flagged.get(url, headers=t["carol"]).status_code == 403


# ---- MCP -----------------------------------------------------------------------------------------------------
async def call(client, name, args=None):
    r = await client.call_tool(name, args or {})
    assert not r.is_error, r.content
    data = r.structured_content
    if data is None:
        return json.loads(r.content[0].text)
    return data["result"] if isinstance(data, dict) and set(data) == {"result"} else data


def test_mcp_proxy_tools_never_bypass_approval(pb):
    async def go():
        async with Client(create_server(pb, "agent")) as c:
            names = {x.name for x in (await c.list_tools()).tools}
            assert {
                "create_proxy_cohort",
                "validate_proxy_cohort",
                "explain_proxy_cohort",
                "compile_proxy_cohort",
                "get_proxy_review_packet",
                "execute_proxy_cohort",
                "compare_proxy_cohorts",
            } <= names
            assert not any(n.startswith(("approve", "review", "reject")) for n in names)
            assert not any("patient" in n for n in names)
            v = await call(c, "validate_proxy_cohort", {"definition_yaml": YAML})
            assert v["valid"]
            saved = await call(c, "create_proxy_cohort", {"definition_yaml": YAML})
            did = saved["cohort_definition_id"]
            assert saved["saved"] and saved["status"] == "draft"
            refused = await call(c, "execute_proxy_cohort", {"definition_id": did})
            assert "error" in refused and "approve" in refused["hint"]
            packet = await call(c, "get_proxy_review_packet", {"definition_id": did})
            assert "Approval is only possible for a human" in packet["approval"]
            assert (await call(c, "compile_proxy_cohort", {"definition_id": did}))["statements"]["cohort"]
            pb.review(did, "bob", "approved")  # a human, outside MCP
            out = await call(c, "execute_proxy_cohort", {"definition_id": did})
            assert out["person_count"] == 4
            assert "subject_id" not in json.dumps(out)

    asyncio.run(go())


# ---- CLI -----------------------------------------------------------------------------------------------------
def test_cli_proxy_commands(tmp_path, monkeypatch, capsys):
    from cohort_builder.cli import main

    b = make_builder(tmp_path)
    b.con.close()
    monkeypatch.setenv("CB_DB_PATH", str(tmp_path / "escc.duckdb"))
    monkeypatch.setenv("CB_ONTOLOGY_DIR", str(tmp_path / "ontology"))
    monkeypatch.setenv("CB_DATASET", "escc_synthetic")
    assert main(["proxy", "validate", str(EXAMPLE)]) == 0
    assert "Exploratory identification algorithm" in capsys.readouterr().out
    assert main(["proxy", "submit", str(EXAMPLE), "--user", "alice"]) == 0
    did = json.loads(capsys.readouterr().out)["cohort_definition_id"]
    with pytest.raises(SystemExit, match="refused"):
        main(["proxy", "execute", str(did)])
    with pytest.raises(SystemExit, match="self-approval"):
        main(["proxy", "approve", str(did), "--reviewer", "alice"])
    assert main(["proxy", "approve", str(did), "--reviewer", "bob"]) == 0
    capsys.readouterr()
    assert main(["proxy", "execute", str(did), "--user", "carol"]) == 0
    assert json.loads(capsys.readouterr().out)["person_count"] == "<10"
    with pytest.raises(SystemExit, match="CB_ALLOW_PATIENT_LEVEL"):
        main(["proxy", "explain-patient", str(did), "5"])
