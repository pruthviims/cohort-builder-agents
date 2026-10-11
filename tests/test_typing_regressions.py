"""Regression tests for the fixes behind the CI type-check failures (mypy 2.4 / mcp 2.3 / pydantic 2.14).

Each group pins the runtime behaviour the stricter types now describe. Synthetic data only."""

from __future__ import annotations

import asyncio
import json
from typing import get_args

import duckdb
import pytest
from mcp import Client
from pydantic import ValidationError

from cohort_builder.agents.explainer import CriterionRenderer, Explainer
from cohort_builder.agents.intent import CohortIntent
from cohort_builder.agents.proxy_explainer import ProxyExplainer
from cohort_builder.agents.proxy_validator import validate_proxy
from cohort_builder.agents.validator import validate
from cohort_builder.compiler import CompiledCohort
from cohort_builder.db import MissingRowError, fetch_count, fetch_one
from cohort_builder.executor import ExecutionError, Executor
from cohort_builder.ir import (
    CLAIM_STATUSES,
    DOMAINS,
    ClaimStatus,
    CohortDefinition,
    Criterion,
    Domain,
    parse_claim_statuses,
    parse_domain,
)
from cohort_builder.mcp_server import READ_ONLY, WRITES, create_server
from cohort_builder.ontology import Ontology
from cohort_builder.proxy import ProxyDefinition
from cohort_builder.proxy_compiler import ProxyCompiler
from cohort_builder.proxy_service import ProxyRunResult
from proxy_fixtures import EXAMPLE, make_builder
from test_proxy_engine import mini
from test_proxy_workflow import LABELS, approved


# ---- A. single-row queries -------------------------------------------------------------------
def test_fetch_helpers_success_and_missing_rows():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t (x INTEGER)")
    assert fetch_count(con, "SELECT count(*) FROM t") == 0  # an empty table still yields one aggregate row
    con.execute("INSERT INTO t VALUES (1), (2)")
    assert fetch_count(con, "SELECT count(*) FROM t WHERE x > ?", [1]) == 1
    assert fetch_one(con, "SELECT min(x), max(x) FROM t") == (1, 2)
    with pytest.raises(MissingRowError, match="no row"):
        fetch_one(con, "SELECT x FROM t WHERE x > 5")  # no row is an invariant violation, not a default
    with pytest.raises(MissingRowError, match="integer count"):
        fetch_count(con, "SELECT max(x) FROM t WHERE x > 5")  # NULL is not silently turned into 0
    with pytest.raises(MissingRowError, match="integer count"):
        fetch_count(con, "SELECT 'a'")


def test_executor_rejects_missing_aggregate_row():
    con = duckdb.connect(":memory:")
    compiled = CompiledCohort(cohort_sql="SELECT 1", attrition_sql="SELECT 1 WHERE FALSE", rule_names=["x"])
    with pytest.raises(ExecutionError, match="no row"):
        Executor(con).attrition(compiled)


def test_vocabulary_version_and_counts_on_empty_vocabulary(tmp_path):
    from cohort_builder.db import init_schemas
    from cohort_builder.vocab import Vocabulary

    con = duckdb.connect(":memory:")
    init_schemas(con)
    v = Vocabulary(con)
    assert v.version().startswith("unknown (0 concepts")
    assert v.is_descendant_or_self(1, 1) is False


def test_definition_sequence_and_manifest_counts(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    first = b.submit_proxy(ProxyDefinition.from_file(EXAMPLE), "alice")["cohort_definition_id"]
    second = b.submit_proxy(ProxyDefinition.from_file(EXAMPLE).model_copy(update={"version": "1.1"}), "alice")[
        "cohort_definition_id"
    ]
    assert second == first + 1
    manifest = b._manifest(
        "no-such-run",
        "q",
        ProxyRunResult(run_id="no-such-run", status="failed"),
        0,
    )
    assert manifest["llm_calls"] == {"total": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
    assert manifest["tool_calls"] == 0
    b.con.close()


# ---- B. proxy model: temporal rules vs tiers ----------------------------------------------------
def test_evidence_roles_follow_temporal_rules_used_in_tiers():
    p = mini(
        temporal_rules=[
            {
                "id": "tr",
                "name": "B within 30 days of A",
                "a": "a",
                "b": "b",
                "relation": "within",
                "days": 30,
                "required": False,
            },
            {"id": "req", "name": "C after A", "a": "a", "b": "c", "relation": "before"},
        ],
        tiers=[{"name": "t1", "rule": {"temporal": "tr"}}, {"name": "t2", "rule": {"evidence": "d"}}],
    )
    roles = p.evidence_roles()
    assert "tier" in roles["a"] and "tier" in roles["b"] and "tier" in roles["d"]
    assert "temporal" in roles["b"] and "temporal_required" in roles["c"]
    assert p.mandatory_evidence() == {"a", "c"}  # the required temporal rule makes both sides mandatory
    with pytest.raises(ValidationError, match="unknown temporal 'nope'"):
        mini(tiers=[{"name": "t", "rule": {"temporal": "nope"}}])
    with pytest.raises(ValidationError, match="duplicate tier names"):
        mini(tiers=[{"name": "t", "rule": {"evidence": "a"}}, {"name": "t", "rule": {"evidence": "b"}}])


# ---- C. explainers ---------------------------------------------------------------------------
def test_cohort_and_proxy_explainers_share_rendering_not_contract(tmp_path, example_ir_path):
    b = make_builder(tmp_path, min_cell=1)
    assert issubclass(Explainer, CriterionRenderer) and issubclass(ProxyExplainer, CriterionRenderer)
    assert not issubclass(ProxyExplainer, Explainer)  # explain() takes different definition types
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())
    cohort_text = Explainer(b.ontology, b.vocab).explain(ir)
    assert cohort_text.startswith(f"### {ir.name}") and "**Entry (index date):**" in cohort_text
    p = ProxyDefinition.from_file(EXAMPLE)
    proxy_text = ProxyExplainer(b.ontology, b.vocab).explain(p)
    assert proxy_text.startswith("### escc_proxy_example v1.0") and "**Evidence tiers" in proxy_text
    assert "PLACEHOLDER esophageal malignancy" in proxy_text  # concept rendering from the shared helper
    assert b.explain(ir) == cohort_text and b.explain(p) == proxy_text  # dispatch by definition type
    b.con.close()


# ---- D. literal types validated at the boundary ------------------------------------------------
def test_domain_and_claim_status_constants_match_literals():
    assert set(DOMAINS) == set(get_args(Domain)) and set(CLAIM_STATUSES) == set(get_args(ClaimStatus))


def test_parse_domain_and_claim_statuses():
    assert parse_domain("Drug") == "Drug"
    with pytest.raises(ValueError, match="unsupported domain 'drug'"):
        parse_domain("drug")  # no case folding or other silent coercion
    assert parse_claim_statuses(["reversed", "paid", "paid"]) == ["paid", "reversed"]
    assert parse_claim_statuses(None) is None and parse_claim_statuses([]) == []
    with pytest.raises(ValueError, match="unsupported claim status 'pending'"):
        parse_claim_statuses(["paid", "pending"])


def test_model_output_with_unknown_claim_status_is_rejected_recoverably():
    base = {"name": "x", "description": "", "mentions": [], "index_mention_key": "m"}
    assert CohortIntent.model_validate({**base, "index_claim_status": ["paid"]}).index_claim_status == ["paid"]
    assert CohortIntent.model_validate(base).index_claim_status is None
    with pytest.raises(ValidationError):  # becomes a schema-repair turn in LLMClient.structured
        CohortIntent.model_validate({**base, "index_claim_status": ["approved"]})


def test_ontology_validates_profile_literals(tmp_path):
    from proxy_fixtures import make_ontology_dir

    ont = Ontology.load(make_ontology_dir(tmp_path), "iqvia_laad")
    assert ont.default_claim_status() == ["paid"]
    assert ont.entity_domain("DrugExposure") == "Drug"
    ont.capabilities["default_claim_status"] = ["paid", "maybe"]
    with pytest.raises(ValueError, match="default_claim_status"):
        ont.default_claim_status()


def test_laad_composer_makes_default_claim_status_explicit(laad_builder):
    from fake_llm import LAAD_SGLT2_QUERY

    r = laad_builder.ask(LAAD_SGLT2_QUERY, "alice")
    assert r.ir is not None and r.ir.index_event.claim_status == ["paid"]


# ---- E/F. validator narrowing and proxy evidence -----------------------------------------------
def test_validator_requires_criteria_for_proxy_definitions(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    p = ProxyDefinition.from_file(EXAMPLE)
    with pytest.raises(TypeError, match="explicit criteria"):
        validate(p, b.ontology, b.vocab)
    with pytest.raises(TypeError, match="compiled SQL"):
        validate(p, b.ontology, b.vocab, b.executor, criteria=[("evidence", e) for e in p.evidence])
    compiled = ProxyCompiler(b.ontology).compile_proxy(p)
    issues, attrition = validate(
        p, b.ontology, b.vocab, b.executor, criteria=[("evidence", e) for e in p.evidence], compiled=compiled
    )  # EvidenceCriterion is accepted where Criterion is expected
    assert attrition is not None and not [i for i in issues if i.severity == "error"]
    issues, _, _ = validate_proxy(p, b.ontology, b.vocab, b.executor)
    assert not [i for i in issues if i.severity == "error"]
    b.con.close()


def test_validator_on_valid_unusual_and_malformed_criteria(builder, example_ir_path):
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())
    issues, attrition = validate(ir, builder.ontology, builder.vocab, builder.executor)
    assert attrition is not None and not [i for i in issues if i.severity == "error"]
    data = ir.model_dump(mode="json")
    data["exclusion"][0]["occurrence"] = "at_most"
    issues, _ = validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)
    assert any("exclusion with occurrence at_most is unusual" in i.message for i in issues)
    data = ir.model_dump(mode="json")
    data["inclusion"][0]["concept_set_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown concept set 'missing'"):
        CohortDefinition.model_validate(data)
    with pytest.raises(ValidationError):
        Criterion.model_validate({"id": "c1", "name": "x", "entity": "Nope", "concept_set_id": "cs"})
    data = ir.model_dump(mode="json")
    data["inclusion"] = []
    data["exclusion"] = []
    issues, _ = validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)
    assert any("not used by any rule" in i.message for i in issues)  # no criteria is valid but flagged


# ---- G. MCP tool annotations -----------------------------------------------------------------
def test_tool_annotation_constants():
    assert (
        READ_ONLY.read_only_hint,
        READ_ONLY.destructive_hint,
        READ_ONLY.idempotent_hint,
        READ_ONLY.open_world_hint,
    ) == (True, False, True, False)
    assert (WRITES.read_only_hint, WRITES.destructive_hint, WRITES.idempotent_hint, WRITES.open_world_hint) == (
        False,
        False,
        False,
        False,
    )
    wire = READ_ONLY.model_dump(by_alias=True, exclude_none=True)
    assert wire == {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}


def test_tools_expose_intended_annotations(builder):
    async def go():
        async with Client(create_server(builder, "tester")) as c:
            tools = {t.name: t.annotations for t in (await c.list_tools()).tools}
            for name in (
                "search_concepts",
                "validate_cohort",
                "get_proxy_status",
                "get_proxy_review_packet",
                "compile_proxy_cohort",
            ):
                a = tools[name]
                assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (
                    True,
                    False,
                    True,
                    False,
                ), name
            for name in ("execute_approved_cohort", "execute_proxy_cohort", "create_proxy_cohort", "build_cohort"):
                a = tools[name]
                assert (a.read_only_hint, a.destructive_hint, a.open_world_hint) == (False, False, False), name
            raw = json.dumps([t.model_dump(by_alias=True, mode="json") for t in (await c.list_tools()).tools])
            assert '"readOnlyHint": true' in raw  # protocol payload keeps the camelCase wire names

    asyncio.run(go())


# ---- governance negative tests (section 4D) -------------------------------------------------------
def test_legacy_validation_record_cannot_back_a_claim(tmp_path):
    from cohort_builder.security import GovernanceError
    from test_proxy_workflow import synthetic_provenance

    b = make_builder(tmp_path, min_cell=1)
    did = approved(b)
    gen = b.execute(did, "carol")["generation_id"]
    p = b.load_proxy(did)[1]
    # a record shaped like the pre-hardening schema: no evaluation/acceptance status, forged review decision
    b.con.execute(
        "INSERT INTO meta.proxy_validation (validation_id, generation_id, cohort_definition_id, reference_name, "
        "positive_tiers, counts_json, metrics_json, created_by, created_at, tenant, semantic_hash, review_decision) "
        "VALUES ('legacy-1', ?, ?, 'old', '[]', '{}', '{\"reported\": true}', 'bob', now(), 'default', ?, 'accepted')",
        [gen, did, p.semantic_hash()],
    )
    claim = synthetic_provenance(version="2.0", classification="clinically_validated", validation_reference="legacy-1")
    with pytest.raises(GovernanceError, match="legacy record without status"):
        b.submit_proxy(claim, "alice")
    b.con.close()


def test_failed_evaluation_query_is_audited_and_not_recorded(tmp_path, monkeypatch):
    b = make_builder(tmp_path, min_cell=1)
    did = approved(b)
    b.execute(did, "carol")
    b.load_reference("chart", LABELS, "synthetic", "root")

    def boom(sql, params=None):
        raise ExecutionError("cohort query failed (error id test)", "test")

    monkeypatch.setattr(b.executor, "query", boom)
    with pytest.raises(ExecutionError):
        b.evaluate_against_reference(did, "chart", "bob")
    assert b.store.validations(did) == []
    failed = [e for e in b.store.audit_events() if e["action"] == "proxy.evaluate_reference"]
    assert failed and failed[0]["outcome"] == "failed"
    b.con.close()
