"""Hardening of proxy-algorithm validation: unique identifiers, metric definitions, the reference
evaluation population, canonical semantic hashing, evaluation vs acceptance vs approval states, and
evidence provenance.

ALL DATA HERE IS SYNTHETIC (placeholder concepts, invented patients and invented reference labels).
Passing these tests says the software behaves as specified; it is not evidence of clinical validity."""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import date

import pytest
import yaml
from fastapi.testclient import TestClient
from mcp import Client
from pydantic import ValidationError

from cohort_builder.api import create_app
from cohort_builder.metrics import classification_metrics, f1_score, ratio, rounded, wilson_interval
from cohort_builder.ontology import Ontology
from cohort_builder.proxy import (
    AcceptanceCriteria,
    DuplicateKeyError,
    ProxyDefinition,
    load_json_strict,
    load_yaml_strict,
)
from cohort_builder.proxy_compiler import ProxyCompiler
from cohort_builder.proxy_evaluation import (
    EligibilityRules,
    ReferenceRecord,
    assess_acceptance,
    collapse_records,
    normalize_label,
)
from cohort_builder.security import GovernanceError, SecurityConfig, issue_token
from proxy_fixtures import EXAMPLE, Events, build_escc_db, make_builder, make_ontology_dir
from test_proxy_engine import mini
from test_proxy_workflow import LABELS, approved, example, synthetic_provenance

YAML = EXAMPLE.read_text()


@pytest.fixture
def pb(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    yield b
    b.con.close()


# =============================== Phase 2: unique identifiers ===================================
def _cs(cid, concept, name="cs"):
    return {"id": cid, "name": name, "domain": "Condition", "items": [{"concept_id": concept}]}


def test_unique_concept_set_ids_accepted():
    p = mini()
    assert len({cs.id for cs in p.concept_sets}) == len(p.concept_sets)


def test_duplicate_concept_set_ids_identical_rejected():
    sets = copy.deepcopy(mini().to_dict()["concept_sets"])
    sets.append(copy.deepcopy(sets[0]))
    with pytest.raises(ValidationError, match=r"repeated identical definitions for \['dx'\]"):
        mini(concept_sets=sets)


def test_duplicate_concept_set_ids_conflicting_rejected():
    sets = copy.deepcopy(mini().to_dict()["concept_sets"])
    sets.append(_cs("dx", 2100000009, "other codes"))
    with pytest.raises(ValidationError, match=r"conflicting definitions for \['dx'\]"):
        mini(concept_sets=sets)


@pytest.mark.parametrize("where", ["index", "evidence"])
def test_reference_to_unknown_concept_set(where):
    data = mini().to_dict()
    if where == "index":
        data["index_event"]["concept_set_id"] = "missing"
    else:
        data["evidence"][1]["concept_set_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown concept set 'missing'"):
        ProxyDefinition.model_validate(data)


def test_nested_expressions_resolve_unambiguously():
    ok = mini(
        groups={"g": {"any": [{"all": [{"evidence": "a"}, {"not": {"evidence": "b"}}]}, {"evidence": "c"}]}},
        tiers=[{"name": "t", "rule": {"at_least": {"n": 1, "of": [{"group": "g"}, {"evidence": "d"}]}}}],
    )
    assert ok.concept_set(ok.evidence_by_id("b").concept_set_id).id == "chemo"
    with pytest.raises(ValidationError, match="unknown evidence 'zz'"):
        mini(groups={"g": {"any": [{"all": [{"evidence": "a"}, {"not": {"evidence": "zz"}}]}]}})


def test_duplicate_ids_cannot_reach_sql_expansion(tmp_path):
    """Regression: two concept sets with one id would be UNIONed into one code list in cs_expanded."""
    p = mini()
    dupe = p.model_copy(
        update={
            "concept_sets": [
                *p.concept_sets,
                p.concept_sets[0].model_copy(update={"items": [p.concept_sets[4].items[0]]}),
            ]
        }
    )  # bypasses validation on purpose
    ont = Ontology.load(make_ontology_dir(tmp_path), "escc_synthetic")
    with pytest.raises(ValueError, match="duplicate concept set ids would merge code lists"):
        ProxyCompiler(ont).compile_proxy(dupe)


@pytest.mark.parametrize(
    "text, key",
    [
        ("algorithm_name: a\nalgorithm_name: b\n", "algorithm_name"),
        ("groups:\n  g: {evidence: a}\n  g: {evidence: b}\n", "g"),
        ("concept_sets:\n  - {id: x, id: y}\n", "id"),
    ],
)
def test_yaml_duplicate_keys_rejected_at_any_depth(text, key):
    with pytest.raises(DuplicateKeyError, match=f"duplicate key '{key}'"):
        load_yaml_strict(text)


def test_example_duplicate_group_key_rejected():
    bad = YAML.replace("groups:\n", "groups:\n  strong_treatment: {evidence: chemo}\n", 1)
    with pytest.raises(DuplicateKeyError, match="strong_treatment"):
        ProxyDefinition.from_yaml(bad)


def test_json_duplicate_keys_rejected():
    with pytest.raises(DuplicateKeyError):
        load_json_strict('{"a": 1, "b": {"c": 1, "c": 2}}')
    assert load_json_strict('{"a": {"b": 1}}') == {"a": {"b": 1}}


# =============================== Phase 3: metrics ==============================================
def test_perfect_classification():
    m = classification_metrics(5, 0, 0, 7)
    assert (m["sensitivity"], m["specificity"], m["ppv"], m["npv"], m["f1"]) == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_all_positive_predictions():
    m = classification_metrics(4, 6, 0, 0)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 0.0 and m["ppv"] == 0.4
    assert m["npv"] is None  # nothing predicted negative
    assert m["f1"] == pytest.approx(2 * 0.4 / 1.4)


def test_no_positive_predictions():
    m = classification_metrics(0, 0, 4, 6)
    assert m["ppv"] is None and m["sensitivity"] == 0.0 and m["specificity"] == 1.0 and m["npv"] == 0.6
    assert m["f1"] is None  # precision undefined -> F1 undefined


def test_zero_precision_nonzero_recall_impossible_and_zero_tp_gives_zero_f1():
    # precision 0 with recall > 0 cannot happen (both need TP); with TP = 0 both are 0 -> F1 is 0.0
    m = classification_metrics(0, 3, 2, 5)
    assert m["ppv"] == 0.0 and m["sensitivity"] == 0.0 and m["f1"] == 0.0


def test_f1_zero_when_either_component_is_zero():
    assert f1_score(0.0, 0.8) == 0.0 and f1_score(0.8, 0.0) == 0.0 and f1_score(0.0, 0.0) == 0.0
    assert f1_score(None, 0.5) is None and f1_score(0.5, None) is None


def test_zero_true_positives():
    m = classification_metrics(0, 2, 3, 5)
    assert m["sensitivity"] == 0.0 and m["ppv"] == 0.0 and m["f1"] == 0.0


def test_no_reference_positive_cases():
    m = classification_metrics(0, 1, 0, 9)
    assert m["sensitivity"] is None and m["ci"]["sensitivity"] is None and m["f1"] is None
    assert m["specificity"] == 0.9


def test_empty_evaluation_population():
    m = classification_metrics(0, 0, 0, 0)
    assert all(m[k] is None for k in ("sensitivity", "specificity", "ppv", "npv", "f1"))
    assert all(v is None for v in m["ci"].values())


def test_undefined_npv_and_specificity():
    m = classification_metrics(3, 0, 0, 0)
    assert m["specificity"] is None and m["npv"] is None and m["ppv"] == 1.0


def test_metrics_never_nan_and_validate_input():
    for counts in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
        out = json.dumps(rounded(classification_metrics(*counts)))
        assert "NaN" not in out and "Infinity" not in out
    with pytest.raises(ValueError):
        classification_metrics(-1, 0, 0, 0)
    with pytest.raises(ValueError):
        ratio(3, 2)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.65 < lo < 1
    assert wilson_interval(0, 0) is None
    lo95, hi95 = wilson_interval(5, 10, 0.95)
    lo99, hi99 = wilson_interval(5, 10, 0.99)
    assert lo99 < lo95 < 0.5 < hi95 < hi99


def test_rounded_serialization_keeps_null_and_zero_distinct():
    out = json.loads(json.dumps(rounded(classification_metrics(0, 0, 4, 6))))
    assert out["ppv"] is None and out["sensitivity"] == 0.0 and out["f1"] is None
    assert out["ci"]["ppv"] is None and isinstance(out["ci"]["npv"], list)


# =============================== Phase 4: evaluation population =================================
def build_population(con):
    """The ESCC fixture (persons 1-7) plus: 9 = short observation, 10 = reference date before observation,
    11 = labelled unknown, 12 = conflicting labels. Person 8 is labelled but absent from the dataset."""
    build_escc_db(con)
    for pid, start, end in [
        (9, "2020-01-01", "2020-03-01"),
        (10, "2015-01-01", "2025-12-31"),
        (11, "2015-01-01", "2025-12-31"),
        (12, "2015-01-01", "2025-12-31"),
    ]:
        con.execute("INSERT INTO cdm.person VALUES (?,8507,1960,1,1,0,0,?)", [pid, f"p{pid}"])
        con.execute("INSERT INTO cdm.observation_period VALUES (?,?,?,?,0)", [pid, pid, start, end])
    ev = Events()
    ev.dx(9, "2020-01-10")  # observed too briefly for the algorithm's 365 + 365 days
    ev.dx(10, "2020-01-10")
    ev.dx(12, "2020-01-10")
    ev.dx(13, "2020-01-10")  # a record for a patient that is not in cdm.person
    ev.write(con)


POP_LABELS = [
    (1, "case"),
    (1, "case"),  # duplicate record: counts once
    (2, True),
    (5, "case"),
    (3, False),
    (4, "non_case"),
    (6, "non_case"),
    (7, "non_case"),
    (8, "case"),  # not in the dataset
    (9, "case"),  # insufficient observation
    (10, "non_case", date(2010, 6, 1)),  # reference date outside observation
    (11, None),  # unknown / indeterminate
    (12, "case"),
    (12, "non_case"),  # conflicting
]


@pytest.fixture
def popb(tmp_path):
    b = make_builder(tmp_path, build=build_population, min_cell=1)
    did = approved(b)
    gen = b.execute(did, "carol")["generation_id"]
    b.load_reference("chart", POP_LABELS, "synthetic chart review (test fixture)", "root")
    yield b, did, gen
    b.con.close()


def test_collapse_records_is_deterministic():
    rows, summary = collapse_records(
        [
            ReferenceRecord(2, "case"),
            ReferenceRecord(1, "unknown"),
            ReferenceRecord(2, "case", date(2020, 1, 2)),
            ReferenceRecord(2, "unknown", date(2020, 1, 1)),
            ReferenceRecord(3, "case"),
            ReferenceRecord(3, "non_case"),
        ]
    )
    assert rows == [(1, "unknown", None, 1), (2, "case", date(2020, 1, 1), 3), (3, "conflicting", None, 2)]
    assert summary["patients_with_duplicate_records"] == 2 and summary["patients_with_several_reference_dates"] == 1
    with pytest.raises(ValueError, match="unrecognised reference label"):
        normalize_label("maybe")


def test_evaluation_population_excludes_unobservable_and_unknown(popb):
    b, did, _ = popb
    r = b.evaluate_against_reference(did, "chart", "bob")
    pop = r["population"]
    assert pop["reference_patients"] == 12  # patients 1..12 (14 records; patient 8 is absent from the data)
    assert pop["labels"] == {"case": 5, "non_case": 5, "unknown": 1, "conflicting": 1}
    assert pop["excluded_by_reason"] == {
        "unknown_reference_label": 1,
        "conflicting_reference_labels": 1,
        "not_in_dataset": 1,
        "insufficient_observation": 1,
        "reference_date_not_observed": 1,
    }
    assert (pop["eligible"], pop["evaluated"], pop["excluded"]) == (7, 7, 5)
    assert (pop["eligible_reference_positive"], pop["eligible_reference_negative"]) == (3, 4)
    # without eligibility filtering 8 and 9 would have been false negatives and 10 a true negative
    assert r["confusion_matrix"] == {"TP": 3, "FP": 1, "FN": 0, "TN": 3}
    m = r["metrics"]
    assert (m["sensitivity"], m["specificity"], m["ppv"], m["npv"]) == (1.0, 0.75, 0.75, 1.0)
    assert m["f1"] == 0.8571 and m["ci"]["sensitivity"][1] == 1.0
    assert r["status"] == {
        **r["status"],
        "evaluation": "completed",
        "acceptance_criteria": "not_assessed",
        "human_review": "pending",
    }


def test_missing_laboratory_data_is_excluded_only_when_required(popb):
    b, did, _ = popb
    r = b.evaluate_against_reference(did, "chart", "bob", eligibility=EligibilityRules(require_data=["Measurement"]))
    # eligible patients without any lab record (2, 3, 4, 6) are excluded, never counted as negatives
    assert r["population"]["excluded_by_reason"]["missing_Measurement_data"] == 4
    assert r["confusion_matrix"] == {"TP": 2, "FP": 0, "FN": 0, "TN": 1}
    assert r["eligibility_rules"]["require_data"] == ["Measurement"]


def test_shorter_observation_requirement_changes_eligibility(popb):
    b, did, _ = popb
    r = b.evaluate_against_reference(did, "chart", "bob", eligibility=EligibilityRules(min_observation_days=30))
    assert r["population"]["excluded_by_reason"]["insufficient_observation"] == 0
    assert r["confusion_matrix"]["FN"] == 1  # person 9 is now observable: a reference case the algorithm missed


def test_unsupported_required_data_is_refused(popb):
    b, did, _ = popb
    b.ontology.capabilities["entities"] = [e for e in b.ontology.capabilities["entities"] if e != "Measurement"]
    with pytest.raises(ValueError, match="does not provide"):
        b.evaluate_against_reference(did, "chart", "bob", eligibility=EligibilityRules(require_data=["Measurement"]))


def test_reference_load_summary_is_suppressed(tmp_path):
    b = make_builder(tmp_path, build=build_population)  # min cell 10
    out = b.load_reference("chart", POP_LABELS, "synthetic", "root")
    assert out["patients"] == 12 and out["records"] == 14
    assert {out[k] for k in ("cases", "non_cases", "unknown", "conflicting")} <= {"<10", "suppressed", 0}


def test_snapshot_change_makes_evaluation_inconclusive(popb):
    b, did, _ = popb
    b.con.execute("UPDATE cdm.cdm_source SET cdm_release_date = DATE '2026-02-01'")
    r = b.evaluate_against_reference(did, "chart", "bob", intended_use="synthetic_demo")
    assert r["status"]["evaluation"] == "inconclusive" and "snapshot changed" in r["status"]["evaluation_reasons"][0]
    assert r["status"]["acceptance_criteria"] == "inconclusive"


# =============================== Phase 5: semantic hash ========================================
def _reordered(data: dict) -> dict:
    out = copy.deepcopy(data)
    for cs in out["concept_sets"]:
        cs["items"] = list(reversed(cs["items"]))
    out["concept_sets"] = list(reversed(out["concept_sets"]))
    out["evidence"] = list(reversed(out["evidence"]))
    out["scoring"]["weights"] = list(reversed(out["scoring"]["weights"]))
    rule = out["tiers"][0]["rule"]["all"]
    out["tiers"][0]["rule"]["all"] = list(reversed(rule))
    return {k: out[k] for k in reversed(list(out))}  # reversed key order too


def test_unordered_collections_and_key_order_do_not_change_hash():
    p = ProxyDefinition.from_file(EXAMPLE)
    q = ProxyDefinition.model_validate(_reordered(p.to_dict()))
    assert q.content_hash() != p.content_hash() and q.semantic_hash() == p.semantic_hash()


def test_duplicate_concept_items_and_equivalent_temporal_spelling():
    data = mini().to_dict()
    data["concept_sets"][0]["items"] = data["concept_sets"][0]["items"] * 2
    assert ProxyDefinition.model_validate(data).semantic_hash() == mini().semantic_hash()
    t1 = mini(
        temporal_rules=[
            {
                "id": "t",
                "name": "x",
                "a": "a",
                "b": "b",
                "relation": "before",
                "days": 30,
                "allow_same_day": True,
                "required": False,
            }
        ]
    )
    t2 = mini(
        temporal_rules=[
            {
                "id": "t",
                "name": "y",
                "a": "a",
                "b": "b",
                "relation": "between",
                "min_days": 0,
                "max_days": 30,
                "required": False,
            }
        ]
    )
    assert t1.semantic_hash() == t2.semantic_hash()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["evidence"][0]["window"].update(end_days=1),  # clinical condition / window
        lambda d: d["temporal_rules"][0].update(days=90),  # temporal relationship
        lambda d: d["evidence"][1].update(count=3),  # evidence threshold
        lambda d: d.update(tiers=[d["tiers"][1], d["tiers"][0], d["tiers"][2]]),  # tier priority
        lambda d: d["concept_sets"][3]["items"].pop(),  # concept set content
        lambda d: d["conflicts"][0].update(action="exclude"),
        lambda d: d["funnel"].reverse(),  # funnel is cumulative: order matters
    ],
)
def test_meaningful_changes_change_hash(change):
    data = ProxyDefinition.from_file(EXAMPLE).to_dict()
    base = ProxyDefinition.model_validate(data).semantic_hash()
    changed = copy.deepcopy(data)
    change(changed)
    assert ProxyDefinition.model_validate(changed).semantic_hash() != base


def test_descriptive_metadata_does_not_change_hash():
    p = ProxyDefinition.from_file(EXAMPLE)
    data = p.to_dict()
    data.update(clinical_notes="other notes", version="9.9", target={"name": "Renamed"}, assumptions=["x"])
    data["evidence"][0]["name"] = "Renamed evidence"
    data["evidence"][0]["provenance"] = {"clinical_rationale": "different text"}
    data["concept_sets"][0]["name"] = "Renamed set"
    data["concept_sets"][0]["version"] = "2.0"
    data["tiers"][0]["label"] = "Renamed label"
    data["acceptance_criteria"] = {}
    assert ProxyDefinition.model_validate(data).semantic_hash() == p.semantic_hash()


def test_hash_is_deterministic_and_pure():
    p = ProxyDefinition.from_file(EXAMPLE)
    before = p.model_dump()
    hashes = {ProxyDefinition.from_file(EXAMPLE).semantic_hash() for _ in range(5)} | {p.semantic_hash()}
    assert len(hashes) == 1 and p.model_dump() == before
    assert p.semantic_hash().startswith("sha256:")


# =============================== Phase 6: evaluation vs acceptance vs approval ====================
def test_acceptance_criteria_have_no_defaults():
    with pytest.raises(ValidationError, match="at least one threshold"):
        AcceptanceCriteria()
    with pytest.raises(ValidationError, match="confidence_level"):
        AcceptanceCriteria(min_ppv=0.5, confidence_level=0.8)


@pytest.fixture
def evb(tmp_path):
    b = make_builder(tmp_path, min_cell=1)
    did = approved(b)
    b.execute(did, "carol")
    b.load_reference("chart", LABELS, "synthetic chart review (test fixture)", "root")
    yield b, did
    b.con.close()


def test_structural_validation_passes_but_acceptance_fails(evb):
    b, did = evb
    r = b.evaluate_against_reference(
        did, "chart", "bob", intended_use="registry_linkage", criteria=AcceptanceCriteria(min_ppv=0.9)
    )
    assert b.proxy_status(did)["definition_validation"]["status"] == "passed"
    assert r["status"]["evaluation"] == "completed" and r["acceptance"]["status"] == "criteria_not_met"
    assert r["acceptance"]["checks"][0] == {
        "criterion": "min_ppv",
        "threshold": 0.9,
        "observed": 0.75,
        "basis": "point estimate",
        "passed": False,
    }
    with pytest.raises(ValueError, match="criteria were met"):
        b.review_evaluation(did, r["validation_id"], "dana", "accepted", "trying to accept a failure")


def test_sample_size_insufficient_is_inconclusive(evb):
    b, did = evb
    r = b.evaluate_against_reference(
        did, "chart", "bob", intended_use="x", criteria=AcceptanceCriteria(min_sensitivity=0.5, min_evaluated=50)
    )
    assert r["status"]["evaluation"] == "completed" and r["acceptance"]["status"] == "inconclusive"
    assert "insufficient sample" in r["acceptance"]["reasons"][0]


def test_sensitivity_threshold_not_met(evb):
    b, did = evb
    r = b.evaluate_against_reference(
        did, "chart", "bob", positive_tiers=["high"], intended_use="x", criteria=AcceptanceCriteria(min_sensitivity=0.9)
    )
    assert r["metrics"]["sensitivity"] == 0.6667 and r["acceptance"]["status"] == "criteria_not_met"


def test_all_criteria_met_then_human_review(evb):
    b, did = evb
    r = b.evaluate_against_reference(did, "chart", "bob", intended_use="synthetic_demo")
    assert r["acceptance"]["status"] == "criteria_met" and r["acceptance"]["criteria_source"].startswith("prespec")
    status = b.proxy_status(did)
    assert status["evaluations"][0]["human_review"]["decision"] == "pending"  # human approval absent
    assert status["accepted_for_intended_uses"] == []
    b.review_evaluation(did, r["validation_id"], "dana", "accepted", "criteria met for synthetic demo")
    status = b.proxy_status(did)
    assert status["accepted_for_intended_uses"] == ["synthetic_demo"]
    assert status["execution_approval"]["status"] == "approved"  # a different, separate status
    assert status["classification"]["value"] == "exploratory"  # acceptance never upgrades the label by itself


def test_prespecified_criteria_cannot_be_replaced(evb):
    b, did = evb
    with pytest.raises(ValueError, match="prespecified"):
        b.evaluate_against_reference(
            did, "chart", "bob", intended_use="synthetic_demo", criteria=AcceptanceCriteria(min_ppv=0.1)
        )


def test_confidence_lower_bound_is_stricter_for_small_samples(evb):
    b, did = evb
    point = b.evaluate_against_reference(
        did, "chart", "bob", intended_use="x", criteria=AcceptanceCriteria(min_sensitivity=0.6)
    )
    lower = b.evaluate_against_reference(
        did,
        "chart",
        "bob",
        intended_use="x",
        criteria=AcceptanceCriteria(min_sensitivity=0.6, use_confidence_lower_bound=True),
    )
    assert point["acceptance"]["status"] == "criteria_met" and lower["acceptance"]["status"] == "criteria_not_met"


def test_inconclusive_evaluation_cannot_be_accepted(tmp_path):
    b = make_builder(tmp_path)  # min cell 10: cells are small -> metrics withheld -> inconclusive
    did = approved(b)
    b.execute(did, "carol")
    b.load_reference("chart", LABELS, "synthetic", "root")
    r = b.evaluate_against_reference(did, "chart", "bob", intended_use="synthetic_demo")
    assert r["status"]["evaluation"] == "inconclusive" and r["acceptance"]["status"] == "inconclusive"
    with pytest.raises(ValueError, match="only a completed evaluation"):
        b.review_evaluation(did, r["validation_id"], "dana", "accepted", "should not be possible")
    b.review_evaluation(did, r["validation_id"], "dana", "rejected", "inconclusive: too few patients")


def test_definition_changed_after_approval_needs_new_approval(evb):
    b, did = evb
    ev = b.evaluate_against_reference(did, "chart", "bob", intended_use="synthetic_demo")
    b.review_evaluation(did, ev["validation_id"], "dana", "accepted", "criteria met for synthetic demo")
    v2 = b.submit_proxy(example(version="1.1", prior_observation_days=200), "alice")
    assert v2["status"] == "draft"
    with pytest.raises(GovernanceError):
        b.execute(v2["cohort_definition_id"], "carol")  # approval of 1.0 does not carry over
    status = b.proxy_status(v2["cohort_definition_id"])
    assert status["evaluations"] == [] and status["accepted_for_intended_uses"] == []
    claim = synthetic_provenance(
        version="1.2",
        prior_observation_days=200,
        classification="clinically_validated",
        validation_reference=ev["validation_id"],
    )
    with pytest.raises(GovernanceError, match="different algorithm logic"):
        b.submit_proxy(claim, "alice")


def test_superseded_versions(evb):
    b, did = evb
    v2 = approved(b, example(version="1.1", prior_observation_days=200))
    status = b.proxy_status(did)["lifecycle"]
    assert status["superseded"] and status["superseded_by_approved_versions"] == ["1.1"]
    assert not b.proxy_status(v2)["lifecycle"]["superseded"]
    out = b.execute(did, "carol")
    assert any("superseded" in c["message"] for c in out["caveats"])
    assert "superseded by v1.1" in b.proxy_status(did)["summary"]


def test_assess_acceptance_states():
    pop = {"evaluated": 10, "eligible_reference_positive": 4, "eligible_reference_negative": 6}
    m = classification_metrics(3, 1, 1, 5)
    assert assess_acceptance(None, pop, m, "completed")["status"] == "not_assessed"
    assert assess_acceptance(AcceptanceCriteria(min_ppv=0.7), pop, m, "completed")["status"] == "criteria_met"
    assert assess_acceptance(AcceptanceCriteria(min_ppv=0.7), pop, m, "inconclusive")["status"] == "inconclusive"
    assert assess_acceptance(AcceptanceCriteria(min_ppv=0.7), pop, None, "completed")["status"] == "inconclusive"
    none_pred = classification_metrics(0, 0, 4, 6)
    undefined = assess_acceptance(AcceptanceCriteria(min_ppv=0.1), pop, none_pred, "completed")
    assert undefined["status"] == "inconclusive" and "ppv is undefined" in undefined["reasons"]


# =============================== Phase 7: provenance ===========================================
def test_provenance_survives_parse_validation_compile_and_report(evb):
    b, did = evb
    p = ProxyDefinition.from_file(EXAMPLE)
    path_ev = p.evidence_by_id("squamous_path")
    assert path_ev.provenance and "Pathology results are often missing" in path_ev.provenance.limitations[0]
    assert p.concept_set("radiation").code_system_version == "synthetic-1"
    v = b.validate_proxy_definition(p)
    prov = [i["message"] for i in v["issues"] if i["stage"] == "provenance"]
    assert any("no clinical rationale recorded" in m for m in prov)  # missing provenance is flagged
    assert any("placeholder provenance" in m for m in prov)  # placeholder text is flagged, not accepted
    compiled = ProxyCompiler(b.ontology).compile_proxy(p)
    meta = {e["id"]: e for e in compiled.metadata["provenance"]["evidence"]}
    assert meta["squamous_path"]["provenance"]["limitations"]
    assert meta["esoph_dx"]["mandatory_for_membership"] and not meta["squamous_path"]["mandatory_for_membership"]
    assert "exclusion" in meta["competing_cancer"]["roles"] and "tier" in meta["squamous_path"]["roles"]
    packet = b.proxy_review_packet(did)
    assert packet["provenance"]["concept_sets"][0]["code_system"].startswith("PLACEHOLDER")
    assert packet["acceptance_criteria"]["synthetic_demo"]["min_sensitivity"] == 0.5
    report = b.evaluate_against_reference(did, "chart", "bob")
    assert {e["id"] for e in report["provenance"]["evidence"]} == {e.id for e in p.evidence}
    assert compiled.sql_hash == ProxyCompiler(b.ontology).compile_proxy(p).sql_hash  # metadata not in the hash


def test_supporting_evidence_is_not_mandatory():
    p = ProxyDefinition.from_file(EXAMPLE)
    assert p.mandatory_evidence() == {"esoph_dx"}  # only the entry rule is mandatory


def test_invalid_provenance_dates_rejected():
    sets = copy.deepcopy(mini().to_dict()["concept_sets"])
    sets[0].update(effective_from="2024-01-01", effective_to="2023-01-01")
    with pytest.raises(ValidationError, match="effective_from is after effective_to"):
        mini(concept_sets=sets)


# =============================== missing dates / overlapping evidence ============================
def build_dates(con):
    from proxy_fixtures import base_db

    base_db(con, [1, 2])
    ev = Events()
    ev.dx(1, 0)
    ev.dx(2, 0)
    ev.write(con)
    con.execute("INSERT INTO cdm.drug_exposure VALUES (900, 1, 2100000004, NULL, NULL, 21, 0, '', NULL)")
    con.execute("INSERT INTO cdm.drug_exposure VALUES (901, 2, 2100000004, DATE '2020-06-05', NULL, 21, 0, '', NULL)")


def test_missing_dates_never_qualify_and_overlapping_evidence(tmp_path):
    b = make_builder(tmp_path, build=build_dates, min_cell=1)
    ev = mini().to_dict()["evidence"]
    ev.append(
        {
            "id": "b2",
            "name": "B again",
            "entity": "DrugExposure",
            "concept_set_id": "chemo",
            "window": {"start_days": 0, "end_days": 30},
        }
    )
    p = mini(evidence=ev, tiers=[{"name": "t", "rule": {"all": [{"evidence": "b"}, {"evidence": "b2"}]}}])
    compiled = ProxyCompiler(b.ontology).compile_proxy(p)
    rows = b.con.execute(compiled.assignment_sql).fetchall()
    assert [r[0] for r in rows] == [2]  # person 1's drug record has no date: it is not evidence
    b.con.close()


# =============================== API / MCP =====================================================
USERS = {
    "alice": (["author"], "research"),
    "bob": (["reviewer"], "research"),
    "dana": (["reviewer"], "research"),
    "carol": (["executor"], "research"),
    "victor": (["viewer"], "research"),
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
    client = TestClient(create_app(b, SecurityConfig(tokens_file=path)), raise_server_exceptions=False)
    yield client, toks, b
    b.con.close()


def test_api_statuses_and_acceptance_review(api):
    c, t, b = api
    did = approved(b, tenant="research")
    b.execute(did, "carol")
    labels = [{"person_id": pid, "label": "case" if case else "non_case"} for pid, case in LABELS]
    labels += [{"person_id": 1, "is_case": True}, {"person_id": 99}]  # duplicate + unknown
    r = c.post(
        "/proxy-references",
        json={"name": "chart", "source": "synthetic chart review", "labels": labels},
        headers=t["root"],
    )
    assert r.status_code == 200 and r.json()["unknown"] == 1
    bad = c.post(
        "/proxy-references",
        json={"name": "x", "source": "synthetic", "labels": [{"person_id": 1, "label": "case", "is_case": True}]},
        headers=t["root"],
    )
    assert bad.status_code == 422
    ev = c.post(
        f"/proxy-cohorts/{did}/reference-validation",
        json={"reference_name": "chart", "intended_use": "synthetic_demo"},
        headers=t["bob"],
    )
    assert ev.status_code == 200, ev.text
    body = ev.json()
    assert body["status"]["evaluation"] == "completed" and body["acceptance"]["status"] == "criteria_met"
    assert body["population"]["excluded_by_reason"]["unknown_reference_label"] == 1
    assert body["metrics"]["f1"] == 0.8571 and body["metrics"]["ci"]["ppv"][0] < 0.75
    none_pos = c.post(
        f"/proxy-cohorts/{did}/reference-validation",
        json={
            "reference_name": "chart",
            "positive_tiers": ["exploratory"],
            "intended_use": "x",
            "criteria": {"min_ppv": 0.5},
        },
        headers=t["bob"],
    ).json()
    assert none_pos["metrics"]["sensitivity"] == 0.0 and none_pos["metrics"]["ppv"] == 0.0
    assert none_pos["metrics"]["f1"] == 0.0  # zero, not null
    vid = body["validation_id"]
    url = f"/proxy-cohorts/{did}/evaluations/{vid}/review"
    review = {"decision": "accepted", "rationale": "criteria met for the synthetic demo"}
    assert c.post(url, json=review, headers=t["alice"]).status_code == 403  # not a reviewer
    assert c.post(url, json=review, headers=t["bob"]).status_code == 403  # ran the evaluation
    assert c.post(url, json={**review, "reviewer": "dana"}, headers=t["dana"]).status_code == 422  # no identity field
    assert c.post(url, json={"decision": "accepted", "rationale": "short"}, headers=t["dana"]).status_code == 422
    ok = c.post(url, json=review, headers=t["dana"])
    assert ok.status_code == 200 and ok.json()["reviewer"] == "dana"
    assert c.post(url, json=review, headers=t["dana"]).status_code == 409  # recorded once
    st = c.get(f"/proxy-cohorts/{did}/status", headers=t["victor"])
    assert st.status_code == 200
    s = st.json()
    assert s["definition_validation"]["status"] == "passed" and s["execution_approval"]["status"] == "approved"
    reviews = {e["validation_id"]: e for e in s["evaluations"]}
    assert reviews[vid]["human_review"]["decision"] == "accepted" and reviews[vid]["acceptance_status"] == (
        "criteria_met"
    )
    assert c.get(f"/proxy-cohorts/{did}/evaluations", headers=t["victor"]).status_code == 403
    assert len(c.get(f"/proxy-cohorts/{did}/evaluations", headers=t["carol"]).json()) == 2
    assert "person_id" not in json.dumps(s) and "subject_id" not in json.dumps(body)


def test_api_rejects_duplicate_yaml_keys(api):
    c, t, _ = api
    bad = YAML.replace("groups:\n", "groups:\n  strong_treatment: {evidence: chemo}\n", 1)
    r = c.post("/proxy-cohorts/validate", json={"yaml": bad}, headers=t["alice"])
    assert r.status_code == 422 and "duplicate key 'strong_treatment'" in r.json()["detail"]


def test_mcp_status_is_read_only_and_no_acceptance_tool(pb):
    async def go():
        async with Client(
            __import__("cohort_builder.mcp_server", fromlist=["create_server"]).create_server(pb, "agent")
        ) as c:
            tools = {x.name: x for x in (await c.list_tools()).tools}
            assert "get_proxy_status" in tools and tools["get_proxy_status"].annotations.read_only_hint
            assert not any(("accept" in n or "review_evaluation" in n) for n in tools)
            did = approved(pb)
            r = await c.call_tool("get_proxy_status", {"definition_id": did})
            data = r.structured_content or json.loads(r.content[0].text)
            assert data["execution_approval"]["status"] == "approved" and data["evaluations"] == []

    asyncio.run(go())


def test_evaluation_endpoints_are_tenant_scoped(api, tmp_path):
    c, t, b = api
    did = approved(b, tenant="research")
    b.execute(did, "carol")
    b.load_reference("chart", LABELS, "synthetic", "root", tenant="research")
    vid = b.evaluate_against_reference(did, "chart", "bob", intended_use="synthetic_demo")["validation_id"]
    token, entry = issue_token("mallory", ["reviewer", "executor"], "other", expires_days=1)
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump({"tokens": [entry]}))
    other = TestClient(create_app(b, SecurityConfig(tokens_file=path)), raise_server_exceptions=False)
    hdr = {"Authorization": f"Bearer {token}"}
    review = {"decision": "accepted", "rationale": "cross-tenant attempt to accept"}
    assert other.post(f"/proxy-cohorts/{did}/evaluations/{vid}/review", json=review, headers=hdr).status_code == 404
    assert other.get(f"/proxy-cohorts/{did}/status", headers=hdr).status_code == 404
    assert (
        other.post(
            f"/proxy-cohorts/{did}/reference-validation", json={"reference_name": "chart"}, headers=hdr
        ).status_code
        == 404
    )
