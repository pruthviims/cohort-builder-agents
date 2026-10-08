"""IR model validation (structural rules) and validator rules (units, genders, vocabulary). Positive + negative."""
from __future__ import annotations

import copy
import json
import math

import pytest
from pydantic import ValidationError

from cohort_builder.agents.composer import compose, format_validation_error
from cohort_builder.agents.validator import validate
from cohort_builder.config import REPO_ROOT
from cohort_builder.ir import CohortDefinition, CohortExit, ConceptSet, Criterion, Demographics, ValueFilter, Window

BASE = json.loads((REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json").read_text())


def ir_with(**changes) -> dict:
    data = copy.deepcopy(BASE)
    for path, value in changes.items():
        node = data
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[int(k)] if k.isdigit() else node[k]
        last = keys[-1]
        node[int(last) if last.isdigit() else last] = value
    return data


def errors_of(data: dict) -> str:
    with pytest.raises(ValidationError) as exc:
        CohortDefinition.model_validate(data)
    return " | ".join(format_validation_error(exc.value))


def crit(**kw) -> dict:
    base = {"id": "c1", "name": "rule", "entity": "Measurement", "concept_set_id": "cs_hba1c"}
    return {**base, **kw}


# ---- ages -------------------------------------------------------------------------
@pytest.mark.parametrize("age_min,age_max", [(18, None), (None, 65), (18, 65), (40, 40), (0, 150), (None, None)])
def test_valid_age_ranges(age_min, age_max):
    assert Demographics(age_min=age_min, age_max=age_max)


@pytest.mark.parametrize("age_min,age_max,msg", [(-1, None, "greater than or equal to 0"),
                                                 (None, -5, "greater than or equal to 0"),
                                                 (70, 18, "age_min (70) must be <= age_max (18)"),
                                                 (200, None, "less than or equal to 150")])
def test_invalid_age_ranges(age_min, age_max, msg):
    with pytest.raises(ValidationError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        Demographics(age_min=age_min, age_max=age_max)


def test_gender_ids_normalized_and_validated(builder):
    assert Demographics(gender_concept_ids=[8532, 8507, 8532]).gender_concept_ids == [8507, 8532]
    with pytest.raises(ValidationError):
        Demographics(gender_concept_ids=[0])
    issues, _ = validate(CohortDefinition.model_validate(ir_with(demographics__gender_concept_ids=[8551])),
                         builder.ontology, builder.vocab)
    assert any("gender concept 8551" in i.message for i in issues if i.severity == "error")


# ---- value filters -------------------------------------------------------------------
@pytest.mark.parametrize("vf", [{"op": ">", "value": 8, "unit_concept_id": 8554},
                                {"op": "between", "value": 7, "value_high": 9, "unit_concept_id": 8554},
                                {"op": "between", "value": 7, "value_high": 7, "unit_concept_id": 8554},
                                {"op": "=", "value": 0, "unit_concept_id": 8554}])
def test_valid_value_filters(vf):
    assert ValueFilter(**vf)


@pytest.mark.parametrize("vf,msg", [
    ({"op": "between", "value": 7, "unit_concept_id": 8554}, "requires value_high"),
    ({"op": "between", "value": 9, "value_high": 7, "unit_concept_id": 8554}, "greater than upper bound"),
    ({"op": ">", "value": 8, "value_high": 9, "unit_concept_id": 8554}, "only used with op 'between'"),
    ({"op": ">", "value": math.nan, "unit_concept_id": 8554}, "finite number"),
    ({"op": ">", "value": math.inf, "unit_concept_id": 8554}, "finite number"),
    ({"op": ">", "value": 8, "unit_concept_id": 0}, "greater than 0"),
    ({"op": "like", "value": 8, "unit_concept_id": 8554}, "Input should be"),
    ({"op": ">", "value": "8; DROP TABLE x", "unit_concept_id": 8554}, "valid number"),
])
def test_invalid_value_filters(vf, msg):
    with pytest.raises(ValidationError, match=msg):
        ValueFilter(**vf)


# ---- windows -------------------------------------------------------------------------
@pytest.mark.parametrize("s,e", [(-365, 0), (0, 0), (0, 90), (None, 0), (-30, None), (None, None), (-36600, 36600)])
def test_valid_windows(s, e):
    assert Window(start_days=s, end_days=e)


@pytest.mark.parametrize("s,e,msg", [(10, -10, "must be <= end_days"), (-40000, 0, "greater than or equal"),
                                     (0, 99999, "less than or equal")])
def test_invalid_windows(s, e, msg):
    with pytest.raises(ValidationError, match=msg):
        Window(start_days=s, end_days=e)


# ---- observation requirements and exit -------------------------------------------------
@pytest.mark.parametrize("field,value", [("prior_observation_days", -1), ("post_observation_days", -30),
                                         ("prior_observation_days", 100000)])
def test_invalid_observation_requirements(field, value):
    assert field in errors_of(ir_with(**{field: value}))


def test_exit_rules():
    assert CohortExit(type="fixed_days", days=30)
    with pytest.raises(ValidationError, match="requires days"):
        CohortExit(type="fixed_days")
    with pytest.raises(ValidationError, match="only used with type 'fixed_days'"):
        CohortExit(type="end_of_observation", days=30)
    with pytest.raises(ValidationError):
        CohortExit(type="fixed_days", days=0)


# ---- occurrence rules -------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"occurrence": "at_least", "count": 1}, {"occurrence": "at_most", "count": 0},
                                {"occurrence": "exactly", "count": 0}, {"occurrence": "exactly", "count": 3},
                                {"occurrence": "at_least", "count": 2, "min_span_days": 30}])
def test_valid_occurrence_rules(kw):
    assert Criterion(**crit(**kw))


@pytest.mark.parametrize("kw,msg", [
    ({"occurrence": "at_least", "count": 0}, "always true"),
    ({"count": -1}, "greater than or equal to 0"),
    ({"count": 10**6}, "less than or equal"),
    ({"occurrence": "at_most", "count": 2, "min_span_days": 30}, "min_span_days needs"),
    ({"occurrence": "at_least", "count": 1, "min_span_days": 30}, "min_span_days needs"),
    ({"occurrence": "at_least", "count": 2, "min_span_days": 0}, "greater than or equal to 1"),
    ({"occurrence": "sometimes"}, "Input should be"),
    ({"claim_status": []}, "at least one status"),
])
def test_invalid_occurrence_rules(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        Criterion(**crit(**kw))


def test_claim_status_normalized():
    assert Criterion(**crit(claim_status=["rejected", "paid", "paid"])).claim_status == ["paid", "rejected"]


# ---- concept sets and references ---------------------------------------------------------
@pytest.mark.parametrize("bad_id", ["x'; DROP TABLE cdm.person; --", "has space", "", "a" * 65, "ünïcode"])
def test_concept_set_ids_cannot_carry_sql(bad_id):
    with pytest.raises(ValidationError):
        ConceptSet(id=bad_id, name="n", domain="Drug", items=[{"concept_id": 1}])


def test_concept_set_item_rules():
    with pytest.raises(ValidationError, match="greater than 0"):
        ConceptSet(id="a", name="n", domain="Drug", items=[{"concept_id": 0}])
    with pytest.raises(ValidationError, match="only excluded items"):
        ConceptSet(id="a", name="n", domain="Drug", items=[{"concept_id": 1, "is_excluded": True}])
    with pytest.raises(ValidationError, match="both included and excluded"):
        ConceptSet(id="a", name="n", domain="Drug", items=[{"concept_id": 1}, {"concept_id": 1, "is_excluded": True}])
    assert ConceptSet(id="a", name="n", domain="Drug", items=[{"concept_id": 2_000_005_001}])  # local ids are fine


def test_references_and_duplicates():
    assert "unknown concept set 'nope'" in errors_of(ir_with(inclusion__0__concept_set_id="nope"))
    assert "index_event references unknown" in errors_of(ir_with(index_event__concept_set_id="nope"))
    dup = copy.deepcopy(BASE)
    dup["concept_sets"].append(copy.deepcopy(dup["concept_sets"][0]))
    assert "duplicate concept set ids" in errors_of(dup)
    dup2 = copy.deepcopy(BASE)
    dup2["exclusion"][0]["id"] = dup2["inclusion"][0]["id"]
    assert "duplicate criterion ids" in errors_of(dup2)


def test_in_place_edits_are_validated():
    ir = CohortDefinition.model_validate(BASE)
    with pytest.raises(ValidationError):
        ir.demographics.age_min = -3
    with pytest.raises(ValidationError):
        ir.inclusion[0].count = -1


def test_error_messages_identify_the_field():
    msg = errors_of(ir_with(inclusion__1__window={"start_days": 5, "end_days": -5}))
    assert msg.startswith("inclusion.1.window") and "start_days (5) must be <= end_days (-5)" in msg


# ---- validator: concepts, units, labs -------------------------------------------------
def test_unknown_and_wrong_domain_concepts(builder):
    ir = CohortDefinition.model_validate(ir_with(concept_sets__0__items=[{"concept_id": 1999999999}]))
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    assert any("not in the loaded vocabulary" in i.message for i in issues)
    ir = CohortDefinition.model_validate(ir_with(concept_sets__3__items=[{"concept_id": 201826}]))  # dx in lab set
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    assert any("domain Condition != concept set domain Measurement" in i.message for i in issues)


def test_classification_concept_needs_descendants(builder):
    ir = CohortDefinition.model_validate(
        ir_with(concept_sets__0__items=[{"concept_id": 2000006001, "include_descendants": False}]))
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    assert any("classification concept" in i.message for i in issues if i.severity == "error")


def test_threshold_in_non_canonical_unit_is_rejected_with_conversion_hint(builder):
    ir = CohortDefinition.model_validate(ir_with(inclusion__1__value_filter={
        "op": ">", "value": 64, "unit_concept_id": 2000001001}))
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    msg = next(i.message for i in issues if "canonical unit" in i.message)
    assert "mmol/mol" in msg and "= 8.007 %" in msg


def test_unit_not_in_ontology_is_rejected(builder):
    ir = CohortDefinition.model_validate(ir_with(inclusion__1__value_filter={
        "op": ">", "value": 8, "unit_concept_id": 999}))
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    assert any("not declared in ontology" in i.message for i in issues)


def test_value_filter_spanning_analytes_with_different_units(builder):
    ir = CohortDefinition.model_validate(ir_with(concept_sets__3__items=[{"concept_id": 3004410},
                                                                         {"concept_id": 3016723}]))
    issues, _ = validate(ir, builder.ontology, builder.vocab)
    assert any("spans analytes with different units" in i.message for i in issues)


def test_measurement_without_unit_metadata_warns(builder, monkeypatch):
    # a lab with no analyte entry: results in other units are excluded, and the reviewer is told so
    monkeypatch.delitem(builder.ontology.analytes, 3004410)
    data = ir_with(concept_sets__3__items=[{"concept_id": 3004410}])
    issues, _ = validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)
    assert any("no unit-conversion metadata" in i.message and i.severity == "warning" for i in issues)


def test_value_filter_on_non_measurement_is_rejected(builder):
    data = ir_with(exclusion__0__value_filter={"op": ">", "value": 1, "unit_concept_id": 8554})
    issues, _ = validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)
    assert any("value filters are only allowed" in i.message for i in issues)


# ---- composer: invalid agent output becomes feedback, not a crash ---------------------------
def test_composer_reports_invalid_intent_as_issues(builder):
    from cohort_builder.agents.intent import CohortIntent
    from cohort_builder.agents.resolver import ResolvedConceptSet
    from cohort_builder.ir import ConceptSetItem

    intent = CohortIntent.model_validate({
        "name": "x", "description": "", "index_mention_key": "met", "age_min": 70, "age_max": 18,
        "mentions": [{"key": "met", "text": "metformin", "entity": "DrugExposure"}], "criteria": []})
    resolved = {"met": ResolvedConceptSet(mention_key="met", name="metformin", domain="Drug", source="resolved",
                                          items=[ConceptSetItem(concept_id=1503297)])}
    ir, issues = compose(intent, resolved, builder.ontology, builder.vocab)
    assert ir is None
    assert any(i.stage == "intent" and "age_min (70) must be <= age_max (18)" in i.message for i in issues)
