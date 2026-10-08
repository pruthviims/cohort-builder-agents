"""Clinical / data-coverage caveats: absence of evidence, lookback beyond observed data, partial capture,
missing observation definitions, and caveats carried into execution metadata."""

from __future__ import annotations

import copy
import json
import shutil

import pytest
import yaml

from cohort_builder.agents.validator import validate
from cohort_builder.config import REPO_ROOT
from cohort_builder.ir import CohortDefinition
from cohort_builder.ontology import Ontology

BASE = json.loads((REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json").read_text())
SERT = json.loads((REPO_ROOT / "eval" / "gold" / "sertraline_depression.json").read_text())


def warnings(issues) -> list[str]:
    return [i.message for i in issues if i.severity == "warning" and i.stage == "data"]


def with_exclusion(base: dict, window: dict, prior: int = 365, post: int = 0) -> CohortDefinition:
    data = copy.deepcopy(base)
    data["prior_observation_days"], data["post_observation_days"] = prior, post
    data["exclusion"] = [
        {
            "id": "x1",
            "name": "prior depression",
            "entity": "ConditionOccurrence",
            "concept_set_id": "cs_depression",
            "window": window,
        }
    ]
    return CohortDefinition.model_validate(data)


def test_absence_of_record_is_flagged_with_dataset_specific_basis(builder, laad_builder):
    ir = with_exclusion(SERT, {"start_days": -180, "end_days": -31})
    omop = warnings(validate(ir, builder.ontology, builder.vocab)[0])
    assert any("'no record' is treated as 'did not happen'" in w and "outside this data source" in w for w in omop)
    laad = warnings(validate(ir, laad_builder.ontology, laad_builder.vocab)[0])
    assert any("inferred from claim activity" in w and "weak evidence" in w for w in laad)


def test_absence_based_inclusion_is_flagged_too(builder):
    data = copy.deepcopy(SERT)
    data["inclusion"][0].update({"occurrence": "exactly", "count": 0})
    w = warnings(validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)[0])
    assert any("'no record' is treated as 'did not happen'" in m for m in w)


def test_presence_based_criteria_have_no_absence_caveat(builder):
    w = warnings(validate(CohortDefinition.model_validate(SERT), builder.ontology, builder.vocab)[0])
    assert not any("did not happen" in m for m in w)


@pytest.mark.parametrize(
    "window,prior,expect",
    [
        ({"start_days": -730, "end_days": 0}, 365, "looks back 730 days but only 365"),
        ({"start_days": None, "end_days": 0}, 0, "history may be empty"),
        ({"start_days": None, "end_days": 0}, 365, "guaranteed to be at least 365 days"),
    ],
)
def test_lookback_beyond_observable_history(builder, window, prior, expect):
    w = warnings(validate(with_exclusion(SERT, window, prior=prior), builder.ontology, builder.vocab)[0])
    assert any(expect in m for m in w), w


def test_no_lookback_caveat_when_observation_covers_window(builder):
    w = warnings(
        validate(with_exclusion(SERT, {"start_days": -365, "end_days": 0}, prior=365), builder.ontology, builder.vocab)[
            0
        ]
    )
    assert not any("looks back" in m for m in w)


def test_follow_up_beyond_required_observation(builder):
    w = warnings(
        validate(with_exclusion(SERT, {"start_days": 0, "end_days": 90}, post=0), builder.ontology, builder.vocab)[0]
    )
    assert any("looks 90 days after index but only 0 days of follow-up" in m for m in w)
    w = warnings(
        validate(with_exclusion(SERT, {"start_days": 0, "end_days": 90}, post=90), builder.ontology, builder.vocab)[0]
    )
    assert not any("days after index" in m for m in w)


def test_presence_lookback_beyond_history_warns_about_undercounting(builder):
    data = copy.deepcopy(SERT)
    data["inclusion"][0]["window"] = {"start_days": -730, "end_days": 0}
    w = warnings(validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab)[0])
    assert any("under-counted" in m for m in w)


def test_partially_captured_labs_are_flagged(builder):
    w = warnings(validate(CohortDefinition.model_validate(BASE), builder.ontology, builder.vocab)[0])
    assert any("only partially captured" in m and "not evidence of a normal result" in m for m in w)


def test_dataset_without_observation_definition_is_an_error(tmp_path):
    target = tmp_path / "ontology"
    shutil.copytree(REPO_ROOT / "ontology", target)
    path = target / "datasets" / "omop_demo.yaml"
    prof = yaml.safe_load(path.read_text())
    prof["mapping"].pop("observation_period")
    prof["capabilities"]["observation"] = "none"
    path.write_text(yaml.safe_dump(prof))
    ont = Ontology.load(target)

    class NoVocab:  # vocabulary checks are not under test here
        def concepts(self, ids):
            return {}

        def version(self):
            return "n/a"

    issues, attrition = validate(CohortDefinition.model_validate(SERT), ont, NoVocab())  # type: ignore[arg-type]
    assert attrition is None
    assert any(i.stage == "dataset" and "defines no observation periods" in i.message for i in issues)


def test_caveats_travel_with_execution_metadata(builder):
    ir = with_exclusion(SERT, {"start_days": -730, "end_days": -31})
    def_id, _ = builder.submit_ir(ir, "alice")
    builder.review(def_id, "bob", "approved")
    out = builder.execute(def_id, "carol")
    assert out["observation"] == "observation_period"
    assert any("did not happen" in c["message"] for c in out["caveats"])
    stored = json.loads(builder.con.execute("SELECT caveats_json FROM meta.cohort_generation").fetchone()[0])
    assert stored == out["caveats"]


def test_unsupported_capabilities_on_claims_data(laad_builder):
    # labs on open claims: hard error from the dataset capabilities, not an empty cohort
    issues, _ = validate(CohortDefinition.model_validate(BASE), laad_builder.ontology, laad_builder.vocab)
    assert any(i.stage == "dataset" and "Measurement" in i.message for i in issues if i.severity == "error")
