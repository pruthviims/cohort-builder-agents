"""Proxy cohort engine: model validation, boolean / N-of-M / temporal semantics with exact
boundaries, tiers, scoring, conflicts, dataset capabilities, absence warnings and SQL safety.

Synthetic data and placeholder concepts only (see proxy_fixtures.py)."""

from __future__ import annotations

import copy
import os
import re

import pytest
from pydantic import ValidationError

from cohort_builder.agents.proxy_validator import ABSENCE_WEAK, truth, validate_proxy
from cohort_builder.executor import guard_select
from cohort_builder.ontology import Ontology
from cohort_builder.proxy import Expr, ProxyDefinition
from cohort_builder.proxy_compiler import ProxyCompiler
from cohort_builder.vocab import Vocabulary
from proxy_fixtures import (
    ADENO,
    CHEMO_A,
    DX,
    EXAMPLE,
    EXPECTED,
    FLAGGED,
    RADIATION,
    SQUAMOUS,
    Events,
    base_db,
    make_builder,
)

CONCEPT_SETS = [
    {"id": "dx", "name": "PH dx", "domain": "Condition", "items": [{"concept_id": DX}]},
    {"id": "chemo", "name": "PH chemo", "domain": "Drug", "items": [{"concept_id": CHEMO_A}]},
    {"id": "rad", "name": "PH radiation", "domain": "Procedure", "items": [{"concept_id": RADIATION}]},
    {"id": "path", "name": "PH squamous", "domain": "Measurement", "items": [{"concept_id": SQUAMOUS}]},
    {"id": "adeno", "name": "PH adeno", "domain": "Measurement", "items": [{"concept_id": ADENO}]},
]


def mini(**over) -> ProxyDefinition:
    """Generic test algorithm: a = dx, b = chemo, c = radiation, d = pathology; unbounded windows."""
    data = {
        "algorithm_name": "mini",
        "version": "1.0",
        "target": {"name": "Synthetic target"},
        "dataset_profile": "escc_synthetic",
        "ontology_version": "1.2.0",
        "vocabulary_version": "test",
        "concept_sets": CONCEPT_SETS,
        "index_event": {"entity": "ConditionOccurrence", "concept_set_id": "dx"},
        "evidence": [
            {"id": "a", "name": "A dx", "entity": "ConditionOccurrence", "concept_set_id": "dx"},
            {
                "id": "b",
                "name": "B chemo",
                "category": "treatment",
                "entity": "DrugExposure",
                "concept_set_id": "chemo",
            },
            {
                "id": "c",
                "name": "C radiation",
                "category": "procedure",
                "entity": "ProcedureOccurrence",
                "concept_set_id": "rad",
            },
            {
                "id": "d",
                "name": "D pathology",
                "category": "pathology",
                "entity": "Measurement",
                "concept_set_id": "path",
            },
            {
                "id": "e",
                "name": "E adeno",
                "category": "conflicting",
                "entity": "Measurement",
                "concept_set_id": "adeno",
            },
        ],
        "tiers": [{"name": "t", "rule": {"evidence": "a"}}],
    }
    data.update(over)
    return ProxyDefinition.model_validate(data)


# ---- scenario database: one person per scenario, index dx on 2020-06-01 (day 0) -------------------
DELTAS = {101: -31, 102: -30, 103: -1, 104: 0, 105: 1, 106: 30, 107: 31}  # chemo day relative to the dx
NOF = {301: "", 302: "b", 303: "bc", 304: "bcd"}


def build_scenarios(con) -> None:
    persons = [*DELTAS, 201, 202, 203, 204, 205, *NOF, 305, 306, 307, 401]
    base_db(con, persons)
    ev = Events()
    for p, delta in DELTAS.items():
        ev.dx(p, 0)
        ev.rx(p, delta)
    # max_span / min_span: second diagnosis at +180 / +181 / (+100, +181) / +29 / +30
    for p, days in {201: [180], 202: [181], 203: [100, 181], 204: [29], 205: [30]}.items():
        ev.dx(p, 0)
        for d in days:
            ev.dx(p, d)
    for p, kinds in NOF.items():
        ev.dx(p, 0)
        if "b" in kinds:
            ev.rx(p, 10)
        if "c" in kinds:
            ev.px(p, 10)
        if "d" in kinds:
            ev.lab(p, 10)
    ev.dx(305, 0)
    ev.rx(305, 10)
    ev.px(305, 101)  # noqa: E702  chemo/radiation 91 days apart
    ev.dx(306, 0)
    ev.rx(306, 10)
    ev.px(306, 100)  # noqa: E702  90 days apart
    ev.dx(307, 0)
    ev.rx(307, 10)
    ev.lab(307, 12, ADENO)  # noqa: E702  conflicting evidence
    ev.dx(401, 0)  # many duplicate records: must not multiply rows
    for i in range(40):
        ev.rx(401, 1 + i % 5)
        ev.dx(401, i % 3)
    ev.write(con)


@pytest.fixture(scope="module")
def sb(tmp_path_factory):
    b = make_builder(tmp_path_factory.mktemp("proxy_scen"), build=build_scenarios, min_cell=1)
    yield b
    b.con.close()


def assign(b, p: ProxyDefinition) -> dict[int, tuple[str, int]]:
    compiled = ProxyCompiler(b.ontology).compile_proxy(p)
    rows = b.con.execute(guard_select(compiled.assignment_sql)).fetchall()
    return {int(r[0]): (r[1], int(r[2])) for r in rows}


def members(b, p: ProxyDefinition, among) -> set[int]:
    return set(assign(b, p)) & set(among)


# ---- model validation ------------------------------------------------------------------------------
def test_example_yaml_loads_and_is_exploratory():
    p = ProxyDefinition.from_file(EXAMPLE)
    assert p.classification == "exploratory"
    assert "validated" not in p.label.lower()
    assert all(c.items[0].concept_id >= 2_100_000_000 for c in p.concept_sets)  # placeholders only


@pytest.mark.parametrize(
    "change, message",
    [
        ({"tiers": [{"name": "t", "rule": {"evidence": "zzz"}}]}, "unknown evidence 'zzz'"),
        ({"tiers": [{"name": "t", "rule": {"evidence": "a", "any": [{"evidence": "b"}]}}]}, "exactly one of"),
        (
            {"tiers": [{"name": "t", "rule": {"at_least": {"n": 3, "of": [{"evidence": "a"}, {"evidence": "b"}]}}}]},
            "can never be satisfied",
        ),
        (
            {"tiers": [{"name": "t", "rule": {"at_most": {"n": 1, "within_days": 5, "of": [{"evidence": "a"}]}}}]},
            "within_days is only supported with at_least",
        ),
        ({"groups": {"g1": {"group": "g2"}, "g2": {"group": "g1"}}}, "cycle"),
        ({"tiers": [{"name": "t", "min_score": 2}]}, "no scoring is defined"),
        ({"classification": "clinically_validated"}, "requires validation_reference"),
        ({"temporal_rules": [{"id": "x", "name": "x", "a": "a", "b": "a", "relation": "before"}]}, "must differ"),
        ({"temporal_rules": [{"id": "x", "name": "x", "a": "a", "b": "b", "relation": "between"}]}, "needs min_days"),
        (
            {
                "temporal_rules": [
                    {"id": "x", "name": "x", "a": "a", "b": "b", "relation": "between", "min_days": 5, "max_days": 1}
                ]
            },
            "empty range",
        ),
        ({"algorithm_name": "Bad-Name"}, "algorithm_name"),
        ({"version": "v1"}, "version"),
    ],
)
def test_invalid_definitions_rejected(change, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        mini(**change)


def test_evidence_ids_cannot_inject_sql():
    with pytest.raises(ValidationError):
        mini(evidence=[{"id": "a; DROP TABLE x", "name": "x", "entity": "ConditionOccurrence", "concept_set_id": "dx"}])
    with pytest.raises(ValidationError, match="invalid attribute code"):
        mini(
            evidence=[
                {
                    "id": "a",
                    "name": "x",
                    "entity": "ConditionOccurrence",
                    "concept_set_id": "dx",
                    "place_of_service": ["11' OR '1'='1"],
                }
            ]
        )


def test_max_span_requires_repeat_count():
    with pytest.raises(ValidationError, match="max_span_days needs"):
        mini(
            evidence=[
                {"id": "a", "name": "x", "entity": "ConditionOccurrence", "concept_set_id": "dx", "max_span_days": 30}
            ]
        )


def test_semantic_hash_ignores_wording_but_not_logic():
    p = mini()
    renamed = mini(target={"name": "Other words"}, clinical_notes="notes", version="2.0")
    assert p.semantic_hash() == renamed.semantic_hash() and p.content_hash() != renamed.content_hash()
    changed = mini(tiers=[{"name": "t", "rule": {"evidence": "b"}}])
    assert changed.semantic_hash() != p.semantic_hash()


def test_not_alias_round_trips():
    p = mini(tiers=[{"name": "t", "rule": {"all": [{"evidence": "a"}, {"not": {"evidence": "d"}}]}}])
    again = ProxyDefinition.model_validate(p.to_dict())
    assert again == p and '"not"' in p.canonical_json()


# ---- temporal semantics and boundaries -----------------------------------------------------------------
@pytest.mark.parametrize(
    "rule, holds",
    [
        ({"relation": "before", "days": 30}, {105, 106}),  # b - a in [1, 30]
        ({"relation": "before", "days": 30, "allow_same_day": True}, {104, 105, 106}),  # [0, 30]
        ({"relation": "before"}, {105, 106, 107}),  # [1, inf)
        ({"relation": "after", "days": 30}, {102, 103}),  # [-30, -1]
        ({"relation": "within", "days": 30}, {102, 103, 104, 105, 106}),  # [-30, 30]
        ({"relation": "same_day"}, {104}),  # [0, 0]
        ({"relation": "between", "min_days": -1, "max_days": 1}, {103, 104, 105}),
        ({"relation": "between", "min_days": 31}, {107}),
    ],
)
def test_temporal_relations_inclusive_boundaries(sb, rule, holds):
    p = mini(
        temporal_rules=[{"id": "tr", "name": "tr", "a": "a", "b": "b", **rule, "required": False}],
        tiers=[{"name": "t", "rule": {"temporal": "tr"}}],
    )
    assert members(sb, p, DELTAS) == holds


def test_required_temporal_rule_is_attrition_step(sb):
    p = mini(
        temporal_rules=[
            {"id": "tr", "name": "Chemo within 30 days", "a": "a", "b": "b", "relation": "within", "days": 30}
        ]
    )
    compiled = ProxyCompiler(sb.ontology).compile_proxy(p)
    assert "Temporal: Chemo within 30 days" in compiled.rule_names
    assert members(sb, p, DELTAS) == {102, 103, 104, 105, 106}


@pytest.mark.parametrize(
    "window, holds",
    [
        ({"start_days": 0, "end_days": 30}, {104, 105, 106}),  # inclusive both ends
        ({"start_days": -30, "end_days": -1}, {102, 103}),
        ({"start_days": None, "end_days": 0}, {101, 102, 103, 104}),  # lookback, unbounded
        ({"start_days": 1, "end_days": None}, {105, 106, 107}),  # follow-up, unbounded
    ],
)
def test_evidence_windows_lookback_followup(sb, window, holds):
    ev = copy.deepcopy(mini().to_dict()["evidence"])
    ev[1]["window"] = window
    p = mini(evidence=ev, tiers=[{"name": "t", "rule": {"evidence": "b"}}])
    assert members(sb, p, DELTAS) == holds


def test_max_span_and_min_span_boundaries(sb):
    ev = copy.deepcopy(mini().to_dict()["evidence"])
    ev.append(
        {
            "id": "rep",
            "name": "2 dx within 180d",
            "entity": "ConditionOccurrence",
            "concept_set_id": "dx",
            "count": 2,
            "max_span_days": 180,
        }
    )
    ev.append(
        {
            "id": "apart",
            "name": "2 dx 30d apart",
            "entity": "ConditionOccurrence",
            "concept_set_id": "dx",
            "count": 2,
            "min_span_days": 30,
        }
    )
    group = [201, 202, 203, 204, 205]
    assert members(sb, mini(evidence=ev, tiers=[{"name": "t", "rule": {"evidence": "rep"}}]), group) == {
        201,
        203,
        204,
        205,
    }  # 202: 181 days apart
    assert members(sb, mini(evidence=ev, tiers=[{"name": "t", "rule": {"evidence": "apart"}}]), group) == {
        201,
        202,
        203,
        205,
    }  # 204: only 29 days apart


# ---- boolean logic and N-of-M -----------------------------------------------------------------------
BCD = [{"evidence": "b"}, {"evidence": "c"}, {"evidence": "d"}]


@pytest.mark.parametrize(
    "rule, holds",
    [
        ({"at_least": {"n": 2, "of": BCD}}, {303, 304}),
        ({"at_most": {"n": 1, "of": BCD}}, {301, 302}),
        ({"exactly": {"n": 2, "of": BCD}}, {303}),
        ({"exactly": {"n": 0, "of": BCD}}, {301}),
        ({"all": [{"evidence": "b"}, {"not": {"evidence": "d"}}]}, {302, 303}),
        ({"any": [{"evidence": "d"}, {"all": [{"evidence": "b"}, {"evidence": "c"}]}]}, {303, 304}),
        ({"not": {"any": BCD}}, {301}),
        (
            {"all": [{"evidence": "a"}, {"at_least": {"n": 1, "of": [{"not": {"evidence": "c"}}, {"evidence": "d"}]}}]},
            {301, 302, 304},
        ),
    ],
)
def test_boolean_and_n_of_m(sb, rule, holds):
    assert members(sb, mini(tiers=[{"name": "t", "rule": rule}]), NOF) == holds


def test_groups_nest(sb):
    p = mini(
        groups={
            "tx": {"any": [{"evidence": "b"}, {"evidence": "c"}]},
            "strong": {"all": [{"group": "tx"}, {"evidence": "d"}]},
        },
        tiers=[{"name": "t", "rule": {"group": "strong"}}],
    )
    assert members(sb, p, NOF) == {304}


def test_n_of_m_within_days_boundary(sb):
    rule = {"at_least": {"n": 2, "within_days": 90, "of": [{"evidence": "b"}, {"evidence": "c"}]}}
    assert members(sb, mini(tiers=[{"name": "t", "rule": rule}]), [305, 306, 303]) == {306, 303}


# ---- tiers, scoring, conflicts ------------------------------------------------------------------------
def test_tiers_first_match_and_scores(sb):
    p = mini(
        scoring={
            "weights": [
                {"ref": {"evidence": "b"}, "points": 1},
                {"ref": {"evidence": "c"}, "points": 2},
                {"ref": {"evidence": "d"}, "points": 4},
            ]
        },
        tiers=[
            {"name": "high", "min_score": 6},
            {"name": "mid", "rule": {"any": [{"evidence": "b"}, {"evidence": "c"}]}, "min_score": 2},
            {"name": "low", "rule": {"evidence": "b"}},
        ],
    )
    got = {k: v for k, v in assign(sb, p).items() if k in NOF}
    assert got == {302: ("low", 1), 303: ("mid", 3), 304: ("high", 7)}  # 301 matches no tier: not a member


def test_conflict_flag_vs_exclude(sb):
    flag = mini(
        conflicts=[{"name": "adeno", "rule": {"evidence": "e"}, "action": "flag"}],
        tiers=[{"name": "t", "rule": {"evidence": "b"}}],
    )
    excl = mini(
        conflicts=[{"name": "adeno", "label": "Adeno histology", "rule": {"evidence": "e"}, "action": "exclude"}],
        tiers=[{"name": "t", "rule": {"evidence": "b"}}],
    )
    assert 307 in members(sb, flag, [307, 302]) and members(sb, excl, [307, 302]) == {302}
    assert "Conflict exclusion: Adeno histology" in ProxyCompiler(sb.ontology).compile_proxy(excl).rule_names


def test_no_row_multiplication(sb):
    p = mini(tiers=[{"name": "t", "rule": {"evidence": "b"}}])
    compiled = ProxyCompiler(sb.ontology).compile_proxy(p)
    rows = sb.con.execute(compiled.assignment_sql).fetchall()
    assert len(rows) == len({r[0] for r in rows})
    cohort = sb.con.execute(compiled.cohort_sql).fetchall()
    assert len(cohort) == len({r[0] for r in cohort}) == len(rows)
    ev_rows = sb.con.execute(compiled.evidence_sql).fetchall()
    assert len(ev_rows) == len(rows) * len(p.evidence)
    assert {r[0] for r in rows if r[0] == 401} == {401}


# ---- the synthetic ESCC-like case (P001-P006 + exclusion case) ---------------------------------------------
@pytest.fixture(scope="module")
def escc(tmp_path_factory):
    b = make_builder(tmp_path_factory.mktemp("proxy_escc"), min_cell=1)
    yield b
    b.con.close()


def test_escc_example_tiers(escc):
    p = ProxyDefinition.from_file(EXAMPLE)
    got = assign(escc, p)
    assert {k: v[0] for k, v in got.items()} == EXPECTED
    assert got[1] == ("high", 9) and got[2] == ("moderate", 6) and got[3] == ("exploratory", 2)
    compiled = ProxyCompiler(escc.ontology).compile_proxy(p)
    keys = {(r[0], r[1]): r[3] for r in escc.con.execute(compiled.evidence_sql).fetchall()}
    flagged = {pid for (pid, k), v in keys.items() if k == "cf_adeno_histology" and v}
    assert flagged == FLAGGED
    attrition = escc.executor.attrition(compiled)
    assert [r["remaining"] for r in attrition.rules] == [6, 6, 6, 6, 6, 5, 4]
    raw = escc.executor.proxy_summary(compiled)
    assert raw["candidates"] == 5 and raw["tier_none"] == 1 and raw["cf_adeno_histology"] == 1
    assert sum(raw[f"tier_{t.name}"] for t in p.tiers) == attrition.final_count
    assert [raw["funnel_1"], raw["funnel_2"], raw["funnel_3"]] == [5, 4, 2]


def test_escc_sql_deterministic_and_safe(escc):
    p = ProxyDefinition.from_file(EXAMPLE)
    a = ProxyCompiler(escc.ontology).compile_proxy(p)
    b = ProxyCompiler(escc.ontology).compile_proxy(ProxyDefinition.from_file(EXAMPLE))
    assert a.sql_hash == b.sql_hash and a.assignment_sql == b.assignment_sql
    for sql in (a.cohort_sql, a.assignment_sql, a.evidence_sql, a.attrition_sql, a.summary_sql):
        guard_select(sql)  # one read-only SELECT each
        for free_text in (p.target.name, p.clinical_notes[:30], p.evidence[0].name):
            assert free_text not in sql


# ---- dataset capabilities and absence semantics ---------------------------------------------------------
@pytest.fixture(scope="module")
def no_path(tmp_path_factory):
    b = make_builder(tmp_path_factory.mktemp("proxy_nopath"), pathology=False, min_cell=1)
    yield b
    b.con.close()


def _msgs(issues, severity):
    return [i.message for i in issues if i.severity == severity]


def test_required_unavailable_evidence_is_an_error(no_path):
    p = ProxyDefinition.from_file(EXAMPLE)
    issues, attrition, compiled = validate_proxy(p, no_path.ontology, no_path.vocab, no_path.executor)
    errors = _msgs(issues, "error")
    assert any(
        "Proxy rule requires pathology evidence 'Squamous histology result'" in m and "does not provide" in m
        for m in errors
    )
    assert attrition is None and compiled is None
    with pytest.raises(ValueError, match="required evidence not available"):
        ProxyCompiler(no_path.ontology).compile_proxy(p)


def test_optional_unavailable_evidence_warns_and_runs(no_path):
    data = ProxyDefinition.from_file(EXAMPLE).to_dict()
    for e in data["evidence"]:
        if e["id"] in ("squamous_path", "adeno_path"):
            e["required"] = False
    p = ProxyDefinition.model_validate(data)
    issues, attrition, compiled = validate_proxy(p, no_path.ontology, no_path.vocab, no_path.executor)
    assert not _msgs(issues, "error")
    warnings = _msgs(issues, "warning")
    assert any("will operate using the remaining evidence" in m for m in warnings)
    assert any("tier 'high' can never be assigned" in m for m in warnings)
    # adenocarcinoma histology is categorised 'conflicting' (a Measurement), so it stays available
    assert compiled is not None and compiled.unavailable_evidence == ("squamous_path",)
    got = {k: v[0] for k, v in assign(no_path, p).items()}
    # without pathology P1 falls back to its treatment pattern; P5 (single dx + chemo only) gets no tier
    assert got == {1: "moderate", 2: "moderate", 3: "exploratory"}


def test_three_valued_reachability():
    p = mini(tiers=[{"name": "t", "rule": {"evidence": "a"}}])
    assert truth(Expr(evidence="d"), p, {"d"}) is False
    assert truth(Expr.model_validate({"not": {"evidence": "d"}}), p, {"d"}) is True
    assert truth(Expr.model_validate({"any": [{"evidence": "d"}, {"evidence": "a"}]}), p, {"d"}) is None
    assert truth(Expr.model_validate({"at_most": {"n": 0, "of": [{"evidence": "d"}]}}), p, {"d"}) is True


def test_absence_warning_depends_on_dataset(escc, no_path):
    p = mini(tiers=[{"name": "t", "rule": {"all": [{"evidence": "a"}, {"not": {"evidence": "b"}}]}}])
    issues, _, _ = validate_proxy(p, escc.ontology, escc.vocab)
    obs = [m for m in _msgs(issues, "warning") if "'No record' only means no record during observed time" in m]
    assert obs  # OMOP-style data with observation periods
    laad = Ontology.load(escc.settings.ontology_dir, "iqvia_laad")
    issues, _, _ = validate_proxy(
        mini(
            dataset_profile="iqvia_laad",
            evidence=[mini().to_dict()["evidence"][i] for i in (0, 1)],
            tiers=[{"name": "t", "rule": {"all": [{"evidence": "a"}, {"not": {"evidence": "b"}}]}}],
        ),
        laad,
        Vocabulary(escc.con),
    )
    assert any(ABSENCE_WEAK in m for m in _msgs(issues, "warning"))


def test_absence_supported_dataset_has_no_absence_warning(escc):
    ont = Ontology.load(escc.settings.ontology_dir, "escc_synthetic")
    ont.capabilities["absence_inference"] = "supported"
    p = mini(tiers=[{"name": "t", "rule": {"not": {"evidence": "b"}}}])
    before, _, _ = validate_proxy(p, escc.ontology, escc.vocab)
    assert [m for m in _msgs(before, "warning") if "'No record'" in m]
    issues, _, _ = validate_proxy(p, ont, escc.vocab)
    assert not [m for m in _msgs(issues, "warning") if "'No record'" in m or ABSENCE_WEAK in m]


def test_laad_has_no_pathology_or_lab_evidence(escc):
    laad = Ontology.load(escc.settings.ontology_dir, "iqvia_laad")
    issues, _, _ = validate_proxy(ProxyDefinition.from_file(EXAMPLE), laad, Vocabulary(escc.con))
    errors = _msgs(issues, "error")
    assert any("pathology evidence" in m for m in errors)
    assert any("designed for dataset 'escc_synthetic'" in m for m in _msgs(issues, "warning"))


# ---- PostgreSQL ---------------------------------------------------------------------------------------
def test_proxy_sql_parses_as_postgresql(escc):
    pglast = pytest.importorskip("pglast")
    compiled = ProxyCompiler(escc.ontology).compile_proxy(ProxyDefinition.from_file(EXAMPLE))
    for sql in (
        compiled.cohort_sql,
        compiled.assignment_sql,
        compiled.evidence_sql,
        compiled.attrition_sql,
        compiled.summary_sql,
    ):
        pglast.parse_sql(sql)


@pytest.mark.skipif(not os.environ.get("CB_TEST_POSTGRES_DSN"), reason="set CB_TEST_POSTGRES_DSN")
def test_proxy_results_identical_on_postgresql(escc):
    from test_compiler_semantics import PostgresEngine

    pg = PostgresEngine()
    for table in ("concept", "concept_ancestor", "concept_synonym"):
        col = "ancestor_concept_id" if table == "concept_ancestor" else "concept_id"
        pg.insert(
            f"vocab.{table}", escc.con.execute(f"SELECT * FROM vocab.{table} WHERE {col} >= 2100000000").fetchall()
        )
    for table in (
        "person",
        "observation_period",
        "condition_occurrence",
        "drug_exposure",
        "procedure_occurrence",
        "measurement",
    ):
        pg.insert(f"cdm.{table}", escc.con.execute(f"SELECT * FROM cdm.{table}").fetchall())
    compiled = ProxyCompiler(escc.ontology).compile_proxy(ProxyDefinition.from_file(EXAMPLE))
    for sql in (
        compiled.cohort_sql,
        compiled.assignment_sql,
        compiled.evidence_sql,
        compiled.attrition_sql,
        compiled.summary_sql,
    ):
        duck = [tuple(r) for r in escc.con.execute(sql).fetchall()]
        assert [tuple(r) for r in pg.query(sql)] == duck


# ---- provider / care-setting capabilities and genericity ------------------------------------------------
def test_attribute_evidence_depends_on_dataset(escc):
    ev = mini().to_dict()["evidence"][:1] + [
        {
            "id": "inpatient_dx",
            "name": "Inpatient diagnosis",
            "category": "care_setting",
            "entity": "ConditionOccurrence",
            "concept_set_id": "dx",
            "place_of_service": ["21"],
        },
        {
            "id": "specialist",
            "name": "Specialist diagnosis",
            "category": "provider",
            "required": False,
            "entity": "ConditionOccurrence",
            "concept_set_id": "dx",
            "provider_specialty": ["oncology"],
        },
    ]
    p = mini(evidence=ev)
    issues, _, _ = validate_proxy(p, escc.ontology, escc.vocab)
    assert any("place of service" in m and "Inpatient diagnosis" in m for m in _msgs(issues, "error"))
    assert any("Specialist diagnosis" in m and "remaining evidence" in m for m in _msgs(issues, "warning"))
    laad = Ontology.load(escc.settings.ontology_dir, "iqvia_laad")
    issues, _, _ = validate_proxy(mini(evidence=ev, dataset_profile="iqvia_laad"), laad, Vocabulary(escc.con))
    assert not [m for m in _msgs(issues, "error") if "Inpatient diagnosis" in m]  # LAAD records place of service
    sql = ProxyCompiler(laad).compile_proxy(mini(evidence=ev, dataset_profile="iqvia_laad")).assignment_sql
    assert "IN ('21')" in sql and "ev_specialist" in sql


def test_core_engine_has_no_disease_specific_logic():
    from cohort_builder.config import REPO_ROOT

    src = REPO_ROOT / "src" / "cohort_builder"
    files = [
        src / "proxy.py",
        src / "proxy_compiler.py",
        src / "proxy_service.py",
        *(src / "agents").glob("proxy_*.py"),
    ]
    for f in files:
        text = f.read_text().lower()
        for word in ("escc", "esophag", "squamous", "adenocarcinoma"):
            assert word not in text, (f.name, word)


def test_other_rare_disease_configuration_runs(sb):
    """A non-oncology shape: lab + specialist-free treatment pattern, score-only tiers."""
    p = mini(
        algorithm_name="rare_metabolic_example",
        target={"name": "Synthetic metabolic disorder"},
        scoring={
            "weights": [
                {"ref": {"evidence": "a"}, "points": 1},
                {"ref": {"evidence": "d"}, "points": 3},
                {"ref": {"at_least": {"n": 1, "of": [{"evidence": "b"}, {"evidence": "c"}]}}, "points": 2},
            ]
        },
        tiers=[{"name": "strong", "min_score": 6}, {"name": "possible", "min_score": 3}],
    )
    got = {k: v for k, v in assign(sb, p).items() if k in NOF}
    assert got == {303: ("possible", 3), 304: ("strong", 6), 302: ("possible", 3)}
