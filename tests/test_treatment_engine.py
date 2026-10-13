"""Treatment-event engine: configuration, canonical events from differently shaped claims sources,
deterministic episodes and transition evidence, provenance, suppression and access control.
SYNTHETIC DATA ONLY (made-up claims, codes and patients)."""

from __future__ import annotations

import dataclasses
import json
import random
from datetime import date, timedelta

import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.cli import main as cli
from cohort_builder.config import REPO_ROOT, Settings
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import GovernanceError, GovernancePolicy, SecurityConfig, issue_token
from cohort_builder.treatment_config import TreatmentRuleSet, TreatmentSourceConfig, load_rules, load_sources
from cohort_builder.treatment_episodes import Event, build
from cohort_builder.treatment_service import TreatmentError, TreatmentService, resolve_rules
from fake_llm import FakeLLM

D = date.fromisoformat

RULES = {
    "rule_set_version": 1,
    "name": "test_rules",
    "version": "1.0.0",
    "drug_groups": {
        "metformin": {"concepts": [1503297]},
        "sglt2": {"concepts": [900], "max_gap_days": 60},  # an ATC-like class: expands to its ingredients
        "drugz": {"concepts": [990001]},
    },
    "include_claim_status": ["paid", "not_applicable"],
    "include_evidence_types": ["dispensing", "administration"],
    "days_supply": {"missing": "assume", "assume_days": {"administration": 21}, "min_valid": 1, "max_valid": 365},
    "episode": {
        "max_gap_days": 45,
        "overlap": "shift_forward",
        "max_carryover_days": 90,
        "same_day_same_group": "max_supply",
        "duplicates": "same_product_same_day",
    },
    "censoring": {"observation": "dataset", "washout_days": 180},
    "transitions": {"add_on_min_overlap_days": 30, "switch_window_days": 60},
}


def rules(**over) -> TreatmentRuleSet:
    data = json.loads(json.dumps(RULES))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return TreatmentRuleSet.model_validate(data)


# ---- 1. configuration -------------------------------------------------------------------------------------------
def test_shipped_configs_validate_and_hash_stably():
    base = REPO_ROOT / "ontology" / "treatment"
    for p in sorted((base / "sources").glob("*.yaml")):
        cfg = load_sources(p.read_text())
        assert cfg.dataset == p.stem and cfg.content_hash() == load_sources(p.read_text()).content_hash()
    r = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
    assert r.gap_for("sglt2_inhibitors") == 60 and r.gap_for("metformin") == 45


def test_rule_sets_have_no_hidden_defaults_and_are_validated():
    data = json.loads(json.dumps(RULES))
    del data["episode"]["max_gap_days"]
    with pytest.raises(ValueError, match="max_gap_days"):  # no global gap threshold: it must be written out
        TreatmentRuleSet.model_validate(data)
    with pytest.raises(ValueError, match="assume_days"):
        rules(days_supply={"missing": "assume", "assume_days": {}, "min_valid": 1, "max_valid": 365})
    with pytest.raises(ValueError):
        rules(drug_groups={"Bad Name": {"concepts": [1]}})
    with pytest.raises(ValueError):
        rules(drug_groups={"g": {"concepts": [-1]}})
    with pytest.raises(ValueError, match="anchors"):
        load_rules("name: &n x\nversion: *n\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_rules(yaml.safe_dump(RULES) + "name: again\n")


def source_cfg(sources: list[dict], dataset: str = "omop_demo") -> TreatmentSourceConfig:
    return TreatmentSourceConfig.model_validate({"dataset": dataset, "sources": sources})


def test_source_configs_reject_unsafe_or_ambiguous_settings():
    ok = {
        "name": "rx",
        "table": "claims.pharmacy",
        "person_column": "member",
        "record_id_column": "claim_id",
        "start_column": "fill_dt",
        "start_date_source": "fill_date",
        "evidence_type": "dispensing",
        "code_column": "ndc",
        "vocabulary": "NDC",
    }
    source_cfg([ok])
    for bad in (
        {"table": "meta.audit_event"},  # internal schemas are never sources
        {"table": 'claims.x"; DROP TABLE y; --'},
        {"person_column": "member; DROP"},
        {"evidence_type": None},  # evidence type must be declared, never guessed
        {"concept_column": "c"},  # concept_column XOR code_column
        {"vocabulary": "ND'C"},
        {"status": {"column": "st", "values": {"X": "reversal"}}},  # reversal needs a reference column
        {"status": {"column": "st", "values": {"P'": "paid"}}},
    ):
        with pytest.raises(ValueError):
            source_cfg([{**ok, **bad}])
    with pytest.raises(ValueError, match="unique"):
        source_cfg([ok, ok])


# ---- 2. pure episode logic ------------------------------------------------------------------------------------------
def ev(eid: str, start: str, days: int, group: str = "metformin", product: int = 801, person: int = 1) -> Event:
    return Event(eid, person, group, product, D(start), days)


OBS = {1: [(D("2019-01-01"), D("2021-12-31"))]}


def test_overlapping_fills_shift_forward_with_a_carryover_cap():
    r = build([ev("a", "2020-01-01", 30), ev("b", "2020-01-25", 30)], OBS, rules())
    b = next(d for d in r.decisions if d.event_id == "b")
    assert b.shifted_days == 6 and b.coverage_start == D("2020-01-31") and b.coverage_end == D("2020-02-29")
    assert len(r.episodes) == 1 and r.episodes[0].end == D("2020-02-29") and r.episodes[0].days_supplied == 60
    capped = build(
        [ev("a", "2020-01-01", 30), ev("b", "2020-01-25", 30)], OBS, rules(episode={"max_carryover_days": 2})
    )
    assert next(d for d in capped.decisions if d.event_id == "b").shifted_days == 2
    ignored = build([ev("a", "2020-01-01", 30), ev("b", "2020-01-25", 30)], OBS, rules(episode={"overlap": "ignore"}))
    assert ignored.episodes[0].end == D("2020-02-23")


def test_same_day_events_and_duplicates():
    evs = [ev("a", "2020-01-01", 30), ev("dup", "2020-01-01", 30), ev("b", "2020-01-01", 60, product=802)]
    r = build(evs, OBS, rules())
    steps = {d.event_id: d.step for d in r.decisions}
    assert steps == {"a": "counted", "b": "same_day_merged", "dup": "duplicate_event"} or steps == {
        "b": "counted",
        "a": "same_day_merged",
        "dup": "duplicate_event",
    }
    assert r.episodes[0].end == D("2020-02-29")  # max supply (60), not 90
    summed = build(evs, OBS, rules(episode={"same_day_same_group": "sum_supply"}))
    assert summed.episodes[0].end == D("2020-03-30")  # 30 + 60
    keep = build(evs, OBS, rules(episode={"duplicates": "same_record"}))
    assert not [d for d in keep.decisions if d.step == "duplicate_event"]


def test_gaps_restarts_discontinuation_and_per_group_gap():
    evs = [ev("a", "2020-01-01", 30), ev("b", "2020-03-10", 30), ev("c", "2020-06-01", 30)]
    r = build(evs, OBS, rules())  # gaps: 39 (bridged, <= 45) then 52 (new episode)
    assert [(e.start, e.end, e.gap_days_bridged) for e in r.episodes] == [
        (D("2020-01-01"), D("2020-04-08"), 39),
        (D("2020-06-01"), D("2020-06-30"), 0),
    ]
    kinds = [(t.kind, t.from_episode, t.to_episode) for t in r.transitions]
    # discontinuation is the episode END STATUS (once per episode), never also a transition
    assert kinds == [("restart", "1:metformin:1", "1:metformin:2")]
    assert [e.end_status for e in r.episodes] == ["discontinued", "discontinued"]
    wider = build([ev(x.event_id, x.start.isoformat(), x.days, group="sglt2") for x in evs], OBS, rules())
    assert len(wider.episodes) == 1  # the sglt2 group allows 60 days


def test_censoring_at_observation_boundaries_and_unknowns():
    obs = {1: [(D("2020-01-01"), D("2020-03-01"))]}
    r = build([ev("a", "2020-01-10", 30)], obs, rules())
    e = r.episodes[0]
    assert e.left_censored is True and e.right_censored is True and e.end_reason == "observation_end"
    assert e.end_status == "censored" and r.transitions == []  # no claim that treatment stopped
    outside = build([ev("a", "2018-01-10", 30)], obs, rules())  # no observed time around it
    assert outside.episodes == [] and outside.decisions[0].step == "outside_observation"
    none = build([ev("a", "2020-01-10", 30)], None, rules(censoring={"observation": "none", "washout_days": 0}))
    assert none.episodes[0].end_reason == "unknown" and none.episodes[0].end_status == "unknown"


def test_switch_add_on_and_concurrent_start():
    r = build(
        [
            ev("m1", "2020-01-01", 90),
            ev("s1", "2020-02-01", 30, group="sglt2", product=805),  # starts during metformin, overlaps 30 days
            ev("m2", "2021-01-01", 30),
            ev("s2", "2021-02-10", 30, group="sglt2", product=805),  # 10 uncovered days after metformin ended
            ev("m3", "2021-06-01", 30),
            ev("z3", "2021-06-01", 21, group="drugz", product=806),
        ],
        OBS,
        rules(),
    )
    kinds = {(t.kind, t.from_group, t.to_group, t.days) for t in r.transitions if t.to_group and t.kind != "restart"}
    assert ("add_on", "metformin", "sglt2", 30) in kinds
    assert ("switch", "metformin", "sglt2", 10) in kinds  # Jan 31 .. Feb 9 uncovered
    assert ("concurrent_start", "drugz", "metformin", 0) in kinds


def test_episode_building_is_deterministic_regardless_of_input_order():
    evs = [ev(f"e{i}", (D("2020-01-01") + timedelta(days=25 * i)).isoformat(), 30, person=1 + i % 3) for i in range(30)]
    obs = {p: [(D("2019-01-01"), D("2022-12-31"))] for p in (1, 2, 3)}
    first = build(evs, obs, rules())
    shuffled = list(evs)
    random.Random(4).shuffle(shuffled)
    again = build(shuffled, obs, rules())
    assert (again.decisions, again.episodes, again.transitions) == (first.decisions, first.episodes, first.transitions)


# ---- 3. canonical events from multiple differently shaped sources -----------------------------------------------
CONCEPTS = [
    (1503297, "metformin", "Drug", "RxNorm", "Ingredient", "S", "6809"),
    (45775965, "empagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1545653"),
    (44785829, "dapagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1488564"),
    (990001, "drugz", "Drug", "RxNorm", "Ingredient", "S", "Z1"),
    (900, "SGLT2 class", "Drug", "ATC", "ATC 4th", "C", "A10BK"),
    (801, "metformin 500 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P801"),
    (802, "empagliflozin 10 MG [Brand X]", "Drug", "RxNorm", "Branded Drug", "S", "P802"),
    (803, "empagliflozin 10 MG [Brand Y]", "Drug", "RxNorm", "Branded Drug", "S", "P803"),
    (804, "empagliflozin / metformin", "Drug", "RxNorm", "Clinical Drug", "S", "P804"),
    (805, "dapagliflozin 10 MG", "Drug", "RxNorm", "Clinical Drug", "S", "P805"),
    (806, "drugz injection", "Drug", "RxNorm", "Clinical Drug", "S", "P806"),
    (851, "Brand X", "Drug", "RxNorm", "Brand Name", None, "BX"),
    (852, "Brand Y", "Drug", "RxNorm", "Brand Name", None, "BY"),
    (701, "ndc metformin", "Drug", "NDC", "11-digit NDC", None, "11111000101"),
    (702, "ndc brand x", "Drug", "NDC", "11-digit NDC", None, "22222000101"),
    (703, "ndc brand y", "Drug", "NDC", "11-digit NDC", None, "33333000101"),
    (704, "ndc combo", "Drug", "NDC", "11-digit NDC", None, "44444000101"),
    (705, "ndc dapa", "Drug", "NDC", "11-digit NDC", None, "55555000101"),
    (706, "drugz admin", "Drug", "HCPCS", "HCPCS", None, "J9999"),
    (707, "office visit code", "Procedure", "CPT4", "CPT4", None, "99213"),
    (950, "office visit", "Procedure", "SNOMED", "Procedure", "S", "185349003"),
    (708, "T2DM code", "Condition", "ICD10CM", "5-char", None, "E11.9"),
    (960, "type 2 diabetes", "Condition", "SNOMED", "Clinical Finding", "S", "44054006"),
]
MAPS = [(701, 801), (702, 802), (703, 803), (704, 804), (705, 805), (706, 806), (707, 950), (708, 960)]
ANCESTORS = [
    (1503297, 801), (45775965, 802), (45775965, 803), (45775965, 804), (1503297, 804), (44785829, 805),
    (990001, 806), (900, 45775965), (900, 44785829), (900, 802), (900, 803), (900, 804), (900, 805),
]  # fmt: skip

PHARMACY = [
    # claim, member, fill date, ndc, days, qty, status, reference
    ("C1", 1, "2020-01-01", "11111000101", 30, 30, "P", None),
    ("C2", 1, "2020-01-25", "11111000101", 30, 30, "P", None),  # early refill
    ("C2", 1, "2020-01-25", "11111000101", 30, 30, "P", None),  # the same claim line delivered twice
    ("C3", 1, "2020-02-20", "11111000101", 30, 30, "P", None),
    ("C3b", 1, "2020-02-20", "11111000101", 30, 30, "P", None),  # same product/day/supply, other claim id
    ("C4", 1, "2020-03-01", "11111000101", 30, 30, "R", None),  # rejected: nothing was dispensed
    ("C5", 1, "2020-04-01", "11111000101", 30, 30, "P", None),  # paid, then reversed below
    ("C5R", 1, "2020-04-03", "11111000101", 30, 30, "X", "C5"),
    ("C6", 1, "2020-06-01", "11111000101", 30, 30, "P", None),  # restart after a gap
    ("C20", 2, "2020-01-01", "22222000101", 30, 30, "P", None),  # brand X
    ("C21", 2, "2020-02-01", "33333000101", 30, 30, "P", None),  # brand Y, same ingredient
    ("C22", 2, "2020-03-02", "55555000101", 30, 30, "P", None),  # other ingredient of the same class
    ("C23", 2, "2020-02-15", "11111000101", 90, 90, "P", None),  # metformin added on
    ("C30", 3, "2020-01-01", "11111000101", 30, 30, "P", None),
    ("C31", 3, "2020-02-10", "55555000101", 30, 30, "P", None),  # switch
    ("C40", 4, "2020-05-01", "44444-0001-01", 30, 30, "P", None),  # combination product, hyphenated NDC
    ("C50", 5, "2020-01-01", "11111000101", None, 30, "P", None),  # missing days supply (dispensing)
    ("C51", 5, "2020-02-01", "11111000101", 999, 30, "P", None),  # implausible days supply
    ("C60", None, "2020-01-01", "11111000101", 30, 30, "P", None),  # no patient id
    ("C61", 6, None, "11111000101", 30, 30, "P", None),  # no date
    ("C62", 6, "2020-01-01", "00000000000", 30, 30, "P", None),  # unknown NDC
    ("C63", 6, "2020-01-01", "11111-0001-01", 30, 30, "P", None),  # hyphenated 5-4-2 NDC
    ("C70", 7, "2020-01-01", "11111000101", 30, 30, "P", None),  # superseded by the adjustment
    ("C71", 7, "2020-01-01", "11111000101", 60, 60, "A", "C70"),
    ("C80", 8, "2020-01-01", "11111000101", 30, 30, "Z", None),  # unknown status code
    ("C90", 9, "2020-01-05", "11111000101", 30, 30, "X", "NOPE"),  # reversal of an unknown claim
]
MEDICAL = [
    ("M1", 10, "2020-01-01", "J9999", "HCPCS"),
    ("M2", 10, "2020-01-22", "J9999", "HCPCS"),
    ("M3", 10, "2020-02-12", "J9999", "HCPCS"),
    ("M4", 10, "2020-02-12", "99213", "CPT"),  # a procedure, not a drug
]
DIAGNOSES = [("D1", 1, "2020-01-01", "E11.9")]
EHR = [
    # id, person, start, end, days, concept, type
    (1, 11, "2021-01-01", "2021-01-30", None, 805, 32869),  # pharmacy claim: duration from the end date
    (2, 11, "2021-03-01", None, 30, 805, 32817),  # 'EHR' type only: evidence type unknown
]
OBSERVATION = {
    1: ("2019-01-01", "2021-12-31"),
    2: ("2019-12-01", "2020-04-15"),
    3: ("2018-01-01", "2022-12-31"),
    4: ("2018-01-01", "2022-12-31"),
    6: ("2018-01-01", "2022-12-31"),
    7: ("2018-01-01", "2022-12-31"),
    10: ("2018-01-01", "2022-12-31"),
    11: ("2018-01-01", "2022-12-31"),
}

SOURCES = [
    {
        "name": "pharmacy",
        "table": "claims.pharmacy",
        "person_column": "member",
        "record_id_column": "claim_id",
        "start_column": "fill_dt",
        "start_date_source": "fill_date",
        "days_supply_column": "dsup",
        "quantity_column": "qty",
        "evidence_type": "dispensing",
        "code_column": "ndc",
        "vocabulary": "NDC",
        "status": {
            "column": "status",
            "values": {"P": "paid", "R": "rejected", "X": "reversal", "A": "adjustment"},
            "reference_column": "ref_claim",
        },
    },
    {
        "name": "medical",
        "table": "claims.medical",
        "person_column": "member",
        "record_id_column": "claim_id",
        "start_column": "svc_dt",
        "start_date_source": "service_date",
        "evidence_type": "administration",
        "code_column": "code",
        "code_type_column": "code_type",
        "code_type_values": {"HCPCS": "HCPCS", "CPT": "CPT4"},
    },
    {  # even a misconfigured diagnosis table cannot produce medication events
        "name": "diagnoses",
        "table": "claims.dx",
        "person_column": "member",
        "record_id_column": "claim_id",
        "start_column": "svc_dt",
        "start_date_source": "service_date",
        "evidence_type": "dispensing",
        "code_column": "dx",
        "vocabulary": "ICD10CM",
    },
    {
        "name": "ehr",
        "table": "ehr.drug_exposure",
        "person_column": "person_id",
        "record_id_column": "id",
        "start_column": "start_dt",
        "start_date_source": "start_date",
        "end_column": "end_dt",
        "days_supply_column": "days",
        "concept_column": "concept",
        "evidence_type_column": "type_concept",
        "evidence_type_values": {"32869": "dispensing"},
    },
]


@pytest.fixture
def tx(tmp_path):
    """A fresh database (omop_demo profile) with a tiny vocabulary and three differently shaped sources."""
    s = dataclasses.replace(Settings(), db_path=tmp_path / "tx.duckdb", local_datasets_dir=tmp_path / "ds")
    b = CohortBuilder(s, backend=FakeLLM({}))
    con = b.con
    con.executemany("INSERT INTO vocab.concept VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)", [list(c) for c in CONCEPTS])
    con.executemany(
        "INSERT INTO vocab.concept_relationship VALUES (?,?,'Maps to',NULL,NULL,NULL)", [list(m) for m in MAPS]
    )
    con.executemany(
        "INSERT INTO vocab.concept_relationship VALUES (?,?,'Has brand name',NULL,NULL,NULL)", [[802, 851], [803, 852]]
    )
    con.executemany("INSERT INTO vocab.concept_ancestor VALUES (?,?,1,1)", [list(a) for a in ANCESTORS])
    con.execute(
        "CREATE SCHEMA claims; CREATE SCHEMA ehr;"
        "CREATE TABLE claims.pharmacy (claim_id VARCHAR, member BIGINT, fill_dt DATE, ndc VARCHAR, dsup INTEGER, "
        "qty DOUBLE, status VARCHAR, ref_claim VARCHAR);"
        "CREATE TABLE claims.medical (claim_id VARCHAR, member BIGINT, svc_dt DATE, code VARCHAR, code_type VARCHAR);"
        "CREATE TABLE claims.dx (claim_id VARCHAR, member BIGINT, svc_dt DATE, dx VARCHAR);"
        "CREATE TABLE ehr.drug_exposure (id BIGINT, person_id BIGINT, start_dt DATE, end_dt DATE, days INTEGER, "
        "concept BIGINT, type_concept BIGINT)"
    )
    con.executemany("INSERT INTO claims.pharmacy VALUES (?,?,?,?,?,?,?,?)", [list(r) for r in PHARMACY])
    con.executemany("INSERT INTO claims.medical VALUES (?,?,?,?,?)", [list(r) for r in MEDICAL])
    con.executemany("INSERT INTO claims.dx VALUES (?,?,?,?)", [list(r) for r in DIAGNOSES])
    con.executemany("INSERT INTO ehr.drug_exposure VALUES (?,?,?,?,?,?,?)", [list(r) for r in EHR])
    con.executemany(
        "INSERT INTO cdm.observation_period VALUES (?,?,?,?,32817)",
        [[i, p, s_, e_] for i, (p, (s_, e_)) in enumerate(sorted(OBSERVATION.items()), start=1)],
    )
    yield b
    b.con.close()


def run(b: CohortBuilder, r: TreatmentRuleSet | None = None) -> dict:
    return TreatmentService(b).run(r or rules(), "analyst", "default", source_cfg(SOURCES))


def events(b: CohortBuilder, run_id: str) -> dict[str, dict]:
    cur = b.con.execute("SELECT * FROM results.treatment_event WHERE run_id = ? ORDER BY event_id", [run_id])
    cols = [d[0] for d in cur.description]
    return {r[1]: dict(zip(cols, r, strict=True)) for r in cur.fetchall()}


def episodes(b: CohortBuilder, run_id: str, person: int) -> list[tuple]:
    return b.con.execute(
        "SELECT drug_group, episode_start, episode_end, n_events, n_products, left_censored, right_censored "
        "FROM results.treatment_episode WHERE run_id = ? AND person_id = ? ORDER BY episode_start, drug_group",
        [run_id, person],
    ).fetchall()


def transitions(b: CohortBuilder, run_id: str, person: int) -> set[tuple]:
    return set(
        b.con.execute(
            "SELECT kind, from_group, to_group FROM results.treatment_transition WHERE run_id = ? AND person_id = ?",
            [run_id, person],
        ).fetchall()
    )


def test_claim_status_reversals_adjustments_and_duplicates(tx):
    out = run(tx)
    assert out["status"] == "completed"
    e = events(tx, out["run_id"])
    reason = {k: v["exclusion_reason"] for k, v in e.items()}
    assert reason["pharmacy:C4:1503297"] == "rejected_claim"
    assert reason["pharmacy:C5:1503297"] == "reversed_claim"  # the paid claim a reversal cancels
    assert reason["pharmacy:C5R:1503297"] == "reversal_transaction"
    assert reason["pharmacy:C2:1503297#2"] == "duplicate_record"  # the repeated claim line
    assert {reason["pharmacy:C3:1503297"], reason["pharmacy:C3b:1503297"]} == {None, "duplicate_event"}
    assert reason["pharmacy:C70:1503297"] == "superseded_by_adjustment"
    assert reason["pharmacy:C71:1503297"] is None and e["pharmacy:C71:1503297"]["claim_status"] == "adjusted"
    assert e["pharmacy:C71:1503297"]["days_supply"] == 60
    assert reason["pharmacy:C80:1503297"] == "claim_status_excluded"
    assert reason["pharmacy:C90:1503297"] == "reversal_transaction"
    assert "orphan_reversal" in e["pharmacy:C90:1503297"]["quality_flags"]
    # without the reversal handling C5 (Apr 1) would bridge the gap; with it, June starts a new episode
    assert [x[:3] for x in episodes(tx, out["run_id"], 1)] == [
        ("metformin", D("2020-01-01"), D("2020-03-30")),
        ("metformin", D("2020-06-01"), D("2020-06-30")),
    ]
    assert ("restart", "metformin", "metformin") in transitions(tx, out["run_id"], 1)
    early = e["pharmacy:C2:1503297"]
    assert early["episode_step"] == "counted" and early["shifted_days"] == 6


def test_record_level_problems_are_excluded_with_reasons(tx):
    out = run(tx)
    e = events(tx, out["run_id"])
    by_record = {(v["source_name"], v["source_record_id"]): v for v in e.values()}
    assert by_record[("pharmacy", "C50")]["exclusion_reason"] == "missing_days_supply"  # dispensing: not assumed
    assert "days_supply_missing" in by_record[("pharmacy", "C50")]["quality_flags"]
    c51 = by_record[("pharmacy", "C51")]
    assert c51["exclusion_reason"] == "missing_days_supply" and "days_supply_out_of_range" in c51["quality_flags"]
    assert by_record[("pharmacy", "C60")]["exclusion_reason"] == "missing_person"
    assert by_record[("pharmacy", "C61")]["exclusion_reason"] == "missing_start_date"
    assert by_record[("pharmacy", "C62")]["exclusion_reason"] == "unmapped_code"
    c63 = by_record[("pharmacy", "C63")]
    assert c63["included"] and c63["source_code"] == "11111-0001-01" and c63["product_concept_id"] == 801


def test_brands_products_and_combinations(tx):
    out = run(tx)
    e = events(tx, out["run_id"])
    assert e["pharmacy:C20:45775965"]["brand_concept_id"] == 851
    assert e["pharmacy:C21:45775965"]["brand_concept_id"] == 852
    assert e["pharmacy:C22:44785829"]["brand_concept_id"] is None
    # two brands of one ingredient and a second class member: one sglt2 episode, three products
    eps = episodes(tx, out["run_id"], 2)
    sg = [x for x in eps if x[0] == "sglt2"]
    assert len(sg) == 1 and sg[0][3] == 3 and sg[0][4] == 3
    assert sg[0][5] is True and sg[0][6] is True  # starts 31 days after observation start; ends near its end
    assert ("add_on", "sglt2", "metformin") in transitions(tx, out["run_id"], 2)
    # a combination product is one record, two ingredient events in two groups
    combo = [v for v in e.values() if v["source_record_id"] == "C40"]
    assert sorted(v["drug_group"] for v in combo) == ["metformin", "sglt2"]
    assert all(v["included"] and "combination_product" in v["quality_flags"] for v in combo)
    assert ("concurrent_start", "metformin", "sglt2") in transitions(tx, out["run_id"], 4)
    assert ("switch", "metformin", "sglt2") in transitions(tx, out["run_id"], 3)


def test_medical_procedure_diagnosis_and_ehr_sources(tx):
    out = run(tx)
    e = events(tx, out["run_id"])
    by_record = {(v["source_name"], v["source_record_id"]): v for v in e.values()}
    admin = by_record[("medical", "M1")]
    assert (
        admin["included"] and admin["evidence_type"] == "administration" and admin["claim_status"] == "not_applicable"
    )
    assert admin["days_supply_source"] == "assumed" and admin["evidence_status"] == "inferred"
    assert by_record[("medical", "M4")]["exclusion_reason"] == "not_a_drug"  # CPT office visit
    assert by_record[("diagnoses", "D1")]["exclusion_reason"] == "not_a_drug"  # a diagnosis is never a drug event
    assert [x[:4] for x in episodes(tx, out["run_id"], 10)] == [("drugz", D("2020-01-01"), D("2020-03-03"), 3)]
    ehr1 = by_record[("ehr", "1")]
    assert ehr1["included"] and ehr1["days_supply"] == 30 and ehr1["days_supply_source"] == "from_end_date"
    assert ehr1["end_date_source"] == "recorded" and ehr1["evidence_type"] == "dispensing"
    assert by_record[("ehr", "2")]["exclusion_reason"] == "evidence_type_excluded"  # 'EHR' type: unknown evidence


def test_repeat_runs_are_identical_and_carry_provenance(tx):
    a, b = run(tx), run(tx)
    assert a["run_id"] != b["run_id"] and a["output_hash"] == b["output_hash"]
    assert (
        a["rule_set_hash"] == rules().content_hash() and a["source_config_hash"] == source_cfg(SOURCES).content_hash()
    )
    assert a["event_schema"] == "treatment_event/2" and a["software_version"] and a["data_snapshot"] is not None
    other = run(tx, rules(episode={"max_gap_days": 120}))
    assert other["output_hash"] != a["output_hash"] and other["rule_set_hash"] != a["rule_set_hash"]
    stored = json.loads(
        tx.con.execute("SELECT rule_set_json FROM meta.treatment_run WHERE run_id = ?", [a["run_id"]]).fetchone()[0]
    )
    assert TreatmentRuleSet.model_validate(stored).content_hash() == a["rule_set_hash"]  # rules reproducible


def test_summary_is_aggregate_and_suppressed(tx):
    out = run(tx)
    s = out["summary"]
    blob = json.dumps(out, default=str)
    for leak in ("C50", "11111000101", "2020-01-25", "member", '"person_id"'):
        assert leak not in blob
    assert s["records"]["by_outcome"]["rejected_claim"] == "<10" and s["drug_groups"]["metformin"]["persons"] == "<10"
    assert s["drug_groups"]["metformin"]["median_episode_days"] is None  # medians only for >= min cell persons
    assert s["unmatched_medication_codes"]["rows"] == "<10"
    events_audit = json.dumps(tx.store.audit_events(), default=str)
    assert "treatment.run" in events_audit and "C50" not in events_audit and "11111000101" not in events_audit


def test_explanations_need_admin_and_patient_level_policy(tx):
    out = run(tx)
    svc = TreatmentService(tx)
    with pytest.raises(GovernanceError):
        svc.explain_episode(out["run_id"], "1:metformin:1", "analyst", is_admin=True)
    assert any(
        e["action"] == "treatment.episode_explanation" and e["outcome"] == "denied" for e in tx.store.audit_events()
    )
    tx.policy = GovernancePolicy(allow_patient_level=True)
    with pytest.raises(GovernanceError):
        svc.explain_episode(out["run_id"], "1:metformin:1", "analyst", is_admin=False)
    x = svc.explain_episode(out["run_id"], "1:metformin:1", "admin", is_admin=True)
    steps = {(ev["source_record_id"], ev["episode_step"]) for ev in x["events"]}
    assert {("C1", "counted"), ("C2", "counted")} <= steps
    assert any(rec in ("C3", "C3b") and step == "duplicate_event" for rec, step in steps)  # dropped duplicate shown
    assert x["rule_set"]["hash"] == rules().content_hash() and x["rules_applied"]["max_gap_days"] == 45
    assert any(t["kind"] == "restart" for t in x["transitions"])
    with pytest.raises(KeyError):
        svc.explain_episode(out["run_id"], "999:none:1", "admin", is_admin=True)
    with pytest.raises(KeyError):
        svc.summary(out["run_id"], "other_tenant")  # tenant isolation


def test_failed_runs_are_recorded_without_data(tx):
    bad = source_cfg([{**SOURCES[0], "table": "claims.missing_table"}])
    with pytest.raises(TreatmentError, match="treatment run failed"):
        TreatmentService(tx).run(rules(), "analyst", "default", bad)
    row = tx.con.execute("SELECT status, errors_json FROM meta.treatment_run ORDER BY created_at DESC LIMIT 1")
    status, errors = row.fetchone()
    assert status == "failed" and json.loads(errors)[0].startswith("CatalogException")
    assert tx.con.execute("SELECT count(*) FROM results.treatment_event").fetchone()[0] == 0  # rolled back
    with pytest.raises(TreatmentError, match="not the active dataset"):
        TreatmentService(tx).run(rules(), "analyst", "default", source_cfg(SOURCES, dataset="iqvia_laad"))


# ---- 4. built-in datasets, CLI and API -------------------------------------------------------------------------------
def test_builtin_demo_datasets_run_end_to_end(builder, laad_builder):
    r = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
    laad = TreatmentService(laad_builder).run(r, "analyst")
    by = laad["summary"]["records"]["by_outcome"]
    assert laad["status"] == "completed" and by["reversed_claim"] != 0 and by["rejected_claim"] != 0
    assert laad["summary"]["records"]["by_source"][0]["source"] in ("medical", "pharmacy")
    omop = TreatmentService(builder).run(r, "analyst")
    assert omop["status"] == "completed" and omop["summary"]["drug_groups"]["metformin"]["episodes"] != 0
    assert TreatmentService(builder).run(r, "analyst")["output_hash"] == omop["output_hash"]


def test_cli_run_summary_and_gated_explain(settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CB_DB_PATH", str(settings.db_path))
    monkeypatch.delenv("CB_ALLOW_PATIENT_LEVEL", raising=False)
    assert cli(["treatment", "run", "--rules", "glucose_lowering_example"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"
    assert cli(["treatment", "summary", out["run_id"]]) == 0
    assert json.loads(capsys.readouterr().out)["output_hash"] == out["output_hash"]
    with pytest.raises(SystemExit, match="patient-level"):
        cli(["treatment", "explain", out["run_id"], "1:metformin:1"])
    with pytest.raises(SystemExit, match="unknown treatment rule set"):
        cli(["treatment", "run", "--rules", "nope"])


def test_api_roles_tenants_and_validation(builder, tmp_path):
    from cohort_builder.api import create_app

    entries, heads = [], {}
    for name, roles, tenant in [
        ("ex", ["executor"], "default"),
        ("au", ["author"], "default"),
        ("rv", ["reviewer"], "other"),
        ("ad", ["admin"], "default"),
    ]:
        token, entry = issue_token(name, roles, tenant, expires_days=1)
        entries.append(entry)
        heads[name] = {"Authorization": f"Bearer {token}"}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    c = TestClient(create_app(builder, SecurityConfig(tokens_file=tmp_path / "t.yaml")), raise_server_exceptions=False)
    body = {"rule_set": "glucose_lowering_example"}
    assert c.post("/treatment-runs", json=body).status_code == 401
    assert c.post("/treatment-runs", json=body, headers=heads["au"]).status_code == 403
    r = c.post("/treatment-runs", json=body, headers=heads["ex"])
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert c.get(f"/treatment-runs/{run_id}", headers=heads["ex"]).json()["output_hash"] == r.json()["output_hash"]
    assert c.get(f"/treatment-runs/{run_id}", headers=heads["rv"]).status_code == 404  # other tenant
    explain = f"/treatment-runs/{run_id}/episodes/1:metformin:1/explanation"
    assert c.get(explain, headers=heads["ex"]).status_code == 403
    assert c.get(explain, headers=heads["ad"]).status_code == 403  # admin, but patient-level output is off
    assert c.post("/treatment-runs", json={}, headers=heads["ex"]).status_code == 422
    both = {"rule_set": "glucose_lowering_example", "rules_yaml": "x: 1"}
    assert c.post("/treatment-runs", json=both, headers=heads["ex"]).status_code == 422
    bad = c.post("/treatment-runs", json={"rules_yaml": "name: x\n"}, headers=heads["ex"])
    assert bad.status_code == 422 and "invalid rule set" in bad.json()["detail"]
    inline = yaml.safe_dump(RULES).replace("test_rules", "inline_rules")
    ok = c.post("/treatment-runs", json={"rules_yaml": inline}, headers=heads["ex"])
    assert ok.status_code == 200, ok.text
    assert ok.json()["rule_set_name"] == "inline_rules"
    # omop_demo records are typed 'EHR' only: with dispensing/administration required nothing is included,
    # and the run says so instead of failing or inventing evidence
    assert ok.json()["summary"]["records"]["by_outcome"]["evidence_type_excluded"] != 0
    assert c.post("/treatment-runs", json={"rule_set": "../x"}, headers=heads["ex"]).status_code == 422
