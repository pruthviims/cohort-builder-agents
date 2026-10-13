"""Hand-verified SYNTHETIC fixture: ten fictitious T2D patients in the native laad schema, run with the
example rule set (illustrative; not clinically validated). Every expected value below was worked out by hand
from the rules (max_gap_days 45 / SGLT2 60, shift_forward with carry-over cap 90, washout 180, switch window
90, add-on overlap >= 30) and the activity-based observation periods of the iqvia_laad profile (first to last
activity, split on gaps > 365 days); the arithmetic is written next to each patient.

Activity: every patient except P10 has diagnosis claims on Jan 1 and Jul 1 of 2019-2023 and on 2023-12-31,
so one observation period 2019-01-01 .. 2023-12-31 (no gap exceeds 365 days)."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from cohort_builder.config import REPO_ROOT, Settings
from cohort_builder.db import connect, init_schemas
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.synthetic import load_demo_vocabulary
from cohort_builder.synthetic_laad import LAAD_DDL, NDC
from cohort_builder.synthetic_t2d import DX_COLS, PATIENT_COLS, RX_COLS
from cohort_builder.treatment_engine import bulk_insert
from cohort_builder.treatment_service import TreatmentService, resolve_rules
from fake_llm import FakeLLM

D = date.fromisoformat
MET, EMPA, DAPA, EMPA_MET = NDC["metformin_500"], NDC["empagliflozin"], NDC["dapagliflozin"], NDC["empa_met"]
REF = D("2026-06-30")


def standard_activity(pid: int) -> list[str]:
    return [f"{y}-{m}" for y in range(2019, 2024) for m in ("01-01", "07-01")] + ["2023-12-31"]


RX: list[tuple] = [
    # P1 persistent: Jan1 30d covers Jan1-Jan30; Jan25 refill is early -> shifted to Jan31 (carry 6 <= 90),
    # covers Jan31-Feb29 (2020 is a leap year); Apr10: uncovered Mar1..Apr9 = 31 + 9 = 40 <= 45 -> bridged;
    # covers Apr10-May9. One episode Jan1-May9, 3 events, 90 days, 40 bridged; May9 + 45 < 2023-12-31 ->
    # discontinued.
    (101, 1, "2020-01-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (102, 1, "2020-01-25", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (103, 1, "2020-04-10", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    # P2 restart: Mar1-Mar30, then Jun1: uncovered Mar31..May31 = 1 + 30 + 31 = 62 > 45 -> new episode
    # Jun1-Jun30; restart on Jun1 with 62 days.
    (201, 2, "2020-03-01", MET, 30, 60.0, "PAID", None, None, "MEDICARE"),
    (202, 2, "2020-06-01", MET, 30, 60.0, "PAID", None, None, "MEDICARE"),
    # P3 switch: metformin Feb1-Mar1 + Mar2-Mar31 (one episode Feb1-Mar31); dapagliflozin Apr20-May19 +
    # May20-Jun18 (episode Apr20-Jun18). No overlap; Apr20 - Mar31 - 1 = 19 <= 90 -> switch on Apr20, 19 days.
    (301, 3, "2020-02-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (302, 3, "2020-03-02", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (303, 3, "2020-04-20", DAPA, 30, 30.0, "PAID", None, None, "COMMERCIAL"),
    (304, 3, "2020-05-20", DAPA, 30, 30.0, "PAID", None, None, "COMMERCIAL"),
    # P4 add-on: metformin 90d Jan1-Mar31, Apr1-Jun29, Jun30-Sep27 (episode Jan1-Sep27, 2021);
    # empagliflozin Mar1-Mar30 + Mar31-Apr29 (episode Mar1-Apr29). Overlap min(Sep27, Apr29) - Mar1 + 1 =
    # 31 + 29 = 60 >= 30 -> add_on metformin -> sglt2 on Mar1 with 60 days.
    (401, 4, "2021-01-01", MET, 90, 180.0, "PAID", None, None, "MEDICAID"),
    (402, 4, "2021-04-01", MET, 90, 180.0, "PAID", None, None, "MEDICAID"),
    (403, 4, "2021-06-30", MET, 90, 180.0, "PAID", None, None, "MEDICAID"),
    (404, 4, "2021-03-01", EMPA, 30, 30.0, "PAID", None, None, "MEDICAID"),
    (405, 4, "2021-03-31", EMPA, 30, 30.0, "PAID", None, None, "MEDICAID"),
    # P5 fixed-dose combination: one claim -> a metformin and an empagliflozin event (different groups), both
    # Jan10-Feb8 2022 -> concurrent_start metformin -> sglt2_inhibitors on Jan10.
    (501, 5, "2022-01-10", EMPA_MET, 30, 30.0, "PAID", None, None, "COMMERCIAL"),
    # P6 adjudication: 601 reversed by 602; 603 rejected then 604 paid the same day; 604 delivered twice.
    # Only one 604 row counts: episode Jun1-Jun30 2020, 1 event.
    (601, 6, "2020-05-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (602, 6, "2020-05-03", MET, 30, 60.0, "REVERSAL", None, 601, "COMMERCIAL"),
    (603, 6, "2020-06-01", MET, 30, 60.0, "REJECTED", "79", None, "COMMERCIAL"),
    (604, 6, "2020-06-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (604, 6, "2020-06-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    # P7 untreated: diagnoses only.
    # P8 unknown and malformed NDCs: excluded as unmapped, never assigned to a drug group.
    (801, 8, "2020-02-02", NDC["unmapped"], 30, 30.0, "PAID", None, None, "COMMERCIAL"),
    (802, 8, "2020-03-03", "12345-678", 30, 30.0, "PAID", None, None, "COMMERCIAL"),
    # P9 right-censored: Dec15 2023 30d covers Dec15-Jan13 2024; observation ends 2023-12-31 and
    # Jan13 + 45 >= Dec31 -> censored, coverage beyond observation, observed_end 2023-12-31.
    (901, 9, "2023-12-15", MET, 30, 60.0, "PAID", None, None, "MEDICARE"),
    # P10 two observation periods: activity 2019-01-01, 2019-02-01 | 2020-06-01 ... 2023-03-01 (gap 486 > 365).
    (1001, 10, "2019-02-01", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
    (1002, 10, "2020-06-10", MET, 30, 60.0, "PAID", None, None, "COMMERCIAL"),
]
P10_DX = ["2019-01-01", "2020-06-01", "2021-05-01", "2022-04-01", "2023-03-01"]

# (episode_id, start, end, n_events, gap_days_bridged, left_censored, right_censored, end_status,
#  observed_end, coverage_beyond_observation)
EPISODES = [
    ("1:metformin:1", "2020-01-01", "2020-05-09", 3, 40, False, False, "discontinued", "2020-05-09", False),
    ("2:metformin:1", "2020-03-01", "2020-03-30", 1, 0, False, False, "discontinued", "2020-03-30", False),
    ("2:metformin:2", "2020-06-01", "2020-06-30", 1, 0, False, False, "discontinued", "2020-06-30", False),
    ("3:metformin:1", "2020-02-01", "2020-03-31", 2, 0, False, False, "discontinued", "2020-03-31", False),
    ("3:sglt2_inhibitors:1", "2020-04-20", "2020-06-18", 2, 0, False, False, "discontinued", "2020-06-18", False),
    ("4:metformin:1", "2021-01-01", "2021-09-27", 3, 0, False, False, "discontinued", "2021-09-27", False),
    ("4:sglt2_inhibitors:1", "2021-03-01", "2021-04-29", 2, 0, False, False, "discontinued", "2021-04-29", False),
    ("5:metformin:1", "2022-01-10", "2022-02-08", 1, 0, False, False, "discontinued", "2022-02-08", False),
    ("5:sglt2_inhibitors:1", "2022-01-10", "2022-02-08", 1, 0, False, False, "discontinued", "2022-02-08", False),
    ("6:metformin:1", "2020-06-01", "2020-06-30", 1, 0, False, False, "discontinued", "2020-06-30", False),
    ("9:metformin:1", "2023-12-15", "2024-01-13", 1, 0, False, True, "censored", "2023-12-31", True),
    # P10 period 1 (2019-01-01..2019-02-01): Feb1 - Jan1 = 31 < 180 -> left-censored; Feb1-Mar2 2019 ends
    # after the period -> right-censored, observed_end Feb1. Period 2 starts 2020-06-01: Jun10 is 9 days in
    # -> left-censored; Jul9 + 45 < 2023-03-01 -> discontinued.
    ("10:metformin:1", "2019-02-01", "2019-03-02", 1, 0, True, True, "censored", "2019-02-01", True),
    ("10:metformin:2", "2020-06-10", "2020-07-09", 1, 0, True, False, "discontinued", "2020-07-09", False),
]
# (person, kind, from, to, on_date, days); P10: 2020-06-10 - 2019-03-02 - 1 = 366 + 100 - 1 = 465
TRANSITIONS = [
    (2, "restart", "2:metformin:1", "2:metformin:2", "2020-06-01", 62),
    (3, "switch", "3:metformin:1", "3:sglt2_inhibitors:1", "2020-04-20", 19),
    (4, "add_on", "4:metformin:1", "4:sglt2_inhibitors:1", "2021-03-01", 60),
    (5, "concurrent_start", "5:metformin:1", "5:sglt2_inhibitors:1", "2022-01-10", 0),
    (10, "resumed_after_unobserved_interval", "10:metformin:1", "10:metformin:2", "2020-06-10", 465),
]
# source record -> (exclusion_reason or None for included, episode step)
RECORD_OUTCOMES = {
    "601": "reversed_claim",
    "602": "reversal_transaction",
    "603": "rejected_claim",
    "801": "unmapped_code",
    "802": "unmapped_code",
}


@pytest.fixture
def fixture_builder(tmp_path):
    path = tmp_path / "fixture.duckdb"
    con = connect(path)
    init_schemas(con)
    load_demo_vocabulary(con)
    con.execute(LAAD_DDL)
    bulk_insert(con, "laad.patient", PATIENT_COLS, [(p, 1960, "F") for p in range(1, 11)], 1000)
    rows = [(r[0], r[1], D(r[2]), *r[3:]) for r in RX]
    bulk_insert(con, "laad.rx_claims", RX_COLS, rows, 1000)
    dx: list[tuple] = []
    for p in range(1, 10):
        dx += [(len(dx) + 1, p, D(d), "11", "E119", None, None, None) for d in standard_activity(p)]
    dx += [(len(dx) + 1, 10, D(d), "11", "E119", None, None, None) for d in P10_DX]
    bulk_insert(con, "laad.dx_claims", DX_COLS, dx, 1000)
    con.close()
    s = dataclasses.replace(Settings(), db_path=path, dataset="iqvia_laad", local_datasets_dir=tmp_path / "ds")
    b = CohortBuilder(s, backend=FakeLLM({}))
    yield b
    b.con.close()


@pytest.mark.parametrize("engine", ["sql", "reference"])
def test_hand_verified_t2d_fixture(fixture_builder, engine):
    b = fixture_builder
    rules = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
    with TreatmentService(b) as svc:
        out = svc.run(rules, "analyst", reference_date=REF, engine=engine)
    rid = out["run_id"]
    got = b.con.execute(
        "SELECT episode_id, episode_start, episode_end, n_events, gap_days_bridged, left_censored, right_censored, "
        "end_status, observed_end, coverage_beyond_observation FROM results.treatment_episode WHERE run_id = ? "
        "ORDER BY person_id, drug_group, episode_start",
        [rid],
    ).fetchall()
    expected = [(e[0], D(e[1]), D(e[2]), *e[3:8], D(e[8]), e[9]) for e in EPISODES]
    assert got == expected
    tr = b.con.execute(
        "SELECT person_id, kind, from_episode, to_episode, on_date, days FROM results.treatment_transition "
        "WHERE run_id = ? ORDER BY person_id, on_date, kind",
        [rid],
    ).fetchall()
    assert tr == [(*t[:4], D(t[4]), t[5]) for t in TRANSITIONS]
    outcomes = dict(
        b.con.execute(
            "SELECT source_record_id, exclusion_reason FROM results.treatment_event WHERE run_id = ? AND "
            "source_record_id IN ('601', '602', '603', '801', '802')",
            [rid],
        ).fetchall()
    )
    assert outcomes == RECORD_OUTCOMES
    dup = b.con.execute(
        "SELECT count(*) FILTER (WHERE included), count(*) FILTER (WHERE exclusion_reason = 'duplicate_record') "
        "FROM results.treatment_event WHERE run_id = ? AND source_record_id = '604'",
        [rid],
    ).fetchone()
    assert dup == (1, 1)
    shifted = b.con.execute(
        "SELECT coverage_start, shifted_days FROM results.treatment_event WHERE run_id = ? "
        "AND source_record_id = '102'",
        [rid],
    ).fetchone()
    assert shifted == (D("2020-01-31"), 6)
    combo = b.con.execute(
        "SELECT count(*), bool_and(quality_flags LIKE '%combination_product%') FROM results.treatment_event "
        "WHERE run_id = ? AND source_record_id = '501'",
        [rid],
    ).fetchone()
    assert combo == (2, True)
    assert out["summary"]["review_status"] == {"rule_set": "illustrative", "source_config": "illustrative"}
    assert any("not 'validated'" in x for x in out["summary"]["limitations"])


def test_fixture_runs_are_reproducible_across_engines(fixture_builder):
    rules = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
    with TreatmentService(fixture_builder) as svc:
        a = svc.run(rules, "analyst", reference_date=REF)
        b = svc.run(rules, "analyst", reference_date=REF, engine="reference")
        c = svc.run(rules, "analyst", reference_date=REF, reuse=True)
    assert a["output_hash"] == b["output_hash"] and c["run_id"] == a["run_id"]
    assert a["input_fingerprint"] != b["input_fingerprint"]  # the engine is part of the recorded inputs
