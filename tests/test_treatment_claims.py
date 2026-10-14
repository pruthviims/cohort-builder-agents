"""Claims adjudication and normalization edge cases for the treatment engine: transaction ordering, late
corrections, reference scope, conflicting duplicates, implausible quantities and supply, rebills, runs
with no qualifying events, and run-to-run diff diagnostics. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import json
import random

import pytest

from cohort_builder.treatment_engine import diff_runs
from cohort_builder.treatment_service import TreatmentService
from test_treatment_engine import rules, source_cfg
from test_treatment_scale import RX_SOURCE, episodes, fills, make_db, observe, run, steps, transitions

OBS = {p: [("2019-01-01", "2022-12-31")] for p in range(1, 40)}


@pytest.fixture
def db(tmp_path):
    b = make_db(tmp_path)
    yield b
    b.con.close()


def flags(b, run_id: str) -> dict[str, str | None]:
    rows = b.con.execute(
        "SELECT source_record_id, quality_flags FROM results.treatment_event WHERE run_id = ? ORDER BY event_id",
        [run_id],
    ).fetchall()
    return {r: f for r, f in rows}


def test_late_and_out_of_order_corrections(db):
    observe(db, OBS)
    rows = [
        ("o1", 1, "2020-03-01", 801, 30),
        ("x1", 1, "2020-02-20", 801, 30, "X", "o1"),  # reversal dated BEFORE the claim it reverses
        ("o2", 1, "2020-06-01", 801, 30),
        ("x2", 1, "2020-09-01", 801, 30, "X", "o2"),  # late-arriving reversal, months later
    ]
    fills(db, rows)
    out = run(db)
    st = steps(db, out["run_id"])
    assert st["o1"][1] == "reversed_claim" and st["o2"][1] == "reversed_claim"
    assert "reversal_before_original" in flags(db, out["run_id"])["x1"]
    assert episodes(db, out["run_id"]) == []  # nothing left to establish coverage
    # the same claims loaded in another order give the same logical output
    db.con.execute("DELETE FROM claims.pharmacy")
    fills(db, list(reversed(rows)))
    assert run(db)["output_hash"] == out["output_hash"]


@pytest.mark.parametrize("scope", ["record", "record_and_person"])
def test_reference_scope_decides_cross_patient_reversals(db, scope):
    observe(db, OBS)
    fills(db, [("c1", 2, "2020-01-01", 801, 30), ("r1", 3, "2020-01-05", 801, 30, "X", "c1")])
    src = {**RX_SOURCE, "status": {**RX_SOURCE["status"], "reference_scope": scope}}
    out = TreatmentService(db).run(rules(), "a", "default", source_cfg([src]), __import__("datetime").date(2026, 1, 1))
    st = steps(db, out["run_id"])
    assert "reference_person_mismatch" in flags(db, out["run_id"])["r1"]
    if scope == "record":  # vendor semantics: claim ids are unique per source, the reversal applies
        assert st["c1"][1] == "reversed_claim"
    else:  # claim ids only unique per patient: another patient's reversal does not cancel this claim
        assert st["c1"][0] == "counted"


def test_conflicting_duplicate_records_are_flagged_and_resolved_deterministically(db):
    observe(db, OBS)
    rows = [("d1", 4, "2020-01-01", 801, 30), ("d1", 4, "2020-01-01", 801, 90)]  # same claim id, other supply
    fills(db, rows)
    out = run(db)
    got = db.con.execute(
        "SELECT event_id, days_supply, exclusion_reason, quality_flags FROM results.treatment_event "
        "WHERE run_id = ? ORDER BY event_id",
        [out["run_id"]],
    ).fetchall()
    assert all("conflicting_duplicate_record" in g[3] for g in got)
    assert [(g[1], g[2]) for g in got] == [(30, None), (90, "duplicate_record")]  # tie-break: lower supply first
    db.con.execute("DELETE FROM claims.pharmacy")
    fills(db, list(reversed(rows)))
    assert run(db)["output_hash"] == out["output_hash"]


def test_invalid_and_extreme_days_supply_and_quantity(db):
    observe(db, OBS)
    fills(
        db,
        [
            ("z0", 5, "2020-01-01", 801, 0),
            ("zn", 5, "2020-02-01", 801, -5),
            ("z366", 5, "2020-03-01", 801, 366),
            ("znull", 5, "2020-04-01", 801, None),
            ("z365", 5, "2020-05-01", 801, 365),
        ],
    )
    out = run(db)
    st, fl = steps(db, out["run_id"]), flags(db, out["run_id"])
    for r in ("z0", "zn", "z366"):
        assert st[r][1] == "missing_days_supply" and "days_supply_out_of_range" in fl[r]
    assert st["znull"][1] == "missing_days_supply" and "days_supply_missing" in fl["znull"]
    assert st["z365"][0] == "counted"
    assert "quantity_invalid" in fl["zn"] and "quantity_invalid" in fl["z0"]  # the harness sets qty = days


def test_rejected_refill_and_reversal_rebill_corrections(db):
    observe(db, OBS)
    fills(
        db,
        [
            ("r1", 6, "2020-01-01", 801, 30, "R"),  # refill-too-soon rejection, then paid the same day
            ("p1", 6, "2020-01-01", 801, 30),
            ("p2", 6, "2020-03-01", 801, 30),  # billed with the wrong supply ...
            ("v2", 6, "2020-03-02", 801, 30, "X", "p2"),  # ... reversed ...
            ("p3", 6, "2020-03-02", 801, 90),  # ... and rebilled
        ],
    )
    out = run(db)
    st = steps(db, out["run_id"])
    assert st["r1"][1] == "rejected_claim" and st["p2"][1] == "reversed_claim" and st["p3"][0] == "counted"
    eps = episodes(db, out["run_id"])
    # Jan 1-30 and the rebilled 90 days from Mar 2: 31 uncovered days (<= 45) are bridged; the reversed claim
    # contributes nothing, so the episode ends May 30, not Mar 30 + ...
    assert [(e[2].isoformat(), e[3].isoformat(), e[10]) for e in eps] == [("2020-01-01", "2020-05-30", 31)]


def test_runs_with_no_qualifying_events_complete_cleanly(db):
    observe(db, OBS)
    fills(db, [("u1", 7, "2020-01-01", "00000000000", 30), ("r1", 7, "2020-01-01", 801, 30, "R")])
    out = run(db)
    assert out["status"] == "completed" and episodes(db, out["run_id"]) == [] and transitions(db, out["run_id"]) == []
    g = out["summary"]["drug_groups"]["metformin"]
    assert g["persons"] == 0 and g["episodes"] == 0 and g["median_episode_days"] is None
    assert out["summary"]["records"]["by_outcome"]["included"] == 0


def test_diff_runs_pinpoints_differences(db):
    observe(db, OBS)
    rng = random.Random(3)
    fills(db, [(f"c{i}", 1 + i % 5, f"2020-{1 + i % 12:02d}-{1 + rng.randint(0, 27):02d}", 801, 30) for i in range(60)])
    a, b = run(db), run(db)
    same = diff_runs(db.con, a["run_id"], b["run_id"])
    assert all(v == {"only_in_a": 0, "only_in_b": 0} for v in same.values())
    c = run(db, rules(episode={"max_gap_days": 0}))
    d = diff_runs(db.con, a["run_id"], c["run_id"], examples=2)
    assert d["episodes"]["only_in_a"] > 0 and d["episodes"]["examples"][0]["side"] in ("a", "b")
    assert {"episode_id", "episode_start", "episode_end"} <= set(d["episodes"]["examples"][0])
    json.dumps(d, default=str)  # serializable for reports


def test_reversal_with_another_payer_is_flagged_but_still_applies(db):
    observe(db, OBS)
    db.con.execute("ALTER TABLE claims.pharmacy ADD COLUMN payer VARCHAR")
    fills(
        db,
        [
            ("p1", 8, "2020-01-01", 801, 30),
            ("x1", 8, "2020-01-03", 801, 30, "X", "p1"),  # reversed by another payer
            ("p2", 8, "2020-02-01", 801, 30),
            ("x2", 8, "2020-02-02", 801, 30, "X", "p2"),  # same payer: no flag
            ("p3", 8, "2020-03-01", 801, 30),
            ("x3", 8, "2020-03-02", 801, 30, "X", "p3"),  # payer unknown on the reversal: nothing to compare
        ],
    )
    db.con.execute(
        "UPDATE claims.pharmacy SET payer = CASE claim_id WHEN 'x1' THEN 'MEDICAID' WHEN 'x3' THEN NULL "
        "ELSE 'COMMERCIAL' END"
    )
    src = {**RX_SOURCE, "payer_column": "payer"}
    out = TreatmentService(db).run(rules(), "a", "default", source_cfg([src]), __import__("datetime").date(2026, 1, 1))
    fl, st = flags(db, out["run_id"]), steps(db, out["run_id"])
    assert "reference_payer_mismatch" in fl["x1"]
    assert "reference_payer_mismatch" not in (fl["x2"] or "") and "reference_payer_mismatch" not in (fl["x3"] or "")
    assert st["p1"][1] == "reversed_claim"  # flagged for review, never silently dropped or re-included
    # the same claims without payer_column: no flag, same inclusion decisions
    plain = run(db)
    assert "reference_payer_mismatch" not in (flags(db, plain["run_id"])["x1"] or "")
    with pytest.raises(ValueError, match="payer_column needs"):
        source_cfg([{**RX_SOURCE, "status": None, "payer_column": "payer"}])
