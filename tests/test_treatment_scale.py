"""Treatment engine v2: observation boundaries, transition semantics, claims integrity, the set-based SQL
engine against the in-memory reference, batch-size independence, the output-hash contract, run lifecycle,
resource limits and reproducibility. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import date, timedelta

import pytest

from cohort_builder import treatment_service as ts
from cohort_builder.config import Settings
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.treatment_config import TreatmentRuleSet
from cohort_builder.treatment_engine import HASH_SECTIONS, OUTPUT_HASH_VERSION, bulk_insert, output_hash
from cohort_builder.treatment_service import TreatmentError, TreatmentLimits, TreatmentService
from fake_llm import FakeLLM
from test_treatment_engine import ANCESTORS, CONCEPTS, MAPS, rules, source_cfg

D = date.fromisoformat
EXTRA = [
    (807, "empagliflozin / dapagliflozin", "Drug", "RxNorm", "Clinical Drug", "S", "P807"),
    (708_1, "ndc empa/dapa", "Drug", "NDC", "11-digit NDC", None, "66666000101"),
]
EXTRA_MAPS = [(708_1, 807)]
EXTRA_ANC = [(45775965, 807), (44785829, 807), (900, 807)]
NDC = {
    801: "11111000101",
    802: "22222000101",
    803: "33333000101",
    804: "44444000101",
    805: "55555000101",
    807: "66666000101",
}
RX_COLS = [
    ("claim_id", "VARCHAR"),
    ("member", "BIGINT"),
    ("fill_dt", "DATE"),
    ("ndc", "VARCHAR"),
    ("dsup", "INTEGER"),
    ("qty", "DOUBLE"),
    ("status", "VARCHAR"),
    ("ref_claim", "VARCHAR"),
]
RX_SOURCE = {
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
}


@pytest.fixture
def db(tmp_path):
    """Empty claims tables, a tiny vocabulary and helpers to load fills and observation periods."""
    s = dataclasses.replace(Settings(), db_path=tmp_path / "s.duckdb", local_datasets_dir=tmp_path / "ds")
    b = CohortBuilder(s, backend=FakeLLM({}))
    con = b.con
    con.executemany(
        "INSERT INTO vocab.concept VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)", [list(c) for c in CONCEPTS + EXTRA]
    )
    con.executemany(
        "INSERT INTO vocab.concept_relationship VALUES (?,?,'Maps to',NULL,NULL,NULL)",
        [list(m) for m in MAPS + EXTRA_MAPS],
    )
    con.executemany(
        "INSERT INTO vocab.concept_relationship VALUES (?,?,'Has brand name',NULL,NULL,NULL)", [[802, 851], [803, 852]]
    )
    con.executemany("INSERT INTO vocab.concept_ancestor VALUES (?,?,1,1)", [list(a) for a in ANCESTORS + EXTRA_ANC])
    con.execute(
        "CREATE SCHEMA claims; CREATE TABLE claims.pharmacy (claim_id VARCHAR, member BIGINT, fill_dt DATE, "
        "ndc VARCHAR, dsup INTEGER, qty DOUBLE, status VARCHAR, ref_claim VARCHAR)"
    )
    yield b
    b.con.close()


def fills(b: CohortBuilder, rows: list[tuple]) -> None:
    """(claim, person, date, product, days[, status, ref])"""
    full = [
        (r[0], r[1], D(r[2]) if isinstance(r[2], str) else r[2], NDC.get(r[3], r[3]), r[4], r[4],
         r[5] if len(r) > 5 else "P", r[6] if len(r) > 6 else None)
        for r in rows
    ]  # fmt: skip
    bulk_insert(b.con, "claims.pharmacy", RX_COLS, full, 10_000)


def observe(b: CohortBuilder, periods: dict[int, list[tuple[str, str]]]) -> None:
    rows = [[p, D(s_), D(e_)] for p, ps in sorted(periods.items()) for s_, e_ in ps]
    for i, r in enumerate(rows, start=1):
        b.con.execute("INSERT INTO cdm.observation_period VALUES (?,?,?,?,32817)", [i, *r])


REF_DATE = D("2026-01-01")


def run(b: CohortBuilder, r=None, engine="sql", batch_size=None, reference_date=REF_DATE) -> dict:
    return TreatmentService(b).run(
        r or rules(), "analyst", "default", source_cfg([RX_SOURCE]), reference_date, engine, batch_size
    )


def table(b: CohortBuilder, run_id: str, name: str, cols: str, order: str) -> list[tuple]:
    return b.con.execute(
        f"SELECT {cols} FROM results.treatment_{name} WHERE run_id = ? ORDER BY {order}", [run_id]
    ).fetchall()


def episodes(b: CohortBuilder, run_id: str, person: int | None = None) -> list[tuple]:
    rows = table(
        b, run_id, "episode",
        "person_id, drug_group, period_id, episode_start, episode_end, end_status, left_censored, right_censored, "
        "observed_end, coverage_beyond_observation, n_events, gap_days_bridged",
        "person_id, drug_group, episode_start",
    )  # fmt: skip
    return [r[1:] for r in rows if person is None or r[0] == person]


def transitions(b: CohortBuilder, run_id: str, person: int | None = None) -> list[tuple]:
    rows = table(
        b, run_id, "transition", "person_id, kind, from_group, to_group, on_date, days, from_products, to_products",
        "person_id, on_date, kind, from_group, to_group",
    )  # fmt: skip
    return [r[1:] for r in rows if person is None or r[0] == person]


def steps(b: CohortBuilder, run_id: str) -> dict[str, tuple]:
    return {
        r[0]: r[1:]
        for r in table(b, run_id, "event", "source_record_id, episode_step, exclusion_reason, episode_id", "event_id")
    }


BOTH = pytest.mark.parametrize("engine", ["sql", "reference"])


# ---- A. observation-period boundaries --------------------------------------------------------------------------------
@BOTH
def test_events_before_during_after_and_between_observation_periods(db, engine):
    observe(db, {1: [("2020-01-01", "2020-06-30"), ("2021-01-01", "2021-12-31")]})
    fills(
        db,
        [
            ("before", 1, "2019-12-01", 801, 30),  # before the first period
            ("p1a", 1, "2020-01-10", 801, 30),
            ("p1b", 1, "2020-02-15", 801, 30),
            ("gap", 1, "2020-09-01", 801, 30),  # between periods: not observed
            ("p2a", 1, "2021-01-05", 801, 30),  # 35 days after the next period starts: within max gap of p1?
            ("after", 1, "2022-03-01", 801, 30),
        ],
    )
    out = run(db, engine=engine)
    st = steps(db, out["run_id"])
    assert st["before"][:2] == ("outside_observation", "outside_observation")
    assert st["gap"][1] == "outside_observation" and st["after"][1] == "outside_observation"
    eps = episodes(db, out["run_id"])
    # the unobserved interval is never bridged, and the second period's episode is a separate one
    assert [(e[1], e[2], e[3]) for e in eps] == [
        (1, D("2020-01-10"), D("2020-03-15")),
        (2, D("2021-01-05"), D("2021-02-03")),
    ]
    assert eps[0][4] == "discontinued" and eps[0][5] is True  # starts 9 days after observation start
    assert eps[1][4] == "discontinued"
    tr = transitions(db, out["run_id"])
    assert [t[0] for t in tr] == ["resumed_after_unobserved_interval"]  # not a 'restart': the gap was unobserved


@BOTH
def test_coverage_beyond_observation_end_is_censored_not_discontinued(db, engine):
    observe(db, {2: [("2020-01-01", "2020-03-31")], 3: [("2020-01-01", "2020-12-31")]})
    fills(
        db,
        [
            ("a", 2, "2020-03-15", 801, 90),  # supply runs past the observation end
            ("b", 3, "2020-10-01", 801, 30),  # ends Oct 30; + 45 gap days reaches Dec 14 < Dec 31 -> observed
            ("c", 3, "2020-11-15", 805, 30),  # ends Dec 14; + 60 > Dec 31 -> censored
        ],
    )
    out = run(db, engine=engine)
    e2 = episodes(db, out["run_id"], 2)[0]
    assert e2[4] == "censored" and e2[6] is True and e2[7] == D("2020-03-31") and e2[8] is True
    e3 = episodes(db, out["run_id"], 3)
    assert [(e[0], e[4]) for e in e3] == [("metformin", "discontinued"), ("sglt2", "censored")]
    # censored episodes never become discontinuations in the summary
    g = out["summary"]["drug_groups"]
    assert g["metformin"]["end_status"]["discontinued"] == "<10" and g["metformin"]["end_status"]["censored"] in (
        "<10",
        "suppressed",
    )


@BOTH
def test_overlapping_and_touching_periods_are_merged(db, engine):
    observe(db, {4: [("2020-01-01", "2020-03-31"), ("2020-04-01", "2020-06-30"), ("2020-05-01", "2020-12-31")]})
    fills(db, [("a", 4, "2020-03-01", 801, 30), ("b", 4, "2020-04-10", 801, 30)])
    out = run(db, engine=engine)
    eps = episodes(db, out["run_id"])
    assert len(eps) == 1 and eps[0][1] == 1 and eps[0][4] == "discontinued"  # one merged period, bridged gap


@BOTH
def test_restart_needs_an_observed_gap(db, engine):
    observe(db, {5: [("2019-01-01", "2022-12-31")]})
    fills(db, [("a", 5, "2020-01-01", 801, 30), ("b", 5, "2020-05-01", 801, 30)])
    out = run(db, engine=engine)
    assert [e[4] for e in episodes(db, out["run_id"])] == ["discontinued", "discontinued"]
    assert transitions(db, out["run_id"]) == [
        ("restart", "metformin", "metformin", D("2020-05-01"), 91, None, None)  # Jan 31 .. Apr 30 uncovered
    ]


# ---- B. transition semantics -----------------------------------------------------------------------------------------
@BOTH
def test_product_and_brand_changes_within_a_group(db, engine):
    observe(db, {6: [("2019-01-01", "2022-12-31")]})
    fills(
        db,
        [
            ("a", 6, "2020-01-01", 802, 30),  # brand X
            ("b", 6, "2020-01-31", 803, 30),  # brand Y, same ingredient
            ("c", 6, "2020-03-01", 803, 30),  # unchanged
            ("d", 6, "2020-03-31", 805, 30),  # other ingredient of the class
            ("e", 6, "2020-03-31", 802, 30),  # same day: the day's product set is {802, 805}
        ],
    )
    out = run(db, engine=engine)
    tr = transitions(db, out["run_id"])
    assert [(t[0], t[3], t[5], t[6]) for t in tr] == [
        ("product_change", D("2020-01-31"), "802", "803"),
        ("product_change", D("2020-03-31"), "803", "802,805"),
    ]
    assert len(episodes(db, out["run_id"])) == 1


@BOTH
def test_switch_add_on_boundaries_and_multiple_candidates(db, engine):
    observe(db, {p: [("2019-01-01", "2022-12-31")] for p in (7, 8, 9, 10)})
    fills(
        db,
        [
            # 7: overlap of exactly 30 days (threshold) -> add_on
            ("a7", 7, "2020-01-01", 801, 60), ("b7", 7, "2020-01-31", 805, 90),
            # 8: overlap of 29 days -> switch (new group lasts beyond the first)
            ("a8", 8, "2020-01-01", 801, 60), ("b8", 8, "2020-02-01", 805, 90),
            # 9: gap of exactly 60 days after the end (switch window) -> switch; 61 -> nothing for drugz
            ("a9", 9, "2020-01-01", 801, 30), ("b9", 9, "2020-03-31", 805, 30),
        ],
    )  # fmt: skip
    out = run(db, engine=engine)
    assert [(t[0], t[1], t[2], t[4]) for t in transitions(db, out["run_id"], 7)] == [
        ("add_on", "metformin", "sglt2", 30)
    ]
    assert [(t[0], t[4]) for t in transitions(db, out["run_id"], 8)] == [("switch", 0)]
    assert [(t[0], t[4]) for t in transitions(db, out["run_id"], 9)] == [("switch", 60)]


@BOTH
def test_multiple_candidate_transitions_are_all_kept(db, engine):
    observe(db, {10: [("2019-01-01", "2022-12-31")]})
    # metformin ends Jan 30; empagliflozin starts Feb 10, dapagliflozin Feb 15 (overlapping empa by 25 days)
    fills(db, [("a", 10, "2020-01-01", 801, 30), ("b", 10, "2020-02-10", 802, 30), ("c", 10, "2020-02-15", 805, 30)])
    groups = {"metformin": {"concepts": [1503297]}, "empa": {"concepts": [45775965]}, "dapa": {"concepts": [44785829]}}
    data = rules().model_dump()
    data["drug_groups"] = groups  # replace (not merge): empagliflozin may belong to one group only
    out = run(db, TreatmentRuleSet.model_validate(data), engine=engine)
    assert {(t[0], t[1], t[2], t[4]) for t in transitions(db, out["run_id"])} == {
        ("switch", "metformin", "empa", 10),
        ("switch", "metformin", "dapa", 15),
        ("switch", "empa", "dapa", 0),
    }


@BOTH
def test_same_day_start_is_concurrent_not_add_on(db, engine):
    observe(db, {11: [("2019-01-01", "2022-12-31")]})
    fills(db, [("a", 11, "2020-01-01", 801, 90), ("b", 11, "2020-01-01", 805, 90)])
    out = run(db, engine=engine)
    assert [t[0] for t in transitions(db, out["run_id"])] == ["concurrent_start"]


# ---- C. claims integrity ---------------------------------------------------------------------------------------------
@BOTH
def test_adjustment_chains_and_same_group_combinations(db, engine):
    observe(db, {12: [("2019-01-01", "2022-12-31")]})
    fills(
        db,
        [
            ("C", 12, "2020-01-01", 801, 30),
            ("A1", 12, "2020-01-01", 801, 60, "A", "C"),
            ("A2", 12, "2020-01-01", 801, 90, "A", "A1"),  # only the last adjustment counts
            ("K", 12, "2020-06-01", 807, 30),  # empa/dapa: two ingredients, both sglt2
        ],
    )
    out = run(db, rules(episode={"same_day_same_group": "sum_supply"}), engine=engine)
    st = steps(db, out["run_id"])
    assert st["C"][1] == "superseded_by_adjustment" and st["A1"][1] == "superseded_by_adjustment"
    assert st["A2"][0] == "counted"
    combo = table(db, out["run_id"], "event", "episode_step", "event_id")
    assert [r[0] for r in combo].count("same_record_merged") == 1  # the 2nd ingredient adds no supply
    eps = episodes(db, out["run_id"])
    met = [e for e in eps if e[0] == "metformin"][0]
    assert (met[2], met[3]) == (D("2020-01-01"), D("2020-03-30"))  # 90 days, not 30 + 60 + 90
    sg = [e for e in eps if e[0] == "sglt2"][0]
    assert sg[3] - sg[2] == timedelta(days=29)  # the combination supplies 30 days once, not 60 (sum rule)


# ---- D. SQL engine == reference engine, any batch size ---------------------------------------------------------------
def random_claims(seed: int, persons: int) -> tuple[list[tuple], dict[int, list[tuple[str, str]]]]:
    rng = random.Random(seed)
    rows: list[tuple] = []
    obs: dict[int, list[tuple[str, str]]] = {}
    n = 0
    for p in range(1, persons + 1):
        base = D("2018-01-01") + timedelta(days=rng.randint(0, 400))
        periods = []
        cursor = base
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            end = cursor + timedelta(days=rng.randint(60, 700))
            periods.append((cursor.isoformat(), end.isoformat()))
            cursor = end + timedelta(days=rng.choice([-30, 0, 1, 2, 90, 200]))  # overlap, touch, gap
        if periods:
            obs[p] = periods
        day = base - timedelta(days=rng.randint(0, 60))
        for _ in range(rng.randint(1, 40)):
            n += 1
            product = rng.choice([801, 801, 802, 803, 804, 805, 807, "00000000000"])
            days = rng.choice([7, 14, 30, 30, 30, 60, 90, None, 400])
            status = rng.choices(["P", "R", "X", "A"], [90, 3, 4, 3])[0]
            ref = f"c{rng.randint(max(1, n - 5), n)}" if status in ("X", "A") else None
            rows.append((f"c{n}", p, day, product, days, status, ref))
            if rng.random() < 0.06:  # exact duplicate claim line
                rows.append((f"c{n}", p, day, product, days, status, ref))
            if rng.random() < 0.06:  # same product, same day, other claim id
                n += 1
                rows.append((f"c{n}", p, day, product, days, "P", None))
            day = day + timedelta(days=rng.choice([0, 1, 5, 20, 25, 28, 30, 45, 61, 120, 400]))
    return rows, obs


EQUIV_RULES = [
    {},
    {"episode": {"max_carryover_days": 10}},  # the cap binds often: exercises the streaming path
    {"episode": {"overlap": "ignore"}},
    {"episode": {"same_day_same_group": "sum_supply", "duplicates": "same_record"}},
    {"censoring": {"observation": "none", "washout_days": 0}},
]


@pytest.mark.parametrize("over", EQUIV_RULES, ids=["default", "cap10", "ignore", "sum_same_record", "no_obs"])
def test_sql_engine_matches_reference_row_by_row(db, over):
    rows, obs = random_claims(7, 150)
    fills(db, rows)
    observe(db, obs)
    r = rules(**over)
    sql, ref = run(db, r, "sql"), run(db, r, "reference")
    assert sql["status"] == ref["status"] == "completed"
    for name, _table, cols, order in HASH_SECTIONS:
        key = ", ".join(order)
        a = table(db, sql["run_id"], _table.split("_", 1)[1], ", ".join(cols), key)
        b = table(db, ref["run_id"], _table.split("_", 1)[1], ", ".join(cols), key)
        assert a == b, name
    assert sql["output_hash"] == ref["output_hash"]
    assert sql["summary"] == ref["summary"]
    if over.get("episode", {}).get("max_carryover_days") == 10:
        assert sql["metrics"]["rows"]["capped_partitions"] > 0  # the streaming path really ran


def test_batch_size_and_person_buckets_do_not_change_results(db):
    rows, obs = random_claims(11, 80)
    fills(db, rows)
    observe(db, obs)
    r = rules(episode={"max_carryover_days": 5})
    hashes = {run(db, r, "sql", bs)["output_hash"] for bs in (1, 3, 17, 100_000)}
    assert len(hashes) == 1
    assert run(db, r, "reference", 2)["output_hash"] in hashes
    for bucket_events in (1, 37, 500):  # 1 = one bucket per event-count unit -> many buckets
        svc = TreatmentService(db)
        svc.limits = dataclasses.replace(svc.limits, bucket_events=bucket_events)
        out = svc.run(r, "analyst", "default", source_cfg([RX_SOURCE]), REF_DATE)
        assert out["metrics"]["rows"]["person_buckets"] > (1 if bucket_events < 500 else 0)
        assert out["output_hash"] in hashes


# ---- E. output hash contract -----------------------------------------------------------------------------------------
def test_output_hash_follows_the_documented_contract(db):
    rows, obs = random_claims(3, 20)
    fills(db, rows)
    observe(db, obs)
    out = run(db)
    h = hashlib.sha256(f"{OUTPUT_HASH_VERSION}\n".encode())
    for name, tab, cols, order in HASH_SECTIONS:
        h.update(f"#{name}\n".encode())
        got = db.con.execute(
            f"SELECT {', '.join(cols)} FROM {tab} WHERE run_id = ? ORDER BY "
            + ", ".join(f"{c} NULLS FIRST" for c in order),
            [out["run_id"]],
        ).fetchall()
        for r in got:
            fields = []
            for v in r:
                if v is None:
                    fields.append("\\N")
                elif isinstance(v, bool):
                    fields.append("true" if v else "false")
                elif isinstance(v, float):
                    fields.append(db.con.execute("SELECT CAST(? AS DOUBLE)::VARCHAR", [v]).fetchone()[0])
                else:
                    fields.append(str(v))
            h.update(("\x1f".join(fields) + "\n").encode())
    assert out["output_hash"] == "sha256:" + h.hexdigest() and out["output_hash_version"] == OUTPUT_HASH_VERSION
    db.con.execute("UPDATE results.treatment_episode SET n_events = n_events + 1 WHERE run_id = ?", [out["run_id"]])
    assert output_hash(db.con, out["run_id"], 1000) != out["output_hash"]  # any changed row changes the hash


# ---- F. reproducibility: reference date and provenance ---------------------------------------------------------------
def test_reference_date_is_recorded_and_used(db):
    observe(db, {13: [("2019-01-01", "2030-12-31")]})
    fills(db, [("now", 13, "2025-06-01", 801, 30), ("future", 13, "2026-03-01", 801, 30)])
    early = run(db, reference_date=D("2025-12-31"))
    late = run(db, reference_date=D("2026-12-31"))
    assert early["reference_date"] == "2025-12-31" and late["reference_date"] == "2026-12-31"
    assert steps(db, early["run_id"])["future"][1] == "implausible_start_date"
    assert steps(db, late["run_id"])["future"][1] is None
    assert run(db, reference_date=D("2025-12-31"))["output_hash"] == early["output_hash"]
    assert early["engine"] == "sql" and early["engine_version"] and early["software_version"]


# ---- G. lifecycle, cleanup and limits --------------------------------------------------------------------------------
def staging_tables(b: CohortBuilder) -> int:
    return b.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'tx_stage'").fetchone()[0]


def test_failure_publishes_nothing_and_cleans_up(db, monkeypatch):
    observe(db, {1: [("2019-01-01", "2022-12-31")]})
    fills(db, [("a", 1, "2020-01-01", 801, 30), ("b", 1, "2020-02-01", 805, 30)])

    def boom(*_a, **_k):
        raise RuntimeError("value 'ZZMARKER9876' broke")

    monkeypatch.setattr(ts, "transitions_sql", boom)
    with pytest.raises(TreatmentError, match="stage 'episodes'"):  # transitions are built per person bucket
        run(db)
    row = db.con.execute("SELECT run_id, status, failed_stage, errors_json FROM meta.treatment_run").fetchone()
    assert row[1:3] == ("failed", "episodes") and "ZZMARKER9876" not in row[3]
    for t in ("event", "episode", "transition"):
        assert (
            db.con.execute(f"SELECT count(*) FROM results.treatment_{t} WHERE run_id = ?", [row[0]]).fetchone()[0] == 0
        )
    assert staging_tables(db) == 0
    svc = TreatmentService(db)
    assert svc.summary(row[0], None)["status"] == "failed"
    db.policy = dataclasses.replace(db.policy, allow_patient_level=True)
    with pytest.raises(KeyError, match="failed"):
        svc.explain_episode(row[0], "1:metformin:1", "admin", True)


def test_publish_failure_rolls_back_the_whole_run(db, monkeypatch):
    observe(db, {1: [("2019-01-01", "2022-12-31")]})
    fills(db, [("a", 1, "2020-01-01", 801, 30)])

    def bad_hash(*_a, **_k):
        raise RuntimeError("hash failed")

    monkeypatch.setattr(ts, "output_hash", bad_hash)
    with pytest.raises(TreatmentError, match="stage 'publish'"):
        run(db)
    assert db.con.execute("SELECT count(*) FROM results.treatment_event").fetchone()[0] == 0  # rolled back
    assert db.con.execute("SELECT status FROM meta.treatment_run").fetchone()[0] == "failed"


def test_timeout_cancels_the_run(db, monkeypatch):
    observe(db, {1: [("2019-01-01", "2022-12-31")]})
    fills(db, [("a", 1, "2020-01-01", 801, 30)])
    svc = TreatmentService(db)
    svc.limits = dataclasses.replace(svc.limits, timeout_seconds=0.001)
    import time

    real = ts.TreatmentService._events

    def slow(self, *a, **k):
        time.sleep(0.05)
        return real(self, *a, **k)

    monkeypatch.setattr(ts.TreatmentService, "_events", slow)
    with pytest.raises(TreatmentError, match="RunCancelled|stage"):
        svc.run(rules(), "analyst", "default", source_cfg([RX_SOURCE]), D("2026-01-01"))
    assert db.con.execute("SELECT status FROM meta.treatment_run").fetchone()[0] == "failed"
    assert staging_tables(db) == 0


def test_concurrency_limit_and_stale_run_cleanup(db):
    svc = TreatmentService(db)
    slot = ts._slot(svc.limits.max_concurrent_runs)
    assert slot.acquire(blocking=False)
    try:
        with pytest.raises(TreatmentError, match="too many concurrent"):
            run(db)
    finally:
        slot.release()
    # a crashed run: 'running' long ago, with a leftover staging table
    db.con.execute(
        "INSERT INTO meta.treatment_run (run_id, status, created_at) VALUES "
        "('00000000-0000-0000-0000-000000000001', 'running', TIMESTAMP '2020-01-01 00:00:00')"
    )
    db.con.execute("CREATE TABLE tx_stage.ev0_00000000000000000000000000000001 (x INTEGER)")
    db.con.execute("CREATE TABLE tx_stage.ev0_ffffffffffffffffffffffffffffffff (x INTEGER)")  # orphan
    got = svc.cleanup_incomplete_runs(60)
    assert got == {"abandoned_runs": 1, "dropped_staging_tables": 2} and staging_tables(db) == 0
    assert svc.summary("00000000-0000-0000-0000-000000000001", None)["status"] == "abandoned"


def test_limits_are_validated(db, monkeypatch):
    for field, bad in (
        ("treatment_batch_size", 5),
        ("treatment_max_concurrent_runs", 0),
        ("treatment_timeout_seconds", 0),
        ("treatment_reference_max_events", 0),
        ("treatment_stale_run_minutes", 0),
    ):
        with pytest.raises(TreatmentError, match="CB_TREATMENT_"):
            TreatmentLimits.from_settings(dataclasses.replace(db.settings, **{field: bad}))
    monkeypatch.setenv("CB_TREATMENT_BATCH_SIZE", "250")
    monkeypatch.setenv("CB_TREATMENT_TIMEOUT_SECONDS", "90")
    lim = TreatmentLimits.from_settings(Settings.from_env())
    assert lim.batch_size == 250 and lim.timeout_seconds == 90
    with pytest.raises(TreatmentError, match="batch size"):
        run(db, batch_size=0)


def test_reference_engine_is_size_limited_and_sql_never_loads_events(db, monkeypatch):
    observe(db, {1: [("2019-01-01", "2022-12-31")]})
    fills(db, [(f"c{i}", 1, D("2020-01-01") + timedelta(days=i * 30), 801, 30) for i in range(5)])
    svc = TreatmentService(db)
    svc.limits = dataclasses.replace(svc.limits, reference_max_events=4)
    with pytest.raises(TreatmentError, match="reference engine is limited"):
        svc.run(rules(), "a", "default", source_cfg([RX_SOURCE]), D("2026-01-01"), "reference")

    def no(*_a, **_k):
        raise AssertionError("the SQL engine must not build episodes in memory")

    monkeypatch.setattr(ts, "build", no)
    assert svc.run(rules(), "a", "default", source_cfg([RX_SOURCE]), D("2026-01-01"), "sql")["status"] == "completed"


def test_summary_counts_discontinuation_once(db):
    observe(db, {p: [("2019-01-01", "2022-12-31")] for p in range(100, 125)})
    fills(
        db,
        [(f"a{p}", p, "2020-01-01", 801, 30) for p in range(100, 125)]
        + [(f"b{p}", p, "2020-06-01", 801, 30) for p in range(100, 125)],
    )
    out = run(db)
    s = out["summary"]
    assert "discontinuation" not in s["transitions"] and s["transitions"]["restart"] == {"metformin -> metformin": 25}
    assert s["drug_groups"]["metformin"]["end_status"] == {"discontinued": 50, "censored": 0, "unknown": 0}
    assert s["drug_groups"]["metformin"]["persons"] == 25 and s["drug_groups"]["metformin"]["episodes"] == 50
    blob = json.dumps(out, default=str)
    assert "a100" not in blob and "11111000101" not in blob
