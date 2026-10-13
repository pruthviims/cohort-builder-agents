"""SYNTHETIC type 2 diabetes claims in the native LAAD-style `laad` schema (placeholder layout).

Every patient is fictitious; identifiers are sequential integers, codes are the demo vocabulary's synthetic
NDCs (labeler 99999) and ICD-10-CM codes, and `laad.synthetic_manifest` labels the database as synthetic
with the generator version, seed and what was injected. Same seed + same version = identical tables.

Each patient has a T2D diagnosis history and one treatment pattern (`ARCHETYPES`): untreated, persistent
metformin, discontinuation, restart after a gap, SGLT2 add-on, switch to SGLT2, fixed-dose combination,
early refills (overlapping supply), SGLT2 first, or sparse/incomplete records. On top of that the generator
injects claims-processing artefacts at fixed rates (`ANOMALY_RATES`): rejected-then-paid claims, reversals,
corrected claims (reversal + rebill), exact and conflicting duplicates, missing or implausible days supply,
negative quantities, unknown transaction types, unmapped and malformed NDCs, orphan and early reversals,
reversals of another patient's claim, reversals with a different payer, future-dated claims, rows without a
patient id, and activity gaps longer than a year (which split activity-based observation periods).

The returned dict and the manifest give exact counts of everything generated and injected, so reports can
cite them rather than estimates. Generating REPLACES the laad.* tables of the target database.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import date, timedelta
from typing import Any

import duckdb

from .synthetic_laad import LAAD_DDL, NDC
from .treatment_engine import bulk_insert

GENERATOR = "synthetic_t2d"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 2026
WINDOW = (date(2018, 1, 1), date(2025, 12, 31))
FUTURE = date(2027, 3, 1)  # after any sensible reference date: an implausible service date

ARCHETYPES = {
    "untreated": 0.12,
    "metformin_persistent": 0.20,
    "metformin_discontinued": 0.12,
    "metformin_restart": 0.08,
    "sglt2_add_on": 0.14,
    "switch_to_sglt2": 0.10,
    "combination_product": 0.05,
    "early_refills": 0.07,
    "sglt2_first": 0.05,
    "sparse_incomplete": 0.07,
}
ANOMALY_RATES = {  # per paid fill
    "rejected_then_paid": 0.03,
    "reversed": 0.025,
    "corrected_rebill": 0.015,
    "exact_duplicate_row": 0.01,
    "conflicting_duplicate": 0.003,
    "missing_days_supply": 0.007,
    "implausible_days_supply": 0.004,
    "negative_quantity": 0.003,
    "unknown_txn_type": 0.003,
}
PATIENT_RATES = {  # per patient
    "activity_gap_over_365_days": 0.10,
    "unmapped_ndc_claim": 0.02,
    "malformed_ndc_claim": 0.005,
    "non_diabetes_drug": 0.30,
    "orphan_reversal": 0.003,
    "reversal_before_original": 0.002,
    "reversal_of_other_patients_claim": 0.002,
    "reversal_payer_mismatch": 0.003,
    "future_dated_claim": 0.002,
}
NULL_PATIENT_ROWS_PER_10K = 5
METFORMIN = [NDC["metformin_500"], NDC["metformin_er"]]
SGLT2 = [NDC["empagliflozin"], NDC["dapagliflozin"], NDC["canagliflozin"]]
OTHER_DRUGS = [NDC["lisinopril"], NDC["sertraline"]]
T2D_DX = ["E119", "E1122"]
FILLER_DX = ["Z0000", "R05", "M545", "J069", "K219"]
MALFORMED_NDC = ["12345-678", "ABCDEFGHIJK", "999990101"]
RX_COLS = [
    ("claim_id", "BIGINT"),
    ("patient_id", "BIGINT"),
    ("svc_dt", "DATE"),
    ("ndc", "VARCHAR"),
    ("days_supply", "INTEGER"),
    ("quantity", "DOUBLE"),
    ("txn_type", "VARCHAR"),
    ("reject_code", "VARCHAR"),
    ("orig_claim_id", "BIGINT"),
    ("payer_type", "VARCHAR"),
]
DX_COLS = [
    ("claim_id", "BIGINT"),
    ("patient_id", "BIGINT"),
    ("svc_dt", "DATE"),
    ("place_of_service", "VARCHAR"),
    ("dx1", "VARCHAR"),
    ("dx2", "VARCHAR"),
    ("dx3", "VARCHAR"),
    ("dx4", "VARCHAR"),
]
PATIENT_COLS = [("patient_id", "BIGINT"), ("birth_year", "INTEGER"), ("gender", "VARCHAR")]


class _Patient:
    """Claims of one synthetic patient; anomalies are drawn from the shared generator RNG."""

    def __init__(self, gen: _Generator, pid: int):
        self.g, self.pid, rng = gen, pid, gen.rng
        self.payer = rng.choices(["COMMERCIAL", "MEDICARE", "MEDICAID"], [0.55, 0.30, 0.15])[0]
        self.start = WINDOW[0] + timedelta(days=rng.randint(0, 1250))
        self.stop = min(WINDOW[1], self.start + timedelta(days=rng.randint(900, 2600)))
        self.gap: tuple[date, date] | None = None
        if rng.random() < PATIENT_RATES["activity_gap_over_365_days"] and (self.stop - self.start).days > 1300:
            g0 = self.start + timedelta(days=rng.randint(300, 500))
            self.gap = (g0, g0 + timedelta(days=rng.randint(400, 650)))
            gen.injected["activity_gap_over_365_days"] += 1

    def visible(self, d: date) -> bool:
        return self.start <= d <= self.stop and not (self.gap and self.gap[0] <= d <= self.gap[1])

    def dx(self, d: date, code: str) -> None:
        if self.visible(d):
            g = self.g
            g.ids["dx"] += 1
            extra = g.rng.sample(FILLER_DX, k=g.rng.randint(0, 2))
            codes: list[str | None] = [code, *extra, None, None, None][:4]
            g.dx_rows.append([g.ids["dx"], self.pid, d, g.rng.choice(["11", "22"]), *codes])

    def _row(self, d: date, ndc: str, supply: int | None, qty: float | None, txn: str, **kw: Any) -> int:
        g = self.g
        g.ids["rx"] += 1
        g.rx_rows.append(
            [
                g.ids["rx"],
                self.pid,
                d,
                ndc,
                supply,
                qty,
                txn,
                kw.get("reject"),
                kw.get("orig"),
                kw.get("payer", self.payer),
            ]
        )
        return g.ids["rx"]

    def fill(self, d: date, ndc: str, supply: int) -> bool:
        """One dispensing with injected artefacts; False when the date is not visible (no claim)."""
        if not self.visible(d):
            return False
        g, rng, hit = self.g, self.g.rng, self.g.hit
        qty = float(supply * (2 if ndc == NDC["metformin_500"] else 1))
        if hit("rejected_then_paid"):
            self._row(d, ndc, supply, qty, "REJECTED", reject="79")
        days: int | None = supply
        if hit("missing_days_supply"):
            days = None
        elif hit("implausible_days_supply"):
            days = rng.choice([0, 999])
        if hit("negative_quantity"):
            qty = -qty
        txn = "PENDING" if hit("unknown_txn_type") else "PAID"
        cid = self._row(d, ndc, days, qty, txn)
        if hit("exact_duplicate_row"):
            g.rx_rows.append(list(g.rx_rows[-1]))
        elif hit("conflicting_duplicate"):
            dup = list(g.rx_rows[-1])
            dup[4] = (days or supply) + 30
            g.rx_rows.append(dup)
        if txn != "PAID":
            return True
        if hit("corrected_rebill"):
            self._row(d, ndc, supply, qty, "REVERSAL", orig=cid)
            fixed = 90 if supply == 30 else 30
            self._row(d, ndc, fixed, float(fixed), "PAID")
        elif hit("reversed"):
            self._row(d + timedelta(days=rng.randint(0, 5)), ndc, supply, qty, "REVERSAL", orig=cid)
        return True

    def therapy(self, first: date, ndc: str, n: int, supply: int = 30, early: bool = False) -> date:
        """n fills from `first`; returns the day after the last fill's supply."""
        d, rng, last = first, self.g.rng, first
        for _ in range(n):
            if d > self.stop:
                break
            if self.fill(d, ndc, supply):
                last = d + timedelta(days=supply)
            d += timedelta(days=supply + (rng.randint(-10, -4) if early else rng.randint(-3, 10)))
        return last

    def fills_until_stop(self, first: date, supply: int = 30) -> int:
        return max(1, (self.stop - first).days // (supply + 3) + 1)


class _Generator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.ids = {"rx": 0, "dx": 0}
        self.rx_rows: list[list[Any]] = []
        self.dx_rows: list[list[Any]] = []
        self.patients: list[list[Any]] = []
        self.injected: Counter[str] = Counter()
        self.archetypes: Counter[str] = Counter()

    def hit(self, name: str) -> bool:
        if self.rng.random() < ANOMALY_RATES[name]:
            self.injected[name] += 1
            return True
        return False

    def patient_hit(self, name: str) -> bool:
        if self.rng.random() < PATIENT_RATES[name]:
            self.injected[name] += 1
            return True
        return False


def _patient(g: _Generator, pid: int) -> None:
    rng = g.rng
    g.patients.append([pid, rng.randint(1940, 1990), rng.choice(["M", "F"])])
    p = _Patient(g, pid)
    for _ in range(rng.randint(2, 6)):  # background activity
        p.dx(p.start + timedelta(days=rng.randint(0, (p.stop - p.start).days)), rng.choice(FILLER_DX))
    dx0 = p.start + timedelta(days=rng.randint(0, 180))
    code = rng.choices(T2D_DX, [0.8, 0.2])[0]
    d = dx0
    while d <= p.stop:  # diagnosis follow-up visits keep the patient observable
        p.dx(d, code)
        d += timedelta(days=rng.randint(60, 200))
    arche = rng.choices(list(ARCHETYPES), list(ARCHETYPES.values()))[0]
    g.archetypes[arche] += 1
    t0 = dx0 + timedelta(days=rng.randint(0, 60))
    met = rng.choice(METFORMIN)
    sglt2 = rng.choice(SGLT2)
    if arche == "metformin_persistent":
        p.therapy(t0, met, p.fills_until_stop(t0))
    elif arche == "metformin_discontinued":
        p.therapy(t0, met, rng.randint(2, 10))
    elif arche == "metformin_restart":
        end = p.therapy(t0, met, rng.randint(2, 8))
        p.therapy(end + timedelta(days=rng.randint(90, 300)), met, rng.randint(2, 10))
    elif arche == "sglt2_add_on":
        p.therapy(t0, met, p.fills_until_stop(t0))
        s0 = t0 + timedelta(days=rng.randint(90, 360))
        p.therapy(s0, sglt2, p.fills_until_stop(s0))
    elif arche == "switch_to_sglt2":
        end = p.therapy(t0, met, rng.randint(3, 10))
        s0 = end + timedelta(days=rng.randint(0, 60))
        p.therapy(s0, sglt2, rng.randint(3, 12))
    elif arche == "combination_product":
        p.therapy(t0, NDC["empa_met"], rng.randint(3, 15))
    elif arche == "early_refills":
        p.therapy(t0, met, rng.randint(4, 14), early=True)
    elif arche == "sglt2_first":
        p.therapy(t0, sglt2, rng.randint(2, 12))
    elif arche == "sparse_incomplete":
        for _ in range(rng.randint(1, 2)):
            day = t0 + timedelta(days=rng.randint(0, 400))
            if p.visible(day):
                p._row(day, met, rng.choice([None, 30]), None, "PAID")
    # patient-level artefacts
    any_day = p.start + timedelta(days=rng.randint(0, (p.stop - p.start).days))
    if p.visible(any_day):
        if g.patient_hit("non_diabetes_drug"):
            p.therapy(any_day, rng.choice(OTHER_DRUGS), rng.randint(1, 6), 90)
        if g.patient_hit("unmapped_ndc_claim"):
            p._row(any_day, NDC["unmapped"], 30, 30.0, "PAID")
        if g.patient_hit("malformed_ndc_claim"):
            p._row(any_day, rng.choice(MALFORMED_NDC), 30, 30.0, "PAID")
        if g.patient_hit("orphan_reversal"):
            p._row(any_day, met, 30, 30.0, "REVERSAL", orig=10**12 + pid)
        if g.patient_hit("reversal_before_original"):
            cid = p._row(any_day, met, 30, 30.0, "PAID")
            p._row(any_day - timedelta(days=3), met, 30, 30.0, "REVERSAL", orig=cid)
        if g.patient_hit("reversal_payer_mismatch"):
            cid = p._row(any_day, met, 30, 30.0, "PAID")
            other = "MEDICAID" if p.payer != "MEDICAID" else "COMMERCIAL"
            p._row(any_day + timedelta(days=1), met, 30, 30.0, "REVERSAL", orig=cid, payer=other)
        if pid > 1 and g.patient_hit("reversal_of_other_patients_claim"):
            target = next((r for r in reversed(g.rx_rows) if r[1] == pid - 1 and r[6] == "PAID"), None)
            if target is not None:
                p._row(any_day, target[3], target[4], target[5], "REVERSAL", orig=target[0])
            else:
                g.injected["reversal_of_other_patients_claim"] -= 1
    if g.patient_hit("future_dated_claim"):
        g.ids["rx"] += 1
        g.rx_rows.append([g.ids["rx"], pid, FUTURE, met, 30, 30.0, "PAID", None, None, p.payer])


def generate_laad_t2d(con: duckdb.DuckDBPyConnection, n_patients: int = 10_000, seed: int = DEFAULT_SEED) -> dict:
    """Create (replace) the laad.* tables with n synthetic T2D patients; returns exact counts."""
    if not 1 <= n_patients <= 5_000_000:
        raise ValueError("n_patients must be 1..5000000")
    g = _Generator(seed)
    for pid in range(1, n_patients + 1):
        _patient(g, pid)
    for k in range(max(1, n_patients * NULL_PATIENT_ROWS_PER_10K // 10_000)):  # unattributable rows
        g.ids["rx"] += 1
        g.rx_rows.append(
            [g.ids["rx"], None, date(2022, 1, 3 + k % 25), METFORMIN[0], 30, 30.0, "PAID", None, None, None]
        )
        g.injected["missing_patient_id"] += 1
    con.execute(LAAD_DDL)
    con.execute("CREATE OR REPLACE TABLE laad.synthetic_manifest (key VARCHAR PRIMARY KEY, value VARCHAR)")
    batch = 50_000
    bulk_insert(con, "laad.patient", PATIENT_COLS, [tuple(r) for r in g.patients], batch)
    bulk_insert(con, "laad.rx_claims", RX_COLS, [tuple(r) for r in g.rx_rows], batch)
    bulk_insert(con, "laad.dx_claims", DX_COLS, [tuple(r) for r in g.dx_rows], batch)
    tables = {
        t: int((con.execute(f"SELECT count(*) FROM laad.{t}").fetchone() or (0,))[0])
        for t in ("patient", "rx_claims", "dx_claims", "px_claims")
    }
    by_txn = dict(con.execute("SELECT txn_type, count(*) FROM laad.rx_claims GROUP BY 1 ORDER BY 1").fetchall())
    stats = {
        "synthetic": True,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "patients": n_patients,
        "tables": tables,
        "rx_claims_by_txn_type": by_txn,
        "archetypes": dict(sorted(g.archetypes.items())),
        "injected": dict(sorted(g.injected.items())),
        "note": "SYNTHETIC DATA - fictitious patients, synthetic NDCs (labeler 99999); not real claims",
    }
    con.executemany(
        "INSERT INTO laad.synthetic_manifest VALUES (?, ?)",
        [[k, json.dumps(v, sort_keys=True) if not isinstance(v, str) else v] for k, v in stats.items()],
    )
    return stats
