"""Synthetic IQVIA LAAD-style claims (open US claims).

The `laad` schema mimics the *shape* of LAAD-like deliveries: pharmacy
transactions with adjudication outcomes (paid / rejected / reversal), medical
claims with wide diagnosis columns (ICD-10-CM without dots), and procedure
claims. Table and column names are placeholders, not IQVIA's real names;
map your delivery in ontology/datasets/iqvia_laad.yaml.

All data is synthetic and deterministic for a given seed.
"""

from __future__ import annotations

import csv
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb

LAAD_DDL = """
CREATE SCHEMA IF NOT EXISTS laad;
CREATE OR REPLACE TABLE laad.patient (patient_id BIGINT PRIMARY KEY, birth_year INTEGER, gender VARCHAR);
CREATE OR REPLACE TABLE laad.rx_claims (
  claim_id BIGINT, patient_id BIGINT, svc_dt DATE, ndc VARCHAR, days_supply INTEGER, quantity DOUBLE,
  txn_type VARCHAR,          -- PAID | REJECTED | REVERSAL (placeholder for the vendor's transaction/status code)
  reject_code VARCHAR,       -- NCPDP-style reject code on rejected claims (e.g. 75 = prior authorization required)
  orig_claim_id BIGINT,      -- for REVERSAL rows: the paid claim being reversed
  payer_type VARCHAR);
CREATE OR REPLACE TABLE laad.dx_claims (
  claim_id BIGINT, patient_id BIGINT, svc_dt DATE, place_of_service VARCHAR,
  dx1 VARCHAR, dx2 VARCHAR, dx3 VARCHAR, dx4 VARCHAR);
CREATE OR REPLACE TABLE laad.px_claims (
  claim_id BIGINT, patient_id BIGINT, svc_dt DATE, px_code VARCHAR, px_code_type VARCHAR);
"""

NDC = {
    "metformin_500": "99999010101",
    "metformin_er": "99999010201",
    "empa_met": "99999010301",
    "empagliflozin": "99999020101",
    "dapagliflozin": "99999020201",
    "canagliflozin": "99999020301",
    "lisinopril": "99999030101",
    "sertraline": "99999040101",
    "atorvastatin": "99999050101",
    "unmapped": "00000000000",
}
FILLER_DX = ["Z0000", "R05", "M545", "J069", "K219"]  # common non-study codes (unmapped in the demo vocabulary)


def _d(rng: random.Random, lo: date, hi: date) -> date:
    return lo if hi <= lo else lo + timedelta(days=rng.randint(0, (hi - lo).days))


def generate_laad(con: duckdb.DuckDBPyConnection, n_patients: int = 4000, seed: int = 11) -> dict:
    rng = random.Random(seed)
    con.execute(LAAD_DDL)
    end = date(2026, 6, 30)
    rows: dict[str, list[list]] = {"patient": [], "rx_claims": [], "dx_claims": [], "px_claims": []}
    ids = {"rx": 0, "dx": 0, "px": 0}

    for pid in range(1, n_patients + 1):
        yob = rng.randint(1940, 2006)
        rows["patient"].append([pid, yob, rng.choice(["M", "F"])])
        payer = rng.choices(["COMMERCIAL", "MEDICARE", "MEDICAID"], [0.55, 0.3, 0.15])[0]
        start = _d(rng, date(2017, 1, 1), date(2023, 6, 30))
        stop = min(end, start + timedelta(days=rng.randint(400, 3000)))
        # ~10% of patients disappear from the data for > 1 year (open-data coverage gap)
        gap = None
        if rng.random() < 0.10 and (stop - start).days > 1200:
            g0 = start + timedelta(days=rng.randint(300, 600))
            gap = (g0, g0 + timedelta(days=rng.randint(420, 600)))

        def visible(d: date) -> bool:
            return start <= d <= stop and not (gap and gap[0] <= d <= gap[1])

        def dx(d: date, codes: list[str], primary: bool = True) -> None:
            if not visible(d):
                return
            codes = list(codes)
            fillers = rng.sample(FILLER_DX, k=rng.randint(0, 2))
            ordered: list[str | None] = list(codes + fillers if primary else fillers[:1] + codes + fillers[1:])
            if not primary and not fillers:
                ordered = ["Z0000", *codes]
            ordered = (ordered + [None] * 4)[:4]
            ids["dx"] += 1
            rows["dx_claims"].append([ids["dx"], pid, d, rng.choice(["11", "22", "21"]), *ordered])

        def px(d: date, code: str) -> None:
            if visible(d):
                ids["px"] += 1
                rows["px_claims"].append([ids["px"], pid, d, code, "CPT"])

        def rx(d: date, ndc: str, status: str = "PAID", reject: str | None = None, supply: int = 30) -> int | None:
            if not visible(d):
                return None
            ids["rx"] += 1
            rows["rx_claims"].append([ids["rx"], pid, d, ndc, supply, float(supply), status, reject, None, payer])
            if status == "PAID" and rng.random() < 0.03:  # reversed a few days later
                rid = ids["rx"]
                ids["rx"] += 1
                rows["rx_claims"].append(
                    [
                        ids["rx"],
                        pid,
                        d + timedelta(days=rng.randint(0, 5)),
                        ndc,
                        supply,
                        float(supply),
                        "REVERSAL",
                        None,
                        rid,
                        payer,
                    ]
                )
            return ids["rx"]

        def therapy(first: date, ndc: str, fills: int, supply: int = 30) -> None:
            d = first
            for _ in range(fills):
                if d > stop:
                    break
                if rng.random() < 0.04:  # occasional rejection (e.g. refill too soon), then paid
                    rx(d, ndc, "REJECTED", "79", supply)
                    d += timedelta(days=rng.randint(1, 4))
                rx(d, ndc, "PAID", None, supply)
                d += timedelta(days=supply + rng.randint(-3, 12))

        # background activity so that observation periods are realistic
        for _ in range(rng.randint(1, 5)):
            dx(_d(rng, start, stop), [rng.choice(FILLER_DX)])
        age = 2022 - yob

        # ---- type 2 diabetes ------------------------------------------------------
        has_t2d = rng.random() < 0.08 + 0.004 * max(0, age - 30)
        if has_t2d:
            dx0 = _d(rng, start, stop - timedelta(days=90))
            code = rng.choices(["E119", "E1122"], [0.8, 0.2])[0]
            for k in range(rng.randint(1, 6)):
                dx(dx0 + timedelta(days=k * rng.randint(20, 150)), [code], primary=rng.random() < 0.65)
            if rng.random() < 0.8:
                therapy(
                    dx0 + timedelta(days=rng.randint(0, 120)),
                    rng.choices([NDC["metformin_500"], NDC["metformin_er"], NDC["empa_met"]], [0.6, 0.3, 0.1])[0],
                    rng.randint(1, 14),
                )
            if rng.random() < 0.35:  # SGLT2 access journey: often PA-rejected first
                t = _d(rng, dx0, stop)
                drug = rng.choice([NDC["empagliflozin"], NDC["dapagliflozin"], NDC["canagliflozin"]])
                if rng.random() < 0.4:
                    rx(t, drug, "REJECTED", rng.choice(["75", "70", "76"]))
                    if rng.random() < 0.7:
                        therapy(t + timedelta(days=rng.randint(3, 60)), drug, rng.randint(1, 8))
                else:
                    therapy(t, drug, rng.randint(1, 8))
        elif rng.random() < 0.02:
            dx(_d(rng, start, stop), ["E109"])

        # ---- heart failure --------------------------------------------------------
        if rng.random() < (0.10 if has_t2d else 0.04):
            d0 = _d(rng, start, stop)
            dx(d0, [rng.choice(["I509", "I5022"])], primary=rng.random() < 0.7)
            px(d0 + timedelta(days=rng.randint(0, 20)), "93306")
            if rng.random() < 0.3:
                therapy(d0 + timedelta(days=rng.randint(0, 120)), NDC["dapagliflozin"], rng.randint(1, 6))

        # ---- hypertension, depression, CABG ---------------------------------------
        if rng.random() < 0.15 + 0.003 * max(0, age - 30):
            d0 = _d(rng, start, stop)
            dx(d0, ["I10"], primary=rng.random() < 0.5)
            if rng.random() < 0.6:
                therapy(d0 + timedelta(days=rng.randint(0, 45)), NDC["lisinopril"], rng.randint(1, 12), 90)
        if rng.random() < 0.10:
            d0 = _d(rng, start, stop)
            dx(d0, [rng.choice(["F329", "F331"])], primary=rng.random() < 0.6)
            if rng.random() < 0.55:
                therapy(d0 + timedelta(days=rng.randint(0, 40)), NDC["sertraline"], rng.randint(1, 8))
        if rng.random() < 0.01:
            px(_d(rng, start, stop), "33533")
        if rng.random() < 0.03:
            rx(_d(rng, start, stop), NDC["unmapped"])  # NDC missing from the vocabulary

    with tempfile.TemporaryDirectory() as td:
        for table, data in rows.items():
            path = Path(td) / f"{table}.csv"
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerows([["" if v is None else v for v in r] for r in data])
            con.execute(
                f"INSERT INTO laad.{table} SELECT * FROM read_csv('{path}', header=false, nullstr='', "
                f"auto_detect=true, all_varchar=true)"
            )
    return {f"laad.{t}": con.execute(f"SELECT count(*) FROM laad.{t}").fetchone()[0] for t in rows}
