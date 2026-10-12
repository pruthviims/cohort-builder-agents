"""Regenerates the SYNTHETIC example CSVs in this folder (deterministic; no real patients).

    python examples/csv/make_example.py

Codes are the demo vocabulary's (ICD-10-CM / synthetic NDC / LOINC), so the import maps to standard
concepts after `cohort-builder init-demo`.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent


def main(n: int = 60, seed: int = 11) -> None:
    rng = random.Random(seed)
    patients, enrollment, dx, rx, labs = [], [], [], [], []
    for i in range(1, n + 1):
        pid = f"P{i:04d}"
        patients.append([pid, rng.choice(["M", "F"]), rng.randint(1945, 1995)])
        start = date(2016, 1, 1) + timedelta(days=rng.randint(0, 365))
        enrollment.append([pid, start.isoformat(), date(2025, 12, 31).isoformat()])
        if i % 3:  # two thirds have type 2 diabetes
            d = start + timedelta(days=rng.randint(400, 1500))
            dx.append([f"D{len(dx) + 1:05d}", pid, d.isoformat(), rng.choice(["E11.9", "E11.22"])])
            lab_day = d + timedelta(days=rng.randint(10, 200))
            labs.append([f"L{len(labs) + 1:05d}", pid, lab_day.isoformat(), "4548-4", round(rng.uniform(6.0, 10.5), 1),
                         "%"])
            if i % 2:
                fill = lab_day + timedelta(days=rng.randint(5, 120))
                rx.append([f"R{len(rx) + 1:05d}", pid, fill.strftime("%m/%d/%Y"), "99999010101", 30])
        if i % 10 == 0:
            dx.append([f"D{len(dx) + 1:05d}", pid, (start + timedelta(days=200)).isoformat(), "E10.9"])
        if i % 7 == 0:
            dx.append([f"D{len(dx) + 1:05d}", pid, (start + timedelta(days=90)).isoformat(), "I10"])
    files = {
        "patients.csv": (["patient_id", "sex", "birth_year"], patients),
        "enrollment.csv": (["patient_id", "start_date", "end_date"], enrollment),
        "diagnoses.csv": (["dx_id", "patient_id", "dx_date", "icd10_code"], dx),
        "prescriptions.csv": (["rx_id", "patient_id", "fill_date", "ndc", "days_supply"], rx),
        "labs.csv": (["lab_id", "patient_id", "lab_date", "loinc_code", "result_value", "result_unit"], labs),
    }
    for name, (header, rows) in files.items():
        with (HERE / name).open("w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)


if __name__ == "__main__":
    main()
