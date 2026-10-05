"""Demo vocabulary + seeded synthetic OMOP patients.

The vocabulary is a small hand-built subset so the project runs offline.
Top-level concepts use their usual OMOP concept IDs; product-level, demo and
local concepts use IDs >= 2,000,000,000 (OMOP's reserved local range).
For real work, load the full Athena download with `cohort-builder load-athena`.

Patient data is fully synthetic and deterministic for a given seed.
"""
from __future__ import annotations

import csv
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb

from .db import init_schemas

DEMO_VOCAB_VERSION = "DEMO-2026-10-01"

# (concept_id, name, domain, vocabulary, class, standard, code, parents, synonyms)
CONCEPTS: list[tuple] = [
    # --- Conditions (SNOMED) ---
    (201820, "Diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "73211009", [], ["DM", "diabetes"]),
    (201826, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "44054006", [201820],
     ["T2DM", "type II diabetes", "non-insulin dependent diabetes mellitus", "adult-onset diabetes"]),
    (2000003001, "Type 2 diabetes mellitus with diabetic nephropathy", "Condition", "SNOMED", "Clinical Finding", "S",
     "D-T2DN", [201826], []),
    (2000003002, "Type 2 diabetes mellitus without complication", "Condition", "SNOMED", "Clinical Finding", "S",
     "D-T2NC", [201826], []),
    (201254, "Type 1 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "46635009", [201820],
     ["T1DM", "type I diabetes", "insulin dependent diabetes mellitus", "juvenile diabetes"]),
    (316139, "Heart failure", "Condition", "SNOMED", "Clinical Finding", "S", "84114007", [],
     ["cardiac failure", "HF"]),
    (319835, "Congestive heart failure", "Condition", "SNOMED", "Clinical Finding", "S", "42343007", [316139], ["CHF"]),
    (2000003003, "Heart failure with reduced ejection fraction", "Condition", "SNOMED", "Clinical Finding", "S",
     "D-HFREF", [316139], ["HFrEF", "systolic heart failure"]),
    (2000003004, "Heart failure with preserved ejection fraction", "Condition", "SNOMED", "Clinical Finding", "S",
     "D-HFPEF", [316139], ["HFpEF", "diastolic heart failure"]),
    (46271022, "Chronic kidney disease", "Condition", "SNOMED", "Clinical Finding", "S", "709044004", [], ["CKD"]),
    (320128, "Essential hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "59621000", [],
     ["hypertension", "high blood pressure", "HTN"]),
    (4329847, "Myocardial infarction", "Condition", "SNOMED", "Clinical Finding", "S", "22298006", [],
     ["heart attack", "MI"]),
    (255573, "Chronic obstructive lung disease", "Condition", "SNOMED", "Clinical Finding", "S", "13645005", [],
     ["COPD", "chronic obstructive pulmonary disease"]),
    (440383, "Depressive disorder", "Condition", "SNOMED", "Clinical Finding", "S", "35489007", [], ["depression"]),
    (2000003005, "Major depressive disorder", "Condition", "SNOMED", "Clinical Finding", "S", "D-MDD", [440383],
     ["MDD", "major depression"]),
    # deprecated concept, kept to exercise the validator
    (2000003099, "Diabetes mellitus type 2 (old code)", "Condition", "SNOMED", "Clinical Finding", None, "D-OLD",
     [], []),
    # --- ICD-10-CM source concepts (non-standard, mapped) ---
    (2000004001, "Type 2 diabetes mellitus without complications", "Condition", "ICD10CM", "5-char billing code", None,
     "E11.9", [], []),
    (2000004002, "Type 1 diabetes mellitus", "Condition", "ICD10CM", "3-char nonbill code", None, "E10", [], []),
    (2000004003, "Heart failure, unspecified", "Condition", "ICD10CM", "4-char billing code", None, "I50.9", [], []),
    (2000004004, "Essential (primary) hypertension", "Condition", "ICD10CM", "3-char billing code", None, "I10", [], []),
    (2000004005, "Chronic kidney disease, stage 3", "Condition", "ICD10CM", "4-char nonbill code", None, "N18.3", [], []),
    (2000004006, "Major depressive disorder, single episode, unspecified", "Condition", "ICD10CM", "4-char billing code",
     None, "F32.9", [], []),
    # --- Drug classes (ATC, classification) ---
    (2000006001, "Biguanides", "Drug", "ATC", "ATC 4th", "C", "A10BA", [], []),
    (2000006002, "Sodium-glucose co-transporter 2 (SGLT2) inhibitors", "Drug", "ATC", "ATC 4th", "C", "A10BK", [],
     ["SGLT2 inhibitors", "gliflozins"]),
    # --- Drug ingredients (RxNorm) ---
    (1503297, "metformin", "Drug", "RxNorm", "Ingredient", "S", "6809", [2000006001], ["metformin hydrochloride"]),
    (45775965, "empagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1545653", [2000006002], ["Jardiance"]),
    (44785829, "dapagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1488564", [2000006002], ["Farxiga"]),
    (43526465, "canagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1373463", [2000006002], ["Invokana"]),
    (1308216, "lisinopril", "Drug", "RxNorm", "Ingredient", "S", "29046", [], ["ACE inhibitor lisinopril"]),
    (739138, "sertraline", "Drug", "RxNorm", "Ingredient", "S", "36437", [], ["Zoloft"]),
    (1545958, "atorvastatin", "Drug", "RxNorm", "Ingredient", "S", "83367", [], ["Lipitor"]),
    # --- Drug products (demo IDs) ---
    (2000005001, "metformin hydrochloride 500 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-MET500",
     [1503297], []),
    (2000005002, "24 HR metformin hydrochloride 1000 MG Extended Release Oral Tablet", "Drug", "RxNorm",
     "Clinical Drug", "S", "P-MET1000ER", [1503297], []),
    (2000005003, "empagliflozin 12.5 MG / metformin hydrochloride 1000 MG Oral Tablet", "Drug", "RxNorm",
     "Clinical Drug", "S", "P-EMPAMET", [1503297, 45775965], []),
    (2000005004, "empagliflozin 10 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-EMPA10", [45775965], []),
    (2000005005, "dapagliflozin 10 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-DAPA10", [44785829], []),
    (2000005006, "canagliflozin 100 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-CANA100", [43526465], []),
    (2000005007, "lisinopril 10 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-LIS10", [1308216], []),
    (2000005008, "sertraline 50 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-SER50", [739138], []),
    (2000005009, "atorvastatin 20 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "P-ATO20", [1545958], []),
    # --- Measurements (LOINC) ---
    (3004410, "Hemoglobin A1c/Hemoglobin.total in Blood", "Measurement", "LOINC", "Lab Test", "S", "4548-4", [],
     ["HbA1c", "A1c", "glycated hemoglobin", "hemoglobin A1c"]),
    (3016723, "Creatinine [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2160-0", [],
     ["serum creatinine", "creatinine"]),
    (2000002001, "Glomerular filtration rate/1.73 sq M.predicted", "Measurement", "LOINC", "Lab Test", "S", "D-EGFR",
     [], ["eGFR", "estimated GFR", "estimated glomerular filtration rate"]),
    (2000002002, "Left ventricular Ejection fraction", "Measurement", "LOINC", "Clinical Observation", "S", "D-LVEF",
     [], ["LVEF", "ejection fraction", "EF"]),
    # --- Procedures ---
    (2000007001, "Echocardiography", "Procedure", "SNOMED", "Procedure", "S", "D-ECHO", [], ["echo", "cardiac ultrasound"]),
    (2000007002, "Coronary artery bypass graft", "Procedure", "SNOMED", "Procedure", "S", "D-CABG", [], ["CABG"]),
    # --- Visits ---
    (9201, "Inpatient Visit", "Visit", "Visit", "Visit", "S", "IP", [], ["hospitalization", "inpatient admission"]),
    (9202, "Outpatient Visit", "Visit", "Visit", "Visit", "S", "OP", [], ["office visit"]),
    (9203, "Emergency Room Visit", "Visit", "Visit", "Visit", "S", "ER", [], ["ED visit", "emergency visit"]),
    # --- Gender & units ---
    (8507, "MALE", "Gender", "Gender", "Gender", "S", "M", [], []),
    (8532, "FEMALE", "Gender", "Gender", "Gender", "S", "F", [], []),
    (8554, "percent", "Unit", "UCUM", "Unit", "S", "%", [], []),
    (8840, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL", [], []),
    (8753, "millimole per liter", "Unit", "UCUM", "Unit", "S", "mmol/L", [], []),
    (2000001001, "millimole per mole", "Unit", "UCUM", "Unit", "S", "mmol/mol", [], []),
    (2000001002, "milliliter per minute per 1.73 square meter", "Unit", "UCUM", "Unit", "S", "mL/min/{1.73_m2}", [], []),
    (2000001003, "micromole per liter", "Unit", "UCUM", "Unit", "S", "umol/L", [], []),
]

MAPS_TO = {
    2000003099: 201826,
    2000004001: 201826,
    2000004002: 201254,
    2000004003: 316139,
    2000004004: 320128,
    2000004005: 46271022,
    2000004006: 2000003005,
}
DEPRECATED = {2000003099}

EHR = 32817  # type concept: EHR


def _ancestor_rows() -> list[tuple[int, int, int, int]]:
    parents = {c[0]: c[7] for c in CONCEPTS}
    rows = []
    for cid in parents:
        # BFS upwards, tracking min/max levels
        levels: dict[int, list[int]] = {cid: [0]}
        frontier = [(cid, 0)]
        while frontier:
            node, lvl = frontier.pop()
            for p in parents.get(node, []):
                levels.setdefault(p, []).append(lvl + 1)
                frontier.append((p, lvl + 1))
        for anc, lv in levels.items():
            standard = next(c[5] for c in CONCEPTS if c[0] == anc)
            if standard in ("S", "C"):
                rows.append((anc, cid, min(lv), max(lv)))
    return sorted(set(rows))


def load_demo_vocabulary(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM vocab.concept; DELETE FROM vocab.concept_relationship; "
                "DELETE FROM vocab.concept_ancestor; DELETE FROM vocab.concept_synonym; DELETE FROM vocab.vocabulary;")
    start, end = date(1970, 1, 1), date(2099, 12, 31)
    for c in CONCEPTS:
        cid, name, dom, voc, cls, std, code = c[:7]
        invalid = "U" if cid in DEPRECATED else None
        vend = date(2020, 1, 1) if invalid else end
        con.execute("INSERT INTO vocab.concept VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [cid, name, dom, voc, cls, std, code, start, vend, invalid])
        for syn in c[8]:
            con.execute("INSERT INTO vocab.concept_synonym VALUES (?,?,4180186)", [cid, syn])
        for p in c[7]:
            con.execute("INSERT INTO vocab.concept_relationship VALUES (?,?,'Is a',?,?,NULL)", [cid, p, start, end])
            con.execute("INSERT INTO vocab.concept_relationship VALUES (?,?,'Subsumes',?,?,NULL)", [p, cid, start, end])
    for src, tgt in MAPS_TO.items():
        con.execute("INSERT INTO vocab.concept_relationship VALUES (?,?,'Maps to',?,?,NULL)", [src, tgt, start, end])
        con.execute("INSERT INTO vocab.concept_relationship VALUES (?,?,'Mapped from',?,?,NULL)", [tgt, src, start, end])
    for c in CONCEPTS:  # standard concepts map to themselves
        if c[5] == "S":
            con.execute("INSERT INTO vocab.concept_relationship VALUES (?,?,'Maps to',?,?,NULL)", [c[0], c[0], start, end])
    con.executemany("INSERT INTO vocab.concept_ancestor VALUES (?,?,?,?)", _ancestor_rows())
    for vid in sorted({c[3] for c in CONCEPTS}):
        con.execute("INSERT INTO vocab.vocabulary VALUES (?,?,?,?,NULL)",
                    [vid, vid, "demo subset", DEMO_VOCAB_VERSION])


# ---------------------------------------------------------------------------
# Synthetic patients
# ---------------------------------------------------------------------------
def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    if hi <= lo:
        return lo
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


class _Writer:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.files: dict[str, tuple[Path, csv.writer, object]] = {}
        self.ids: dict[str, int] = {}

    def row(self, table: str, values: list) -> int:
        if table not in self.files:
            p = self.tmp / f"{table}.csv"
            fh = open(p, "w", newline="")
            self.files[table] = (p, csv.writer(fh), fh)
        self.ids[table] = self.ids.get(table, 0) + 1
        self.files[table][1].writerow(["" if v is None else v for v in values])
        return self.ids[table]

    def next_id(self, table: str) -> int:
        return self.ids.get(table, 0) + 1

    def flush(self, con: duckdb.DuckDBPyConnection) -> None:
        for table, (p, _, fh) in self.files.items():
            fh.close()
            con.execute(f"INSERT INTO cdm.{table} SELECT * FROM read_csv('{p}', header=false, all_varchar=false, "
                        f"nullstr='', auto_detect=true)")


def generate_patients(con: duckdb.DuckDBPyConnection, n_persons: int = 5000, seed: int = 42) -> None:
    rng = random.Random(seed)
    for t in ("person", "observation_period", "visit_occurrence", "condition_occurrence", "drug_exposure",
              "measurement", "procedure_occurrence", "cdm_source"):
        con.execute(f"DELETE FROM cdm.{t}")
    data_end = date(2026, 6, 30)
    with tempfile.TemporaryDirectory() as td:
        w = _Writer(Path(td))

        for pid in range(1, n_persons + 1):
            gender = rng.choice([8507, 8532])
            yob = rng.randint(1935, 2005)
            w.row("person", [pid, gender, yob, rng.randint(1, 12), rng.randint(1, 28), 0, 0, f"SRC{pid:06d}"])
            op_start = _rand_date(rng, date(2012, 1, 1), date(2020, 1, 1))
            op_end = min(data_end, op_start + timedelta(days=rng.randint(365, 365 * 14)))
            w.row("observation_period", [pid, pid, op_start, op_end, EHR])
            age = 2020 - yob

            def visit(d: date, kind: int = 9202, los: int = 0) -> int:
                return w.row("visit_occurrence", [w.next_id("visit_occurrence"), pid, kind, d,
                                                  d + timedelta(days=los), EHR])

            def condition(cid: int, d: date, src: str, src_cid: int = 0, kind: int = 9202) -> None:
                v = visit(d, kind, 3 if kind == 9201 else 0)
                w.row("condition_occurrence", [w.next_id("condition_occurrence"), pid, cid, d, None, EHR,
                                               src, src_cid, v])

            def drug(cid: int, d: date, n_fills: int, src: str) -> None:
                for i in range(n_fills):
                    s = d + timedelta(days=90 * i + rng.randint(0, 10))
                    if s > op_end:
                        break
                    w.row("drug_exposure", [w.next_id("drug_exposure"), pid, cid, s, s + timedelta(days=89), 90,
                                            EHR, src, None])

            def meas(cid: int, d: date, value: float, unit: int, src: str) -> None:
                if op_start <= d <= op_end:
                    w.row("measurement", [w.next_id("measurement"), pid, cid, d, round(value, 2), unit, EHR, src,
                                          None])

            def proc(cid: int, d: date, src: str) -> None:
                w.row("procedure_occurrence", [w.next_id("procedure_occurrence"), pid, cid, d, EHR, src, None])

            # routine visits
            for _ in range(rng.randint(1, 6)):
                visit(_rand_date(rng, op_start, op_end))

            # ---- Type 2 diabetes -------------------------------------------
            p_t2d = 0.08 + 0.004 * max(0, age - 30)
            has_t2d = rng.random() < p_t2d
            if has_t2d:
                dx = _rand_date(rng, op_start, op_end - timedelta(days=60))
                code = rng.choices([201826, 2000003002, 2000003001], [0.6, 0.3, 0.1])[0]
                condition(code, dx, "E11.9", 2000004001)
                condition(201826, _rand_date(rng, dx, op_end), "E11.9", 2000004001)  # repeat dx
                baseline = rng.gauss(7.9, 1.4)
                d = dx - timedelta(days=rng.randint(0, 300))
                while d <= op_end:
                    val = max(5.0, baseline + rng.gauss(0, 0.5))
                    if rng.random() < 0.2:
                        meas(3004410, d, (val - 2.152) / 0.09148, 2000001001, "HbA1c mmol/mol")
                    else:
                        meas(3004410, d, val, 8554, "HbA1c %")
                    d += timedelta(days=rng.randint(120, 240))
                if rng.random() < 0.78:
                    start = dx + timedelta(days=rng.randint(0, 150))
                    product = rng.choices([2000005001, 2000005002, 2000005003], [0.6, 0.3, 0.1])[0]
                    drug(product, start, rng.randint(1, 12), "metformin")
                if rng.random() < 0.25:
                    drug(rng.choice([2000005004, 2000005005, 2000005006]),
                         _rand_date(rng, dx, op_end), rng.randint(1, 8), "sglt2")
            else:
                for _ in range(rng.randint(0, 2)):
                    meas(3004410, _rand_date(rng, op_start, op_end), rng.uniform(4.8, 6.2), 8554, "HbA1c %")

            # ---- Type 1 diabetes -------------------------------------------
            if not has_t2d and rng.random() < 0.02:
                condition(201254, _rand_date(rng, op_start, op_end), "E10", 2000004002)

            # ---- Heart failure ---------------------------------------------
            if rng.random() < (0.12 if has_t2d else 0.05):
                dx = _rand_date(rng, op_start, op_end)
                hftype = rng.choices([316139, 319835, 2000003003, 2000003004], [0.3, 0.2, 0.3, 0.2])[0]
                condition(hftype, dx, "I50.9", 2000004003, kind=rng.choice([9201, 9202, 9203]))
                ef = rng.uniform(20, 39) if hftype == 2000003003 else (
                    rng.uniform(50, 65) if hftype == 2000003004 else rng.uniform(25, 65))
                ed = dx + timedelta(days=rng.randint(-20, 20))
                meas(2000002002, ed, ef, 8554, "LVEF")
                proc(2000007001, ed, "echo")
                if rng.random() < 0.35:
                    drug(rng.choice([2000005004, 2000005005]), dx + timedelta(days=rng.randint(0, 200)),
                         rng.randint(1, 6), "sglt2")

            # ---- CKD -------------------------------------------------------
            if rng.random() < (0.15 if has_t2d else 0.05):
                dx = _rand_date(rng, op_start, op_end)
                condition(46271022, dx, "N18.3", 2000004005)
                for k in range(rng.randint(1, 4)):
                    d = dx + timedelta(days=180 * k)
                    meas(2000002001, d, rng.uniform(15, 59), 2000001002, "eGFR")
                    if rng.random() < 0.5:
                        meas(3016723, d, rng.uniform(1.4, 3.5), 8840, "creatinine mg/dL")
                    else:
                        meas(3016723, d, rng.uniform(1.4, 3.5) / 0.0113, 2000001003, "creatinine umol/L")

            # ---- Hypertension ---------------------------------------------
            if rng.random() < 0.15 + 0.003 * max(0, age - 30):
                dx = _rand_date(rng, op_start, op_end)
                condition(320128, dx, "I10", 2000004004)
                if rng.random() < 0.6:
                    drug(2000005007, dx + timedelta(days=rng.randint(0, 60)), rng.randint(1, 10), "lisinopril")
                if rng.random() < 0.4:
                    drug(2000005009, _rand_date(rng, dx, op_end), rng.randint(1, 10), "atorvastatin")

            # ---- MI, COPD, depression -------------------------------------
            if rng.random() < 0.03:
                d = _rand_date(rng, op_start, op_end)
                condition(4329847, d, "I21.9", 0, kind=9201)
                if rng.random() < 0.2:
                    proc(2000007002, d + timedelta(days=rng.randint(1, 30)), "cabg")
            if rng.random() < 0.06:
                condition(255573, _rand_date(rng, op_start, op_end), "J44.9", 0)
            if rng.random() < 0.10:
                dx = _rand_date(rng, op_start, op_end)
                condition(rng.choice([440383, 2000003005]), dx, "F32.9", 2000004006)
                if rng.random() < 0.5:
                    drug(2000005008, dx + timedelta(days=rng.randint(0, 30)), rng.randint(1, 8), "sertraline")

            # ---- unmapped source codes (data-quality realism) -------------
            if rng.random() < 0.02:
                condition(0, _rand_date(rng, op_start, op_end), "LOCAL-XYZ", 0)

        w.flush(con)
    con.execute("INSERT INTO cdm.cdm_source VALUES ('Synthetic demo CDM', DATE '2026-07-01', 'v5.4', ?)",
                [DEMO_VOCAB_VERSION])


def build_demo_database(con: duckdb.DuckDBPyConnection, n_persons: int = 5000, seed: int = 42) -> dict:
    init_schemas(con)
    load_demo_vocabulary(con)
    generate_patients(con, n_persons=n_persons, seed=seed)
    counts = {t: con.execute(f"SELECT count(*) FROM cdm.{t}").fetchone()[0]
              for t in ("person", "condition_occurrence", "drug_exposure", "measurement", "procedure_occurrence",
                        "visit_occurrence")}
    counts["concepts"] = con.execute("SELECT count(*) FROM vocab.concept").fetchone()[0]
    return counts


def load_athena(con: duckdb.DuckDBPyConnection, athena_dir: Path) -> dict:
    """Load a real OHDSI Athena vocabulary download (tab-delimited CSVs) into the vocab schema."""
    init_schemas(con)
    athena_dir = Path(athena_dir)
    tables = {
        "concept": "CONCEPT.csv",
        "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
        "concept_ancestor": "CONCEPT_ANCESTOR.csv",
        "concept_synonym": "CONCEPT_SYNONYM.csv",
        "vocabulary": "VOCABULARY.csv",
    }
    counts = {}
    for table, fname in tables.items():
        path = athena_dir / fname
        if not path.exists():
            raise FileNotFoundError(path)
        con.execute(f"DELETE FROM vocab.{table}")
        cols = [r[0] for r in con.execute(f"DESCRIBE vocab.{table}").fetchall()]
        select = ", ".join(
            f"strptime({c}, '%Y%m%d')::DATE AS {c}" if c.endswith("_date") else c for c in cols
        )
        con.execute(
            f"INSERT INTO vocab.{table} SELECT {select} FROM read_csv('{path}', delim='\\t', header=true, "
            f"quote='', all_varchar=true)"
        )
        counts[table] = con.execute(f"SELECT count(*) FROM vocab.{table}").fetchone()[0]
    return counts
