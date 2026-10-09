"""Synthetic ESCC-like fixture for proxy cohort tests. Synthetic data only; placeholder concepts.

Placeholder concepts live in the local id range (2,100,000,000+). They are NOT clinical codes.
The dataset profile `escc_synthetic` is a copy of the OMOP demo profile that declares pathology
data (Measurement-based histology results) so proxy rules needing pathology can run.
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb
import yaml

from cohort_builder.config import REPO_ROOT
from cohort_builder.db import init_schemas
from cohort_builder.synthetic import load_demo_vocabulary

EXAMPLE = REPO_ROOT / "examples" / "proxy" / "escc_proxy_example.yaml"
D = date.fromisoformat

# (id, name, domain, vocabulary, class, parent)
PLACEHOLDER_CONCEPTS = [
    (2100000001, "PLACEHOLDER esophageal malignancy", "Condition", "SNOMED", "Clinical Finding", None),
    (
        2100000002,
        "PLACEHOLDER esophageal malignancy, lower third",
        "Condition",
        "SNOMED",
        "Clinical Finding",
        2100000001,
    ),
    (2100000003, "PLACEHOLDER squamous histology finding", "Measurement", "LOINC", "Lab Test", None),
    (2100000004, "PLACEHOLDER chemotherapy agent A", "Drug", "RxNorm", "Ingredient", None),
    (2100000005, "PLACEHOLDER chemotherapy agent B", "Drug", "RxNorm", "Ingredient", None),
    (2100000006, "PLACEHOLDER radiation therapy", "Procedure", "SNOMED", "Procedure", None),
    (2100000007, "PLACEHOLDER esophagectomy", "Procedure", "SNOMED", "Procedure", None),
    (2100000008, "PLACEHOLDER adenocarcinoma histology finding", "Measurement", "LOINC", "Lab Test", None),
    (2100000009, "PLACEHOLDER competing malignancy", "Condition", "SNOMED", "Clinical Finding", None),
]
SYNONYMS = {
    2100000001: ["esophageal cancer", "esophageal malignancy"],
    2100000003: ["squamous pathology"],
    2100000004: ["escc chemotherapy"],
    2100000006: ["radiation therapy"],
}
DX, DX_SUB, SQUAMOUS, CHEMO_A, CHEMO_B, RADIATION, SURGERY, ADENO, COMPETING = (c[0] for c in PLACEHOLDER_CONCEPTS)

# expected classification (person_id -> tier) for the example algorithm; None = not in cohort
EXPECTED = {1: "high", 2: "moderate", 3: "exploratory", 5: "high"}
NOT_IN_COHORT = {
    4: "treatment but no qualifying diagnosis (never indexed)",
    6: "diagnosis but evidence outside the required windows",
    7: "excluded: competing malignancy in the prior year",
}
FLAGGED = {5}


def add_placeholder_vocabulary(con: duckdb.DuckDBPyConnection) -> None:
    start, end = D("1970-01-01"), D("2099-12-31")
    for cid, name, dom, voc, cls, parent in PLACEHOLDER_CONCEPTS:
        con.execute(
            "INSERT INTO vocab.concept VALUES (?,?,?,?,?,'S',?,?,?,NULL)",
            [cid, name, dom, voc, cls, f"PH{cid}", start, end],
        )
        con.execute("INSERT INTO vocab.concept_ancestor VALUES (?,?,0,0)", [cid, cid])
        if parent:
            con.execute("INSERT INTO vocab.concept_ancestor VALUES (?,?,1,1)", [parent, cid])
        for syn in SYNONYMS.get(cid, []):
            con.execute("INSERT INTO vocab.concept_synonym VALUES (?,?,4180186)", [cid, syn])


class Events:
    """Collects synthetic CDM rows; `day` is an ISO date or an int offset from `anchor`."""

    def __init__(self, anchor: str = "2020-06-01"):
        self.anchor = D(anchor)
        self.cond: list[tuple] = []
        self.drug: list[tuple] = []
        self.proc: list[tuple] = []
        self.meas: list[tuple] = []

    def _d(self, day):
        return self.anchor + timedelta(days=day) if isinstance(day, int) else D(day)

    def dx(self, p, day, concept=DX):
        self.cond.append((len(self.cond) + 1, p, concept, self._d(day), None, 0, "", 0, None))

    def rx(self, p, day, concept=CHEMO_A):
        self.drug.append((len(self.drug) + 1, p, concept, self._d(day), self._d(day), 21, 0, "", None))

    def px(self, p, day, concept=RADIATION):
        self.proc.append((len(self.proc) + 1, p, concept, self._d(day), 0, "", None))

    def lab(self, p, day, concept=SQUAMOUS):
        self.meas.append((len(self.meas) + 1, p, concept, self._d(day), None, None, 0, "", None))

    def write(self, con: duckdb.DuckDBPyConnection) -> None:
        for table, rows, n in (
            ("condition_occurrence", self.cond, 9),
            ("drug_exposure", self.drug, 9),
            ("procedure_occurrence", self.proc, 7),
            ("measurement", self.meas, 9),
        ):
            if rows:
                con.executemany(f"INSERT INTO cdm.{table} VALUES ({','.join('?' * n)})", rows)


def base_db(con: duckdb.DuckDBPyConnection, persons) -> None:
    """Schemas, demo + placeholder vocabulary, persons observed 2015-01-01 .. 2025-12-31."""
    init_schemas(con)
    load_demo_vocabulary(con)
    add_placeholder_vocabulary(con)
    persons = list(persons)
    con.executemany("INSERT INTO cdm.person VALUES (?,8507,1960,1,1,0,0,?)", [(p, f"p{p}") for p in persons])
    con.executemany(
        "INSERT INTO cdm.observation_period VALUES (?,?,'2015-01-01','2025-12-31',0)", [(p, p) for p in persons]
    )
    con.execute("INSERT INTO cdm.cdm_source VALUES ('Synthetic ESCC fixture', DATE '2026-01-01', 'v5.4', 'test')")


def build_escc_db(con: duckdb.DuckDBPyConnection) -> None:
    base_db(con, range(1, 8))
    ev = Events()
    dx, rx, px, lab = ev.dx, ev.rx, ev.px, ev.lab

    # P1: diagnosis + pathology + treatment -> high
    dx(1, "2020-01-10")
    dx(1, "2020-03-01", DX_SUB)
    lab(1, "2020-01-20")
    rx(1, "2020-02-01")  # noqa: E702
    px(1, "2020-02-10")
    # P2: diagnosis + strong treatment pattern (chemo + radiation within 90 days), no pathology -> moderate
    dx(2, "2020-01-10")
    dx(2, "2020-02-15")
    rx(2, "2020-02-01", CHEMO_B)
    px(2, "2020-02-05")  # noqa: E702
    # P3: repeated diagnosis only -> exploratory
    dx(3, "2020-01-10")
    dx(3, "2020-05-01")  # noqa: E702
    # P4: treatment but no qualifying diagnosis -> never indexed
    rx(4, "2020-02-01")
    px(4, "2020-02-05")  # noqa: E702
    # P5: diagnosis + pathology + treatment + conflicting adenocarcinoma histology -> high, flagged
    dx(5, "2020-01-10")
    lab(5, "2020-01-15")
    lab(5, "2020-01-16", ADENO)
    rx(5, "2020-01-20")  # noqa: E702
    # P6: diagnosis, but repeat diagnosis and treatment fall outside the windows -> no tier
    dx(6, "2020-01-10")
    dx(6, "2020-12-01")
    rx(6, "2020-09-01")  # noqa: E702
    # P7: diagnosis + pathology + treatment, but competing malignancy 7 months before -> excluded
    dx(7, "2020-01-10")
    lab(7, "2020-01-12")
    rx(7, "2020-01-20")
    dx(7, "2019-06-01", COMPETING)  # noqa: E702
    ev.write(con)


def make_ontology_dir(tmp: Path, pathology: bool = True) -> Path:
    """Copy of the repo ontology plus an `escc_synthetic` profile (OMOP tables; pathology optional)."""
    target = tmp / "ontology"
    if not target.exists():
        shutil.copytree(REPO_ROOT / "ontology", target)
    prof = yaml.safe_load((target / "datasets" / "omop_demo.yaml").read_text())
    prof["name"] = "escc_synthetic"
    prof["description"] = "Synthetic ESCC-like fixture on OMOP tables (placeholder concepts)."
    prof["capabilities"]["extra_data"] = ["pathology"] if pathology else []
    prof["capabilities"]["partial_entities"] = []
    (target / "datasets" / "escc_synthetic.yaml").write_text(yaml.safe_dump(prof, sort_keys=False))
    return target


def make_builder(
    tmp: Path,
    build=build_escc_db,
    pathology: bool = True,
    dataset: str = "escc_synthetic",
    policy=None,
    backend=None,
    min_cell: int | None = None,
    name: str = "escc.duckdb",
):
    """A CohortBuilder over a fresh synthetic database (placeholder concepts only)."""
    import dataclasses

    from cohort_builder.config import Settings
    from cohort_builder.db import connect
    from cohort_builder.orchestrator import CohortBuilder

    db = tmp / name
    con = connect(db)
    build(con)
    con.close()
    settings = dataclasses.replace(
        Settings(),
        db_path=db,
        ontology_dir=make_ontology_dir(tmp, pathology),
        dataset=dataset,
        model="fake-model",
        llm_mode="cached",
    )
    b = CohortBuilder(settings, backend=backend, policy=policy)
    if min_cell is not None:
        b.executor.min_cell = min_cell
    return b
