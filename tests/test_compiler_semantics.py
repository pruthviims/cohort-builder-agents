"""Compiler semantics on tiny hand-built datasets with hand-computed expected patients.

OMOP-shaped scenarios run on DuckDB and, when CB_TEST_POSTGRES_DSN is set (CI sets it via a
PostgreSQL service), on a real PostgreSQL server too: the same compiled SQL must give the same
patients on both engines. Claims scenarios use the LAAD-style profile (DuckDB semantic views).

Documented semantics exercised here:
  * windows are inclusive at both ends, in calendar days relative to the index date
  * first_occurrence_only: the person's earliest qualifying event EVER is the index; if it falls
    outside an observation period the person does not enter (no fallback to a later event)
  * age at index = calendar year of index - year of birth
  * occurrence counts records by default; count_by="dates" counts distinct event dates
  * overlapping observation periods: earliest-starting period wins, then latest-ending
"""
from __future__ import annotations

import os
import re
from datetime import date

import duckdb
import pytest

from cohort_builder.compiler import Compiler
from cohort_builder.config import REPO_ROOT
from cohort_builder.db import CDM_DDL, VOCAB_DDL, init_schemas
from cohort_builder.executor import ExecutionError, guard_select
from cohort_builder.ir import CohortDefinition
from cohort_builder.ontology import Ontology
from cohort_builder.synthetic import load_demo_vocabulary
from cohort_builder.synthetic_laad import LAAD_DDL

D = date.fromisoformat
METFORMIN_500, METFORMIN_ER, LISINOPRIL = 2000005001, 2000005002, 2000005007
T2DM, T2DM_NEPHRO, T2DM_NOCOMP, T1DM = 201826, 2000003001, 2000003002, 201254
HBA1C, PCT, MMOL_MOL, MG_DL = 3004410, 8554, 2000001001, 8840
PG_DSN = os.environ.get("CB_TEST_POSTGRES_DSN")


# ---- engines -------------------------------------------------------------------------
class Engine:
    name = "duckdb"

    def __init__(self) -> None:
        self.con = duckdb.connect(":memory:")
        init_schemas(self.con)
        load_demo_vocabulary(self.con)

    def insert(self, table: str, rows: list[tuple]) -> None:
        if rows:
            marks = ",".join("?" * len(rows[0]))
            self.con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)

    def query(self, sql: str) -> list[tuple]:
        return self.con.execute(sql).fetchall()


class PostgresEngine(Engine):
    name = "postgres"
    _vocab_rows: dict[str, list[tuple]] = {}

    def __init__(self) -> None:
        import psycopg

        self.pg = psycopg.connect(PG_DSN, autocommit=True)
        with self.pg.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS cdm CASCADE; DROP SCHEMA IF EXISTS vocab CASCADE")
            vocab_ddl = re.sub(r"\bDOUBLE\b", "DOUBLE PRECISION", VOCAB_DDL)
            cur.execute(vocab_ddl)
            cur.execute(re.sub(r"\bDOUBLE\b", "DOUBLE PRECISION", CDM_DDL))
        if not PostgresEngine._vocab_rows:
            src = Engine()
            for t in ("concept", "concept_relationship", "concept_ancestor", "concept_synonym", "vocabulary"):
                PostgresEngine._vocab_rows[t] = src.query(f"SELECT * FROM vocab.{t}")
        for t, rows in PostgresEngine._vocab_rows.items():
            self.insert(f"vocab.{t}", rows)

    def insert(self, table: str, rows: list[tuple]) -> None:
        if rows:
            marks = ",".join(["%s"] * len(rows[0]))
            with self.pg.cursor() as cur:
                cur.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)

    def query(self, sql: str) -> list[tuple]:
        with self.pg.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()


ENGINES = ["duckdb"] + (["postgres"] if PG_DSN else [])


@pytest.fixture(params=ENGINES)
def engine(request) -> Engine:
    return PostgresEngine() if request.param == "postgres" else Engine()


@pytest.fixture(scope="module")
def ont() -> Ontology:
    return Ontology.load(REPO_ROOT / "ontology")


# ---- builders --------------------------------------------------------------------------
def person(pid: int, yob: int = 1970, gender: int = 8507) -> tuple:
    return (pid, gender, yob, 1, 1, 0, 0, f"p{pid}")


def op(pid: int, start: str = "2010-01-01", end: str = "2025-12-31", opid: int | None = None) -> tuple:
    return (opid or pid, pid, D(start), D(end), 0)


def drug(rid: int, pid: int, concept: int, day: str) -> tuple:
    return (rid, pid, concept, D(day), D(day), 30, 0, "", None)


def cond(rid: int, pid: int, concept: int, day: str) -> tuple:
    return (rid, pid, concept, D(day), None, 0, "", 0, None)


def meas(rid: int, pid: int, value: float | None, unit: int, day: str, concept: int = HBA1C) -> tuple:
    return (rid, pid, concept, D(day), value, unit, 0, "", None)


def cs(cid: str, domain: str, *concepts: int, desc: bool = True, exclude: tuple[int, ...] = ()) -> dict:
    return {"id": cid, "name": cid, "domain": domain,
            "items": [{"concept_id": c, "include_descendants": desc} for c in concepts]
            + [{"concept_id": c, "include_descendants": True, "is_excluded": True} for c in exclude]}


def make_ir(**kw) -> CohortDefinition:
    base = {"ontology_version": "1.1.0", "vocabulary_version": "test", "name": "t",
            "concept_sets": [cs("met", "Drug", 1503297)],
            "index_event": {"entity": "DrugExposure", "concept_set_id": "met"}}
    base.update(kw)
    return CohortDefinition.model_validate(base)


def run(engine: Engine, ont: Ontology, ir: CohortDefinition) -> dict[int, tuple[date, date]]:
    compiled = Compiler(ont).compile(ir)
    return {r[0]: (r[1], r[2]) for r in engine.query(compiled.cohort_sql)}


def attrition(engine: Engine, ont: Ontology, ir: CohortDefinition) -> list[int]:
    return [int(x) for x in engine.query(Compiler(ont).compile(ir).attrition_sql)[0]]


# ---- index events --------------------------------------------------------------------------
def test_first_index_ties_duplicates_and_multiple_concepts(engine, ont):
    engine.insert("cdm.person", [person(i) for i in (1, 2, 3, 4)])
    engine.insert("cdm.observation_period", [op(1), op(2), op(3), op(4, start="2010-01-01")])
    engine.insert("cdm.drug_exposure", [
        drug(1, 1, METFORMIN_500, "2018-03-01"), drug(2, 1, METFORMIN_ER, "2018-01-15"),   # 2 concepts: earliest
        drug(3, 2, METFORMIN_500, "2018-05-05"), drug(4, 2, METFORMIN_ER, "2018-05-05"),   # same-day tie
        drug(5, 2, METFORMIN_500, "2018-05-05"),                                           # exact duplicate
        drug(6, 3, LISINOPRIL, "2018-01-01"),                                              # not in concept set
        drug(7, 4, METFORMIN_500, "2009-06-01"), drug(8, 4, METFORMIN_500, "2012-01-01"),  # first is pre-observation
    ])
    got = run(engine, ont, make_ir())
    assert {k: v[0] for k, v in got.items()} == {1: D("2018-01-15"), 2: D("2018-05-05")}
    # without first_occurrence_only, person 4 enters on their earliest qualifying (observed) event
    got_all = run(engine, ont, make_ir(index_event={"entity": "DrugExposure", "concept_set_id": "met",
                                                     "first_occurrence_only": False}))
    assert {k: v[0] for k, v in got_all.items()} == {1: D("2018-01-15"), 2: D("2018-05-05"), 4: D("2012-01-01")}


def test_identical_ir_gives_identical_sql_regardless_of_concept_set_order(ont):
    a = make_ir(concept_sets=[cs("met", "Drug", 1503297), cs("t2", "Condition", T2DM)])
    b = make_ir(concept_sets=[cs("t2", "Condition", T2DM), cs("met", "Drug", 1503297)])
    assert Compiler(ont).compile(a).cohort_sql == Compiler(ont).compile(a).cohort_sql
    assert Compiler(ont).compile(a).sql_hash == Compiler(ont).compile(b).sql_hash


# ---- observation periods ---------------------------------------------------------------------
def test_observation_period_boundaries_missing_and_overlapping(engine, ont):
    engine.insert("cdm.person", [person(i) for i in range(1, 8)])
    engine.insert("cdm.observation_period", [
        op(1, "2017-01-01", "2020-12-31"),                   # index - start = 365 days
        op(2, "2017-01-02", "2020-12-31"),                   # 364 days
        op(3, "2018-01-01", "2018-12-31"),                   # index exactly on start
        op(4, "2016-01-01", "2018-01-01"),                   # index exactly on end
        op(5, "2010-01-01", "2017-12-31"),                   # index one day after end
        # person 6: no observation period at all
        op(7, "2015-01-01", "2019-01-01", opid=70), op(7, "2017-06-01", "2022-01-01", opid=71),  # overlapping
    ])
    engine.insert("cdm.drug_exposure", [drug(i, i, METFORMIN_500, "2018-01-01") for i in range(1, 8)])
    no_prior = make_ir()
    got = run(engine, ont, no_prior)
    assert set(got) == {1, 2, 3, 4, 7}
    assert got[4] == (D("2018-01-01"), D("2018-01-01"))           # end-of-observation exit on the boundary
    assert got[7] == (D("2018-01-01"), D("2019-01-01"))           # deterministic: earliest-starting period
    # 365 required: p1 has exactly 365 days, p2 364, p3 0, p4 731, p7 1096 (earliest-starting period)
    assert set(run(engine, ont, make_ir(prior_observation_days=365))) == {1, 4, 7}
    # attrition makes the lost people visible: 7 have an index event, 5 are inside an observation period
    assert attrition(engine, ont, no_prior)[:2] == [7, 5]


def test_post_observation_requirement(engine, ont):
    engine.insert("cdm.person", [person(1), person(2)])
    engine.insert("cdm.observation_period", [op(1, end="2018-04-01"), op(2, end="2018-03-31")])
    engine.insert("cdm.drug_exposure", [drug(1, 1, METFORMIN_500, "2018-01-01"), drug(2, 2, METFORMIN_500, "2018-01-01")])
    assert set(run(engine, ont, make_ir(post_observation_days=90))) == {1}  # 2018-04-01 - 2018-01-01 = 90


# ---- inclusion / exclusion windows ----------------------------------------------------------------
def _window_fixture(engine: Engine) -> None:
    engine.insert("cdm.person", [person(i) for i in range(1, 9)])
    engine.insert("cdm.observation_period", [op(i) for i in range(1, 9)])
    engine.insert("cdm.drug_exposure", [drug(i, i, METFORMIN_500, "2020-01-01") for i in range(1, 9)])
    engine.insert("cdm.condition_occurrence", [
        cond(1, 1, T2DM, "2019-01-01"),   # -365: inside
        cond(2, 2, T2DM, "2018-12-31"),   # -366: outside
        cond(3, 3, T2DM, "2020-01-01"),   # 0: inside
        cond(4, 4, T2DM, "2020-01-02"),   # +1: outside
        # person 5: no condition records at all
        cond(6, 6, T2DM, "2019-12-22"), cond(7, 6, T1DM, "2020-03-31"),   # T1D at +90 -> excluded
        cond(8, 7, T2DM, "2019-12-22"), cond(9, 7, T1DM, "2020-04-01"),   # T1D at +91 -> kept
        cond(10, 8, T2DM_NOCOMP, "2019-06-01"),                           # descendant concept counts
    ])


def test_inclusive_window_boundaries_and_exclusions(engine, ont):
    _window_fixture(engine)
    sets = [cs("met", "Drug", 1503297), cs("t2", "Condition", T2DM), cs("t1", "Condition", T1DM)]
    inc = {"id": "i1", "name": "T2D", "entity": "ConditionOccurrence", "concept_set_id": "t2",
           "window": {"start_days": -365, "end_days": 0}}
    exc = {"id": "e1", "name": "T1D after", "entity": "ConditionOccurrence", "concept_set_id": "t1",
           "window": {"start_days": 0, "end_days": 90}}
    assert set(run(engine, ont, make_ir(concept_sets=sets, inclusion=[inc]))) == {1, 3, 6, 7, 8}
    assert set(run(engine, ont, make_ir(concept_sets=sets, inclusion=[inc], exclusion=[exc]))) == {1, 3, 7, 8}


def test_leap_year_day_arithmetic(engine, ont):
    engine.insert("cdm.person", [person(1), person(2)])
    engine.insert("cdm.observation_period", [op(1), op(2)])
    engine.insert("cdm.drug_exposure", [drug(1, 1, METFORMIN_500, "2020-03-01"), drug(2, 2, METFORMIN_500, "2020-03-01")])
    engine.insert("cdm.condition_occurrence", [cond(1, 1, T2DM, "2019-03-02"),   # 365 days before (2020 is leap)
                                               cond(2, 2, T2DM, "2019-03-01")])  # 366 days before
    ir = make_ir(concept_sets=[cs("met", "Drug", 1503297), cs("t2", "Condition", T2DM)],
                 inclusion=[{"id": "i", "name": "x", "entity": "ConditionOccurrence", "concept_set_id": "t2",
                             "window": {"start_days": -365, "end_days": 0}}])
    assert set(run(engine, ont, ir)) == {1}


def test_concept_set_descendants_and_exclusions(engine, ont):
    engine.insert("cdm.person", [person(i) for i in (1, 2, 3)])
    engine.insert("cdm.observation_period", [op(i) for i in (1, 2, 3)])
    engine.insert("cdm.drug_exposure", [drug(i, i, METFORMIN_500, "2020-01-01") for i in (1, 2, 3)])
    engine.insert("cdm.condition_occurrence", [cond(1, 1, T2DM_NOCOMP, "2019-01-01"),
                                               cond(2, 2, T2DM_NEPHRO, "2019-01-01"),
                                               cond(3, 3, T2DM, "2019-01-01")])

    def with_set(t2: dict) -> CohortDefinition:
        return make_ir(concept_sets=[cs("met", "Drug", 1503297), t2],
                       inclusion=[{"id": "i", "name": "t2", "entity": "ConditionOccurrence", "concept_set_id": "t2"}])

    assert set(run(engine, ont, with_set(cs("t2", "Condition", T2DM)))) == {1, 2, 3}
    assert set(run(engine, ont, with_set(cs("t2", "Condition", T2DM, exclude=(T2DM_NEPHRO,))))) == {1, 3}
    assert set(run(engine, ont, with_set(cs("t2", "Condition", T2DM, desc=False)))) == {3}


def test_age_is_calendar_year_based(engine, ont):
    engine.insert("cdm.person", [person(1, yob=2000), person(2, yob=2001)])
    engine.insert("cdm.observation_period", [op(1), op(2)])
    engine.insert("cdm.drug_exposure", [drug(1, 1, METFORMIN_500, "2018-01-01"), drug(2, 2, METFORMIN_500, "2018-12-31")])
    assert set(run(engine, ont, make_ir(demographics={"age_min": 18}))) == {1}


# ---- occurrence rules, spans, duplicates ------------------------------------------------------------
def _occurrence_fixture(engine: Engine) -> None:
    engine.insert("cdm.person", [person(i) for i in range(1, 6)])
    engine.insert("cdm.observation_period", [op(i) for i in range(1, 6)])
    engine.insert("cdm.drug_exposure", [drug(i, i, METFORMIN_500, "2020-01-01") for i in range(1, 6)])
    engine.insert("cdm.condition_occurrence", [
        cond(1, 1, T2DM, "2019-06-01"), cond(2, 1, T2DM, "2019-07-01"),   # 2 records 30 days apart
        cond(3, 2, T2DM, "2019-06-01"), cond(4, 2, T2DM, "2019-06-01"),   # exact duplicate records
        cond(5, 3, T2DM, "2019-06-01"),                                   # one record
        cond(6, 4, T2DM, "2019-06-01"), cond(7, 4, T2DM, "2019-06-30"),   # 29 days apart
        # person 5: none
    ])


def _occ(**kw) -> CohortDefinition:
    rule = {"id": "i", "name": "t2", "entity": "ConditionOccurrence", "concept_set_id": "t2",
            "window": {"start_days": -365, "end_days": 0}, **kw}
    return make_ir(concept_sets=[cs("met", "Drug", 1503297), cs("t2", "Condition", T2DM)], inclusion=[rule])


def test_occurrence_rules_and_duplicates(engine, ont):
    _occurrence_fixture(engine)
    assert set(run(engine, ont, _occ(occurrence="at_least", count=2))) == {1, 2, 4}            # duplicates count
    assert set(run(engine, ont, _occ(occurrence="at_least", count=2, count_by="dates"))) == {1, 4}
    assert set(run(engine, ont, _occ(occurrence="at_least", count=2, min_span_days=30))) == {1}
    assert set(run(engine, ont, _occ(occurrence="at_least", count=2, min_span_days=29))) == {1, 4}
    assert set(run(engine, ont, _occ(occurrence="at_most", count=1))) == {3, 5}
    assert set(run(engine, ont, _occ(occurrence="exactly", count=0))) == {5}
    assert set(run(engine, ont, _occ(occurrence="exactly", count=2))) == {1, 2, 4}


# ---- laboratory values -------------------------------------------------------------------------------
def _lab_fixture(engine: Engine) -> None:
    engine.insert("cdm.person", [person(i) for i in range(1, 8)])
    engine.insert("cdm.observation_period", [op(i) for i in range(1, 8)])
    engine.insert("cdm.drug_exposure", [drug(i, i, METFORMIN_500, "2020-01-01") for i in range(1, 8)])
    engine.insert("cdm.measurement", [
        meas(1, 1, 8.0, PCT, "2019-06-01"),        # exactly on the threshold
        meas(2, 2, None, PCT, "2019-06-01"),       # missing value
        meas(3, 3, 75.0, MMOL_MOL, "2019-06-01"),  # 75 mmol/mol = 9.0 %
        meas(4, 4, 9.0, MG_DL, "2019-06-01"),      # incompatible unit for HbA1c: never compared
        meas(5, 5, 64.0, MMOL_MOL, "2019-06-01"),  # 64 mmol/mol = 8.007 %
        meas(6, 6, 8.1, PCT, "2018-11-27"),        # outside the window (-400 days)
        meas(7, 7, 7.2, PCT, "2019-06-01"),
    ])


def _lab(op: str, value: float, high: float | None = None) -> CohortDefinition:
    vf = {"op": op, "value": value, "unit_concept_id": PCT, **({"value_high": high} if high is not None else {})}
    return make_ir(concept_sets=[cs("met", "Drug", 1503297), cs("a1c", "Measurement", HBA1C)],
                   inclusion=[{"id": "i", "name": "a1c", "entity": "Measurement", "concept_set_id": "a1c",
                               "window": {"start_days": -365, "end_days": 0}, "value_filter": vf}])


def test_lab_thresholds_units_and_missing_values(engine, ont):
    _lab_fixture(engine)
    assert set(run(engine, ont, _lab(">", 8))) == {3, 5}
    assert set(run(engine, ont, _lab(">=", 8))) == {1, 3, 5}
    assert set(run(engine, ont, _lab("between", 7, 8))) == {1, 7}      # inclusive bounds
    assert set(run(engine, ont, _lab("<", 8))) == {7}                  # null and mg/dL never match
    no_value = make_ir(concept_sets=[cs("met", "Drug", 1503297), cs("a1c", "Measurement", HBA1C)],
                       exclusion=[{"id": "e", "name": "any a1c", "entity": "Measurement", "concept_set_id": "a1c",
                                   "window": {"start_days": -365, "end_days": 0}}])
    assert set(run(engine, ont, no_value)) == {6}  # any record counts without a value filter, even null/odd units


# ---- SQL safety ----------------------------------------------------------------------------------------
HOSTILE = "x'); DROP TABLE cdm.person; --"


def test_free_text_never_reaches_sql(engine, ont):
    engine.insert("cdm.person", [person(1)])
    engine.insert("cdm.observation_period", [op(1)])
    engine.insert("cdm.drug_exposure", [drug(1, 1, METFORMIN_500, "2020-01-01")])
    sets = [{**cs("met", "Drug", 1503297), "name": HOSTILE, "source": HOSTILE}]
    ir = make_ir(name=HOSTILE, description=HOSTILE, assumptions=[HOSTILE], concept_sets=sets,
                 inclusion=[{"id": "i", "name": HOSTILE, "entity": "DrugExposure", "concept_set_id": "met"}])
    compiled = Compiler(ont).compile(ir)
    for sql in (compiled.cohort_sql, compiled.attrition_sql):
        assert "DROP" not in sql and HOSTILE not in sql
        assert set(re.findall(r"'([^']*)'", sql)) <= {"met"}   # the only string literals are validated ids
        guard_select(sql)
    assert set(run(engine, ont, ir)) == {1}
    assert engine.query("SELECT count(*) FROM cdm.person") == [(1,)]


def test_only_validated_literals_and_claim_statuses_in_sql():
    laad = Ontology.load(REPO_ROOT / "ontology", "iqvia_laad")
    ir = make_ir(index_event={"entity": "DrugExposure", "concept_set_id": "met", "claim_status": ["rejected"]},
                 inclusion=[{"id": "i", "name": "paid", "entity": "DrugExposure", "concept_set_id": "met",
                             "claim_status": ["paid"]}])
    sql = Compiler(laad).compile(ir).cohort_sql
    assert set(re.findall(r"'([^']*)'", sql)) <= {"met", "paid", "rejected"}


@pytest.mark.parametrize("bad", ["cdm.person; DROP TABLE x", "cdm.person p JOIN secrets s ON 1=1", "1table",
                                 "cdm.\"person\""])
def test_malicious_dataset_profile_identifiers_are_rejected(tmp_path, bad):
    import shutil

    import yaml

    target = tmp_path / "ontology"
    shutil.copytree(REPO_ROOT / "ontology", target)
    prof_path = target / "datasets" / "omop_demo.yaml"
    prof = yaml.safe_load(prof_path.read_text())
    prof["mapping"]["entities"]["DrugExposure"]["table"] = bad
    prof_path.write_text(yaml.safe_dump(prof))
    with pytest.raises(ValueError, match="not a plain SQL identifier"):
        Ontology.load(target)


@pytest.mark.parametrize("sql", ["SELECT 1; DROP TABLE cdm.person", "DELETE FROM cdm.person",
                                 "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
                                 "SELECT * FROM read_csv('/etc/passwd'); ", "ATTACH 'x.db'"])
def test_execution_guard_rejects_non_select_sql(sql):
    with pytest.raises(ExecutionError):
        guard_select(sql)


def test_guard_ignores_keywords_inside_literals():
    assert guard_select("SELECT 'load' AS cs_id")


@pytest.mark.skipif(not PG_DSN, reason="set CB_TEST_POSTGRES_DSN to run against PostgreSQL")
def test_postgres_engine_is_really_used():
    assert PostgresEngine().query("SELECT version()")[0][0].startswith("PostgreSQL")


def test_compiled_sql_parses_as_postgresql(ont):
    pglast = pytest.importorskip("pglast")
    laad = Ontology.load(REPO_ROOT / "ontology", "iqvia_laad")
    irs = [_lab(">", 8), _occ(occurrence="at_least", count=2, min_span_days=30, count_by="dates"),
           make_ir(exit={"type": "fixed_days", "days": 30}, demographics={"age_min": 18, "gender_concept_ids": [8532]},
                   prior_observation_days=365, post_observation_days=30)]
    for o in (ont, laad):
        for ir in irs:
            if o is laad and ir.inclusion and ir.inclusion[0].entity == "Measurement":
                continue
            compiled = Compiler(o).compile(ir)
            pglast.parse_sql(compiled.cohort_sql)
            pglast.parse_sql(compiled.attrition_sql)


# ---- claims (LAAD-style profile) -------------------------------------------------------------------------
@pytest.fixture
def laad_db():
    con = duckdb.connect(":memory:")
    init_schemas(con)
    load_demo_vocabulary(con)
    con.execute(LAAD_DDL)
    laad = Ontology.load(REPO_ROOT / "ontology", "iqvia_laad")
    return con, laad


def _laad_rows(con) -> None:
    SGLT2 = "99999020101"
    con.executemany("INSERT INTO laad.patient VALUES (?,?,?)", [(i, 1970, "F") for i in range(1, 7)])
    con.executemany("INSERT INTO laad.rx_claims VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, 1, D("2019-01-10"), SGLT2, 30, 30, "PAID", None, None, "COMMERCIAL"),
        (1, 1, D("2019-01-10"), SGLT2, 30, 30, "PAID", None, None, "COMMERCIAL"),     # duplicate delivery
        (2, 2, D("2019-02-01"), SGLT2, 30, 30, "PAID", None, None, "COMMERCIAL"),
        (3, 2, D("2019-02-03"), SGLT2, 30, 30, "REVERSAL", None, 2, "COMMERCIAL"),    # reverses claim 2
        (4, 2, D("2019-03-01"), SGLT2, 30, 30, "PAID", None, None, "COMMERCIAL"),
        (5, 3, D("2019-04-01"), SGLT2, 30, 30, "PAID", None, None, "COMMERCIAL"),
        (6, 3, D("2019-04-02"), SGLT2, 30, 30, "REVERSAL", None, 999, "COMMERCIAL"),  # orphan reversal
        (7, 4, D("2019-05-01"), SGLT2, 30, 30, "REJECTED", "75", None, "MEDICARE"),
        (8, 4, D("2019-05-20"), SGLT2, 30, 30, "PAID", None, None, "MEDICARE"),
        (9, None, D("2019-06-01"), SGLT2, 30, 30, "PAID", None, None, "MEDICARE"),    # no patient id
        (10, 77, D("2019-06-01"), SGLT2, 30, 30, "PAID", None, None, "MEDICARE"),     # patient not on file
        (11, 5, D("2019-06-01"), SGLT2, 30, 30, "PAID", None, None, "MEDICARE"),
    ])
    con.executemany("INSERT INTO laad.dx_claims VALUES (?,?,?,?,?,?,?,?)", [
        (20, 1, D("2018-06-01"), "11", "E119", None, None, None),
        (20, 1, D("2018-06-01"), "11", "E119", None, None, None),     # second line of the same claim
        (21, 1, D("2018-08-01"), "11", "I10", "R05", "E119", None),   # secondary position
        (22, 5, D("2019-03-01"), "11", "E119", "E119", None, None),   # same code twice on one claim
        (23, 6, D("2019-03-01"), "11", "E119", None, None, None),
    ])


def test_claims_duplicates_reversals_and_rejections(laad_db):
    con, laad = laad_db
    _laad_rows(con)
    for s in laad.setup_statements():
        con.execute(s)
    eng = Engine.__new__(Engine)
    eng.con = con
    sglt2 = [cs("s", "Drug", 45775965)]
    paid = make_ir(concept_sets=sglt2, index_event={"entity": "DrugExposure", "concept_set_id": "s",
                                                    "claim_status": ["paid"]})
    got = {k: v[0] for k, v in run(eng, laad, paid).items()}
    # p2's first paid claim was reversed -> index is the next paid claim; p3's orphan reversal is ignored
    assert got == {1: D("2019-01-10"), 2: D("2019-03-01"), 3: D("2019-04-01"), 4: D("2019-05-20"),
                   5: D("2019-06-01")}
    rejected = make_ir(concept_sets=sglt2, index_event={"entity": "DrugExposure", "concept_set_id": "s",
                                                        "claim_status": ["rejected"]})
    assert {k: v[0] for k, v in run(eng, laad, rejected).items()} == {4: D("2019-05-01")}
    # patient 77 has an index claim but no patient record: counted at step 0, dropped at step 1
    a = attrition(eng, laad, paid)
    assert a[0] == 6 and a[1] == 5
    assert con.execute("SELECT count(*) FROM sem.drug_exposure WHERE claim_id = 1").fetchone()[0] == 1


def test_claims_diagnosis_positions_and_claim_lines(laad_db):
    con, laad = laad_db
    _laad_rows(con)
    for s in laad.setup_statements():
        con.execute(s)
    eng = Engine.__new__(Engine)
    eng.con = con
    sets = [cs("s", "Drug", 45775965), cs("t2", "Condition", T2DM)]

    def t2d(**kw) -> CohortDefinition:
        rule = {"id": "i", "name": "t2d", "entity": "ConditionOccurrence", "concept_set_id": "t2",
                "window": {"start_days": -365, "end_days": 0}, **kw}
        return make_ir(concept_sets=sets, inclusion=[rule],
                       index_event={"entity": "DrugExposure", "concept_set_id": "s", "claim_status": ["paid"]})

    assert set(run(eng, laad, t2d(occurrence="at_least", count=2))) == {1}   # 2 distinct claims, not 3 lines
    assert set(run(eng, laad, t2d(occurrence="at_least", count=1, dx_position="primary"))) == {1, 5}
    assert set(run(eng, laad, t2d(occurrence="at_least", count=2, dx_position="primary"))) == set()
    rows = con.execute("SELECT claim_id, dx_position FROM sem.condition_occurrence WHERE person_id = 5").fetchall()
    assert rows == [(22, 1)]   # the code listed twice on one claim is one diagnosis, in its best position
