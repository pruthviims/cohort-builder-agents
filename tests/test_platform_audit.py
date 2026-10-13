"""Platform audit regressions (synthetic data only): numeric range and code-name inference, binary
content, concurrent version allocation, dry-run isolation, date plausibility, strict mapping YAML,
redacted error logging, API body limits, and one end-to-end path from messy CSV files to an executed,
suppressed aggregate result."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from pathlib import Path

import duckdb
import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.agents.validator import validate
from cohort_builder.config import REPO_ROOT
from cohort_builder.csv_ingest import IngestError, profile_csv, profile_text
from cohort_builder.dataset_mapping import DatasetMapping
from cohort_builder.dataset_registry import DatasetRegistry, Upload, load_mapping
from cohort_builder.db import connect, redact_error
from cohort_builder.executor import ExecutionError, Executor
from cohort_builder.ir import CohortDefinition
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import ConfigError, SecurityConfig, issue_token

EXAMPLE = REPO_ROOT / "examples" / "csv"
EXAMPLE_FILES = ["patients", "enrollment", "diagnoses", "prescriptions", "labs"]
MARKER = "ZZMARKER9876"  # a synthetic cell value that must never appear in any report, log or output


def write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_bytes(text.encode())
    return p


def col_types(report: dict) -> dict[str, tuple[str, str]]:
    assert report["ok"], report.get("errors")
    return {c["name"]: (c["type"], c["kind"]) for c in report["columns"]}


@pytest.fixture
def reg(tmp_path):
    con = connect(tmp_path / "db.duckdb")
    r = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    yield r
    con.close()


def simple_mapping(**entity) -> DatasetMapping:
    return DatasetMapping.model_validate(
        {
            "person": {"table": "people", "person_column": "pid", "year_of_birth_column": "yob"},
            "observation": {"mode": "activity", "absence_inference": "weak", "max_gap_days": 365},
            "entities": {
                "ConditionOccurrence": {
                    "table": "events",
                    "person_column": "pid",
                    "start_column": "day",
                    "code_column": "dx_code",
                    "vocabulary": "ICD10CM",
                    **entity,
                }
            },
        }
    )


# ---- type inference: values a DOUBLE cannot hold, decimal leading zeros, code-like names -------------------
@pytest.mark.parametrize("value", ["1e999", "-1e999", "1e-400", "4.9e-324"])
def test_out_of_range_numbers_stay_exact_text(value):
    # DuckDB would turn these into inf / 0 / a rounded subnormal: a silent change of value
    assert col_types(profile_text("t.csv", f"v\n{value}\n2\n"))["v"][0] == "VARCHAR"
    bad = profile_text("t.csv", f"v\n{value}\n", column_types={"v": "DOUBLE"})
    assert not bad["ok"] and bad["errors"][0]["code"] == "invalid_override"


def test_in_range_numbers_still_infer_numeric():
    t = col_types(profile_text("t.csv", "a,b,c\n1.5e10,0.5,0e5\n-2.25,0.0,1\n"))
    assert t == {"a": ("DOUBLE", "measurement"), "b": ("DOUBLE", "measurement"), "c": ("DOUBLE", "measurement")}


def test_decimals_with_leading_zeros_and_code_like_names_stay_text():
    t = col_types(
        profile_text(
            "t.csv",
            "v,dx,dx2,proc_1,dx_date,process,diagnosis_code\n007.5,250.00,401.9,12.50,2020-01-01,3.5,250.00\n",
        )
    )
    assert t["v"] == ("VARCHAR", "code")  # '007.5' keeps its zeros, like '007'
    assert t["dx"] == ("VARCHAR", "code") and t["dx2"] == ("VARCHAR", "code") and t["proc_1"] == ("VARCHAR", "code")
    assert t["dx_date"] == ("DATE", "date")  # a date name is not a code
    assert t["process"] == ("DOUBLE", "measurement")  # 'proc' only as a whole word
    assert t["diagnosis_code"] == ("VARCHAR", "code")


def test_nul_bytes_and_invalid_unicode_are_rejected(tmp_path):
    r = profile_csv(write(tmp_path, "b.csv", "a,b\n1,2\n3,\x004\n"))
    assert not r["ok"] and r["errors"][0] == {"code": "invalid_encoding", "message": "NUL byte in text", "line": 3}
    r = profile_text("b.csv", "a\nx\x00\n")
    assert not r["ok"] and r["errors"][0]["line"] == 2
    r = profile_text("s.csv", "a\n\ud800\n")
    assert not r["ok"] and r["errors"][0]["code"] == "invalid_encoding"


# ---- concurrency ---------------------------------------------------------------------------------------------
def run_parallel(base: duckdb.DuckDBPyConnection, local: Path, jobs: list[tuple[str, bool]]) -> list[tuple]:
    text = "pid,yob\n" + "".join(f"{i},{1950 + i % 40}\n" for i in range(2000))
    results: list[tuple] = [()] * len(jobs)
    barrier = threading.Barrier(len(jobs))

    def work(i: int, name: str, new_version: bool) -> None:
        con = base.cursor()  # separate connection to the same database (own transactions)
        r = DatasetRegistry(con, REPO_ROOT / "ontology", local)
        barrier.wait()
        try:
            out = r.ingest(name, [(None, Upload("people.csv", text))], f"user{i}", new_version=new_version)
            results[i] = ("ok", out["version"])
        except IngestError as exc:
            results[i] = ("refused", exc.problems[0]["code"] if exc.problems else "")
        except Exception as exc:  # noqa: BLE001 - any other error is a test failure, recorded for the assert
            results[i] = ("error", type(exc).__name__)
        finally:
            con.close()

    threads = [threading.Thread(target=work, args=(i, n, nv)) for i, (n, nv) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_new_versions_get_distinct_numbers(reg):
    results = run_parallel(reg.con, reg.local_dir, [("conc", True)] * 6)
    assert all(r[0] == "ok" for r in results), results  # a commit-time conflict is retried, not raised
    assert sorted(r[1] for r in results) == [1, 2, 3, 4, 5, 6]
    rows = reg.con.execute("SELECT version, status FROM meta.dataset_version WHERE dataset = 'conc'").fetchall()
    assert sorted(rows) == [(v, "staged") for v in range(1, 7)]


def test_concurrent_first_imports_admit_exactly_one(reg):
    results = run_parallel(reg.con, reg.local_dir, [("first", False)] * 4)
    assert [r[0] for r in results].count("ok") == 1, results
    assert all(r == ("refused", "dataset_exists") for r in results if r[0] != "ok")
    live = reg.con.execute(
        "SELECT count(*) FROM meta.dataset_version WHERE dataset = 'first' AND status <> 'failed'"
    ).fetchone()[0]
    assert live == 1


def test_dry_run_uses_its_own_schema_and_keeps_nothing(reg, tmp_path):
    p = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    r = reg.ingest("dry", [(None, p)], "alice", dry_run=True)
    assert r["status"] == "dry_run" and r["staging_schema"].startswith("stg_dry_v1_dry")
    left = reg.con.execute("SELECT count(*) FROM information_schema.schemata WHERE schema_name LIKE '%dry%'")
    assert left.fetchone()[0] == 0 and reg.versions("dry") == []
    assert reg.ingest("dry", [(None, p)], "alice")["version"] == 1


# ---- data quality: implausible dates, two-digit years ----------------------------------------------------------
def test_implausible_dates_and_birth_years_are_reported(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n2,1850\n3,2999\n")
    events = write(
        tmp_path, "events.csv", "pid,day,dx_code\n1,2020-01-01,E11.9\n2,1850-01-01,E11.9\n3,2999-01-01,E11.9\n"
    )
    r = reg.ingest("dates", [(None, people), (None, events)], "alice", mapping=simple_mapping())
    checks = r["mapping"]["checks"]
    assert checks["person"]["implausible_year_of_birth"] == 2
    assert checks["ConditionOccurrence"]["rows"]["start_date_out_of_range"] == 2
    assert checks["ConditionOccurrence"]["rows"]["usable_rows"] == 3  # reported, never altered or dropped
    w = " | ".join(r["mapping"]["warnings"])
    assert "2 person(s) have a year of birth before 1900 or in the future" in w
    assert "2 usable row(s) start before 1900-01-01 or in the future" in w


def test_two_digit_year_formats_warn(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    events = write(tmp_path, "events.csv", "pid,day,dx_code\n1,01/02/20,E11.9\n")
    r = reg.ingest("yy", [(None, people), (None, events)], "alice", mapping=simple_mapping(date_format="%m/%d/%y"))
    assert any("%y (two-digit year)" in w for w in r["mapping"]["warnings"])


# ---- mapping YAML -----------------------------------------------------------------------------------------------
def test_mapping_yaml_rejects_anchors_aliases_oversize_and_syntax_errors():
    good = (EXAMPLE / "mapping.yaml").read_text()
    assert load_mapping(good).entities
    aliased = good.replace("person:\n", "person: &p\n", 1) + "\nnotes: *p\n"
    with pytest.raises(IngestError, match="anchors or aliases"):
        load_mapping(aliased)
    with pytest.raises(IngestError, match="larger than"):
        load_mapping(good + "\n# " + "x" * 200_001)
    with pytest.raises(IngestError, match="not valid YAML"):
        load_mapping("person: [unclosed\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_mapping(good + "\ndata_type: ehr\n")


# ---- logs never carry data values ---------------------------------------------------------------------------------
def test_redact_error_removes_quoted_values():
    con = duckdb.connect()
    with pytest.raises(duckdb.Error) as info:
        con.execute(f"SELECT CAST('{MARKER}' AS INTEGER)")
    assert MARKER in str(info.value)  # DuckDB quotes the offending value...
    assert MARKER not in redact_error(info.value)  # ...the log line does not
    assert redact_error(info.value).startswith("ConversionException")
    assert MARKER not in redact_error(ValueError(f'duplicate key "{MARKER}, 3"'))


def test_executor_logs_redacted_database_errors(caplog):
    caplog.set_level(logging.DEBUG)
    ex = Executor(duckdb.connect(), min_cell_count=10)
    with pytest.raises(ExecutionError) as info:
        ex._run(f"SELECT CAST('{MARKER}' AS INTEGER)")
    assert MARKER not in str(info.value) and MARKER not in caplog.text
    assert "cohort query failed" in caplog.text and "<redacted>" in caplog.text


def tokens(tmp_path: Path, users: dict[str, list[str]]) -> tuple[Path, dict[str, dict[str, str]]]:
    entries, heads = [], {}
    for name, roles in users.items():
        token, entry = issue_token(name, roles, "default", expires_days=1)
        entries.append(entry)
        heads[name] = {"Authorization": f"Bearer {token}"}
    path = tmp_path / "tokens.yaml"
    path.write_text(yaml.safe_dump({"tokens": entries}))
    return path, heads


def test_api_unhandled_errors_are_logged_without_values(builder, tmp_path, caplog, monkeypatch):
    from cohort_builder.api import create_app

    path, heads = tokens(tmp_path, {"vic": ["viewer"]})
    c = TestClient(create_app(builder, SecurityConfig(tokens_file=path)), raise_server_exceptions=False)

    def boom(*_a, **_k):
        raise RuntimeError(f"failed on value '{MARKER}'")

    monkeypatch.setattr(builder.vocab, "search_concepts", boom)
    caplog.set_level(logging.DEBUG)
    r = c.get("/concepts/search", params={"q": "diabetes"}, headers=heads["vic"])
    assert r.status_code == 500 and r.json()["detail"] == "internal error" and MARKER not in r.text
    assert "unhandled error" in caplog.text and MARKER not in caplog.text


# ---- API request size ---------------------------------------------------------------------------------------------
def test_api_rejects_oversized_bodies_before_authentication(builder, tmp_path):
    from cohort_builder.api import create_app

    path, heads = tokens(tmp_path, {"root": ["admin"]})
    app = create_app(builder, SecurityConfig(tokens_file=path, max_body_bytes=4096))
    c = TestClient(app, raise_server_exceptions=False)
    big = {"files": [{"file_name": "a.csv", "content": "a\n" + "1\n" * 5000}]}
    assert c.post("/datasets/x/ingest", json=big).status_code == 413  # no token: still never buffered
    assert c.post("/datasets/x/ingest", json=big, headers=heads["root"]).status_code == 413

    def chunks():  # no Content-Length: streamed bytes are counted
        for _ in range(10):
            yield b"x" * 1000

    r = c.post("/datasets/x/ingest", content=chunks(), headers={**heads["root"], "content-type": "application/json"})
    assert r.status_code == 413
    small = {"files": [{"file_name": "a.csv", "content": "a\n1\n"}]}
    assert c.post("/datasets/x/ingest", json=small, headers=heads["root"]).status_code == 200
    with pytest.raises(ConfigError):
        SecurityConfig(max_body_bytes=10).check()
    with pytest.raises(ConfigError, match="integer"):
        SecurityConfig.from_env({"CB_API_MAX_BODY_BYTES": "lots"})


# ---- end to end: messy CSV -> ... -> executed, suppressed aggregate result -----------------------------------------
def messy_copy(tmp: Path) -> list[tuple[str | None, Path]]:
    """The synthetic example extract plus rows a real extract contains: duplicates, missing ids, invalid
    dates, unmatched and hyphenated codes. None of the added rows can change cohort membership."""
    out = tmp / "messy"
    out.mkdir()
    extra = {
        "patients": "\nP0001,F,1980",  # exact duplicate person row (same attributes)
        "diagnoses": f"\nD90001,,2020-01-01,E11.9\nD90002,P0001,,E11.9\nD90003,P0002,2020-02-02,{MARKER}",
        "prescriptions": f"\nR90001,P0001,13/45/2020,99999010101,30\nR90002,P0001,08/12/2020,99999-0101-01,30"
        f"\nR90003,P0001,08/12/2020,{MARKER},30\nR00001,P0001,08/12/2020,99999010101,30",  # last: duplicate row
    }
    files = []
    for n in EXAMPLE_FILES:
        text = (EXAMPLE / f"{n}.csv").read_text().rstrip("\n") + extra.get(n, "") + "\n"
        files.append((None, write(out, f"{n}.csv", text)))
    return files


def test_end_to_end_messy_csv_to_suppressed_execution(settings, tmp_path, example_ir_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    local = tmp_path / "datasets"
    files = messy_copy(tmp_path)
    # 1. profiling: structure only
    profiles = [profile_csv(p) for _, p in files]
    assert all(p["ok"] for p in profiles)
    rx = next(p for p in profiles if p["table"] == "prescriptions")
    assert rx["duplicate_rows"] == 1 and any("duplicate" in w for w in rx["warnings"])
    assert dict((c["name"], c["type"]) for c in rx["columns"])["ndc"] == "VARCHAR"

    # 2. staging, then mapping + registration (with a simulated interruption and recovery)
    con = connect(settings.db_path)
    reg = DatasetRegistry(con, REPO_ROOT / "ontology", local)
    staged = reg.ingest("messy", files, "alice")
    assert staged["status"] == "staged"

    def interrupted(self, final, body):
        raise OSError("disk full")

    monkeypatch.setattr(DatasetRegistry, "_write_file", interrupted)
    with pytest.raises(IngestError, match="reconcile"):
        reg.register_mapping("messy", load_mapping((EXAMPLE / "mapping.yaml").read_text()), "alice")
    monkeypatch.undo()
    assert reg.versions("messy")[0]["status"] == "profile_failed" and not reg.list_datasets()[0]["selectable"]
    assert not reg.reconcile("ops", "messy")["problems"]
    v = reg.versions("messy")[0]
    assert v["status"] == "registered" and reg.list_datasets()[0]["selectable"]
    checks = v["validation"]["checks"]
    assert checks["ConditionOccurrence"]["rows"]["missing_patient_id"] == 1
    assert checks["ConditionOccurrence"]["rows"]["missing_start_date"] == 1
    assert checks["ConditionOccurrence"]["concept_mapping"]["distinct_unmatched_codes"] == 1
    drug = checks["DrugExposure"]
    assert drug["rows"]["invalid_start_date"] == 1
    assert drug["concept_mapping"]["unmapped_rows"] == 1  # the unmatched code; the hyphenated NDC maps
    # a repeated import is a new immutable version, never an overwrite
    with pytest.raises(IngestError, match="already exists"):
        reg.ingest("messy", files, "alice")
    con.close()

    # 3. selection, definition, validation, deterministic SQL
    s = dataclasses.replace(settings, dataset="messy", local_datasets_dir=local)
    b = CohortBuilder(s)
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())
    issues, attrition = validate(ir, b.ontology, b.vocab, b.executor)
    assert not [i for i in issues if i.severity == "error"] and attrition is not None
    first, second = b.compiler.compile(ir), b.compiler.compile(ir)
    assert first.cohort_sql == second.cohort_sql and first.sql_hash == second.sql_hash
    assert "sem_messy_v1" in first.cohort_sql and "stg_messy_v1" not in first.cohort_sql  # mapped views only

    # 4. governance + execution: suppressed aggregate output only
    def_id, _ = b.submit_ir(ir, "alice")
    b.review(def_id, "rita", "approved")
    out = b.execute(def_id, "carol")
    assert out["dataset"] == "messy" and out["sql_hash"] == first.sql_hash
    min_cell = b.executor.min_cell
    for step in out["attrition"]:  # small cells never leave the executor
        assert isinstance(step["remaining"], str) or step["remaining"] == 0 or step["remaining"] >= min_cell
    b.con.close()

    # the same cohort on the clean example extract gives the same result: the messy rows changed nothing
    con = connect(settings.db_path)
    reg = DatasetRegistry(con, REPO_ROOT / "ontology", local)
    reg.ingest(
        "clean",
        [(None, EXAMPLE / f"{n}.csv") for n in EXAMPLE_FILES],
        "alice",
        mapping=load_mapping((EXAMPLE / "mapping.yaml").read_text()),
    )
    con.close()
    clean = CohortBuilder(dataclasses.replace(settings, dataset="clean", local_datasets_dir=local))
    def_id, _ = clean.submit_ir(ir, "alice")
    clean.review(def_id, "rita", "approved")
    out_clean = clean.execute(def_id, "carol")
    assert out_clean["person_count"] == out["person_count"] and out_clean["attrition"] == out["attrition"]
    assert out_clean["sql_hash"] != out["sql_hash"]  # different dataset views, same semantics
    audit = json.dumps(clean.store.audit_events(), default=str)
    gens = json.dumps(clean.con.execute("SELECT * FROM meta.cohort_generation").fetchall(), default=str)
    clean.con.close()

    # 5. no synthetic cell value anywhere we write
    blobs = [audit, gens, caplog.text, json.dumps(out, default=str)]
    blobs += [p.read_text() for p in local.glob("*.yaml")]
    assert not any(MARKER in x for x in blobs)
