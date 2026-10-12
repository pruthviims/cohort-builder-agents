"""CSV ingestion: file checks, parsing, type inference, profiling, staging, mapping, registration,
versioning, rollback, dataset selection, API/CLI and privacy. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path

import duckdb
import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.cli import main as cli
from cohort_builder.config import REPO_ROOT, Settings
from cohort_builder.csv_ingest import IngestError, profile_csv, profile_text
from cohort_builder.dataset_mapping import DatasetMapping
from cohort_builder.dataset_registry import DatasetRegistry, Upload, load_mapping
from cohort_builder.db import connect
from cohort_builder.ir import CohortDefinition
from cohort_builder.ontology import Ontology
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import GovernanceError, SecurityConfig, issue_token

EXAMPLE = REPO_ROOT / "examples" / "csv"
EXAMPLE_FILES = ["patients", "enrollment", "diagnoses", "prescriptions", "labs"]
MARKER = "ZZMARKER9876"  # a cell value that must never appear in any report, log or file we write


def write(tmp: Path, name: str, text: str | bytes) -> Path:
    p = tmp / name
    p.write_bytes(text if isinstance(text, bytes) else text.encode())
    return p


def mapping(**over) -> DatasetMapping:
    data = yaml.safe_load((EXAMPLE / "mapping.yaml").read_text())
    data.update(over)
    return DatasetMapping.model_validate(data)


def example_files() -> list[tuple[str | None, Path]]:
    return [(None, EXAMPLE / f"{n}.csv") for n in EXAMPLE_FILES]


@pytest.fixture
def reg(tmp_path):
    con = connect(tmp_path / "db.duckdb")
    r = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    yield r
    con.close()


@pytest.fixture
def demo_reg(settings, tmp_path):
    """Registry on a copy of the demo database (vocabulary present, so source codes map)."""
    con = connect(settings.db_path)
    r = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    yield r, settings
    con.close()


# ---- file checks and parsing ----------------------------------------------------------------------
def test_profile_valid_file_reports_structure_without_values(tmp_path):
    p = write(
        tmp_path,
        "people.csv",
        f"Person ID,Birth Date,Code,Score,Flag,Note\n1,2001-02-03,00123,1.5,true,{MARKER}\n"
        "2,1999-12-31,00456,2,false,x\n3,,00789,,TRUE,\n",
    )
    r = profile_csv(p)
    assert r["ok"] and r["rows"] == 3 and r["table"] == "people"
    cols = {c["name"]: c for c in r["columns"]}
    assert [c["source_name"] for c in r["columns"]][0] == "Person ID"
    # identifiers stay exact text (never numeric), whatever their values look like
    assert cols["person_id"]["type"] == "VARCHAR" and cols["person_id"]["kind"] == "identifier"
    assert "patient_identifier" in cols["person_id"]["flags"]
    assert cols["birth_date"]["type"] == "DATE" and cols["birth_date"]["null_count"] == 1
    assert cols["birth_date"]["null_pct"] == 33.33
    assert cols["code"]["type"] == "VARCHAR"  # leading zeros are kept as text
    assert cols["score"]["type"] == "DOUBLE" and cols["flag"]["type"] == "BOOLEAN"
    assert "person_id" in r["candidate_keys"]
    assert MARKER not in json.dumps(r)


def test_decimals_are_never_truncated_to_integers(tmp_path):
    p = write(tmp_path, "labs.csv", "v,w,x\n7.3,1,0x1F\n8,2,1_000\n")
    cols = {c["name"]: c["type"] for c in profile_csv(p)["columns"]}
    assert cols == {"v": "DOUBLE", "w": "BIGINT", "x": "VARCHAR"}


def test_mixed_types_fall_back_to_text_with_warning(tmp_path):
    p = write(tmp_path, "m.csv", "value\n1\n2\n3\nunknown\n")
    r = profile_csv(p)
    assert r["columns"][0]["type"] == "VARCHAR"
    assert any("mixed values: 3 of 4 look numeric" in w for w in r["warnings"])


def test_missing_values_and_custom_null_tokens(tmp_path):
    p = write(tmp_path, "n.csv", "a,b\n1,NA\n2,\n3,5\n")
    assert profile_csv(p)["columns"][1]["type"] == "VARCHAR"  # 'NA' is a value by default
    r = profile_csv(p, null_values=("NA",))
    assert r["columns"][1]["type"] == "BIGINT" and r["columns"][1]["null_count"] == 2


@pytest.mark.parametrize(
    "name, content, code",
    [
        ("empty.csv", "", "empty_file"),
        ("header.csv", "a,b\n", "no_data_rows"),
        ("dup.csv", "a,A,b\n1,2,3\n", "duplicate_column"),
        ("blank.csv", "a,,b\n1,2,3\n", "empty_column_name"),
        ("bad.csv", "a,b\n1,2\n3\n4,5,6\n", "malformed_row"),
        ("quote.csv", 'a,b\n1,"unterminated\n', "malformed_row"),
        ("data.txt", "a\n1\n", "unsupported_extension"),
    ],
)
def test_invalid_files_are_rejected_with_codes(tmp_path, name, content, code):
    r = profile_csv(write(tmp_path, name, content))
    assert not r["ok"] and code in {p["code"] for p in r["errors"]}


def test_malformed_rows_report_line_numbers_not_values(tmp_path):
    p = write(tmp_path, "bad.csv", f"a,b\n1,2\n3\n{MARKER},5,6\n7,8\n")
    r = profile_csv(p)
    assert [e["line"] for e in r["errors"]] == [3, 4]
    assert MARKER not in json.dumps(r)


def test_invalid_encoding_and_missing_file(tmp_path):
    r = profile_csv(write(tmp_path, "latin.csv", "name\nok\nJos\xe9\n".encode("latin-1")))
    assert r["errors"] == [{"code": "invalid_encoding", "message": "not valid UTF-8", "line": 3}]
    assert profile_csv(tmp_path / "missing.csv")["errors"][0]["code"] == "not_found"


def test_bom_and_tsv_are_supported(tmp_path):
    r = profile_csv(write(tmp_path, "t.tsv", "﻿id\tvalue\n1\t2\n".encode()))
    assert r["ok"] and [c["name"] for c in r["columns"]] == ["id", "value"]


def test_file_size_limit(tmp_path):
    p = write(tmp_path, "big.csv", "a\n" + "1\n" * 100)
    assert profile_csv(p, max_bytes=50)["errors"][0]["code"] == "too_large"


@pytest.mark.parametrize("table", ["Person", "1abc", "a-b", "x;drop", "__meta", "a__b", "t" * 49])
def test_unsafe_table_names_rejected(tmp_path, table):
    p = write(tmp_path, "ok.csv", "a\n1\n")
    assert profile_csv(p, table=table)["errors"][0]["code"] == "invalid_name"


def test_duplicate_rows_and_key_violations(tmp_path):
    p = write(tmp_path, "c.csv", "condition_occurrence_id,patient_id,code\n1,10,A\n1,10,A\n2,11,B\n")
    r = profile_csv(p)
    assert r["duplicate_rows"] == 1
    assert any("candidate key 'condition_occurrence_id' has 1 duplicate value" in w for w in r["warnings"])
    assert not any("patient_id" in w and "candidate key" in w for w in r["warnings"])  # repeats are normal


def test_upload_names_cannot_carry_paths():
    assert profile_text("../../etc/passwd.csv", "a\n1\n")["errors"][0]["code"] == "invalid_file_name"
    assert profile_text("/tmp/x.csv", "a\n1\n")["errors"][0]["code"] == "invalid_file_name"
    assert profile_text("ok.csv", "a\n1\n")["ok"]


# ---- staging, versioning, rollback ---------------------------------------------------------------------
def test_ingest_multiple_files_into_isolated_staging(reg):
    r = reg.ingest("clinic", example_files(), "alice")
    assert r["status"] == "staged" and r["staging_schema"] == "stg_clinic_v1"
    tables = {
        t
        for (t,) in reg.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'stg_clinic_v1'"
        ).fetchall()
    }
    assert tables == set(EXAMPLE_FILES)
    assert reg.con.execute("SELECT count(*) FROM stg_clinic_v1.patients").fetchone()[0] == 60
    row = reg.versions("clinic")[0]
    assert row["status"] == "staged" and row["software_version"] and len(row["files"]) == 5
    assert all(f["sha256"].startswith("sha256:") for f in row["files"])
    assert not (reg.local_dir / "clinic.yaml").exists()  # staged, not selectable yet


def test_repeated_import_needs_new_version(reg):
    reg.ingest("clinic", example_files()[:1], "alice")
    with pytest.raises(IngestError, match="already exists .* --new-version"):
        reg.ingest("clinic", example_files()[:1], "alice")
    r = reg.ingest("clinic", example_files()[:2], "alice", new_version=True)
    assert r["version"] == 2 and r["staging_schema"] == "stg_clinic_v2"
    assert reg.con.execute("SELECT count(*) FROM stg_clinic_v1.patients").fetchone()[0] == 60  # v1 untouched


@pytest.mark.parametrize("name", ["omop_demo", "iqvia_laad", "meta", "cdm", "csv", "Bad", "a__b"])
def test_reserved_and_invalid_dataset_names(reg, name):
    with pytest.raises(IngestError):
        reg.ingest(name, example_files()[:1], "alice")


def test_failed_ingestion_rolls_back_everything(reg, tmp_path):
    bad = write(tmp_path, "bad.csv", "a,b\n1,2\n3\n")
    with pytest.raises(IngestError, match="malformed"):
        reg.ingest("clinic", [*example_files(), ("bad", bad)], "alice")
    schemas = {s for (s,) in reg.con.execute("SELECT schema_name FROM information_schema.schemata").fetchall()}
    assert "stg_clinic_v1" not in schemas
    [row] = reg.versions("clinic")
    assert row["status"] == "failed" and row["errors"][0]["line"] == 3 and row["staging_schema"] is None
    assert reg.list_datasets()[0]["status"] == "failed" and not list(reg.local_dir.glob("*.yaml"))
    r = reg.ingest("clinic", example_files(), "alice")  # a failed attempt does not block a clean import
    assert r["status"] == "staged"


def test_dry_run_keeps_nothing(reg):
    r = reg.ingest("clinic", example_files(), "alice", mapping=mapping(), dry_run=True)
    assert r["status"] == "dry_run" and r["mapping"]["valid"]
    assert reg.versions("clinic") == [] and not list(reg.local_dir.glob("*.yaml"))
    assert "stg_clinic_v1" not in {
        s for (s,) in reg.con.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
    }


def test_duplicate_table_names_in_one_session(reg):
    with pytest.raises(IngestError, match="used for two files"):
        reg.ingest("clinic", [("people", EXAMPLE / "patients.csv"), ("people", EXAMPLE / "labs.csv")], "alice")


# ---- mapping validation ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change, message",
    [
        ({"person": {"table": "patients", "person_column": "nope"}}, "column 'nope' does not exist"),
        ({"person": {"table": "missing", "person_column": "patient_id"}}, "table 'missing' was not imported"),
        (
            {
                "observation": {
                    "mode": "table",
                    "absence_inference": "weak",
                    "table": "enrollment",
                    "person_column": "patient_id",
                    "start_column": "patient_id",
                    "end_column": "end_date",
                }
            },
            "is VARCHAR, not an ISO date",
        ),
    ],
)
def test_mapping_errors_against_staged_schema(reg, change, message):
    with pytest.raises(IngestError, match=message):
        reg.ingest("clinic", example_files(), "alice", mapping=mapping(**change))
    assert reg.versions("clinic")[0]["status"] == "failed"


def test_mapping_model_rejects_ambiguous_or_unsafe_definitions():
    base = yaml.safe_load((EXAMPLE / "mapping.yaml").read_text())
    with pytest.raises(ValueError, match="exactly one of concept_id_column or code_column"):
        DatasetMapping.model_validate(
            {
                **base,
                "entities": {
                    "ConditionOccurrence": {
                        "table": "diagnoses",
                        "person_column": "patient_id",
                        "start_column": "dx_date",
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="max_gap_days"):
        DatasetMapping.model_validate({**base, "observation": {"mode": "activity", "absence_inference": "weak"}})
    with pytest.raises(ValueError):  # absence semantics must be stated, never assumed
        DatasetMapping.model_validate({**base, "observation": {"mode": "none"}})
    with pytest.raises(ValueError, match="not a gender concept"):
        DatasetMapping.model_validate({**base, "person": {**base["person"], "gender_values": {"M": 1}}})
    with pytest.raises(ValueError, match="unsupported source value"):
        DatasetMapping.model_validate({**base, "person": {**base["person"], "gender_values": {"M'; DROP": 8507}}})
    with pytest.raises(ValueError, match="Input should be"):
        DatasetMapping.model_validate({**base, "entities": {"Diagnosis": base["entities"]["ConditionOccurrence"]}})
    with pytest.raises(ValueError, match="unsupported date_format"):
        DatasetMapping.model_validate(
            {**base, "entities": {"DrugExposure": {**base["entities"]["DrugExposure"], "date_format": "%Y'); DROP"}}}
        )
    with pytest.raises(ValueError, match="duplicate key"):
        load_mapping((EXAMPLE / "mapping.yaml").read_text() + "\nperson: {table: x, person_column: y}\n")


def test_measurement_requires_numeric_values(reg):
    m = yaml.safe_load((EXAMPLE / "mapping.yaml").read_text())
    m["entities"]["Measurement"]["value_column"] = "result_unit"
    with pytest.raises(IngestError, match="result_unit is VARCHAR, expected BIGINT or DOUBLE"):
        reg.ingest("clinic", example_files(), "alice", mapping=DatasetMapping.model_validate(m))
    del m["entities"]["Measurement"]["value_column"]
    with pytest.raises(IngestError, match="needs value_column"):
        reg.ingest("clinic", example_files(), "alice", mapping=DatasetMapping.model_validate(m), new_version=True)


def test_data_checks_reject_conflicting_people_and_unusable_entities(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,sex,yob\n1,M,1970\n1,F,1971\n2,F,1980\n")
    events = write(tmp_path, "events.csv", "pid,day,condition_concept_id\n1,1/2/20x,201826\n2,,201826\n")
    m = DatasetMapping.model_validate(
        {
            "person": {
                "table": "people",
                "person_column": "pid",
                "gender_column": "sex",
                "gender_values": {"M": 8507, "F": 8532},
                "year_of_birth_column": "yob",
            },
            "observation": {"mode": "activity", "absence_inference": "weak", "max_gap_days": 365},
            "entities": {
                "ConditionOccurrence": {
                    "table": "events",
                    "person_column": "pid",
                    "start_column": "day",
                    "date_format": "%Y-%m-%d",
                    "concept_id_column": "condition_concept_id",
                }
            },
        }
    )
    with pytest.raises(IngestError) as info:
        reg.ingest("tiny", [(None, people), (None, events)], "alice", mapping=m)
    msgs = " ".join(p["message"] for p in info.value.problems)
    assert "conflicting gender or birth year" in msgs and "no usable rows" in msgs


# ---- registration, profiles, selection ------------------------------------------------------------------
def test_register_creates_valid_profiles_and_semantic_views(demo_reg):
    reg, _ = demo_reg
    r = reg.ingest("clinic", example_files(), "alice", mapping=mapping())
    assert r["status"] == "registered" and r["profile"] == "clinic"
    checks = r["mapping"]["checks"]
    assert checks["ConditionOccurrence"]["concept_mapping"]["unmapped_rows"] == 0 and checks["person"]["persons"] == 60
    assert r["mapping"]["surrogate_person_ids"] is True
    latest = reg.local_dir / "clinic.yaml"
    pinned = reg.local_dir / "clinic__v1.yaml"
    assert latest.is_file() and pinned.is_file()
    ont = Ontology.load(REPO_ROOT / "ontology", "clinic", [reg.local_dir])
    assert ont.capabilities["entities"] == ["ConditionOccurrence", "DrugExposure", "Measurement"]
    assert ont.capabilities["absence_inference"] == "observed_period"
    assert "Not available in this dataset: ProcedureOccurrence, VisitOccurrence." in ont.capabilities["notes"]
    assert yaml.safe_load(pinned.read_text())["name"] == "clinic__v1"
    for name in ("omop_demo", "iqvia_laad"):  # built-in profiles are unaffected
        assert Ontology.load(REPO_ROOT / "ontology", name, [reg.local_dir]).dataset_name == name


def test_stage_then_map_and_mappings_are_immutable(demo_reg):
    reg, _ = demo_reg
    reg.ingest("clinic", example_files(), "alice")
    out = reg.register_mapping("clinic", mapping(), "alice")
    assert out["status"] == "registered" and out["version"] == 1
    assert reg.register_mapping("clinic", mapping(), "alice")["unchanged"]
    with pytest.raises(IngestError, match="already registered with a different mapping"):
        reg.register_mapping("clinic", mapping(description="changed"), "alice")


def test_registered_dataset_drives_cohort_validation(demo_reg, example_ir_path):
    reg, settings = demo_reg
    reg.ingest("clinic", example_files(), "alice", mapping=mapping())
    reg.con.close()
    s = dataclasses.replace(settings, dataset="clinic", local_datasets_dir=reg.local_dir)
    b = CohortBuilder(s)
    b.executor.min_cell = 1
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())
    from cohort_builder.agents.validator import validate

    issues, attrition = validate(ir, b.ontology, b.vocab, b.executor)
    assert attrition is not None and attrition.rules[0]["remaining"] == 20
    assert not [i for i in issues if i.severity == "error"]
    assert b.data_snapshot().startswith("csv:clinic v1 (sha256:")
    # an entity the import does not provide is refused with an actionable message
    data = ir.model_dump(mode="json")
    data["concept_sets"].append(
        {"id": "cs_proc", "name": "proc", "domain": "Procedure", "items": [{"concept_id": 4000000}]}
    )
    data["inclusion"].append(
        {"id": "p1", "name": "procedure", "entity": "ProcedureOccurrence", "concept_set_id": "cs_proc"}
    )
    issues, _ = validate(CohortDefinition.model_validate(data), b.ontology, b.vocab)
    assert any(i.severity == "error" and "ProcedureOccurrence" in i.message and "clinic" in i.message for i in issues)
    # approval safeguards still apply to imported data
    def_id, _ = b.submit_ir(ir, "alice")
    with pytest.raises(GovernanceError):
        b.execute(def_id, "carol")
    b.con.close()


def test_profile_pointing_to_missing_tables_fails_clearly(demo_reg, tmp_path):
    reg, settings = demo_reg
    reg.ingest("clinic", example_files(), "alice", mapping=mapping())
    reg.con.close()
    other = tmp_path / "other.duckdb"
    shutil.copy(settings.db_path, other)
    con = duckdb.connect(str(other))
    con.execute("DROP SCHEMA stg_clinic_v1 CASCADE")
    con.close()
    s = dataclasses.replace(settings, db_path=other, dataset="clinic", local_datasets_dir=reg.local_dir)
    with pytest.raises(ValueError, match="Was it ingested into this database"):
        CohortBuilder(s)


def test_unknown_dataset_selection_lists_imports(tmp_path):
    (tmp_path / "x.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match="unknown dataset 'nope'.*'x'"):
        Ontology.load(REPO_ROOT / "ontology", "nope", [tmp_path])
    with pytest.raises(ValueError, match="invalid dataset name"):
        Ontology.load(REPO_ROOT / "ontology", "../etc/passwd", [tmp_path])


def test_observation_none_is_refused_by_the_validator(demo_reg, example_ir_path):
    reg, settings = demo_reg
    m = mapping(observation={"mode": "none", "absence_inference": "weak"})
    r = reg.ingest("noobs", example_files(), "alice", mapping=m)
    assert any("observation mode 'none'" in w for w in r["mapping"]["warnings"])
    reg.con.close()
    b = CohortBuilder(dataclasses.replace(settings, dataset="noobs", local_datasets_dir=reg.local_dir))
    from cohort_builder.agents.validator import validate

    issues, _ = validate(CohortDefinition.model_validate_json(example_ir_path.read_text()), b.ontology, b.vocab)
    assert any(i.stage == "dataset" and "no observation periods" in i.message for i in issues)
    with pytest.raises(ValueError, match="defines no observation periods"):
        from cohort_builder.compiler import Compiler

        Compiler(b.ontology).compile(CohortDefinition.model_validate_json(example_ir_path.read_text()))
    b.con.close()


# ---- privacy ---------------------------------------------------------------------------------------------
def test_no_cell_values_in_reports_audit_profiles_or_logs(reg, tmp_path, caplog):
    people = write(tmp_path, "people.csv", f"pid,sex,yob,note\n1,M,1970,{MARKER}\n2,F,1980,x\n")
    events = write(tmp_path, "events.csv", f"pid,day,concept\n1,2020-01-01,201826\n2,2020-02-01,{MARKER}\n")
    m = DatasetMapping.model_validate(
        {
            "person": {
                "table": "people",
                "person_column": "pid",
                "gender_column": "sex",
                "gender_values": {"M": 8507, "F": 8532},
                "year_of_birth_column": "yob",
            },
            "observation": {"mode": "activity", "absence_inference": "weak", "max_gap_days": 365},
            "entities": {
                "ConditionOccurrence": {
                    "table": "events",
                    "person_column": "pid",
                    "start_column": "day",
                    "concept_id_column": "concept",
                }
            },
        }
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(IngestError) as info:  # concept column is text because of the secret value
        reg.ingest("tiny", [(None, people), (None, events)], "alice", mapping=m)
    r = reg.ingest("tiny2", [(None, people)], "alice")
    blobs = [
        json.dumps(info.value.problems),
        str(info.value),
        json.dumps(r, default=str),
        caplog.text,
        json.dumps(reg.store.audit_events(), default=str),
        json.dumps(reg.versions("tiny"), default=str),
        json.dumps(reg.info("tiny2"), default=str),
    ]
    blobs += [p.read_text() for p in reg.local_dir.glob("*.yaml")] if reg.local_dir.exists() else []
    assert not any(MARKER in b for b in blobs)


# ---- API -------------------------------------------------------------------------------------------------
def test_api_dataset_endpoints(builder, tmp_path):
    builder.settings = dataclasses.replace(builder.settings, local_datasets_dir=tmp_path / "datasets")
    entries, toks = [], {}
    for name, roles in {"alice": ["author"], "victor": ["viewer"], "root": ["admin"]}.items():
        token, entry = issue_token(name, roles, "default", expires_days=1)
        entries.append(entry)
        toks[name] = {"Authorization": f"Bearer {token}"}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    c = TestClient(create_app_for(builder, tmp_path / "t.yaml"), raise_server_exceptions=False)
    files = [{"file_name": f"{n}.csv", "content": (EXAMPLE / f"{n}.csv").read_text()} for n in EXAMPLE_FILES]
    body = {"files": files, "mapping_yaml": (EXAMPLE / "mapping.yaml").read_text()}
    assert c.post("/datasets/clinic/ingest", json=body).status_code == 401
    assert c.post("/datasets/clinic/ingest", json=body, headers=toks["alice"]).status_code == 403
    traversal = {"files": [{"file_name": "../x.csv", "content": "a\n1\n"}]}
    r = c.post("/datasets/clinic/ingest", json=traversal, headers=toks["root"])
    assert r.status_code == 422 and r.json()["detail"]["problems"][0]["code"] == "invalid_file_name"
    assert (
        c.post(
            "/datasets/clinic/ingest",
            json={"files": [{"file_name": "x.csv", "content": "a\n1\n", "path": "/etc/passwd"}]},
            headers=toks["root"],
        ).status_code
        == 422
    )  # no server paths accepted
    ok = c.post("/datasets/clinic/ingest", json=body, headers=toks["root"])
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "registered"
    listing = c.get("/datasets", headers=toks["victor"]).json()
    assert listing["imported"][0]["dataset"] == "clinic" and "omop_demo" in listing["built_in"]
    info = c.get("/datasets/clinic", headers=toks["victor"]).json()
    assert info["capabilities"]["entities"] == ["ConditionOccurrence", "DrugExposure", "Measurement"]
    v = ok.json()["version"]  # failed attempts keep their own version numbers
    assert v == 2 and c.get("/datasets/clinic/versions/1", headers=toks["victor"]).json()["status"] == "failed"
    assert c.get(f"/datasets/clinic/versions/{v}", headers=toks["victor"]).json()["status"] == "registered"
    assert c.get("/datasets/nope", headers=toks["victor"]).status_code == 404
    prof = c.post("/datasets/csv/profile", json={"file_name": "p.csv", "content": "a,b\n1,2\n"}, headers=toks["alice"])
    assert prof.status_code == 200 and prof.json()["rows"] == 1
    assert (
        c.post(
            "/datasets/csv/profile", json={"file_name": "p.csv", "content": "a\n1\n"}, headers=toks["victor"]
        ).status_code
        == 403
    )
    again = c.post("/datasets/clinic/ingest", json=body, headers=toks["root"])
    assert again.status_code == 422 and "already exists" in again.json()["detail"]["message"]
    actions = {(e["action"], e["outcome"]) for e in builder.store.audit_events()}
    assert ("dataset.ingest", "success") in actions and ("dataset.ingest", "failed") in actions


def create_app_for(builder, tokens_file):
    from cohort_builder.api import create_app

    return create_app(builder, SecurityConfig(tokens_file=tokens_file))


# ---- CLI -------------------------------------------------------------------------------------------------
def test_cli_profile_ingest_info_and_selection(settings, tmp_path, monkeypatch, capsys, example_ir_path):
    monkeypatch.setenv("CB_DB_PATH", str(settings.db_path))
    monkeypatch.setenv("CB_LOCAL_DATASETS_DIR", str(tmp_path / "datasets"))
    assert cli(["csv", "profile", str(EXAMPLE / "labs.csv")]) == 0
    assert "result_value" in capsys.readouterr().out
    assert cli(["csv", "profile", str(write(tmp_path, "bad.csv", "a\n1,2\n"))]) == 1
    args = ["csv", "ingest", "--dataset", "clinic", "--mapping", str(EXAMPLE / "mapping.yaml")]
    for n in EXAMPLE_FILES:
        args += ["--file", f"{n}={EXAMPLE / (n + '.csv')}"]
    assert cli(args + ["--dry-run"]) == 0
    assert "nothing kept" in capsys.readouterr().out
    assert cli(args) == 0
    assert "cohort-builder --dataset clinic" in capsys.readouterr().out
    assert cli(args) == 1
    assert "--new-version" in capsys.readouterr().err
    assert cli(["csv", "info", "clinic"]) == 0
    assert json.loads(capsys.readouterr().out)["versions"][0]["status"] == "registered"
    assert cli(["datasets"]) == 0
    assert "clinic (vcsv-1" in capsys.readouterr().out
    assert cli(["--dataset", "clinic", "dry-run", str(example_ir_path)]) == 0
    assert "Persons with a qualifying index event" in capsys.readouterr().out
    assert cli(["csv", "info", "nope"]) == 1


def test_settings_default_local_dir_is_git_ignored():
    assert Settings().local_datasets_dir == REPO_ROOT / "data" / "datasets"
    assert "data/*" in (REPO_ROOT / ".gitignore").read_text()


def test_upload_and_path_sources_give_the_same_profile(reg):
    a = reg.ingest("from_path", [(None, EXAMPLE / "labs.csv")], "alice")
    raw = (EXAMPLE / "labs.csv").read_bytes().decode("utf-8")  # keep line endings: same bytes, same hash
    b = reg.ingest("from_text", [(None, Upload("labs.csv", raw))], "alice")
    assert a["files"][0]["columns"] == b["files"][0]["columns"] and a["files"][0]["sha256"] == b["files"][0]["sha256"]
