"""CSV ingestion integrity: exact type inference, data-quality metrics, vocabulary-aware code
normalization, registration states / profile consistency, reconciliation, version allocation, and an
end-to-end path to cohort validation. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import duckdb
import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.agents.validator import validate
from cohort_builder.cli import main as cli
from cohort_builder.config import REPO_ROOT
from cohort_builder.csv_ingest import IngestError, profile_csv, profile_text
from cohort_builder.dataset_mapping import DatasetMapping, normalize_sql
from cohort_builder.dataset_registry import (
    DatasetRegistry,
    Upload,
    load_mapping,
    parse_type_overrides,
    verify_registered_profile,
)
from cohort_builder.db import connect, init_schemas
from cohort_builder.ir import CohortDefinition
from cohort_builder.metadata import now
from cohort_builder.ontology import Ontology
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import SecurityConfig, issue_token

EXAMPLE = REPO_ROOT / "examples" / "csv"
EXAMPLE_FILES = ["patients", "enrollment", "diagnoses", "prescriptions", "labs"]
MARKER = "ZZMARKER9876"  # a cell value that must never appear in any report, log or file we write
BIG_ID = "98765432109876543210123"  # 23 digits: beyond BIGINT and beyond exact DOUBLE


def write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_bytes(text.encode())
    return p


def example_mapping(**over) -> DatasetMapping:
    data = yaml.safe_load((EXAMPLE / "mapping.yaml").read_text())
    data.update(over)
    return DatasetMapping.model_validate(data)


def example_files() -> list[tuple[str | None, Path]]:
    return [(None, EXAMPLE / f"{n}.csv") for n in EXAMPLE_FILES]


def cols(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["columns"]}


@pytest.fixture
def reg(tmp_path):
    con = connect(tmp_path / "db.duckdb")
    r = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    yield r
    con.close()


@pytest.fixture
def demo_reg(settings, tmp_path):
    con = connect(settings.db_path)
    r = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    yield r, settings
    con.close()


def code_mapping(table: str = "events", vocabulary: str = "ICD10CM", **entity) -> DatasetMapping:
    return DatasetMapping.model_validate(
        {
            "person": {"table": "people", "person_column": "pid", "year_of_birth_column": "yob"},
            "observation": {"mode": "activity", "absence_inference": "weak", "max_gap_days": 365},
            "entities": {
                "ConditionOccurrence": {
                    "table": table,
                    "person_column": "pid",
                    "start_column": "day",
                    "code_column": "dx_code",
                    "vocabulary": vocabulary,
                    **entity,
                }
            },
        }
    )


def drug_mapping(**entity) -> DatasetMapping:
    return DatasetMapping.model_validate(
        {
            "person": {"table": "people", "person_column": "pid", "year_of_birth_column": "yob"},
            "observation": {"mode": "activity", "absence_inference": "weak", "max_gap_days": 365},
            "entities": {
                "DrugExposure": {
                    "table": "fills",
                    "person_column": "pid",
                    "start_column": "day",
                    "code_column": "ndc",
                    "vocabulary": "NDC",
                    **entity,
                }
            },
        }
    )


# ---- 1. type inference: exact values, never lossy -----------------------------------------------------
def test_very_large_integer_ids_stay_exact_text(tmp_path):
    # 'value' carries no identifier hint: the digit count alone keeps it as text
    r = profile_csv(write(tmp_path, "t.csv", f"value,small\n{BIG_ID},1\n12,2\n"))
    c = cols(r)
    assert c["value"]["type"] == "VARCHAR" and c["value"]["kind"] == "identifier"
    assert c["small"]["type"] == "BIGINT"
    # the 18-digit boundary still fits BIGINT exactly
    r = profile_csv(write(tmp_path, "u.csv", "value\n999999999999999999\n-5\n"))
    assert cols(r)["value"]["type"] == "BIGINT"
    r = profile_csv(write(tmp_path, "v.csv", "value\n9999999999999999999\n"))  # 19 digits
    assert cols(r)["value"]["type"] == "VARCHAR"


def test_leading_zeros_are_preserved(tmp_path):
    r = profile_csv(write(tmp_path, "t.csv", "x,y\n00123,0\n45,10\n"))
    c = cols(r)
    assert c["x"]["type"] == "VARCHAR" and "leading zeros" in " ".join(c["x"]["notes"])
    assert c["y"]["type"] == "BIGINT"  # a lone '0' is not a leading zero


def test_long_numeric_codes_and_identifier_names_stay_text(tmp_path):
    text = "member_id,npi,ndc,dx_code,condition_concept_id,qty\n1,1234567893,99999010101,250.00,201826,30\n"
    c = cols(profile_csv(write(tmp_path, "t.csv", text)))
    assert (c["member_id"]["type"], c["member_id"]["kind"]) == ("VARCHAR", "identifier")
    assert (c["npi"]["type"], c["npi"]["kind"]) == ("VARCHAR", "identifier")
    assert (c["ndc"]["type"], c["ndc"]["kind"]) == ("VARCHAR", "code")
    assert (c["dx_code"]["type"], c["dx_code"]["kind"]) == ("VARCHAR", "code")  # '250.00' keeps its zeros
    assert (c["condition_concept_id"]["type"], c["condition_concept_id"]["kind"]) == ("BIGINT", "concept_id")
    assert (c["qty"]["type"], c["qty"]["kind"]) == ("BIGINT", "quantity")


def test_decimal_measurements_and_precision_limit(tmp_path):
    c = cols(profile_csv(write(tmp_path, "t.csv", "result,precise\n7.25,0.1234567890123456789\n1e3,1\n")))
    assert (c["result"]["type"], c["result"]["kind"]) == ("DOUBLE", "measurement")
    assert c["precise"]["type"] == "VARCHAR"  # > 15 significant digits would not survive a DOUBLE


def test_mixed_type_columns_fall_back_to_text_with_warning(tmp_path):
    r = profile_csv(write(tmp_path, "t.csv", "v,d\n1,2020-01-01\n2,2020-02-30\nabc,2020-03-01\n"))
    c = cols(r)
    assert c["v"]["type"] == "VARCHAR" and any("mixed values" in n for n in c["v"]["notes"])
    assert c["d"]["type"] == "VARCHAR"  # one impossible date: never coerced


def test_type_overrides_are_validated(tmp_path):
    p = write(tmp_path, "t.csv", f"member_id,code\n1,007\n2,{MARKER}\n")
    ok = profile_csv(write(tmp_path, "u.csv", "member_id\n1\n2\n"), column_types={"member_id": "BIGINT"})
    c = cols(ok)["member_id"]
    assert c["type"] == "BIGINT" and c["type_source"] == "override" and c["kind"] == "identifier"
    bad = profile_csv(p, column_types={"code": "BIGINT"})
    assert not bad["ok"] and bad["errors"][0]["code"] == "invalid_override"
    assert MARKER not in json.dumps(bad) and "007" not in json.dumps(bad)
    assert profile_csv(p, column_types={"nope": "VARCHAR"})["errors"][0]["code"] == "invalid_override"
    assert profile_csv(p, column_types={"code": "MONEY"})["errors"][0]["code"] == "invalid_override"
    assert parse_type_overrides(["claims.member_id=varchar"]) == {"claims": {"member_id": "VARCHAR"}}
    with pytest.raises(IngestError):
        parse_type_overrides(["member_id=VARCHAR"])


def test_registry_overrides_unknown_table_rolls_back(reg, tmp_path):
    p = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    with pytest.raises(IngestError, match="not imported"):
        reg.ingest("x", [(None, p)], "alice", column_types={"other": {"pid": "VARCHAR"}})
    assert reg.versions("x")[0]["status"] == "failed"
    assert not reg.con.execute(
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name LIKE 'stg_x_%'"
    ).fetchone()[0]


def test_exact_values_survive_ingestion_and_mapping(reg, tmp_path):
    reg.con.execute(
        "INSERT INTO vocab.concept VALUES (201826, 'T2DM', 'Condition', 'SNOMED', 'Clinical Finding', 'S', "
        "'44054006', NULL, NULL, NULL), (900001, 'E11.9 src', 'Condition', 'ICD10CM', '5-char', NULL, 'E11.9', "
        "NULL, NULL, NULL)"
    )
    reg.con.execute("INSERT INTO vocab.concept_relationship VALUES (900001, 201826, 'Maps to', NULL, NULL, NULL)")
    people = write(tmp_path, "people.csv", f"pid,yob\n000123,1970\n{BIG_ID},1980\n")
    events = write(tmp_path, "events.csv", f"pid,day,dx_code\n000123,2020-01-01,E11.9\n{BIG_ID},2020-02-01, e119 \n")
    r = reg.ingest("exact", [(None, people), (None, events)], "alice", mapping=code_mapping())
    assert r["status"] == "registered"
    staged = reg.con.execute("SELECT pid, _source_row FROM stg_exact_v1.people ORDER BY _source_row").fetchall()
    assert staged == [("000123", 2), (BIG_ID, 3)]
    rows = reg.con.execute(
        "SELECT p.source_id, v.concept_id, v.source_value, v.source_code_normalized, v.source_table, v.source_row "
        "FROM sem_exact_v1.condition_occurrence v JOIN stg_exact_v1.person_map p USING (person_id) "
        "ORDER BY v.source_row"
    ).fetchall()
    assert rows == [
        ("000123", 201826, "E11.9", "E119", "events", 2),
        (BIG_ID, 201826, " e119 ", "E119", "events", 3),  # source value untouched; lookup key normalized
    ]
    checks = r["mapping"]["checks"]["ConditionOccurrence"]["concept_mapping"]
    assert checks["codes_with_several_spellings"] == 1 and checks["coverage_pct"] == 100.0


# ---- 2. data-quality metrics -------------------------------------------------------------------------------
def test_dropped_rows_are_reported_for_code_mappings(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n2,1980\n")
    events = write(
        tmp_path,
        "events.csv",
        "pid,day,dx_code\n"
        "1,01/02/2020,E11.9\n"  # usable, unmatched (no vocabulary loaded)
        ",01/03/2020,E11.9\n"  # missing patient id
        "2,,E11.9\n"  # missing date
        f"2,13/45/2020,{MARKER}\n"  # invalid date
        f"9,01/05/2020,{MARKER}\n",  # usable, orphan patient, unmatched
    )
    r = reg.ingest("dq", [(None, people), (None, events)], "alice", mapping=code_mapping(date_format="%m/%d/%Y"))
    c = r["mapping"]["checks"]["ConditionOccurrence"]
    assert c["rows"] == {
        "staged_rows": 5,
        "missing_patient_id": 1,
        "missing_start_date": 1,
        "invalid_start_date": 1,
        "usable_rows": 2,
        "invalid_end_date": 0,
        "end_before_start": 0,
        "start_date_out_of_range": 0,
    }
    rows = c["rows"]
    assert rows["staged_rows"] == sum(
        rows[k] for k in ("missing_patient_id", "missing_start_date", "invalid_start_date", "usable_rows")
    )
    cm = c["concept_mapping"]
    assert (cm["mapped_rows"], cm["unmapped_rows"], cm["distinct_codes"], cm["distinct_unmatched_codes"]) == (
        0,
        2,
        2,
        2,
    )
    assert cm["coverage_pct"] == 0.0 and c["persons_missing_from_person_table"] == 1
    w = " | ".join(r["mapping"]["warnings"])
    assert "3 of 5 row(s) excluded (1 without patient id, 1 without start date, 1 with an invalid start date)" in w
    assert "2 distinct source code(s) have no match" in w and "vocabulary ICD10CM is not loaded" in w
    assert "1 patient(s) are missing from the person table" in w
    summary = r["mapping"]["summary"]
    assert summary["entities_without_concepts"] == ["ConditionOccurrence"]
    assert any("will find nobody" in x for x in summary["limitations"])
    assert r["mapping"]["errors"] == [] and r["mapping"]["vocabularies"] == {"ICD10CM": None}
    blob = json.dumps(r, default=str) + json.dumps(reg.store.audit_events(), default=str)
    blob += "".join(p.read_text() for p in reg.local_dir.glob("*.yaml"))
    assert MARKER not in blob


def test_concept_id_mapping_reports_unknown_and_non_standard_ids(reg, tmp_path):
    reg.con.execute(
        "INSERT INTO vocab.concept VALUES (201826, 'T2DM', 'Condition', 'SNOMED', 'Clinical Finding', 'S', "
        "'44054006', NULL, NULL, NULL), (900001, 'src', 'Condition', 'ICD10CM', '5-char', NULL, 'E11.9', NULL, "
        "NULL, NULL)"
    )
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    events = write(
        tmp_path,
        "events.csv",
        "pid,day,condition_concept_id\n1,2020-01-01,201826\n1,2020-01-02,123\n1,2020-01-03,900001\n1,2020-01-04,\n",
    )
    m = code_mapping().model_dump()
    e = m["entities"]["ConditionOccurrence"]
    e.update(code_column=None, vocabulary=None, code_normalization=None, concept_id_column="condition_concept_id")
    r = reg.ingest("cid", [(None, people), (None, events)], "alice", mapping=DatasetMapping.model_validate(m))
    cm = r["mapping"]["checks"]["ConditionOccurrence"]["concept_mapping"]
    assert cm["method"] == "concept_id" and cm["unknown_concept_id_rows"] == 1 and cm["missing_concept_rows"] == 1
    assert cm["non_standard_concept_rows"] == 1 and cm["mapped_rows"] == 2 and cm["unmapped_rows"] == 2
    assert any("not in the vocabulary" in w for w in r["mapping"]["warnings"])


def test_entity_with_no_usable_rows_is_a_hard_error(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    events = write(tmp_path, "events.csv", "pid,day,dx_code\n,2020-01-01,E11.9\n")
    with pytest.raises(IngestError) as info:
        reg.ingest("bad", [(None, people), (None, events)], "alice", mapping=code_mapping())
    assert info.value.problems[0]["code"] == "invalid_data"
    assert "1 without patient id" in info.value.problems[0]["message"]
    assert reg.versions("bad")[0]["status"] == "failed"


# ---- 3. vocabulary-aware normalization ---------------------------------------------------------------------
def test_hyphenated_ndc_formats_map_and_ambiguous_formats_do_not(demo_reg, tmp_path):
    reg, _ = demo_reg
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    fills = write(
        tmp_path,
        "fills.csv",
        "pid,day,ndc\n1,2020-01-01,99999010101\n1,2020-01-02,99999-0101-01\n1,2020-01-03,99999-101-01\n"
        "1,2020-01-04,9999901010\n1,2020-01-05,99999-0101-1\n",
    )
    r = reg.ingest("ndc", [(None, people), (None, fills)], "alice", mapping=drug_mapping())
    cm = r["mapping"]["checks"]["DrugExposure"]["concept_mapping"]
    assert cm["normalization"] == "ndc11" and cm["unrecognized_format_rows"] == 1  # 10 digits: ambiguous
    rows = reg.con.execute(
        "SELECT source_value, source_code_normalized, concept_id > 0 FROM sem_ndc_v1.drug_exposure ORDER BY source_row"
    ).fetchall()
    assert rows[0] == ("99999010101", "99999010101", True)
    assert rows[1] == ("99999-0101-01", "99999010101", True)
    assert rows[2] == ("99999-101-01", "99999010101", True)  # 5-3-2 padded in the product segment
    assert rows[3] == ("9999901010", None, False)
    assert rows[4] == ("99999-0101-1", "99999010101", True)  # 5-4-1 padded in the package segment
    assert r["mapping"]["code_normalization"]["DrugExposure"]["strategy"] == "ndc11"
    assert r["mapping"]["vocabularies"]["NDC"]  # vocabulary version recorded


def test_ndc_padding_rules():
    con = duckdb.connect()
    sql = f"SELECT {normalize_sql('x', 'ndc11')} FROM (SELECT ?::VARCHAR AS x)"
    assert con.execute(sql, ["1234-5678-90"]).fetchone()[0] == "01234567890"
    assert con.execute(sql, ["12345-678-90"]).fetchone()[0] == "12345067890"
    assert con.execute(sql, ["12345-6789-0"]).fetchone()[0] == "12345678900"
    for bad in ("1234567890", "12345-67890-1", "abc", "", "1234-567-89"):
        assert con.execute(sql, [bad]).fetchone()[0] is None
    with pytest.raises(ValueError):
        normalize_sql("x", "fuzzy")


def test_normalization_never_merges_distinct_vocabulary_codes(reg, tmp_path):
    reg.con.execute(
        "INSERT INTO vocab.concept VALUES "
        "(1, 'a', 'Condition', 'TESTVOC', 'c', 'S', '12.34', NULL, NULL, NULL), "
        "(2, 'b', 'Condition', 'TESTVOC', 'c', 'S', '123.4', NULL, NULL, NULL), "
        "(3, 'c', 'Condition', 'TESTVOC', 'c', 'S', 'X1', NULL, NULL, NULL)"
    )
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    events = write(tmp_path, "events.csv", "pid,day,dx_code\n1,2020-01-01,12.34\n1,2020-01-02,1234\n1,2020-01-03,x1\n")
    m = code_mapping(vocabulary="TESTVOC", code_normalization="strip_dots")
    r = reg.ingest("coll", [(None, people), (None, events)], "alice", mapping=m)
    got = reg.con.execute("SELECT source_value, concept_id FROM sem_coll_v1.condition_occurrence ORDER BY source_row")
    assert got.fetchall() == [("12.34", 1), ("1234", 0), ("x1", 3)]  # the ambiguous key matches exact spellings only
    cm = r["mapping"]["checks"]["ConditionOccurrence"]["concept_mapping"]
    assert cm["ambiguous_key_rows"] == 2 and cm["multi_concept_rows"] == 0
    assert any("shared by several vocabulary codes" in w for w in r["mapping"]["warnings"])
    # 'exact' compares the stored string as-is (case-sensitive): 'x1' does not match 'X1'
    r2 = reg.ingest(
        "coll",
        [(None, people), (None, events)],
        "alice",
        new_version=True,
        mapping=code_mapping(vocabulary="TESTVOC", code_normalization="exact"),
    )
    got = reg.con.execute(f"SELECT concept_id FROM sem_coll_v{r2['version']}.condition_occurrence ORDER BY source_row")
    assert [x[0] for x in got.fetchall()] == [1, 0, 0]


def test_mapping_model_rejects_invalid_normalization_settings():
    base = code_mapping().model_dump()
    bad = json.loads(json.dumps(base))
    bad["entities"]["ConditionOccurrence"]["code_normalization"] = "fuzzy"
    with pytest.raises(ValueError):
        DatasetMapping.model_validate(bad)
    bad = json.loads(json.dumps(base))
    e = bad["entities"]["ConditionOccurrence"]
    e.update(code_column=None, concept_id_column="dx_code")  # vocabulary left set
    with pytest.raises(ValueError, match="only apply to code_column"):
        DatasetMapping.model_validate(bad)
    bad = json.loads(json.dumps(base))
    bad["entities"]["ConditionOccurrence"]["person_column"] = "_source_row"  # system column
    with pytest.raises(ValueError, match="not a valid"):
        DatasetMapping.model_validate(bad)
    bad = json.loads(json.dumps(base))
    bad["entities"]["ConditionOccurrence"]["vocabulary"] = "ICD'; DROP TABLE x; --"
    with pytest.raises(ValueError):
        DatasetMapping.model_validate(bad)


def test_integer_typed_code_columns_are_flagged(reg, tmp_path):
    people = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    events = write(tmp_path, "events.csv", "pid,day,dx_code\n1,2020-01-01,123\n")
    r = reg.ingest(
        "intcode",
        [(None, people), (None, events)],
        "alice",
        mapping=code_mapping(),
        column_types={"events": {"dx_code": "BIGINT"}},
    )
    assert any("leading zeros cannot be recovered" in w for w in r["mapping"]["warnings"])


# ---- 4. registration states, profile files and recovery -----------------------------------------------------
def no_temp_files(reg) -> bool:
    return not list(reg.local_dir.glob(".tmp-*")) if reg.local_dir.exists() else True


def activate(settings, reg, dataset):
    return CohortBuilder(dataclasses.replace(settings, dataset=dataset, local_datasets_dir=reg.local_dir))


def test_profile_creation_failure_leaves_nothing_selectable_then_reconcile(demo_reg, monkeypatch):
    reg, settings = demo_reg

    def boom(self, final, body):
        raise OSError("disk full")

    monkeypatch.setattr(DatasetRegistry, "_write_file", boom)
    with pytest.raises(IngestError) as info:
        reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())
    assert info.value.problems[0]["code"] == "profile_publish_failed" and "NOT selectable" in str(info.value)
    assert reg.versions("clinic")[0]["status"] == "profile_failed"
    assert not (reg.local_dir / "clinic.yaml").exists() and no_temp_files(reg)
    assert reg.list_datasets()[0]["selectable"] is False
    assert ("dataset.ingest", "partial") in {(e["action"], e["outcome"]) for e in reg.store.audit_events()}
    # the same mapping cannot be re-registered around the failure; reconcile publishes the stored profile
    with pytest.raises(IngestError, match="reconcile"):
        reg.register_mapping("clinic", example_mapping(), "alice")
    monkeypatch.undo()
    out = reg.reconcile("ops")
    assert {"dataset": "clinic", "version": 1, "action": "published"} in out["actions"] and not out["problems"]
    assert reg.versions("clinic")[0]["status"] == "registered" and reg.list_datasets()[0]["selectable"]
    reg.con.close()
    b = activate(settings, reg, "clinic")
    assert b.ontology.dataset_name == "clinic"
    b.con.close()


def test_profile_replace_failure_cleans_temp_files(demo_reg, monkeypatch):
    reg, _ = demo_reg

    def fail_replace(src, dst):
        raise PermissionError("read-only")

    monkeypatch.setattr(DatasetRegistry, "_replace", staticmethod(fail_replace))
    with pytest.raises(IngestError, match="could not be published"):
        reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())
    assert no_temp_files(reg) and not list(reg.local_dir.glob("*.yaml"))
    assert reg.versions("clinic")[0]["status"] == "profile_failed"
    monkeypatch.undo()
    assert not reg.reconcile("ops")["problems"]
    assert sorted(p.name for p in reg.local_dir.glob("*.yaml")) == ["clinic.yaml", "clinic__v1.yaml"]


def test_commit_failure_rolls_back_everything(demo_reg, monkeypatch):
    reg, _ = demo_reg
    real = DatasetRegistry._tx

    def tx(self, stmt):
        if stmt == "COMMIT":
            raise duckdb.IOException("simulated commit failure")
        real(self, stmt)

    monkeypatch.setattr(DatasetRegistry, "_tx", tx)
    with pytest.raises(IngestError, match="nothing was registered"):
        reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())
    monkeypatch.undo()
    v = reg.versions("clinic")[0]
    assert v["status"] == "failed" and v["errors"][0]["code"] == "internal"
    schemas = reg.con.execute(
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name IN ('stg_clinic_v1', 'sem_clinic_v1')"
    ).fetchone()[0]
    assert schemas == 0 and not list(reg.local_dir.glob("*")) if reg.local_dir.exists() else schemas == 0
    r = reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())  # a failed attempt blocks nothing
    assert r["version"] == 2 and r["status"] == "registered"


def test_pointer_failure_keeps_pinned_version_and_refuses_stale_pointer(demo_reg, monkeypatch):
    reg, settings = demo_reg
    reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())

    def boom(self, dataset):
        raise OSError("pointer")

    monkeypatch.setattr(DatasetRegistry, "_update_pointer", boom)
    with pytest.raises(IngestError) as info:
        reg.ingest("clinic", example_files(), "alice", mapping=example_mapping(description="v2"), new_version=True)
    assert info.value.problems[0]["code"] == "pointer_update_failed"
    monkeypatch.undo()
    assert reg.versions("clinic")[0]["status"] == "registered"
    stale = yaml.safe_load((reg.local_dir / "clinic.yaml").read_text())
    with pytest.raises(ValueError, match="latest registered version is v2"):
        verify_registered_profile(reg.con, stale, "clinic")
    pinned = yaml.safe_load((reg.local_dir / "clinic__v2.yaml").read_text())
    verify_registered_profile(reg.con, pinned, "clinic__v2")  # the pinned profile is valid
    assert reg.list_datasets()[0]["selectable"] is False
    assert {"dataset": "clinic", "action": "pointer_written"} in reg.reconcile("ops")["actions"]
    verify_registered_profile(reg.con, yaml.safe_load((reg.local_dir / "clinic.yaml").read_text()), "clinic")
    reg.con.close()
    with pytest.raises(ValueError, match="does not match"):
        p = reg.local_dir / "clinic__v1.yaml"
        body = yaml.safe_load(p.read_text())
        body["description"] = "edited by hand"
        p.write_text(yaml.safe_dump(body))
        activate(settings, reg, "clinic__v1")


def test_registering_an_older_version_does_not_move_the_latest_pointer(demo_reg):
    reg, _ = demo_reg
    reg.ingest("clinic", example_files(), "alice")  # v1 staged only
    reg.ingest("clinic", example_files(), "alice", mapping=example_mapping(), new_version=True)  # v2
    reg.register_mapping("clinic", example_mapping(description="old"), "alice", version=1)
    assert yaml.safe_load((reg.local_dir / "clinic.yaml").read_text())["source"]["version"] == 2
    assert (reg.local_dir / "clinic__v1.yaml").is_file()


def test_profile_from_another_database_is_refused(demo_reg, tmp_path):
    reg, settings = demo_reg
    reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())
    profile = yaml.safe_load((reg.local_dir / "clinic.yaml").read_text())
    other = duckdb.connect(str(tmp_path / "other.duckdb"))
    init_schemas(other)
    with pytest.raises(ValueError, match="not registered in this database"):
        verify_registered_profile(other, profile, "clinic")
    other.close()
    verify_registered_profile(reg.con, {"name": "omop_demo", "source": None}, "omop_demo")  # built-ins: no-op


def test_reconcile_cleans_abandoned_rows_and_stale_files(demo_reg):
    reg, _ = demo_reg
    reg.ingest("clinic", example_files(), "alice", mapping=example_mapping())
    reg.con.execute(
        "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status, created_by, created_at) "
        "VALUES ('crashed', 'clinic', 2, 'allocating', 'alice', ?)",
        [now()],
    )
    stale = dict(yaml.safe_load((reg.local_dir / "clinic__v1.yaml").read_text()))
    stale["source"] = {**stale["source"], "version": 7}
    (reg.local_dir / "clinic__v7.yaml").write_text(yaml.safe_dump({**stale, "name": "clinic__v7"}))
    (reg.local_dir / "clinic__v8.yaml").write_text("name: clinic__v8\nnote: hand-written, not generated\n")
    (reg.local_dir / ".tmp-abc.yaml").write_text("partial")
    (reg.local_dir / "clinic__v1.yaml").unlink()  # registered version lost its file
    left = reg.reconcile("ops", "clinic")  # default: a just-started import may still be running
    acts = {(a.get("version"), a["action"]) for a in left["actions"]}
    assert {(2, "left_in_progress"), (7, "removed_stale_profile"), (1, "rewrote_profile")} <= acts
    assert (reg.local_dir / ".tmp-abc.yaml").exists() and reg.versions("clinic")[0]["status"] == "allocating"
    out = reg.reconcile("ops", "clinic", abandoned_after_minutes=0)  # operator: nothing is running
    acts = {(a.get("version"), a["action"]) for a in out["actions"]}
    assert acts == {(2, "marked_failed"), (None, "removed_temp_file")}
    assert (reg.local_dir / "clinic__v8.yaml").is_file()  # files we did not generate are never deleted
    assert (reg.local_dir / "clinic__v1.yaml").is_file() and no_temp_files(reg)
    assert {v["version"]: v["status"] for v in reg.versions("clinic")} == {1: "registered", 2: "failed"}
    assert reg.reconcile("ops", "clinic")["actions"] == []  # idempotent
    with pytest.raises(IngestError):
        reg.reconcile("ops", "../etc")


def test_versions_are_unique_and_concurrent_names_are_refused(reg, tmp_path):
    p = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    reg.ingest("uniq", [(None, p)], "alice")
    with pytest.raises(duckdb.ConstraintException):
        reg.con.execute(
            "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status) VALUES ('x', 'uniq', 1, 'staged')"
        )
    # an import still 'allocating' (e.g. running in another process) occupies the name
    reg.con.execute(
        "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status) VALUES ('y', 'other', 1, 'allocating')"
    )
    with pytest.raises(IngestError, match="already exists"):
        reg.ingest("other", [(None, p)], "alice")
    assert reg.ingest("other", [(None, p)], "alice", new_version=True)["version"] == 2


def test_allocation_retries_when_a_version_is_taken(reg, tmp_path, monkeypatch):
    from cohort_builder import dataset_registry as dr

    p = write(tmp_path, "people.csv", "pid,yob\n1,1970\n")
    reg.ingest("race", [(None, p)], "alice")
    calls = {"n": 0}
    real = dr.fetch_count

    def stale_max(con, sql, params=None):  # first read sees an outdated maximum (another writer won)
        if "max(version)" in sql and calls["n"] == 0:
            calls["n"] += 1
            return 0
        return real(con, sql, params)

    monkeypatch.setattr(dr, "fetch_count", stale_max)
    assert reg.ingest("race", [(None, p)], "alice", new_version=True)["version"] == 2 and calls["n"] == 1


def test_duplicate_versions_in_an_old_database_block_new_imports(tmp_path):
    con = connect(tmp_path / "old.duckdb")
    init_schemas(con)
    for i in ("a", "b"):
        con.execute(
            "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status) VALUES (?, 'd', 1, 'staged')", [i]
        )
    reg = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    assert reg.unique_versions is False
    with pytest.raises(IngestError, match="duplicate"):
        reg.ingest("e", [(None, write(tmp_path, "p.csv", "pid\n1\n"))], "alice")
    con.close()


def test_versions_staged_before_source_rows_still_register(demo_reg):
    reg, _ = demo_reg
    reg.ingest("clinic", example_files(), "alice")
    for t in EXAMPLE_FILES:
        reg.con.execute(f"ALTER TABLE stg_clinic_v1.{t} DROP COLUMN _source_row")
    out = reg.register_mapping("clinic", example_mapping(), "alice")
    assert out["status"] == "registered"
    assert reg.con.execute("SELECT count(source_row) FROM sem_clinic_v1.condition_occurrence").fetchone()[0] == 0


# ---- 5. end to end: CSV -> profile -> staging -> mapping -> registration -> selection -> validation -------------
def test_end_to_end_csv_to_cohort_validation(settings, tmp_path, example_ir_path):
    for n in EXAMPLE_FILES:  # profiling: every example file is ingestible, reports carry no values
        rep = profile_csv(EXAMPLE / f"{n}.csv")
        assert rep["ok"] and not rep["errors"]
    con = connect(settings.db_path)
    reg = DatasetRegistry(con, REPO_ROOT / "ontology", tmp_path / "datasets")
    staged = reg.ingest("clinic", example_files(), "alice", column_types={"labs": {"result_unit": "VARCHAR"}})
    assert staged["status"] == "staged" and staged["profile"] is None
    assert reg.list_datasets()[0]["selectable"] is False
    with pytest.raises(ValueError, match="unknown dataset"):
        Ontology.load(REPO_ROOT / "ontology", "clinic", [reg.local_dir])
    out = reg.register_mapping("clinic", load_mapping((EXAMPLE / "mapping.yaml").read_text()), "alice")
    assert out["status"] == "registered" and out["mapping"]["summary"]["suitable_for_cohorts"]
    assert reg.list_datasets()[0]["selectable"] is True
    con.close()

    b = CohortBuilder(dataclasses.replace(settings, dataset="clinic", local_datasets_dir=tmp_path / "datasets"))
    b.executor.min_cell = 1
    ir = CohortDefinition.model_validate_json(example_ir_path.read_text())
    issues, attrition = validate(ir, b.ontology, b.vocab, b.executor)
    assert not [i for i in issues if i.severity == "error"] and attrition is not None
    assert attrition.rules[0]["remaining"] == 20
    assert b.data_snapshot().startswith("csv:clinic v1 (sha256:")
    data = ir.model_dump(mode="json")  # an entity the import does not provide is refused, not silently empty
    data["concept_sets"].append({"id": "cs_v", "name": "visit", "domain": "Visit", "items": [{"concept_id": 9201}]})
    data["inclusion"].append({"id": "v1", "name": "visit", "entity": "VisitOccurrence", "concept_set_id": "cs_v"})
    issues, _ = validate(CohortDefinition.model_validate(data), b.ontology, b.vocab)
    assert any(i.severity == "error" and "VisitOccurrence" in i.message for i in issues)
    b.con.close()

    # built-in datasets still activate next to the import; LAAD (claims, no lab results) still refuses the
    # HbA1c criterion instead of silently returning an empty cohort
    bb = CohortBuilder(dataclasses.replace(settings, dataset="omop_demo", local_datasets_dir=tmp_path / "datasets"))
    bb.executor.min_cell = 1
    issues, attrition = validate(ir, bb.ontology, bb.vocab, bb.executor)
    assert not [i for i in issues if i.severity == "error"] and attrition is not None
    assert attrition.rules[0]["remaining"] > 0
    bb.con.close()
    bb = CohortBuilder(dataclasses.replace(settings, dataset="iqvia_laad", local_datasets_dir=tmp_path / "datasets"))
    issues, _ = validate(ir, bb.ontology, bb.vocab)
    errors = [i for i in issues if i.severity == "error"]
    assert errors and all(i.stage == "dataset" and "Measurement" in i.message for i in errors)
    bb.con.close()


# ---- 6. CLI and API surfaces ------------------------------------------------------------------------------------
def test_cli_type_overrides_and_reconcile(settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CB_DB_PATH", str(settings.db_path))
    monkeypatch.setenv("CB_LOCAL_DATASETS_DIR", str(tmp_path / "datasets"))
    p = write(tmp_path, "people.csv", "member_id,yob\n1,1970\n2,1980\n")
    assert cli(["csv", "profile", str(p), "--type", "member_id=BIGINT", "--json"]) == 0
    assert cols(json.loads(capsys.readouterr().out))["member_id"]["type"] == "BIGINT"
    assert cli(["csv", "profile", str(p), "--type", "member_id"]) == 1
    assert cli(["csv", "ingest", "--dataset", "ppl", "--file", f"people={p}", "--type", "people.yob=DATE"]) == 1
    assert "invalid_override" in capsys.readouterr().err
    assert cli(["csv", "ingest", "--dataset", "ppl", "--file", f"people={p}", "--type", "people.yob=VARCHAR"]) == 0
    capsys.readouterr()
    assert cli(["csv", "reconcile"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []


def test_api_column_types_and_reconcile_authorization(builder, tmp_path):
    from cohort_builder.api import create_app

    builder.settings = dataclasses.replace(builder.settings, local_datasets_dir=tmp_path / "datasets")
    entries, toks = [], {}
    for name, roles in {"alice": ["author"], "root": ["admin"]}.items():
        token, entry = issue_token(name, roles, "default", expires_days=1)
        entries.append(entry)
        toks[name] = {"Authorization": f"Bearer {token}"}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    c = TestClient(create_app(builder, SecurityConfig(tokens_file=tmp_path / "t.yaml")), raise_server_exceptions=False)
    body = {"files": [{"file_name": "p.csv", "content": "member_id,yob\n1,1970\n", "column_types": {"yob": "DATE"}}]}
    r = c.post("/datasets/ppl/ingest", json=body, headers=toks["root"])
    assert r.status_code == 422 and r.json()["detail"]["problems"][0]["code"] == "invalid_override"
    body["files"][0]["column_types"] = {"yob": "MONEY"}
    assert c.post("/datasets/ppl/ingest", json=body, headers=toks["root"]).status_code == 422
    body["files"][0]["column_types"] = {"member_id": "BIGINT"}
    ok = c.post("/datasets/ppl/ingest", json=body, headers=toks["root"])
    assert ok.status_code == 200 and cols(ok.json()["files"][0])["member_id"]["type"] == "BIGINT"
    prof = c.post(
        "/datasets/csv/profile",
        json={"file_name": "p.csv", "content": "member_id\n1\n", "column_types": {"member_id": "BIGINT"}},
        headers=toks["alice"],
    )
    assert prof.status_code == 200 and cols(prof.json())["member_id"]["type_source"] == "override"
    assert c.post("/datasets/reconcile").status_code == 401
    assert c.post("/datasets/reconcile", headers=toks["alice"]).status_code == 403
    rec = c.post("/datasets/reconcile", headers=toks["root"])
    assert rec.status_code == 200 and rec.json()["problems"] == []


def test_upload_profile_matches_text_profile_types(tmp_path):
    text = f"member_id,ndc,qty\n{BIG_ID},00002-1433-80,1.5\n"
    a = profile_text("x.csv", text)
    b = profile_csv(write(tmp_path, "x.csv", text))
    assert [(c["name"], c["type"], c["kind"]) for c in a["columns"]] == [
        (c["name"], c["type"], c["kind"]) for c in b["columns"]
    ]
    assert Upload("x.csv", text).name == "x.csv"
