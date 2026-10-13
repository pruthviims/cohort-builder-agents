"""`ontology validate`: schema, reference, conflict, provenance and vocabulary checks with file / field /
value in every issue; review status in run summaries; export cell neutralization."""

from __future__ import annotations

import json
import shutil

import duckdb
import pytest
import yaml

from cohort_builder.cli import main as cli
from cohort_builder.config import REPO_ROOT
from cohort_builder.ontology_validate import validate_ontology
from cohort_builder.treatment_config import Provenance, load_rules
from cohort_builder.treatment_export import _cell

RULES = REPO_ROOT / "ontology" / "treatment" / "rules" / "glucose_lowering_example.yaml"
SOURCES = REPO_ROOT / "ontology" / "treatment" / "sources" / "iqvia_laad.yaml"


@pytest.fixture
def onto(tmp_path):
    dst = tmp_path / "ontology"
    shutil.copytree(REPO_ROOT / "ontology", dst)
    return dst


def issues(report, severity="error"):
    return [(i["file"], i["field"], i["value"], i["message"]) for i in report["issues"] if i["severity"] == severity]


def write_rules(onto, name, mutate):
    data = yaml.safe_load(RULES.read_text())
    data["name"] = name
    mutate(data)
    path = onto / "treatment" / "rules" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_repository_ontology_is_valid_with_and_without_vocabulary(settings):
    report = validate_ontology(REPO_ROOT / "ontology")
    assert report["errors"] == 0, report["issues"]
    assert report["review_status"]["rules/glucose_lowering_example"] == "illustrative"
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        with_vocab = validate_ontology(REPO_ROOT / "ontology", con)
    finally:
        con.close()
    assert with_vocab["errors"] == 0, with_vocab["issues"]
    assert with_vocab["checked"]["vocabulary"] is True


def test_structural_problems_are_reported_with_file_field_and_value(onto):
    def dup_and_conflict(d):
        d["drug_groups"]["metformin"]["concepts"] = [1503297, 1503297]
        d["drug_groups"]["sglt2_inhibitors"]["concepts"].append(1503297)
        d["drug_groups"]["metformin"]["max_gap_days"] = 30  # no justification

    write_rules(onto, "bad_groups", dup_and_conflict)
    write_rules(onto, "no_prov", lambda d: d.pop("provenance"))
    write_rules(onto, "fake_validated", lambda d: d["provenance"].update(review_status="validated"))
    write_rules(onto, "unknown_field", lambda d: d["episode"].update(max_gap_dayz=3))
    p = write_rules(onto, "mismatch", lambda d: None)
    p.rename(p.with_name("other_name.yaml"))
    src = yaml.safe_load(SOURCES.read_text())
    src["dataset"] = "nowhere"
    (onto / "treatment" / "sources" / "nowhere.yaml").write_text(yaml.safe_dump(src))
    (onto / "treatment" / "rules" / "broken.yaml").write_text("name: [unclosed\n")
    report = validate_ontology(onto)
    errs = issues(report)
    assert ("treatment/rules/bad_groups.yaml", "drug_groups.metformin.concepts", "1503297",
            "concept id listed twice in one group") in errs  # fmt: skip
    assert any(e[0].endswith("bad_groups.yaml") and "conflicting groups" in e[3] and e[2] == "1503297" for e in errs)
    assert (
        "treatment/rules/no_prov.yaml",
        "provenance",
        None,
        "missing provenance block (review_status, source)",
    ) in errs
    assert any(e[0].endswith("fake_validated.yaml") and e[1] == "provenance" and "reviewed_by" in e[3] for e in errs)
    assert any(e[0].endswith("unknown_field.yaml") and e[1] == "episode.max_gap_dayz" and e[2] == "3" for e in errs)
    assert any(e[0].endswith("other_name.yaml") and e[1] == "name" and e[2] == "mismatch" for e in errs)
    assert any(e[0].endswith("nowhere.yaml") and e[1] == "dataset" and "no dataset profile" in e[3] for e in errs)
    assert any(e[0].endswith("broken.yaml") and e[1] == "(file)" for e in errs)
    warns = issues(report, "warning")
    assert any(w[1] == "drug_groups.metformin.max_gap_days" and "justification" in w[3] for w in warns)


def test_vocabulary_checks(onto, settings, tmp_path):
    db = tmp_path / "v.duckdb"
    shutil.copy(settings.db_path, db)
    con = duckdb.connect(str(db))
    clinical_drug = con.execute(
        "SELECT min(concept_id) FROM vocab.concept c WHERE domain_id = 'Drug' AND concept_class_id = 'Clinical Drug' "
        "AND NOT EXISTS (SELECT 1 FROM vocab.concept_ancestor a JOIN vocab.concept i ON i.concept_id = "
        "a.descendant_concept_id AND i.concept_class_id = 'Ingredient' WHERE a.ancestor_concept_id = c.concept_id)"
    ).fetchone()[0]
    assert clinical_drug is not None
    con.execute(
        "INSERT INTO vocab.concept VALUES (990001, 'bad ndc [synthetic]', 'Drug', 'NDC', '11-digit NDC', NULL, "
        "'12345-67', DATE '2000-01-01', DATE '2099-12-31', NULL)"
    )
    con.close()

    def groups(d):
        d["drug_groups"]["ghost"] = {"concepts": [987654321]}
        d["drug_groups"]["condition"] = {"concepts": [201826]}  # type 2 diabetes: not a drug
        d["drug_groups"]["orphan"] = {"concepts": [int(clinical_drug)]}

    write_rules(onto, "vocab_bad", groups)
    src = yaml.safe_load(SOURCES.read_text())
    src["provenance"] = {
        "review_status": "validated",
        "source": "x",
        "reviewed_by": "someone",
        "reviewed_on": "2026-01-01",
        "evidence": "ticket-1",
    }
    src["sources"][0]["days_supply_column"] = "no_such_column"
    (onto / "treatment" / "sources" / "iqvia_laad.yaml").write_text(yaml.safe_dump(src))
    ro = duckdb.connect(str(db), read_only=True)
    try:
        report = validate_ontology(onto, ro)
    finally:
        ro.close()
    errs = issues(report)
    assert any(
        e[1] == "drug_groups.ghost.concepts" and e[2] == "987654321" and "not in the vocabulary" in e[3] for e in errs
    )
    assert any(e[1] == "drug_groups.condition.concepts" and "not a Drug concept" in e[3] for e in errs)
    assert any(e[1] == "drug_groups.orphan" and "orphaned group" in e[3] for e in errs)
    assert any(e[2] == "12345-67" and "NDC" in e[3] for e in errs)
    assert any(e[1] == "provenance.review_status" and "placeholder" in e[3] for e in errs)
    assert any(e[1] == "sources.pharmacy.days_supply_column" and e[2] == "no_such_column" for e in errs)
    assert any("no 'Maps to'" in w[3] and w[2] == "12345-67" for w in issues(report, "warning"))


def test_cli_exit_codes(onto, monkeypatch, capsys):
    assert cli(["ontology", "validate"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == 0
    write_rules(onto, "no_prov", lambda d: d.pop("provenance"))
    assert cli(["ontology", "validate", "--ontology-dir", str(onto)]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == 1
    monkeypatch.setenv("CB_DB_PATH", str(onto / "missing.duckdb"))
    with pytest.raises(SystemExit, match="no database"):
        cli(["ontology", "validate", "--vocabulary"])


def test_provenance_model_and_hash_stability():
    with pytest.raises(ValueError, match="reviewed_by"):
        Provenance(review_status="validated", source="x")
    ok = Provenance(review_status="validated", source="x", reviewed_by="r", reviewed_on="2026-01-01", evidence="e")
    assert ok.review_status == "validated"
    data = yaml.safe_load(RULES.read_text())
    without = dict(data)
    without.pop("provenance")
    a, b = load_rules(yaml.safe_dump(without)), load_rules(yaml.safe_dump(data))
    assert a.provenance is None and b.provenance is not None
    assert a.content_hash() != b.content_hash()  # the review record is part of the recorded configuration
    assert "provenance" not in json.dumps(a.model_dump(exclude_none=True))


@pytest.mark.parametrize(
    ("value", "expected"),
    [("=cmd()", "'=cmd()"), ("+1", "'+1"), ("-x", "'-x"), ("@a", "'@a"), ("plain", "plain"), (-5, -5), (None, "")],
)
def test_export_neutralizes_spreadsheet_formulas(value, expected):
    assert _cell(value) == expected
