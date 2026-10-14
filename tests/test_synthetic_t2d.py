"""The SYNTHETIC T2D generator (native laad schema): determinism, labeling, injected artefacts reconciled
against the treatment engine's per-record outcomes, the CLI generate -> run -> export path, and (slow) the
10,000-patient end-to-end run."""

from __future__ import annotations

import dataclasses
import json

import duckdb
import pytest

from cohort_builder.config import REPO_ROOT, Settings
from cohort_builder.db import connect, init_schemas
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.synthetic import load_demo_vocabulary
from cohort_builder.synthetic_t2d import ARCHETYPES, generate_laad_t2d
from cohort_builder.treatment_service import TreatmentService, resolve_rules
from fake_llm import FakeLLM

REF = "2026-06-30"


def table_digest(con: duckdb.DuckDBPyConnection) -> dict[str, tuple]:
    return {
        t: con.execute(f"SELECT count(*), bit_xor(hash(x)), sum(hash(x)::HUGEINT) FROM laad.{t} x").fetchone()
        for t in ("patient", "rx_claims", "dx_claims", "px_claims")
    }


def test_generator_is_deterministic_and_labeled():
    a, b, c = duckdb.connect(), duckdb.connect(), duckdb.connect()
    sa, sb = generate_laad_t2d(a, 400, seed=7), generate_laad_t2d(b, 400, seed=7)
    sc = generate_laad_t2d(c, 400, seed=8)
    assert sa == sb and table_digest(a) == table_digest(b)
    assert table_digest(a) != table_digest(c) and sa["injected"] != sc["injected"]
    assert sa["synthetic"] is True and "SYNTHETIC" in sa["note"]
    manifest = dict(a.execute("SELECT key, value FROM laad.synthetic_manifest").fetchall())
    assert manifest["synthetic"] == "true" and json.loads(manifest["seed"]) == 7
    assert sa["tables"]["patient"] == 400 and sum(sa["archetypes"].values()) == 400
    assert set(sa["archetypes"]) <= set(ARCHETYPES)
    # every patient has a type 2 diabetes diagnosis; no patient id outside 1..n except the unattributable rows
    assert a.execute(
        "SELECT count(DISTINCT patient_id) FROM laad.dx_claims WHERE dx1 IN ('E119', 'E1122')"
    ).fetchone() == (400,)
    assert a.execute("SELECT min(patient_id), max(patient_id) FROM laad.rx_claims").fetchone() == (1, 400)
    # reversal rows = every reversal the generator injected
    inj = sa["injected"]
    reversals = sum(
        inj.get(k, 0)
        for k in (
            "reversed",
            "corrected_rebill",
            "orphan_reversal",
            "reversal_before_original",
            "reversal_of_other_patients_claim",
            "reversal_payer_mismatch",
        )
    )
    assert sa["rx_claims_by_txn_type"]["REVERSAL"] == reversals
    assert sa["rx_claims_by_txn_type"]["REJECTED"] == inj["rejected_then_paid"]
    with pytest.raises(ValueError):
        generate_laad_t2d(a, 0)


def build(tmp_path, n: int, seed: int) -> tuple[CohortBuilder, dict]:
    path = tmp_path / "t2d.duckdb"
    con = connect(path)
    init_schemas(con)
    load_demo_vocabulary(con)
    stats = generate_laad_t2d(con, n, seed)
    con.close()
    s = dataclasses.replace(Settings(), db_path=path, dataset="iqvia_laad", local_datasets_dir=tmp_path / "ds")
    return CohortBuilder(s, backend=FakeLLM({})), stats


def reconcile(b: CohortBuilder, run_id: str, stats: dict) -> dict[str, int]:
    """Per-record outcomes that must equal what the generator injected (single-ingredient codes only)."""
    q = b.con.execute(
        "SELECT "
        "count(*) FILTER (WHERE exclusion_reason = 'unmapped_code'), "
        "count(*) FILTER (WHERE exclusion_reason = 'missing_person'), "
        "count(*) FILTER (WHERE quality_flags LIKE '%orphan_reversal%'), "
        "count(*) FILTER (WHERE quality_flags LIKE '%reference_person_mismatch%'), "
        "count(*) FILTER (WHERE exclusion_reason = 'implausible_start_date') "
        "FROM results.treatment_event WHERE run_id = ?",
        [run_id],
    ).fetchone()
    inj = stats["injected"]
    got = dict(zip(["unmapped", "missing_person", "orphan", "cross_patient", "future"], q, strict=True))
    assert got["unmapped"] == inj.get("unmapped_ndc_claim", 0) + inj.get("malformed_ndc_claim", 0)
    assert got["missing_person"] == inj.get("missing_patient_id", 0)
    assert got["orphan"] == inj.get("orphan_reversal", 0)
    assert got["cross_patient"] == inj.get("reversal_of_other_patients_claim", 0)
    # payer consistency, recomputed independently from the raw laad rows (injected pairs plus cross-patient
    # reversals that happen to name another payer)
    expected_payer = b.con.execute(
        "SELECT count(*) FROM laad.rx_claims v JOIN (SELECT claim_id, min(payer_type) AS p, "
        "count(DISTINCT payer_type) AS n FROM laad.rx_claims GROUP BY claim_id) o "
        "ON o.claim_id = v.orig_claim_id WHERE v.txn_type = 'REVERSAL' AND v.payer_type IS NOT NULL "
        "AND o.p IS NOT NULL AND (o.n > 1 OR o.p <> v.payer_type)"
    ).fetchone()[0]
    payer = b.con.execute(
        "SELECT count(*) FROM results.treatment_event WHERE run_id = ? "
        "AND quality_flags LIKE '%reference_payer_mismatch%'",
        [run_id],
    ).fetchone()[0]
    assert payer == expected_payer >= inj.get("reversal_payer_mismatch", 0)
    got["payer_mismatch"] = payer
    # a future-dated claim is one event, or two for the (never future-dated) combination product
    assert got["future"] == inj.get("future_dated_claim", 0)
    return got


def test_engine_on_generated_data_reconciles_with_injected_artefacts(tmp_path):
    b, stats = build(tmp_path, 1500, 31)
    try:
        rules = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
        from datetime import date

        with TreatmentService(b) as svc:
            out = svc.run(rules, "analyst", reference_date=date.fromisoformat(REF))
            again = svc.run(rules, "analyst", reference_date=date.fromisoformat(REF))
        assert out["status"] == "completed" and out["output_hash"] == again["output_hash"]
        assert out["input_fingerprint"] == again["input_fingerprint"]
        reconcile(b, out["run_id"], stats)
        untreated = b.con.execute(
            "SELECT count(*) FROM laad.patient p WHERE NOT EXISTS (SELECT 1 FROM results.treatment_episode e "
            "WHERE e.run_id = ? AND e.person_id = p.patient_id)",
            [out["run_id"]],
        ).fetchone()[0]
        assert untreated >= stats["archetypes"]["untreated"]  # untreated patients never get an episode
        kinds = {k for k in out["summary"]["transitions"]}
        assert {"add_on", "switch", "restart"} <= kinds
    finally:
        b.con.close()


def test_cli_generate_run_export(tmp_path, monkeypatch, capsys):
    from cohort_builder.cli import main as cli

    db = tmp_path / "cli.duckdb"
    monkeypatch.setenv("CB_DB_PATH", str(db))
    monkeypatch.setenv("CB_DATASET", "iqvia_laad")
    assert cli(["init-t2d-demo", "--patients", "300", "--seed", "5"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["patients"] == 300 and stats["synthetic"] is True
    with pytest.raises(SystemExit, match="exists"):
        cli(["init-t2d-demo", "--patients", "10"])  # never overwrites a database
    assert cli(["treatment", "run", "--rules", "glucose_lowering_example", "--reference-date", REF]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "completed"
    assert cli(["treatment", "export", run["run_id"], "--out", str(tmp_path / "agg")]) == 0
    agg = json.loads(capsys.readouterr().out)
    assert agg["data"] == "synthetic" and agg["patient_level"] is False and list(agg["files"]) == ["summary.json"]
    monkeypatch.delenv("CB_ALLOW_PATIENT_LEVEL", raising=False)
    with pytest.raises(SystemExit, match="patient-level"):
        cli(["treatment", "export", run["run_id"], "--out", str(tmp_path / "pl"), "--patient-level"])
    monkeypatch.setenv("CB_ALLOW_PATIENT_LEVEL", "true")
    assert cli(["treatment", "export", run["run_id"], "--out", str(tmp_path / "pl"), "--patient-level"]) == 0
    pl = json.loads(capsys.readouterr().out)
    assert pl["files"]["episodes.csv"]["rows"] == run["metrics"]["rows"]["episodes"]
    assert pl["files"]["events.csv"]["rows"] == run["metrics"]["rows"]["events"]
    with pytest.raises(SystemExit, match="new or empty"):
        cli(["treatment", "export", run["run_id"], "--out", str(tmp_path / "pl")])
    header = (tmp_path / "pl" / "events.csv").read_text().splitlines()[0]
    assert header.startswith("run_id,event_id,person_id")


@pytest.mark.slow
def test_ten_thousand_patient_end_to_end(tmp_path):
    """The documented 10k scenario: generate, run twice (identical hash), reconcile injected artefacts."""
    from datetime import date

    b, stats = build(tmp_path, 10_000, 2026)
    try:
        rules = resolve_rules(REPO_ROOT / "ontology", "glucose_lowering_example")
        with TreatmentService(b) as svc:
            out = svc.run(rules, "analyst", reference_date=date.fromisoformat(REF))
            again = svc.run(rules, "analyst", reference_date=date.fromisoformat(REF), batch_size=997)
        assert out["status"] == "completed" and out["output_hash"] == again["output_hash"]
        reconcile(b, out["run_id"], stats)
        assert stats["tables"]["patient"] == 10_000
    finally:
        b.con.close()
