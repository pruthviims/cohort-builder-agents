"""Treatment runs under failure and concurrency: failure classification with mocked resource exhaustion,
cancellation (API, registry, signals), concurrent runs on separate connections, input fingerprints and
idempotent reuse, retention with holds / tenants / verified archives, safe directories, and structured logs
without patient data. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import csv
import dataclasses
import errno
import gzip
import json
import logging
import os
import signal
import stat
import threading
import time
from datetime import timedelta

import duckdb
import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder import treatment_retention as tr
from cohort_builder import treatment_service as ts
from cohort_builder.cli import _cancel_on_signals
from cohort_builder.metadata import now
from cohort_builder.safe_paths import UnsafePathError, safe_directory
from cohort_builder.security import SecurityConfig, issue_token
from cohort_builder.treatment_engine import RunCancelled
from cohort_builder.treatment_retention import RetentionError, apply_retention, set_hold
from cohort_builder.treatment_service import (
    TreatmentRunFailed,
    TreatmentService,
    active_runs,
    cancel_all_active,
    classify_failure,
)
from test_treatment_engine import rules, source_cfg
from test_treatment_scale import REF_DATE, RX_SOURCE, fills, make_db, observe

OBS = {p: [("2019-01-01", "2022-12-31")] for p in range(1, 30)}
CLAIMS = [(f"c{i}", 1 + i % 7, f"2020-{1 + i % 12:02d}-{1 + (i * 7) % 27:02d}", 801 + i % 3, 30) for i in range(80)]


@pytest.fixture
def db(tmp_path):
    b = make_db(tmp_path)
    observe(b, OBS)
    fills(b, CLAIMS)
    yield b
    b.con.close()


def count(b, sql: str, params=None) -> int:
    return int(b.con.execute(sql, params or []).fetchone()[0])


def results_rows(b, run_id: str) -> int:
    return sum(
        count(b, f"SELECT count(*) FROM results.treatment_{t} WHERE run_id = ?", [run_id])
        for t in ("event", "episode", "transition")
    )


def staging(b) -> int:
    return count(b, "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'tx_stage'")


def service(b, **limits) -> TreatmentService:
    svc = TreatmentService(b)
    if limits:
        svc.limits = dataclasses.replace(svc.limits, **limits)
    return svc


def go(svc: TreatmentService, r=None, tenant="default", **kw) -> dict:
    return svc.run(r or rules(), "analyst", tenant, source_cfg([RX_SOURCE]), REF_DATE, **kw)


# ---- failure classification ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "reason", "kind"),
    [
        (duckdb.OutOfMemoryException("Out of Memory Error: could not allocate block"), None, "memory_limit"),
        (
            duckdb.OutOfMemoryException(
                "failed to offload data block. This limit was set by 'max_temp_directory_size'"
            ),
            None,
            "disk_limit",
        ),
        (MemoryError(), None, "memory_limit"),
        (OSError(errno.ENOSPC, "No space left on device"), None, "disk_full"),
        (duckdb.IOException("IO Error: Could not write file: No space left on device"), None, "disk_full"),
        (RunCancelled("stopped"), None, "timeout"),
        (duckdb.InterruptException("INTERRUPT Error"), "cancelled", "cancelled"),
        (duckdb.InterruptException("INTERRUPT Error"), "timeout", "timeout"),
        (RunCancelled("stopped"), "cancelled", "cancelled"),
        (RuntimeError("boom"), None, "error"),
        (RuntimeError("boom"), "cancelled", "error"),  # an unrelated failure keeps its own kind
    ],
)
def test_failure_classification(exc, reason, kind):
    assert classify_failure(exc, reason) == kind
    assert kind in ts.FAILURE_HINTS


@pytest.mark.parametrize(
    ("where", "exc", "kind"),
    [
        ("episodes", duckdb.OutOfMemoryException("Out of Memory Error: failed to allocate"), "memory_limit"),
        ("episodes", duckdb.OutOfMemoryException("... 'max_temp_directory_size' setting ..."), "disk_limit"),
        ("publish", OSError(errno.ENOSPC, "No space left on device"), "disk_full"),
        ("publish", duckdb.IOException("Could not write file: No space left on device"), "disk_full"),
    ],
)
def test_mocked_resource_exhaustion_publishes_nothing(db, monkeypatch, where, exc, kind):
    def boom(*_a, **_k):
        raise exc

    if where == "episodes":
        monkeypatch.setattr(ts.TreatmentService, "_sql_episodes", boom)
    else:
        monkeypatch.setattr(ts, "output_hash", boom)  # inside the publishing transaction
    with pytest.raises(TreatmentRunFailed) as info:
        go(service(db))
    err = info.value
    assert err.kind == kind and err.status == "failed" and err.stage == where
    assert ts.FAILURE_HINTS[kind] in str(err) and "No space" not in str(err)  # actionable, no raw detail
    s = TreatmentService(db).summary(err.run_id, None)
    assert s["status"] == "failed" and s["failure_kind"] == kind and s["failure_hint"] == ts.FAILURE_HINTS[kind]
    assert results_rows(db, err.run_id) == 0 and staging(db) == 0 and active_runs() == []


@pytest.mark.parametrize(
    ("kind", "setting", "sql"),
    [
        # a string aggregate cannot spill: DuckDB's own out-of-memory error
        (
            "memory_limit",
            "SET memory_limit = '64MB'",
            "SELECT length(string_agg(r::VARCHAR || 'abcdefghij', ',')) FROM range(20000000) t(r)",
        ),
        # a hash aggregate spills, and the spill directory is capped at 1MB
        (
            "disk_limit",
            "SET memory_limit = '64MB'; SET max_temp_directory_size = '1MB'",
            "SELECT count(DISTINCT r::VARCHAR || 'abcdefghij') FROM range(20000000) t(r)",
        ),
    ],
)
def test_real_resource_limits_are_classified(db, monkeypatch, kind, setting, sql):
    """Not mocked exceptions: real DuckDB resource errors raised by a statement inside the run travel through
    the run's failure path and are classified."""
    real = ts.TreatmentService._events

    def heavy(self, *a, **k):
        real(self, *a, **k)
        self.con.execute(sql).fetchone()

    monkeypatch.setattr(ts.TreatmentService, "_events", heavy)
    db.con.execute(setting)
    try:
        with pytest.raises(TreatmentRunFailed) as info:
            go(service(db))
    finally:
        db.con.execute("SET memory_limit = '2GB'; RESET max_temp_directory_size")
    assert info.value.kind == kind and info.value.stage == "events", str(info.value)
    assert TreatmentService(db).summary(info.value.run_id, None)["failure_kind"] == kind
    assert staging(db) == 0 and results_rows(db, info.value.run_id) == 0


# ---- cancellation ----------------------------------------------------------------------------------------
def _blocking_events(monkeypatch, entered: threading.Event, release: threading.Event, slow_sql: bool = False):
    real = ts.TreatmentService._events

    def blocked(self, *a, **k):
        real(self, *a, **k)
        entered.set()
        if slow_sql:  # a long statement: only an interrupt can stop it
            self.con.execute("SELECT count(*) FROM range(1000000000000) t(r) WHERE r % 7 = 3").fetchone()
        else:
            release.wait(30)

    monkeypatch.setattr(ts.TreatmentService, "_events", blocked)


def _start(db, **kw) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def target():
        try:
            box["result"] = go(service(db, **kw))
        except Exception as exc:  # noqa: BLE001 - collected for the assertions
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, box


@pytest.mark.parametrize("slow_sql", [False, True])
def test_cancel_stops_a_running_run(db, monkeypatch, slow_sql):
    entered, release = threading.Event(), threading.Event()
    _blocking_events(monkeypatch, entered, release, slow_sql)
    t, box = _start(db)
    assert entered.wait(30)
    (run_id,) = active_runs()
    other = TreatmentService(db)
    with pytest.raises(KeyError):
        other.cancel(run_id, "intruder", "other-tenant")  # tenant isolation
    began = time.monotonic()
    assert other.cancel(run_id, "analyst", "default") == {"run_id": run_id, "status": "cancelling", "cancelled": True}
    release.set()
    t.join(30)
    assert not t.is_alive() and time.monotonic() - began < 20
    err = box["error"]
    assert isinstance(err, TreatmentRunFailed) and err.kind == "cancelled" and err.status == "cancelled"
    s = other.summary(run_id, None)
    assert s["status"] == "cancelled" and s["failure_kind"] == "cancelled"
    assert results_rows(db, run_id) == 0 and staging(db) == 0 and active_runs() == []
    assert count(db, "SELECT count(*) FROM meta.audit_event WHERE action = 'treatment.cancel'") == 1
    assert other.cancel(run_id, "analyst")["cancelled"] is False  # finished runs stay as they are
    # the database is fully usable afterwards
    monkeypatch.undo()
    assert go(service(db))["status"] == "completed"


def test_cancel_of_an_orphaned_running_row_marks_it_abandoned(db):
    run_id = "00000000-0000-0000-0000-0000000000aa"
    db.con.execute(
        "INSERT INTO meta.treatment_run (run_id, tenant, status, created_at) VALUES (?, 'default', 'running', ?)",
        [run_id, now()],
    )
    db.con.execute(f"CREATE TABLE tx_stage.ev0_{run_id.replace('-', '')} (x INTEGER)")
    assert TreatmentService(db).cancel(run_id, "ops")["status"] == "abandoned"
    assert TreatmentService(db).summary(run_id, None)["status"] == "abandoned" and staging(db) == 0


def test_signal_handler_cancels_active_runs():
    handle = ts._ActiveRun("r", "default")
    with ts._ACTIVE_LOCK:
        ts._ACTIVE["r"] = handle
    try:
        with _cancel_on_signals():
            os.kill(os.getpid(), signal.SIGINT)
            for _ in range(100):
                if handle.event.is_set():
                    break
                time.sleep(0.01)
        assert handle.event.is_set() and handle.reason == "cancelled"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored
    finally:
        with ts._ACTIVE_LOCK:
            ts._ACTIVE.pop("r", None)
    assert cancel_all_active() == 0


def test_timeout_is_classified(db, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    _blocking_events(monkeypatch, entered, release, slow_sql=True)
    with pytest.raises(TreatmentRunFailed) as info:
        go(service(db, timeout_seconds=1.0))
    assert info.value.kind == "timeout" and info.value.status == "failed"
    assert staging(db) == 0


# ---- concurrency -----------------------------------------------------------------------------------------
def test_concurrent_runs_on_separate_connections_match_sequential_runs(db):
    loose = rules(episode={"max_gap_days": 90})
    seq_a, seq_b = go(service(db)), go(service(db), loose)
    out: dict = {}
    barrier = threading.Barrier(2)

    def worker(name, r):
        barrier.wait()
        try:
            out[name] = go(service(db, max_concurrent_runs=2), r)
        except Exception as exc:  # noqa: BLE001
            out[name] = exc

    threads = [threading.Thread(target=worker, args=a) for a in (("a", None), ("b", loose))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert isinstance(out["a"], dict) and isinstance(out["b"], dict), out
    assert out["a"]["output_hash"] == seq_a["output_hash"] and out["b"]["output_hash"] == seq_b["output_hash"]
    assert out["a"]["summary"] == seq_a["summary"] and out["b"]["summary"] == seq_b["summary"]
    assert staging(db) == 0 and active_runs() == []
    assert count(db, "SELECT count(*) FROM meta.treatment_run WHERE status = 'completed'") == 4


def test_concurrency_limit_still_applies_across_threads(db, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    _blocking_events(monkeypatch, entered, release)
    t, box = _start(db)
    assert entered.wait(30)
    with pytest.raises(ts.TreatmentError, match="too many concurrent"):
        go(service(db))
    release.set()
    t.join(30)
    assert box["result"]["status"] == "completed"


# ---- fingerprint and reuse -------------------------------------------------------------------------------
def test_input_fingerprint_is_order_invariant_and_content_sensitive(db):
    svc = TreatmentService(db)
    src = source_cfg([RX_SOURCE])

    def fp():
        return svc.input_fingerprint(rules(), src, REF_DATE, "sql")

    base = fp()
    db.con.execute("DELETE FROM claims.pharmacy")
    fills(db, list(reversed(CLAIMS)))
    assert fp() == base  # same rows, other physical order
    fills(db, [CLAIMS[0]])  # an exact duplicate row
    dup = fp()
    assert dup["data"] != base["data"]
    db.con.execute("DELETE FROM claims.pharmacy")
    fills(db, CLAIMS[:-1] + [(*CLAIMS[-1][:4], 31)])  # one value changed
    assert fp()["data"] != base["data"]
    assert svc.input_fingerprint(rules(episode={"max_gap_days": 1}), src, REF_DATE, "sql") != base


def test_reuse_returns_the_completed_run_for_identical_inputs(db):
    first = go(service(db))
    assert first["input_fingerprint"]
    again = go(service(db), reuse=True)
    assert again["run_id"] == first["run_id"]
    assert count(db, "SELECT count(*) FROM meta.treatment_run") == 1
    other_tenant = go(service(db), tenant="t2", reuse=True)
    assert other_tenant["run_id"] != first["run_id"]  # never another tenant's run
    fills(db, [("new", 3, "2021-05-05", 801, 30)])
    changed = go(service(db), reuse=True)
    assert changed["run_id"] not in (first["run_id"], other_tenant["run_id"])
    assert changed["input_fingerprint"] != first["input_fingerprint"]
    assert go(service(db))["run_id"] != first["run_id"]  # without reuse: always a new run


# ---- retention -------------------------------------------------------------------------------------------
def age(b, run_id: str, days: int) -> None:
    b.con.execute("UPDATE meta.treatment_run SET finished_at = ? WHERE run_id = ?", [now() - timedelta(days), run_id])


def test_retention_dry_run_apply_holds_tenants_and_idempotency(db):
    old_a, old_b, held = go(service(db)), go(service(db)), go(service(db))
    other = go(service(db), tenant="t2")
    recent = go(service(db))
    for r in (old_a, old_b, held, other):
        age(db, r["run_id"], 40)
    svc = TreatmentService(db)
    set_hold(svc, held["run_id"], "legal", True, "litigation 2026-17")
    claims_before = count(db, "SELECT count(*) FROM claims.pharmacy")
    rows_a = results_rows(db, old_a["run_id"])

    dry = apply_retention(svc, 30, "ops", tenant="default")
    assert dry["dry_run"] and dry["eligible"] == 2 and dry["purged"] == 0
    assert (dry["examined"], dry["skipped_hold"], dry["skipped_recent"]) == (4, 1, 1)
    assert sorted(dry["run_ids"]) == sorted([old_a["run_id"], old_b["run_id"]])
    assert results_rows(db, old_a["run_id"]) == rows_a  # nothing changed
    assert sum(dry["rows"].values()) == rows_a + results_rows(db, old_b["run_id"])

    done = apply_retention(svc, 30, "ops", tenant="default", dry_run=False)
    assert done["purged"] == 2 and done["archived"] == 0
    assert results_rows(db, old_a["run_id"]) == 0 and results_rows(db, old_b["run_id"]) == 0
    assert results_rows(db, held["run_id"]) > 0 and results_rows(db, recent["run_id"]) > 0
    assert results_rows(db, other["run_id"]) > 0  # another tenant: untouched
    assert count(db, "SELECT count(*) FROM claims.pharmacy") == claims_before  # source data never deleted
    s = svc.summary(old_a["run_id"], None)
    assert s["purged_at"] and s["output_hash"] == old_a["output_hash"] and s["purge"]["rows"]["events"] > 0
    db.policy = dataclasses.replace(db.policy, allow_patient_level=True)
    with pytest.raises(KeyError, match="removed by retention"):
        svc.explain_episode(old_a["run_id"], "x", "admin", True)

    again = apply_retention(svc, 30, "ops", tenant="default", dry_run=False)
    assert again["purged"] == 0 and again["skipped_already_purged"] == 2  # idempotent
    set_hold(svc, held["run_id"], "legal", False)
    assert apply_retention(svc, 30, "ops", dry_run=False)["purged"] == 2  # released + other tenant
    audit = count(db, "SELECT count(*) FROM meta.audit_event WHERE action = 'treatment.retention.purge'")
    assert audit == 4
    with pytest.raises(RetentionError, match=">= 1"):
        apply_retention(svc, 0, "ops")
    with pytest.raises(RetentionError, match="reason"):
        set_hold(svc, recent["run_id"], "legal", True, " ")
    with pytest.raises(KeyError):
        set_hold(svc, recent["run_id"], "legal", True, "x", tenant="t2")


def test_retention_skips_running_runs(db):
    done = go(service(db))
    age(db, done["run_id"], 40)
    db.con.execute("UPDATE meta.treatment_run SET status = 'running' WHERE run_id = ?", [done["run_id"]])
    got = apply_retention(TreatmentService(db), 30, "ops", dry_run=False)
    assert got["skipped_active"] == 1 and got["purged"] == 0 and results_rows(db, done["run_id"]) > 0


def test_retention_archive_is_verified_private_and_retry_safe(db, tmp_path, monkeypatch):
    root = tmp_path / "archives"
    root.mkdir(mode=0o700)
    done = go(service(db))
    age(db, done["run_id"], 40)
    expected = {
        "events": count(db, "SELECT count(*) FROM results.treatment_event WHERE run_id = ?", [done["run_id"]]),
        "episodes": count(db, "SELECT count(*) FROM results.treatment_episode WHERE run_id = ?", [done["run_id"]]),
    }
    svc = TreatmentService(db)
    with pytest.raises(RetentionError, match="CB_TREATMENT_ARCHIVE_ROOT"):
        apply_retention(svc, 30, "ops", dry_run=False, archive_dir=root / "a")
    db.settings = dataclasses.replace(db.settings, treatment_archive_root=root)
    with pytest.raises(RetentionError, match="inside the allowed root"):
        apply_retention(svc, 30, "ops", dry_run=False, archive_dir=tmp_path / "elsewhere")
    (tmp_path / "real").mkdir()
    (root / "link").symlink_to(tmp_path / "real")
    with pytest.raises(RetentionError, match="symbolic link"):
        apply_retention(svc, 30, "ops", dry_run=False, archive_dir=root / "link")

    # crash between archive and purge: the archive is written, the rows stay
    def crash(*_a, **_k):
        raise RuntimeError("power cut")

    monkeypatch.setattr(tr, "_purge_one", crash)
    with pytest.raises(RuntimeError):
        apply_retention(svc, 30, "ops", dry_run=False, archive_dir=root / "a")
    assert results_rows(db, done["run_id"]) > 0
    monkeypatch.undo()
    got = apply_retention(svc, 30, "ops", dry_run=False, archive_dir=root / "a")  # retry reuses the archive
    assert got["archived"] == 1 and got["purged"] == 1 and results_rows(db, done["run_id"]) == 0
    run_dir = root / "a" / done["run_id"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["output_hash"] == done["output_hash"]
    assert manifest["files"]["events"]["rows"] == expected["events"]
    assert manifest["files"]["episodes"]["rows"] == expected["episodes"]
    for f in run_dir.iterdir():
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f.name
    assert stat.S_IMODE((root / "a").stat().st_mode) == 0o700
    with gzip.open(run_dir / "events.csv.gz", "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == expected["events"] and {r["run_id"] for r in rows} == {done["run_id"]}
    assert not [p for p in (root / "a").iterdir() if p.name.startswith(".")]  # no partial directories left
    assert TreatmentService(db).summary(done["run_id"], None)["purge"]["archive"]["files"] == manifest["files"]


def test_tampered_archive_blocks_the_purge(db, tmp_path):
    root = tmp_path / "archives"
    root.mkdir(mode=0o700)
    db.settings = dataclasses.replace(db.settings, treatment_archive_root=root)
    done = go(service(db))
    age(db, done["run_id"], 40)
    svc = TreatmentService(db)
    counts = tr._row_counts(svc.con, done["run_id"])
    tr.archive_run(svc, done["run_id"], safe_directory(root / "a", "t", root), counts)
    (root / "a" / done["run_id"] / "events.csv.gz").write_bytes(b"tampered")
    with pytest.raises(RetentionError, match="does not match"):
        apply_retention(svc, 30, "ops", dry_run=False, archive_dir=root / "a")
    assert results_rows(db, done["run_id"]) > 0


# ---- safe directories and the spill directory --------------------------------------------------------------
def test_safe_directory_rules(tmp_path):
    with pytest.raises(UnsafePathError, match="absolute"):
        safe_directory("relative/dir", "X")
    with pytest.raises(UnsafePathError, match=r"'\.\.'"):
        safe_directory(tmp_path / "a" / ".." / "b", "X")
    (tmp_path / "file").write_text("x")
    with pytest.raises(UnsafePathError, match="not a directory"):
        safe_directory(tmp_path / "file", "X")
    loose = tmp_path / "loose"
    loose.mkdir()
    loose.chmod(0o777)
    with pytest.raises(UnsafePathError, match="writable by other"):
        safe_directory(loose, "X")
    (tmp_path / "ln").symlink_to(tmp_path)
    with pytest.raises(UnsafePathError, match="symbolic link"):
        safe_directory(tmp_path / "ln" / "sub", "X")
    with pytest.raises(UnsafePathError, match="does not exist"):
        safe_directory(tmp_path / "nope", "X", create=False)
    made = safe_directory(tmp_path / "new" / "deep", "X", root=tmp_path)
    assert made.is_dir() and stat.S_IMODE(made.stat().st_mode) == 0o700


def test_duckdb_temp_directory_setting(tmp_path):
    from cohort_builder.config import Settings
    from cohort_builder.orchestrator import CohortBuilder
    from fake_llm import FakeLLM

    spill = tmp_path / "spill"
    s = dataclasses.replace(Settings(), db_path=tmp_path / "t.duckdb", duckdb_temp_directory=spill)
    b = CohortBuilder(s, backend=FakeLLM({}))
    try:
        assert b.con.execute("SELECT current_setting('temp_directory')").fetchone()[0] == str(spill)
        assert spill.is_dir()
    finally:
        b.con.close()
    (tmp_path / "ln").symlink_to(tmp_path)
    bad = dataclasses.replace(s, db_path=tmp_path / "u.duckdb", duckdb_temp_directory=tmp_path / "ln" / "x")
    with pytest.raises(ValueError, match="CB_DUCKDB_TEMP_DIRECTORY"):
        CohortBuilder(bad, backend=FakeLLM({}))


def test_settings_read_retention_and_spill_environment(monkeypatch, tmp_path):
    from cohort_builder.config import Settings

    monkeypatch.setenv("CB_DUCKDB_TEMP_DIRECTORY", str(tmp_path / "spill"))
    monkeypatch.setenv("CB_TREATMENT_RETENTION_DAYS", "45")
    monkeypatch.setenv("CB_TREATMENT_ARCHIVE_ROOT", str(tmp_path / "arch"))
    s = Settings.from_env()
    assert s.duckdb_temp_directory == tmp_path / "spill" and s.treatment_retention_days == 45
    assert s.treatment_archive_root == tmp_path / "arch"


# ---- observability ---------------------------------------------------------------------------------------
def test_structured_logs_carry_no_patient_data(db, caplog):
    caplog.set_level(logging.INFO, logger="cohort_builder.treatment")
    out = go(service(db))
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "cohort_builder.treatment"]
    events = [r["event"] for r in records]
    assert events[0] == "treatment.run.started" and events[-1] == "treatment.run.completed"
    assert "treatment.run.stage" in events
    allowed = {"event", "run_id", "tenant", "engine", "input_fingerprint", "stage", "seconds", "rows", "output_hash"}
    text = " ".join(r.getMessage() for r in caplog.records)
    for r in records:
        assert set(r) <= allowed, r
    for claim in ("c1", "c17", "11111000101", "2020-0"):  # record ids, codes and dates never appear
        assert claim not in text
    assert out["run_id"] in text


# ---- API --------------------------------------------------------------------------------------------------
def test_api_cancel_hold_release_roles_and_tenants(builder, tmp_path):
    from cohort_builder.api import create_app

    entries, heads = [], {}
    for name, roles, tenant in [
        ("ex", ["executor"], "default"),
        ("ex2", ["executor"], "other"),
        ("ad", ["admin"], "default"),
    ]:
        token, entry = issue_token(name, roles, tenant, expires_days=1)
        entries.append(entry)
        heads[name] = {"Authorization": f"Bearer {token}"}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    c = TestClient(create_app(builder, SecurityConfig(tokens_file=tmp_path / "t.yaml")), raise_server_exceptions=False)
    body = {"rule_set": "glucose_lowering_example", "reference_date": "2026-06-30"}
    first = c.post("/treatment-runs", json=body, headers=heads["ex"]).json()
    again = c.post("/treatment-runs", json={**body, "reuse": True}, headers=heads["ex"]).json()
    assert again["run_id"] == first["run_id"]
    rid = first["run_id"]
    assert c.post(f"/treatment-runs/{rid}/cancel", headers=heads["ex2"]).status_code == 404  # other tenant
    r = c.post(f"/treatment-runs/{rid}/cancel", headers=heads["ex"])
    assert r.status_code == 200 and r.json() == {"run_id": rid, "status": "completed", "cancelled": False}
    assert c.post(f"/treatment-runs/{rid}/hold", json={"reason": "audit"}, headers=heads["ex"]).status_code == 403
    assert c.post(f"/treatment-runs/{rid}/hold", json={}, headers=heads["ad"]).status_code == 422
    h = c.post(f"/treatment-runs/{rid}/hold", json={"reason": "audit"}, headers=heads["ad"])
    assert h.status_code == 200 and h.json()["legal_hold"] is True
    assert c.get(f"/treatment-runs/{rid}", headers=heads["ex"]).json()["hold"]["reason"] == "audit"
    assert c.post(f"/treatment-runs/{rid}/release", headers=heads["ad"]).json()["legal_hold"] is False


def test_api_maps_run_failures_to_actionable_errors(builder, tmp_path, monkeypatch):
    from cohort_builder.api import create_app

    token, entry = issue_token("ex", ["executor"], "default", expires_days=1)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": [entry]}))
    c = TestClient(create_app(builder, SecurityConfig(tokens_file=tmp_path / "t.yaml")), raise_server_exceptions=False)

    def oom(*_a, **_k):
        raise duckdb.OutOfMemoryException("Out of Memory Error")

    monkeypatch.setattr(ts.TreatmentService, "_sql_episodes", oom)
    r = c.post(
        "/treatment-runs", json={"rule_set": "glucose_lowering_example"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["failure_kind"] == "memory_limit" and "CB_DUCKDB_MEMORY_LIMIT" in detail["message"]


def test_cli_cancel_hold_retention(settings, monkeypatch, capsys, tmp_path):
    from cohort_builder.cli import main as cli

    monkeypatch.setenv("CB_DB_PATH", str(settings.db_path))
    root = tmp_path / "arch"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("CB_TREATMENT_ARCHIVE_ROOT", str(root))
    args = ["treatment", "run", "--rules", "glucose_lowering_example", "--reference-date", "2026-06-30"]
    assert cli(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli([*args, "--reuse"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == first["run_id"]
    rid = first["run_id"]
    assert cli(["treatment", "cancel", rid]) == 0
    assert json.loads(capsys.readouterr().out)["cancelled"] is False
    assert cli(["treatment", "hold", rid, "--reason", "review"]) == 0
    assert json.loads(capsys.readouterr().out)["legal_hold"] is True
    con = duckdb.connect(str(settings.db_path))
    con.execute("UPDATE meta.treatment_run SET finished_at = finished_at - INTERVAL 400 DAY")
    con.close()
    assert cli(["treatment", "retention", "--older-than-days", "30"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] and dry["skipped_hold"] == 1 and dry["eligible"] == 0
    assert cli(["treatment", "release", rid]) == 0
    capsys.readouterr()
    retention = ["treatment", "retention", "--older-than-days", "30", "--apply", "--archive-dir", str(root / "x")]
    assert cli(retention) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["purged"] == 1 and got["archived"] == 1 and (root / "x" / rid / "manifest.json").is_file()
    with pytest.raises(SystemExit, match=">= 1"):
        cli(["treatment", "retention"])  # CB_TREATMENT_RETENTION_DAYS defaults to 0: refuse, never purge all


# ---- determinism ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [5, 23])
def test_row_order_and_thread_count_do_not_change_output(tmp_path, seed):
    """Same rows loaded in three physical orders, run with 1 and 2 DuckDB threads: one output hash and one
    input fingerprint."""
    import random

    from test_treatment_scale import random_claims

    rows, obs = random_claims(seed, 120)
    hashes, prints = set(), set()
    for k, order in enumerate(("as_is", "reversed", "shuffled")):
        b = make_db(tmp_path / f"{seed}_{order}")
        try:
            observe(b, obs)
            data = list(rows)
            if order == "reversed":
                data.reverse()
            elif order == "shuffled":
                random.Random(seed).shuffle(data)
            fills(b, data)
            b.con.execute(f"SET threads = {1 + k % 2}")
            out = go(service(b), rules(episode={"max_carryover_days": 5}))
            hashes.add(out["output_hash"])
            prints.add(out["input_fingerprint"])
        finally:
            b.con.close()
    assert len(hashes) == 1 and len(prints) == 1
