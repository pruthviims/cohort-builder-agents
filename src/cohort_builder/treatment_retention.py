"""Retention of DERIVED treatment results: holds, optional archive, and deletion.

What retention touches, and what it never touches:

- It deletes only the rows of results.treatment_event / treatment_episode / treatment_transition that belong
  to an expired run. Source claims, observation periods, vocabularies and every other table are never read
  for deletion, let alone deleted ("never delete source data merely because derived results expire").
- The run's meta.treatment_run row is kept (provenance, hashes, input fingerprint, summary) and marked with
  purged_at and a purge record; audit events are never deleted.
- Runs that are 'running' (or active in this process), under a legal hold, newer than the cutoff, or of
  another tenant (when a tenant is given) are skipped and counted.

Safety properties:

- dry_run (the default everywhere) reports what would happen and changes nothing;
- each run is purged in its own transaction that re-checks the run's state, so a crash or a concurrent
  hold leaves every run either fully present or fully purged; re-running is idempotent (purged runs are
  skipped and counted);
- with an archive directory, the run's rows are written as gzip CSV files plus meta.json and a manifest
  (row counts, SHA-256) and VERIFIED by reading them back before anything is deleted. Archives are only
  written below CB_TREATMENT_ARCHIVE_ROOT, in a directory validated by safe_paths.safe_directory (no
  symbolic links, owner-only), with files created 0600. Archives contain patient-level derived rows and must
  be protected like the database itself.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from .metadata import dumps, now
from .safe_paths import UnsafePathError, safe_directory

if TYPE_CHECKING:
    from .treatment_service import TreatmentService

RESULT_TABLES = {
    "events": ("results.treatment_event", "event_id"),
    "episodes": ("results.treatment_episode", "episode_id"),
    "transitions": ("results.treatment_transition", "person_id, kind, on_date, from_episode, to_episode"),
}
ARCHIVE_FORMAT = "treatment-archive/1"


class RetentionError(ValueError):
    """Retention refused (bad arguments, unsafe archive path); the message has no data values."""


def _run_row(svc: TreatmentService, run_id: str, tenant: str | None) -> tuple:
    row = svc.con.execute(
        "SELECT run_id, tenant, status, legal_hold, purged_at FROM meta.treatment_run WHERE run_id = ?", [run_id]
    ).fetchone()
    if row is None or (tenant is not None and row[1] != tenant):
        raise KeyError(f"treatment run {run_id} not found")
    return row


def set_hold(
    svc: TreatmentService, run_id: str, actor: str, hold: bool, reason: str = "", tenant: str | None = None
) -> dict[str, Any]:
    """Place or release a legal hold: a held run is never purged by retention."""
    if hold and not reason.strip():
        raise RetentionError("a hold needs a reason (recorded in the run's provenance and the audit log)")
    if len(reason) > 500:
        raise RetentionError("hold reason must be at most 500 characters")
    row = _run_row(svc, run_id, tenant)
    record = {"by": actor, "at": str(now()), "reason": reason.strip()} if hold else None
    svc.con.execute(
        "UPDATE meta.treatment_run SET legal_hold = ?, hold_json = ? WHERE run_id = ?",
        [hold, dumps(record) if record else None, run_id],
    )
    svc.store.audit(
        actor, "treatment.hold" if hold else "treatment.release", "treatment_run", run_id, "success", row[1], {}
    )
    svc._log("treatment.hold", run_id=run_id, hold=hold)
    return {"run_id": run_id, "legal_hold": hold, "purged": row[4] is not None}


# ---- archive --------------------------------------------------------------------------------------------
def _open_private(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v.isoformat() if hasattr(v, "isoformat") else v


def _write_table(con: duckdb.DuckDBPyConnection, table: str, order: str, run_id: str, path: Path, batch: int) -> int:
    cur = con.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order}", [run_id])
    n = 0
    with (
        os.fdopen(_open_private(path), "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as text,
    ):
        w = csv.writer(text, lineterminator="\n")
        w.writerow([d[0] for d in cur.description])
        while rows := cur.fetchmany(batch):
            w.writerows([_cell(v) for v in r] for r in rows)
            n += len(rows)
    return n


def _count_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def _verify_archive(run_dir: Path, expected: dict[str, int]) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RetentionError("archive manifest missing")
    manifest = json.loads(manifest_path.read_text())
    for part, n in expected.items():
        info = manifest["files"].get(part)
        f = run_dir / f"{part}.csv.gz"
        if info is None or not f.is_file():
            raise RetentionError(f"archive file for {part} missing")
        if info["rows"] != n or _sha256(f) != info["sha256"] or _count_csv_rows(f) != n:
            raise RetentionError(f"archive file for {part} does not match the database ({n} rows expected)")
    return manifest


def archive_run(svc: TreatmentService, run_id: str, archive_dir: Path, counts: dict[str, int]) -> dict[str, Any]:
    """Write and verify the archive of one run; returns the manifest. An existing archive of the run is
    verified and reused (retry after a crash between archive and delete), never overwritten."""
    run_dir = archive_dir / str(uuid.UUID(run_id))
    if run_dir.exists():
        return _verify_archive(safe_directory(run_dir, "archive run directory", archive_dir, create=False), counts)
    partial = safe_directory(
        archive_dir / f".{uuid.UUID(run_id)}.partial-{uuid.uuid4().hex[:8]}", "archive directory", archive_dir
    )
    files: dict[str, Any] = {}
    for part, (table, order) in RESULT_TABLES.items():
        f = partial / f"{part}.csv.gz"
        n = _write_table(svc.con, table, order, run_id, f, svc.limits.batch_size)
        files[part] = {"file": f.name, "rows": n, "sha256": _sha256(f)}
    meta = svc.summary(run_id, None)
    with os.fdopen(_open_private(partial / "meta.json"), "w") as fh:
        fh.write(json.dumps(meta, sort_keys=True, default=str, indent=1))
    manifest = {
        "format": ARCHIVE_FORMAT,
        "run_id": run_id,
        "tenant": meta["tenant"],
        "created_at": str(now()),
        "output_hash": meta["output_hash"],
        "input_fingerprint": meta["input_fingerprint"],
        "files": files,
        "meta_sha256": _sha256(partial / "meta.json"),
        "contains": "patient-level derived treatment rows; protect like the source database",
    }
    with os.fdopen(_open_private(partial / "manifest.json"), "w") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=1))
    _verify_archive(partial, counts)
    os.rename(partial, run_dir)  # atomic on one filesystem: an archive directory is complete or absent
    return manifest


# ---- retention ------------------------------------------------------------------------------------------
def _row_counts(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, int]:
    out = {}
    for part, (table, _) in RESULT_TABLES.items():
        row = con.execute(f"SELECT count(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()
        out[part] = int(row[0]) if row else 0
    return out


def apply_retention(
    svc: TreatmentService,
    older_than_days: int,
    actor: str,
    tenant: str | None = None,
    dry_run: bool = True,
    archive_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Purge derived results of runs that finished more than `older_than_days` ago (see module docstring)."""
    from .treatment_service import active_runs

    if not isinstance(older_than_days, int) or older_than_days < 1:
        raise RetentionError("older_than_days must be a whole number of days >= 1")
    target: Path | None = None
    if archive_dir is not None:
        root = svc.b.settings.treatment_archive_root
        if root is None:
            raise RetentionError("archiving needs CB_TREATMENT_ARCHIVE_ROOT (the directory archives may go below)")
        try:
            target = safe_directory(Path(archive_dir), "archive directory", Path(root), create=not dry_run)
        except UnsafePathError as exc:
            raise RetentionError(str(exc)) from exc
    cutoff = now() - timedelta(days=older_than_days)
    live = set(active_runs())
    counts: dict[str, Any] = {
        "dry_run": dry_run,
        "older_than_days": older_than_days,
        "tenant": tenant,
        "examined": 0,
        "eligible": 0,
        "archived": 0,
        "purged": 0,
        "skipped_active": 0,
        "skipped_hold": 0,
        "skipped_recent": 0,
        "skipped_already_purged": 0,
        "skipped_conflict": 0,
        "rows": {part: 0 for part in RESULT_TABLES},
        "run_ids": [],
    }
    runs = svc.con.execute(
        "SELECT run_id, tenant, status, coalesce(finished_at, created_at), coalesce(legal_hold, false), purged_at "
        "FROM meta.treatment_run WHERE (CAST(? AS VARCHAR) IS NULL OR tenant = ?) ORDER BY created_at, run_id",
        [tenant, tenant],
    ).fetchall()
    for run_id, run_tenant, status, finished, held, purged_at in runs:
        counts["examined"] += 1
        if purged_at is not None:
            counts["skipped_already_purged"] += 1
            continue
        if status == "running" or run_id in live:
            counts["skipped_active"] += 1
            continue
        if finished is None or finished >= cutoff:
            counts["skipped_recent"] += 1
            continue
        if held:
            counts["skipped_hold"] += 1
            continue
        counts["eligible"] += 1
        counts["run_ids"].append(run_id)
        rows = _row_counts(svc.con, run_id)
        if dry_run:
            for part, n in rows.items():
                counts["rows"][part] += n
            continue
        manifest = None
        if target is not None:
            manifest = archive_run(svc, run_id, target, rows)
            counts["archived"] += 1
        if _purge_one(svc, run_id, actor, rows, manifest):
            counts["purged"] += 1
            for part, n in rows.items():
                counts["rows"][part] += n
            svc.store.audit(
                actor,
                "treatment.retention.purge",
                "treatment_run",
                run_id,
                "success",
                run_tenant,
                {"rows": rows, "archived": manifest is not None},
            )
        else:
            counts["skipped_conflict"] += 1
    svc.store.audit(
        actor,
        "treatment.retention",
        "treatment_run",
        None,
        "dry_run" if dry_run else "applied",
        tenant,
        {k: v for k, v in counts.items() if k != "run_ids"},
    )
    svc._log("treatment.retention", **{k: v for k, v in counts.items() if k != "run_ids"})
    return counts


def _purge_one(
    svc: TreatmentService, run_id: str, actor: str, rows: dict[str, int], manifest: dict[str, Any] | None
) -> bool:
    """Delete one run's derived rows and mark it purged in ONE transaction, re-checking its state inside the
    transaction. False when the run changed meanwhile (hold placed, purged, restarted) - nothing deleted."""
    con = svc.con
    con.execute("BEGIN")
    try:
        row = con.execute(
            "SELECT status, coalesce(legal_hold, false), purged_at FROM meta.treatment_run WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None or row[0] == "running" or row[1] or row[2] is not None:
            con.execute("ROLLBACK")
            return False
        for table, _ in RESULT_TABLES.values():
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
        record = {
            "by": actor,
            "rows": rows,
            "archive": None
            if manifest is None
            else {"format": manifest["format"], "files": manifest["files"], "meta_sha256": manifest["meta_sha256"]},
        }
        con.execute(
            "UPDATE meta.treatment_run SET purged_at = ?, purge_json = ? WHERE run_id = ?",
            [now(), dumps(record), run_id],
        )
        con.execute("COMMIT")
        return True
    except duckdb.TransactionException:  # a concurrent writer changed the run: leave it for the next pass
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:  # pragma: no cover
            pass
        return False
    except BaseException:
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:  # pragma: no cover
            pass
        raise
