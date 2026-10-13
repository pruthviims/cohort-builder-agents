"""Export of one completed treatment run: the suppressed aggregate summary always, patient-level tables only
when patient-level output is authorized (admin principal AND CB_ALLOW_PATIENT_LEVEL=true; audited).

Files (UTF-8 CSV, header row, deterministic row order, created 0600 in a new owner-only directory):
  summary.json                    provenance + the suppressed aggregate summary (no patient rows)
  episodes.csv / events.csv / transitions.csv   patient-level, only with --patient-level
  manifest.json                   run id, hashes, row counts, SHA-256 of every file, synthetic label

Text cells that a spreadsheet would treat as a formula (starting with = + - @ or a tab / carriage return)
are prefixed with a single quote; numbers and dates are written as they are.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from .metadata import now
from .safe_paths import UnsafePathError, safe_directory
from .security import GovernanceError

if TYPE_CHECKING:
    from .treatment_service import TreatmentService

TABLES = {
    "episodes": ("results.treatment_episode", "person_id, drug_group, episode_start, episode_id"),
    "events": ("results.treatment_event", "event_id"),
    "transitions": ("results.treatment_transition", "person_id, on_date, kind, from_episode, to_episode"),
}
EXPORT_FORMAT = "treatment-export/1"
_FORMULA = ("=", "+", "-", "@", "\t", "\r")


class ExportError(ValueError):
    """Export refused (run not exportable, unsafe directory); no data values in the message."""


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return "'" + v if v.startswith(_FORMULA) else v
    return v.isoformat() if hasattr(v, "isoformat") else v


def _private_file(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_label(con: duckdb.DuckDBPyConnection) -> str:
    """'synthetic' when the database carries a synthetic-data manifest, else 'not declared'."""
    for sql in (
        "SELECT value FROM laad.synthetic_manifest WHERE key = 'synthetic'",
        "SELECT cdm_source_name FROM cdm.cdm_source WHERE cdm_source_name ILIKE '%synthetic%' LIMIT 1",
    ):
        try:
            if con.execute(sql).fetchone():
                return "synthetic"
        except duckdb.Error:
            continue
    return "not declared"


def export_run(
    svc: TreatmentService,
    run_id: str,
    out_dir: Path | str,
    actor: str,
    is_admin: bool,
    patient_level: bool = False,
    tenant: str | None = None,
) -> dict[str, Any]:
    run = svc.summary(run_id, tenant)
    if run["status"] != "completed":
        raise ExportError(f"treatment run {run_id} is {run['status']}, not completed")
    if run["purged_at"] is not None:
        raise ExportError(f"treatment run {run_id} results were removed by retention")
    if patient_level and not (is_admin and svc.b.policy.allow_patient_level):
        svc.store.audit(
            actor, "treatment.export", "treatment_run", run_id, "denied", run["tenant"], {"patient_level": True}
        )
        raise GovernanceError("patient-level export requires an admin principal and CB_ALLOW_PATIENT_LEVEL=true")
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ExportError("the export directory must be new or empty")
    try:
        out = safe_directory(out, "export directory")
    except UnsafePathError as exc:
        raise ExportError(str(exc)) from exc
    label = synthetic_label(svc.con)
    files: dict[str, Any] = {}
    summary_path = out / "summary.json"
    with os.fdopen(_private_file(summary_path), "w") as fh:
        json.dump({"data": label, **run}, fh, sort_keys=True, indent=1, default=str)
    files["summary.json"] = {"rows": None, "sha256": _sha256(summary_path)}
    if patient_level:
        for name, (table, order) in TABLES.items():
            path = out / f"{name}.csv"
            cur = svc.con.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order}", [run_id])
            n = 0
            with os.fdopen(_private_file(path), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow([d[0] for d in cur.description])
                while rows := cur.fetchmany(svc.limits.batch_size):
                    w.writerows([_cell(v) for v in r] for r in rows)
                    n += len(rows)
            files[path.name] = {"rows": n, "sha256": _sha256(path)}
    manifest = {
        "format": EXPORT_FORMAT,
        "data": label,
        "run_id": run_id,
        "tenant": run["tenant"],
        "dataset": run["dataset"],
        "exported_at": str(now()),
        "exported_by": actor,
        "output_hash": run["output_hash"],
        "input_fingerprint": run["input_fingerprint"],
        "rule_set": {"name": run["rule_set_name"], "version": run["rule_set_version"], "hash": run["rule_set_hash"]},
        "patient_level": patient_level,
        "files": files,
    }
    with os.fdopen(_private_file(out / "manifest.json"), "w") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=1)
    svc.store.audit(
        actor,
        "treatment.export",
        "treatment_run",
        run_id,
        "success",
        run["tenant"],
        {"patient_level": patient_level, "files": sorted(files)},
    )
    svc._log("treatment.export", run_id=run_id, patient_level=patient_level, files=len(files))
    return {"directory": str(out), **manifest}
