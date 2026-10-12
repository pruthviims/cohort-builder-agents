"""Registry of imported (CSV) datasets: staging, versioning, mapping, registration.

Lifecycle of one ingestion (one immutable version of a dataset):

    files --(validate, stream, type, profile)--> stg_<dataset>_v<N>.<table>        status 'staged'
    mapping --(validate against staged schema, build semantic views, data checks)-->
        sem_<dataset>_v<N>.<entity views> + profile data/datasets/<dataset>.yaml     status 'registered'

* Every step runs in ONE database transaction; any failure rolls back, so no partial tables or
  registrations remain. The failed attempt is recorded (status 'failed', problems without cell values).
* Versions are immutable. Re-using a dataset name requires `new_version=True`; nothing is overwritten.
  `<dataset>.yaml` points at the latest registered version; `<dataset>__v<N>.yaml` pins a version.
* Built-in dataset names (ontology/datasets) and internal schema names are reserved.
* A registered dataset is a normal dataset profile: select it with --dataset / CB_DATASET. All cohort
  safeguards (validation, approval, deterministic compilation, guards, suppression, audit) apply unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

from . import __version__
from .csv_ingest import (
    FileInfo,
    IngestError,
    TableProfile,
    check_file,
    check_text,
    default_table_name,
    max_bytes_from_env,
    open_text,
    qi,
    stage_table,
    text_stream,
    validate_name,
)
from .dataset_mapping import (
    DatasetMapping,
    Schema,
    build_profile,
    needs_person_map,
    person_map_sql,
    semantic_sql,
    validate_mapping,
)
from .db import fetch_count, init_schemas
from .metadata import MetadataStore, dumps, now, sha
from .ontology import Ontology

RESERVED = {
    "cdm",
    "csv",
    "vocab",
    "sem",
    "meta",
    "results",
    "laad",
    "main",
    "temp",
    "preview",
    "information_schema",
    "pg_catalog",
    "system",
    "default",
}


@dataclass
class Upload:
    """In-memory CSV content (API uploads). `name` must be a bare file name."""

    name: str
    text: str


def load_mapping(text: str) -> DatasetMapping:
    from .proxy import load_yaml_strict  # strict loader rejects duplicate keys

    data = load_yaml_strict(text)
    if not isinstance(data, dict):
        raise IngestError("a mapping file must be a YAML mapping")
    return DatasetMapping.model_validate(data)


class DatasetRegistry:
    def __init__(
        self, con: duckdb.DuckDBPyConnection, ontology_dir: Path, local_dir: Path, store: MetadataStore | None = None
    ):
        self.con = con
        init_schemas(con)
        self.ontology_dir = Path(ontology_dir)
        self.local_dir = Path(local_dir)
        self.store = store or MetadataStore(con)
        self.ont = Ontology.load(self.ontology_dir)  # entities, vocab schema (dataset-independent parts)

    # ---- names and versions --------------------------------------------------------------------
    def builtin_names(self) -> set[str]:
        return {p.stem for p in (self.ontology_dir / self.ont.domain["datasets_dir"]).glob("*.yaml")}

    def check_dataset_name(self, name: str) -> str:
        validate_name(name, "dataset name")
        if len(name) > 40:
            raise IngestError("dataset names are limited to 40 characters")
        if name in RESERVED or name in self.builtin_names():
            raise IngestError(
                f"dataset name {name!r} is reserved (built-in profile or internal schema)",
                [{"code": "reserved_name", "message": "reserved dataset name"}],
            )
        return name

    def versions(self, name: str) -> list[dict[str, Any]]:
        cur = self.con.execute(
            "SELECT ingest_id, dataset, version, status, created_by, created_at, registered_at, staging_schema, "
            "semantic_schema, files_json, mapping_hash, validation_json, profile_hash, snapshot, software_version, "
            "errors_json FROM meta.dataset_version WHERE dataset = ? ORDER BY version DESC, created_at DESC",
            [name],
        )
        cols = [d[0] for d in cur.description]
        out = []
        for r in cur.fetchall():
            row = dict(zip(cols, r, strict=True))
            for k in ("files_json", "validation_json", "errors_json"):
                row[k.removesuffix("_json")] = json.loads(row.pop(k) or "null")
            out.append(row)
        return out

    def _usable(self, name: str) -> list[dict[str, Any]]:
        return [v for v in self.versions(name) if v["status"] in ("staged", "registered")]

    # ---- ingestion -----------------------------------------------------------------------------
    def ingest(
        self,
        dataset: str,
        files: Sequence[tuple[str | None, Path | Upload]],
        actor: str,
        mapping: DatasetMapping | None = None,
        new_version: bool = False,
        dry_run: bool = False,
        null_values: tuple[str, ...] = (),
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        self.check_dataset_name(dataset)
        if not files:
            raise IngestError("give at least one CSV file")
        existing = self._usable(dataset)
        if existing and not new_version and not dry_run:
            v = existing[0]
            raise IngestError(
                f"dataset {dataset!r} already exists (v{v['version']}, {v['status']}); pass "
                "--new-version to import a new immutable version (nothing is overwritten)",
                [{"code": "dataset_exists", "message": "dataset exists"}],
            )
        version = max((v["version"] for v in self.versions(dataset)), default=0) + 1
        stg, sem = f"stg_{dataset}_v{version}", f"sem_{dataset}_v{version}"
        limit = max_bytes or max_bytes_from_env()
        ingest_id = str(uuid.uuid4())
        report: dict[str, Any] = {
            "dataset": dataset,
            "version": version,
            "ingest_id": ingest_id,
            "dry_run": dry_run,
            "staging_schema": stg,
            "files": [],
            "mapping": None,
        }
        tables: list[str] = []
        profile: dict[str, Any] = {}
        pending: list[tuple[Path, Path]] = []
        self.con.execute("BEGIN")
        try:
            self.con.execute(f"CREATE SCHEMA {qi(stg)}")
            for table, src in files:
                if isinstance(src, Upload):
                    info: FileInfo = check_text(src.name, src.text, limit)
                    name = validate_name(table or default_table_name(Path(src.name)))
                else:
                    info = check_file(Path(src), limit)
                    name = validate_name(table or default_table_name(Path(src)))
                if name in tables:
                    raise IngestError(
                        f"table name {name!r} is used for two files",
                        [{"code": "duplicate_table", "message": "two files map to one table"}],
                    )
                tables.append(name)
                stream = text_stream(src.text) if isinstance(src, Upload) else open_text(Path(src))
                try:
                    with stream:
                        prof: TableProfile = stage_table(self.con, stg, name, stream, info, null_values)
                except IngestError as exc:
                    raise IngestError(f"{info.name}: {exc}", [{**p, "file": info.name} for p in exc.problems]) from exc
                report["files"].append(prof.as_dict())
            status = "staged"
            validation = None
            if mapping is not None:
                validation, profile = self._register(dataset, version, stg, sem, mapping, report["files"], ingest_id)
                report["mapping"] = validation
                status = "registered"
            report["status"] = "dry_run" if dry_run else status
            if dry_run:
                self.con.execute("ROLLBACK")
                return report
            self.con.execute(
                "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status, created_by, created_at, "
                "registered_at, staging_schema, semantic_schema, files_json, mapping_json, mapping_hash, "
                "validation_json, profile_hash, snapshot, software_version, errors_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    ingest_id,
                    dataset,
                    version,
                    status,
                    actor,
                    now(),
                    now() if mapping else None,
                    stg,
                    sem if mapping else None,
                    dumps(report["files"]),
                    dumps(mapping.model_dump(mode="json")) if mapping else None,
                    mapping.content_hash() if mapping else None,
                    dumps(validation),
                    sha(profile) if profile else None,
                    profile.get("snapshot_sql"),
                    __version__,
                    None,
                ],
            )
            if mapping is not None:
                pending = self._write_profiles(dataset, version, profile)
            self.con.execute("COMMIT")
        except Exception as exc:
            self.con.execute("ROLLBACK")
            self._discard(pending)
            problems = (
                exc.problems
                if isinstance(exc, IngestError)
                else [{"code": "internal", "message": f"{type(exc).__name__} during ingestion"}]
            )
            if not dry_run:
                self._record_failure(ingest_id, dataset, version, actor, problems, report["files"])
            self.store.audit(
                actor,
                "dataset.ingest",
                "dataset",
                f"{dataset}:v{version}",
                "failed",
                None,
                {"problems": len(problems), "dry_run": dry_run},
            )
            if isinstance(exc, IngestError):
                raise
            raise IngestError(f"ingestion failed ({type(exc).__name__}); nothing was registered", problems) from exc
        self._commit_profiles(pending)
        self.store.audit(
            actor,
            "dataset.ingest",
            "dataset",
            f"{dataset}:v{version}",
            "success",
            None,
            {"status": status, "tables": tables, "files": [f["sha256"] for f in report["files"]]},
        )
        report["profile"] = f"{dataset}" if mapping is not None else None
        return report

    def register_mapping(
        self, dataset: str, mapping: DatasetMapping, actor: str, version: int | None = None
    ) -> dict[str, Any]:
        """Register a mapping for a staged version (same version, no new data)."""
        usable = self._usable(dataset)
        row = next((v for v in usable if version is None or v["version"] == version), None)
        if row is None:
            raise IngestError(f"no staged version of dataset {dataset!r}" + (f" v{version}" if version else ""))
        if row["status"] == "registered":
            if row["mapping_hash"] == mapping.content_hash():
                return {"dataset": dataset, "version": row["version"], "status": "registered", "unchanged": True}
            raise IngestError(
                f"{dataset} v{row['version']} is already registered with a different mapping; ingest "
                "a new version to change it (registered mappings are immutable)",
                [{"code": "mapping_exists", "message": "version already registered"}],
            )
        v, stg = row["version"], row["staging_schema"]
        sem = f"sem_{dataset}_v{v}"
        pending: list[tuple[Path, Path]] = []
        self.con.execute("BEGIN")
        try:
            validation, profile = self._register(dataset, v, stg, sem, mapping, row["files"], row["ingest_id"])
            self.con.execute(
                "UPDATE meta.dataset_version SET status='registered', registered_at=?, semantic_schema=?, "
                "mapping_json=?, mapping_hash=?, validation_json=?, profile_hash=?, snapshot=? WHERE ingest_id=?",
                [
                    now(),
                    sem,
                    dumps(mapping.model_dump(mode="json")),
                    mapping.content_hash(),
                    dumps(validation),
                    sha(profile),
                    profile["snapshot_sql"],
                    row["ingest_id"],
                ],
            )
            pending = self._write_profiles(dataset, v, profile)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            self._discard(pending)
            self.store.audit(actor, "dataset.register", "dataset", f"{dataset}:v{v}", "failed", None, {})
            raise
        self._commit_profiles(pending)
        self.store.audit(
            actor,
            "dataset.register",
            "dataset",
            f"{dataset}:v{v}",
            "success",
            None,
            {"mapping_hash": mapping.content_hash()},
        )
        return {"dataset": dataset, "version": v, "status": "registered", "mapping": validation}

    # ---- mapping -> semantic layer + checks -----------------------------------------------------
    def _schema(self, stg: str) -> Schema:
        rows = self.con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = ? "
            "ORDER BY table_name, ordinal_position",
            [stg],
        ).fetchall()
        out: Schema = {}
        for t, c, typ in rows:
            out.setdefault(t, {})[c] = str(typ).upper()
        return out

    def _register(
        self,
        dataset: str,
        version: int,
        stg: str,
        sem: str,
        m: DatasetMapping,
        files: list[dict[str, Any]],
        ingest_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {t: c for t, c in self._schema(stg).items() if t != "person_map"}
        errors, warnings = validate_mapping(m, schema, self.ont)
        if errors:
            raise IngestError(
                "mapping does not fit the imported tables: " + "; ".join(errors),
                [{"code": "invalid_mapping", "message": e} for e in errors],
            )
        use_map = needs_person_map(m, schema)
        if use_map:
            exists = fetch_count(
                self.con,
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = 'person_map'",
                [stg],
            )
            if not exists:
                self.con.execute(person_map_sql(m, stg))
        statements = semantic_sql(m, stg, sem, use_map, self.ont.vocab_schema())
        for stmt in statements:
            self.con.execute(stmt)
        checks, more_errors, more_warnings = self._data_checks(m, stg, sem, use_map)
        if more_errors:
            raise IngestError(
                "mapped data cannot be used: " + "; ".join(more_errors),
                [{"code": "invalid_data", "message": e} for e in more_errors],
            )
        source = {
            "kind": "csv",
            "dataset": dataset,
            "version": version,
            "ingest_id": ingest_id,
            "files": [
                {"table": f["table"], "file": f["file"], "sha256": f["sha256"], "rows": f["rows"]} for f in files
            ],
            "mapping_hash": m.content_hash(),
            "content_hash": sha(sorted(f["sha256"] for f in files) + [m.content_hash()]),
        }
        profile = build_profile(m, dataset, version, stg, sem, use_map, self.ont, source)
        validation = {
            "valid": True,
            "warnings": warnings + more_warnings,
            "checks": checks,
            "entities": sorted(m.entities),
            "observation": m.observation.mode,
            "absence_inference": m.observation.absence_inference,
            "surrogate_person_ids": use_map,
        }
        return validation, profile

    def _data_checks(
        self, m: DatasetMapping, stg: str, sem: str, use_map: bool
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Counts only (never values): duplicates, unusable rows, unmapped concepts, orphans, observation."""
        con, errors, warnings = self.con, [], []
        checks: dict[str, Any] = {}
        persons = fetch_count(con, f"SELECT count(DISTINCT person_id) FROM {qi(sem)}.person")
        conflicting = fetch_count(con, f"SELECT count(*) - count(DISTINCT person_id) FROM {qi(sem)}.person")
        checks["person"] = {"persons": persons, "conflicting_records": conflicting}
        if persons == 0:
            errors.append("the person table has no usable rows")
        if conflicting:
            errors.append(f"{conflicting} person id(s) have conflicting gender or birth year across rows")
        from .dataset_mapping import _view

        for entity, e in sorted(m.entities.items()):
            view = f"{qi(sem)}.{qi(_view(entity))}"
            staged = fetch_count(con, f"SELECT count(*) FROM {qi(stg)}.{qi(e.table)}")
            usable = fetch_count(con, f"SELECT count(*) FROM {view}")
            unmapped = fetch_count(con, f"SELECT count(*) FROM {view} WHERE concept_id = 0")
            orphans = fetch_count(
                con,
                f"SELECT count(DISTINCT person_id) FROM {view} v WHERE NOT EXISTS "
                f"(SELECT 1 FROM {qi(sem)}.person p WHERE p.person_id = v.person_id)",
            )
            checks[entity] = {
                "staged_rows": staged,
                "usable_rows": usable,
                "unmapped_concept_rows": unmapped,
                "persons_missing_from_person_table": orphans,
            }
            if usable == 0:
                errors.append(f"{entity}: no usable rows (every row lacks a patient id or a valid date)")
                continue
            if usable < staged and not e.code_column:
                warnings.append(f"{entity}: {staged - usable} row(s) dropped (missing patient id or invalid date)")
            if unmapped == usable:
                warnings.append(
                    f"{entity}: no row maps to a standard concept; criteria on {entity} will find nothing "
                    "(check the vocabulary or concept column)"
                )
            elif unmapped:
                warnings.append(f"{entity}: {unmapped} of {usable} row(s) do not map to a standard concept")
            if orphans:
                warnings.append(
                    f"{entity}: {orphans} patient(s) are missing from the person table and cannot enter cohorts"
                )
        if m.observation.mode != "none":
            observed = fetch_count(con, f"SELECT count(DISTINCT person_id) FROM {qi(sem)}.observation_period")
            checks["observation"] = {"persons_with_observation": observed}
            if observed == 0:
                errors.append("no usable observation periods")
            elif observed < persons:
                warnings.append(f"{persons - observed} person(s) have no observation period and cannot enter cohorts")
        for r in m.relationships:
            orphan = fetch_count(
                con,
                f"SELECT count(*) FROM {qi(stg)}.{qi(r.from_table)} a WHERE a.{qi(r.from_column)} IS NOT NULL "
                f"AND NOT EXISTS (SELECT 1 FROM {qi(stg)}.{qi(r.to_table)} b WHERE CAST(b.{qi(r.to_column)} AS "
                f"VARCHAR) = CAST(a.{qi(r.from_column)} AS VARCHAR))",
            )
            checks[f"relationship {r.from_table}.{r.from_column}->{r.to_table}.{r.to_column}"] = {"orphans": orphan}
            if orphan:
                warnings.append(
                    f"relationship {r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}: "
                    f"{orphan} row(s) have no match"
                )
        checks["surrogate_person_ids"] = use_map
        return checks, errors, warnings

    # ---- profile files (written only after the transaction commits) ------------------------------
    def _write_profiles(self, dataset: str, version: int, profile: dict[str, Any]) -> list[tuple[Path, Path]]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for name in (f"{dataset}__v{version}", dataset):
            body = {**profile, "name": name}
            fd, tmp = tempfile.mkstemp(dir=self.local_dir, prefix=".tmp-", suffix=".yaml")
            with os.fdopen(fd, "w") as fh:
                fh.write("# Generated by cohort-builder csv ingestion. Do not edit: register a new version instead.\n")
                yaml.safe_dump(body, fh, sort_keys=False)
            pending.append((Path(tmp), self.local_dir / f"{name}.yaml"))
        return pending

    @staticmethod
    def _commit_profiles(pending: list[tuple[Path, Path]]) -> None:
        for tmp, final in pending:
            os.replace(tmp, final)

    @staticmethod
    def _discard(pending: list[tuple[Path, Path]]) -> None:
        for tmp, _ in pending:
            tmp.unlink(missing_ok=True)

    def _record_failure(
        self,
        ingest_id: str,
        dataset: str,
        version: int,
        actor: str,
        problems: list[dict[str, Any]],
        files: list[dict[str, Any]],
    ) -> None:
        self.con.execute(
            "INSERT INTO meta.dataset_version (ingest_id, dataset, version, status, created_by, created_at, "
            "files_json, software_version, errors_json) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                ingest_id,
                dataset,
                version,
                "failed",
                actor,
                now(),
                dumps([{k: f[k] for k in ("table", "file", "sha256", "rows")} for f in files]),
                __version__,
                dumps(problems[:50]),
            ],
        )

    # ---- reads -----------------------------------------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        names = [
            r[0] for r in self.con.execute("SELECT DISTINCT dataset FROM meta.dataset_version ORDER BY 1").fetchall()
        ]
        out = []
        for n in names:
            versions = self.versions(n)
            usable = [v for v in versions if v["status"] in ("staged", "registered")]
            latest = usable[0] if usable else versions[0]
            out.append(
                {
                    "dataset": n,
                    "latest_version": latest["version"],
                    "status": latest["status"],
                    "versions": len(versions),
                    "selectable": (self.local_dir / f"{n}.yaml").is_file(),
                }
            )
        return out

    def info(self, dataset: str) -> dict[str, Any]:
        versions = self.versions(dataset)
        if not versions:
            raise KeyError(f"dataset {dataset!r} not found")
        profile_path = self.local_dir / f"{dataset}.yaml"
        profile = yaml.safe_load(profile_path.read_text()) if profile_path.is_file() else None
        return {
            "dataset": dataset,
            "selectable_as": dataset if profile else None,
            "capabilities": profile["capabilities"] if profile else None,
            "source": profile.get("source") if profile else None,
            "versions": [
                {
                    k: v[k]
                    for k in (
                        "version",
                        "status",
                        "created_by",
                        "created_at",
                        "registered_at",
                        "staging_schema",
                        "semantic_schema",
                        "mapping_hash",
                        "snapshot",
                        "software_version",
                        "files",
                        "validation",
                        "errors",
                    )
                }
                for v in versions
            ],
        }
