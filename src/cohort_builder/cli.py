"""Command-line interface: `cohort-builder <command>`."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from .config import Settings
from .db import connect


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _attrition_table(rows: list[dict]) -> str:
    width = max((len(r["name"]) for r in rows), default=10)
    return "\n".join(f"  {r['sequence']:>2}  {r['name']:<{width}}  {r['remaining']:>8}" for r in rows)


def _builder(settings: Settings):
    import duckdb

    from .orchestrator import CohortBuilder
    from .security import ConfigError, GovernancePolicy, SecurityConfig

    if not settings.db_path.exists():
        sys.exit(f"Database {settings.db_path} not found. Run `cohort-builder init-demo` (or load-athena) first.")
    try:
        policy = GovernancePolicy.from_security(SecurityConfig.from_env())
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    try:
        return CohortBuilder(settings, policy=policy)
    except duckdb.IOException as exc:
        # DuckDB allows one writing process per database file (e.g. `serve` and `mcp` at the same time)
        sys.exit(f"cannot open {settings.db_path}: {str(exc).splitlines()[0]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cohort-builder", description=__doc__)
    p.add_argument("--dataset", help="dataset profile from ontology/datasets/ (default: CB_DATASET or omop_demo)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="list dataset profiles (built-in and imported) and what each can answer")
    s = sub.add_parser("csv", help="import CSV files as a dataset (profile, ingest, map, info)")
    cs = s.add_subparsers(dest="csv_cmd", required=True)
    t = cs.add_parser("profile", help="profile one CSV (types, nulls, keys, data-quality warnings); writes nothing")
    t.add_argument("file", type=Path)
    t.add_argument("--table", help="destination table name to validate (default: from the file name)")
    t.add_argument("--null-value", action="append", default=[], help="extra text meaning 'missing' (repeatable)")
    t.add_argument(
        "--type",
        action="append",
        default=[],
        metavar="COLUMN=TYPE",
        help="explicit column type (VARCHAR, BIGINT, DOUBLE, DATE, BOOLEAN); every value must convert exactly",
    )
    t.add_argument("--json", action="store_true")
    t = cs.add_parser(
        "ingest",
        help="load CSV files into an isolated staging schema as a new immutable dataset version",
        description="Each --file becomes one staging table. With --mapping the version is also registered as a "
        "dataset profile selectable with --dataset. Failures roll back completely (exit code 1).",
    )
    t.add_argument("--dataset", required=True, help="dataset name (lower-case letters, digits, underscores)")
    t.add_argument(
        "--file",
        action="append",
        required=True,
        metavar="[TABLE=]PATH",
        help="CSV/TSV file, optionally prefixed with the staging table name (repeatable)",
    )
    t.add_argument("--mapping", type=Path, help="mapping YAML (see examples/csv/mapping.yaml)")
    t.add_argument("--new-version", action="store_true", help="required when the dataset name already exists")
    t.add_argument("--dry-run", action="store_true", help="validate everything, then roll back (nothing kept)")
    t.add_argument("--null-value", action="append", default=[])
    t.add_argument(
        "--type",
        action="append",
        default=[],
        metavar="TABLE.COLUMN=TYPE",
        help="explicit column type, e.g. claims.member_id=VARCHAR (repeatable; validated against every value)",
    )
    t.add_argument("--user", default="cli")
    t.add_argument("--json", action="store_true")
    t = cs.add_parser("map", help="register a mapping for a staged (unmapped) dataset version")
    t.add_argument("--dataset", required=True)
    t.add_argument("--mapping", type=Path, required=True)
    t.add_argument("--version", type=int)
    t.add_argument("--user", default="cli")
    t = cs.add_parser("info", help="versions, files, columns, validation results and capabilities of a dataset")
    t.add_argument("dataset")
    t = cs.add_parser(
        "reconcile",
        help="repair profile files after an interrupted import (publish pending versions, remove stale files)",
        description="Run when no import is in progress. Uses only what the database recorded; never re-reads CSVs.",
    )
    t.add_argument("dataset", nargs="?", help="one dataset (default: all imported datasets)")
    t.add_argument(
        "--abandoned-after-minutes",
        type=int,
        default=60,
        help="treat unfinished imports and temp files older than this as abandoned (default 60; 0 = all)",
    )
    t.add_argument("--user", default="cli")

    s = sub.add_parser("init-demo", help="create the demo vocabulary and synthetic patients")
    s.add_argument("--persons", type=int, default=5000, help="patients per demo source (OMOP and LAAD-style)")
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("load-athena", help="load an OHDSI Athena vocabulary download")
    s.add_argument("directory", type=Path)

    s = sub.add_parser("ask", help="build a cohort definition from a natural-language request")
    s.add_argument("query")
    s.add_argument("--user", default="cli")
    s.add_argument("--json", action="store_true")
    sub.add_parser("list", help="list cohort definitions")
    s = sub.add_parser("show", help="show a cohort definition")
    s.add_argument("definition_id", type=int)
    for name in ("approve", "reject"):
        s = sub.add_parser(name, help=f"{name} a cohort definition")
        s.add_argument("definition_id", type=int)
        s.add_argument("--reviewer", required=True)
        s.add_argument("--comments", default="")
    s = sub.add_parser("submit-ir", help="save a hand-written or edited IR JSON file as a draft")
    s.add_argument("file", type=Path)
    s.add_argument("--user", default="cli")
    s.add_argument("--parent", type=int)
    s = sub.add_parser("dry-run", help="validate an IR file and print attrition (no LLM, nothing saved)")
    s.add_argument("file", type=Path)
    s = sub.add_parser("compile", help="print the SQL for a definition")
    s.add_argument("definition_id", type=int)
    s = sub.add_parser("execute", help="materialize an approved definition into results.cohort")
    s.add_argument("definition_id", type=int)
    s.add_argument("--user", default="cli")
    s.add_argument(
        "--allow-draft",
        action="store_true",
        help="run a draft (only honored when CB_ENV=development and CB_ALLOW_DRAFT_EXECUTION=true)",
    )
    s = sub.add_parser("run", help="show a run and its manifest")
    s.add_argument("run_id")
    s = sub.add_parser("replay", help="replay a run from recorded LLM responses and compare")
    s.add_argument("run_id")
    s = sub.add_parser("search", help="search the vocabulary")
    s.add_argument("text")
    s.add_argument("--domain")
    s = sub.add_parser("eval", help="run the golden-case evaluation")
    s.add_argument("--cases", type=Path, default=Path("eval/golden_cases.yaml"))
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--case", action="append", dest="case_ids")
    s.add_argument("--mode", default="live", choices=["live", "cached", "replay"])
    s.add_argument("--out", type=Path)
    s = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    s.add_argument("--http", action="store_true", help="serve Streamable HTTP at /mcp instead of stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("auth", help="API credentials")
    auth = s.add_subparsers(dest="auth_cmd", required=True)
    t = auth.add_parser("issue-token", help="create a bearer token; prints it once plus the hashed file entry")
    t.add_argument("--subject", required=True, help="user identity, e.g. alice@example.org")
    t.add_argument("--roles", required=True, help="comma-separated: viewer,author,reviewer,executor,admin")
    t.add_argument("--tenant", default="default")
    t.add_argument("--expires-days", type=int, default=90, help="0 = no expiry (not recommended)")
    t.add_argument("--id", dest="token_id", help="label shown in audit records (default: subject + random)")
    t.add_argument("--tokens-file", type=Path, help="append the hashed entry to this token file (created 0600)")
    s = sub.add_parser(
        "treatment",
        help="treatment events, episodes and transition evidence (configurable rule sets; aggregate output)",
    )
    tx = s.add_subparsers(dest="treatment_cmd", required=True)
    t = tx.add_parser("run", help="build events and episodes for the active dataset with a rule set")
    t.add_argument("--rules", required=True, help="rule set name (ontology/treatment/rules) or a YAML file path")
    t.add_argument("--sources", type=Path, help="treatment source config (default: ontology/treatment/sources)")
    t.add_argument("--reference-date", help="fixed 'as of' date YYYY-MM-DD (default: today; recorded)")
    t.add_argument(
        "--engine", choices=["sql", "reference"], default="sql", help="'reference' = in-memory oracle (small runs)"
    )
    t.add_argument("--batch-size", type=int, help="rows per fetch/write batch (default CB_TREATMENT_BATCH_SIZE)")
    t.add_argument(
        "--reuse", action="store_true", help="return a completed run with the same input fingerprint, if any"
    )
    t.add_argument("--user", default="cli")
    t = tx.add_parser("cancel", help="cancel a run; a 'running' run with no live process is marked abandoned")
    t.add_argument("run_id")
    t.add_argument("--user", default="cli")
    for name, hlp in (("hold", "place a legal hold (retention never purges)"), ("release", "release a hold")):
        t = tx.add_parser(name, help=hlp)
        t.add_argument("run_id")
        if name == "hold":
            t.add_argument("--reason", required=True)
        t.add_argument("--user", default="cli")
    t = tx.add_parser(
        "retention",
        help="purge derived results of old runs (dry run unless --apply; source data is never deleted)",
    )
    t.add_argument("--older-than-days", type=int, help="default CB_TREATMENT_RETENTION_DAYS (must be >= 1)")
    t.add_argument("--tenant", help="only this tenant's runs")
    t.add_argument("--archive-dir", type=Path, help="verified archive before purging (below CB_TREATMENT_ARCHIVE_ROOT)")
    t.add_argument("--apply", action="store_true", help="actually purge (default: dry run, nothing changes)")
    t.add_argument("--user", default="cli")
    t = tx.add_parser("summary", help="provenance and suppressed aggregate summary of a run")
    t.add_argument("run_id")
    t = tx.add_parser("cleanup", help="mark stale 'running' runs abandoned and drop leftover staging tables")
    t.add_argument("--older-than-minutes", type=int, default=None, help="default CB_TREATMENT_STALE_RUN_MINUTES")
    t.add_argument("--user", default="cli")
    t = tx.add_parser("explain", help="one episode with its source events and rules (needs CB_ALLOW_PATIENT_LEVEL)")
    t.add_argument("run_id")
    t.add_argument("episode_id")
    t.add_argument("--user", default="cli")

    s = sub.add_parser("proxy", help="proxy (indirect) identification algorithms, e.g. for rare diseases")
    px = s.add_subparsers(dest="proxy_cmd", required=True)
    for name, hlp in (
        ("validate", "validate + dry-run a proxy YAML/JSON file (nothing saved)"),
        ("submit", "save a proxy YAML/JSON file as an immutable draft version"),
    ):
        t = px.add_parser(name, help=hlp)
        t.add_argument("file", type=Path)
        t.add_argument("--user", default="cli")
        t.add_argument("--parent", type=int)
    t = px.add_parser("ask", help="draft a proxy algorithm from natural language (never executes)")
    t.add_argument("query")
    t.add_argument("--user", default="cli")
    px.add_parser("list", help="list proxy definitions")
    for name in ("show", "review-packet", "sql", "versions"):
        t = px.add_parser(name)
        t.add_argument("definition_id", type=int)
    for name in ("approve", "reject"):
        t = px.add_parser(name)
        t.add_argument("definition_id", type=int)
        t.add_argument("--reviewer", required=True)
        t.add_argument("--comments", default="")
    t = px.add_parser("execute", help="run an approved proxy definition")
    t.add_argument("definition_id", type=int)
    t.add_argument("--user", default="cli")
    t.add_argument("--allow-draft", action="store_true")
    t = px.add_parser("results", help="suppressed attrition and evidence summary of the latest (or given) run")
    t.add_argument("definition_id", type=int)
    t.add_argument("--generation")
    t = px.add_parser("compare", help="privacy-safe overlap of 2-6 generations")
    t.add_argument("generation_ids", nargs="+")
    t.add_argument("--user", default="cli")
    t = px.add_parser(
        "load-reference",
        help="load EXTERNAL reference-standard labels (CSV: person_id, label or is_case, optional reference_date)",
    )
    t.add_argument("name")
    t.add_argument("csv", type=Path)
    t.add_argument("--source", required=True, help="what the reference standard is, e.g. 'chart review 2026'")
    t.add_argument("--user", default="cli")
    t = px.add_parser("validate-reference", help="evaluate against a loaded reference (metrics + criteria check)")
    t.add_argument("definition_id", type=int)
    t.add_argument("--reference", required=True)
    t.add_argument("--generation")
    t.add_argument("--tiers", help="comma-separated tiers counted as positive (default: all)")
    t.add_argument("--intended-use", help="intended use; uses criteria prespecified in the definition")
    t.add_argument("--criteria", type=Path, help="JSON/YAML acceptance criteria (if not prespecified)")
    t.add_argument("--require-data", default="", help="comma-separated entities patients must have, e.g. Measurement")
    t.add_argument("--min-observation-days", type=int)
    t.add_argument("--user", default="cli")
    t = px.add_parser("status", help="definition, approval, evaluation, acceptance and lifecycle status")
    t.add_argument("definition_id", type=int)
    t = px.add_parser("review-evaluation", help="human acceptance decision for one evaluation")
    t.add_argument("definition_id", type=int)
    t.add_argument("validation_id")
    t.add_argument("--reviewer", required=True)
    t.add_argument("--decision", required=True, choices=["accepted", "rejected"])
    t.add_argument("--rationale", required=True)
    t = px.add_parser("explain-patient", help="per-patient evidence (needs CB_ALLOW_PATIENT_LEVEL=true)")
    t.add_argument("definition_id", type=int)
    t.add_argument("subject_id", type=int)
    t.add_argument("--generation")
    t.add_argument("--user", default="cli")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    settings = Settings.from_env()
    if a.dataset:
        settings = dataclasses.replace(settings, dataset=a.dataset)

    if a.cmd == "auth":
        import os

        import yaml

        from .security import issue_token

        token, entry = issue_token(
            a.subject,
            [r.strip() for r in a.roles.split(",") if r.strip()],
            a.tenant,
            a.expires_days or None,
            a.token_id,
        )
        print("Bearer token (shown once; give it to the user over a secure channel, never commit it):", file=sys.stderr)
        print(token, file=sys.stderr)
        if a.tokens_file:
            doc = yaml.safe_load(a.tokens_file.read_text()) if a.tokens_file.exists() else None
            doc = doc or {"tokens": []}
            doc.setdefault("tokens", []).append(entry)
            fd = os.open(a.tokens_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                yaml.safe_dump(doc, fh, sort_keys=False)
            print(f"\nEntry for {a.subject} written to {a.tokens_file} (hash only).", file=sys.stderr)
        else:
            print("\nAdd this entry under `tokens:` in the file named by CB_AUTH_TOKENS_FILE:\n", file=sys.stderr)
            print(yaml.safe_dump([entry], sort_keys=False).rstrip())
        return 0

    if a.cmd == "csv":
        return _csv(settings, a)
    if a.cmd == "datasets":
        import yaml

        from .ontology import Ontology

        base = Ontology.load(settings.ontology_dir)
        builtin = sorted((settings.ontology_dir / base.domain["datasets_dir"]).glob("*.yaml"))
        local = sorted(settings.local_datasets_dir.glob("*.yaml")) if settings.local_datasets_dir.is_dir() else []
        for path in builtin + [p for p in local if "__v" not in p.stem]:
            prof = yaml.safe_load(path.read_text())
            caps = prof["capabilities"]
            print(f"{prof['name']} (v{prof.get('version')}, {prof.get('data_type')}): {prof.get('description', '')}")
            print(f"    entities: {', '.join(caps['entities'])}")
            print(
                f"    attributes: {', '.join(caps.get('attributes', [])) or '-'}; "
                f"observation: {caps.get('observation')}"
            )
        return 0

    if a.cmd == "init-demo":
        from .synthetic import build_demo_database

        if settings.db_path.exists():
            settings.db_path.unlink()
        con = connect(settings.db_path)
        _print(build_demo_database(con, a.persons, a.seed))
        print(f"Demo database written to {settings.db_path}")
        return 0
    if a.cmd == "load-athena":
        from .synthetic import load_athena

        _print(load_athena(connect(settings.db_path), a.directory))
        return 0
    if a.cmd == "serve":
        import uvicorn

        from .api import create_app
        from .security import ConfigError

        try:
            app = create_app()
        except ConfigError as exc:
            sys.exit(f"refusing to start: {exc}")
        uvicorn.run(app, host=a.host, port=a.port)
        return 0
    if a.cmd == "mcp":
        from .mcp_server import create_server, run_http

        builder = _builder(settings)  # open the database now: fail fast (e.g. file locked by another process)
        if a.http:
            run_http(a.host, a.port, builder)
        else:
            create_server(builder).run("stdio")
        return 0
    if a.cmd == "eval":
        from .evaluation import run_eval
        from .orchestrator import CohortBuilder

        b = CohortBuilder(dataclasses.replace(settings, llm_mode=a.mode))
        report = run_eval(b, a.cases, a.repeats, a.case_ids)
        if a.out:
            a.out.write_text(json.dumps(report, indent=2, default=str))
        _print({k: report[k] for k in ("eval_run_id", "summary", "thresholds", "passed", "failed_metrics")})
        return 0 if report["passed"] else 1

    b = _builder(settings)
    if a.cmd == "proxy":
        return _proxy(b, a)
    if a.cmd == "treatment":
        return _treatment(b, a)
    if a.cmd == "ask":
        r = b.ask(a.query, a.user)
        if a.json:
            _print(r.as_dict())
            return 0 if r.status != "failed" else 1
        print(f"run {r.run_id}  status={r.status}  definition={r.cohort_definition_id}\n")
        if r.explanation:
            print(r.explanation + "\n")
        if r.attrition:
            print("Attrition (dry run, small cells suppressed):\n" + _attrition_table(r.attrition) + "\n")
        for i in r.issues:
            print(f"[{i['severity']}/{i['stage']}] {i['message']}")
        if r.critic_notes:
            print(f"Reviewer notes: {r.critic_notes}")
        if r.status == "draft":
            print(f"\nNext: cohort-builder approve {r.cohort_definition_id} --reviewer <name>")
        return 0 if r.status != "failed" else 1
    if a.cmd == "list":
        _print(b.store.list_definitions())
    elif a.cmd == "show":
        row, ir = b.load_definition(a.definition_id)
        print(b.explain(ir) + "\n")
        _print({k: v for k, v in row.items() if k != "ir"})
    elif a.cmd in ("approve", "reject"):
        from .orchestrator import GovernanceError

        try:
            b.review(a.definition_id, a.reviewer, "approved" if a.cmd == "approve" else "rejected", a.comments)
        except (GovernanceError, ValueError) as exc:
            sys.exit(f"refused: {exc}")
        print(f"definition {a.definition_id} {a.cmd}d by {a.reviewer}")
    elif a.cmd == "submit-ir":
        from .ir import CohortDefinition

        ir = CohortDefinition.model_validate_json(a.file.read_text())
        def_id, issues = b.submit_ir(ir, a.user, a.parent)
        _print({"cohort_definition_id": def_id, "issues": issues})
    elif a.cmd == "dry-run":
        from .agents.validator import validate
        from .ir import CohortDefinition

        ir = CohortDefinition.model_validate_json(a.file.read_text())
        issues, attrition = validate(ir, b.ontology, b.vocab, b.executor)
        print(b.explainer.explain(ir) + "\n")
        if attrition:
            print("Attrition:\n" + _attrition_table(attrition.suppressed(b.executor.min_cell)))
        for i in issues:
            print(f"[{i.severity}/{i.stage}] {i.message}")
        print(f"\ncontent hash  {ir.content_hash()}\nsemantic hash {ir.semantic_hash()}")
    elif a.cmd == "compile":
        try:
            print(b.compile_sql(a.definition_id))
        except (KeyError, ValueError) as exc:
            sys.exit(f"cannot compile: {str(exc).strip(chr(39))}")
    elif a.cmd == "execute":
        from .orchestrator import GovernanceError

        try:
            r = b.execute(a.definition_id, a.user, a.allow_draft)
        except (GovernanceError, ValueError) as exc:
            sys.exit(f"refused: {exc}")
        print(
            f"generation {r['generation_id']}: {r['person_count']} people written to results.cohort "
            f"(counts below {r['min_cell_count']} suppressed)\n"
        )
        print(_attrition_table(r["attrition"]))
        for c in r["caveats"]:
            print(f"[caveat/{c['stage']}] {c['message']}")
    elif a.cmd == "run":
        _print(b.store.get_run(a.run_id))
    elif a.cmd == "replay":
        _print(b.replay(a.run_id))
    elif a.cmd == "search":
        _print(b.vocab.search_concepts(a.text, a.domain))
    return 0


def _csv(settings: Settings, a) -> int:
    from .csv_ingest import IngestError, profile_csv

    if a.csv_cmd == "profile":
        types: dict[str, str] = {}
        for spec in a.type:
            col, sep, typ = spec.partition("=")
            if not sep or not col.strip() or not typ.strip():
                print("invalid input: --type looks like COLUMN=TYPE", file=sys.stderr)
                return 1
            types[col.strip()] = typ.strip().upper()
        try:
            report = profile_csv(a.file, a.table, tuple(a.null_value), column_types=types or None)
        except IngestError as exc:
            print(f"profiling failed: {exc}", file=sys.stderr)
            return 1
        if a.json:
            _print(report)
        elif not report["ok"]:
            print(f"{report['file']}: cannot be ingested")
            for p in report["errors"]:
                print(f"  [{p['code']}]{' line ' + str(p['line']) if p.get('line') else ''} {p['message']}")
        else:
            print(
                f"{report['file']} -> table {report['table']}: {report['rows']} rows, {len(report['columns'])} "
                f"columns, {report['duplicate_rows']} duplicate rows\n"
            )
            for c in report["columns"]:
                print(
                    f"  {c['name']:<28} {c['type']:<8} {c['kind']:<11} nulls {c['null_pct']:>6}%  distinct "
                    f"{c['distinct_count']:>8}  {','.join(c['flags'])}"
                )
            for w in report["warnings"]:
                print(f"[warning] {w}")
            print("\nSuggested roles (confirm them in a mapping file): " + json.dumps(report["suggestions"]))
        return 0 if report["ok"] else 1

    from .dataset_registry import DatasetRegistry, load_mapping, parse_type_overrides
    from .db import connect

    con = connect(settings.db_path)
    try:
        reg = DatasetRegistry(con, settings.ontology_dir, settings.local_datasets_dir)
        try:
            if a.csv_cmd == "ingest":
                files = []
                for spec in a.file:
                    table, sep, path = spec.partition("=")
                    files.append((table, Path(path)) if sep else (None, Path(spec)))
                mapping = load_mapping(a.mapping.read_text()) if a.mapping else None
                r = reg.ingest(
                    a.dataset,
                    files,
                    a.user,
                    mapping=mapping,
                    new_version=a.new_version,
                    dry_run=a.dry_run,
                    null_values=tuple(a.null_value),
                    column_types=parse_type_overrides(a.type) or None,
                )
                if a.json:
                    _print(r)
                else:
                    verb = "validated (dry run, nothing kept)" if a.dry_run else r["status"]
                    print(f"dataset {r['dataset']} v{r['version']}: {verb}")
                    for f in r["files"]:
                        print(f"  {f['file']} -> {r['staging_schema']}.{f['table']}: {f['rows']} rows")
                        for w in f["warnings"]:
                            print(f"    [warning] {w}")
                    if r["mapping"]:
                        for w in r["mapping"]["warnings"]:
                            print(f"  [mapping warning] {w}")
                    if r.get("profile"):
                        print(f"\nSelect it with: cohort-builder --dataset {r['profile']} ...")
                    elif not a.dry_run:
                        print(f"\nNext: cohort-builder csv map --dataset {r['dataset']} --mapping mapping.yaml")
            elif a.csv_cmd == "map":
                _print(reg.register_mapping(a.dataset, load_mapping(a.mapping.read_text()), a.user, a.version))
            elif a.csv_cmd == "info":
                _print(reg.info(a.dataset))
            elif a.csv_cmd == "reconcile":
                result = reg.reconcile(a.user, a.dataset, a.abandoned_after_minutes)
                _print(result)
                return 1 if result["problems"] else 0
        except IngestError as exc:
            print(f"ingestion failed: {exc}", file=sys.stderr)
            for p in exc.problems:
                where = (f" {p['file']}" if p.get("file") else "") + (f" line {p['line']}" if p.get("line") else "")
                print(f"  [{p['code']}]{where}: {p.get('message', '')}", file=sys.stderr)
            return 1
        except KeyError as exc:
            print(str(exc).strip("'\""), file=sys.stderr)
            return 1
        except ValueError as exc:  # invalid mapping file (pydantic) and similar
            print(f"invalid input: {exc}", file=sys.stderr)
            return 1
    finally:
        con.close()
    return 0


@contextlib.contextmanager
def _cancel_on_signals() -> Iterator[None]:
    """SIGINT/SIGTERM during a treatment run cancel it cleanly (status 'cancelled', staging dropped, nothing
    published) instead of killing the process mid-write. A second signal falls back to the default."""
    import signal

    from .treatment_service import cancel_all_active

    def handler(signum: int, _frame: object) -> None:
        signal.signal(signum, previous[signal.Signals(signum)])
        print("cancelling the treatment run (send the signal again to force)", file=sys.stderr)
        cancel_all_active("cancelled")

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous:
        signal.signal(s, handler)
    try:
        yield
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def _treatment(b, a) -> int:
    from datetime import date

    from .security import GovernanceError
    from .treatment_config import load_sources
    from .treatment_service import TreatmentRunFailed, TreatmentService, resolve_rules

    svc = TreatmentService(b)
    try:
        if a.treatment_cmd == "run":
            rules = resolve_rules(b.settings.ontology_dir, a.rules)
            sources = load_sources(a.sources.read_text()) if a.sources else None
            ref = date.fromisoformat(a.reference_date) if a.reference_date else None
            with _cancel_on_signals():
                _print(svc.run(rules, a.user, None, sources, ref, a.engine, a.batch_size, a.reuse))
        elif a.treatment_cmd == "cancel":
            _print(svc.cancel(a.run_id, a.user, None))
        elif a.treatment_cmd in ("hold", "release"):
            from .treatment_retention import set_hold

            hold = a.treatment_cmd == "hold"
            _print(set_hold(svc, a.run_id, a.user, hold, a.reason if hold else "", None))
        elif a.treatment_cmd == "retention":
            from .treatment_retention import apply_retention

            days = a.older_than_days if a.older_than_days is not None else b.settings.treatment_retention_days
            _print(apply_retention(svc, days, a.user, a.tenant, not a.apply, a.archive_dir))
        elif a.treatment_cmd == "cleanup":
            minutes = a.older_than_minutes if a.older_than_minutes is not None else svc.limits.stale_run_minutes
            _print(svc.cleanup_incomplete_runs(minutes, a.user))
        elif a.treatment_cmd == "summary":
            _print(svc.summary(a.run_id, None))
        elif a.treatment_cmd == "explain":
            # the CLI runs with direct database access (operator = admin); the policy flag still applies
            _print(svc.explain_episode(a.run_id, a.episode_id, a.user, True, None))
    except TreatmentRunFailed as exc:
        sys.exit(f"{exc.status}: {exc}")
    except (GovernanceError, ValueError, KeyError) as exc:
        sys.exit(f"refused: {str(exc).strip(chr(39))}")
    finally:
        svc.close()
    return 0


def _proxy(b, a) -> int:  # noqa: C901 - flat command dispatch
    import csv

    from pydantic import ValidationError

    from .agents.composer import format_validation_error
    from .proxy import DuplicateKeyError, load_json_strict
    from .proxy_service import VersionConflict
    from .security import GovernanceError

    c = a.proxy_cmd
    try:
        if c in ("validate", "submit"):
            text = a.file.read_text()
            try:
                p = b.parse_proxy_payload(text if a.file.suffix in (".yaml", ".yml") else load_json_strict(text))
            except DuplicateKeyError as exc:
                sys.exit(f"invalid proxy definition: {exc}")
            except ValidationError as exc:
                sys.exit("invalid proxy definition:\n- " + "\n- ".join(format_validation_error(exc)))
            if c == "validate":
                r = b.validate_proxy_definition(p)
                print(r["explanation"] + "\n")
                if r["attrition"]:
                    print("Attrition (dry run, small cells suppressed):\n" + _attrition_table(r["attrition"]) + "\n")
                for i in r["issues"]:
                    print(f"[{i['severity']}/{i['stage']}] {i['message']}")
                print(f"\ncontent hash  {r['content_hash']}\nsemantic hash {r['semantic_hash']}")
                return 0 if r["valid"] else 1
            r = b.submit_proxy(p, a.user, parent_id=a.parent)
            _print({k: r.get(k) for k in ("cohort_definition_id", "status", "existing", "issues")})
        elif c == "ask":
            _print(b.ask_proxy(a.query, a.user).as_dict())
        elif c == "list":
            _print(b.store.list_definitions(kind="proxy"))
        elif c == "show":
            row, p = b.load_proxy(a.definition_id)
            print(b.explain(p) + "\n")
            _print({k: v for k, v in row.items() if k != "ir"})
        elif c == "review-packet":
            _print(b.proxy_review_packet(a.definition_id))
        elif c == "sql":
            for name, sql in b.compile_proxy(a.definition_id)["statements"].items():
                print(f"-- {name}\n{sql}\n")
        elif c == "versions":
            _print(b.proxy_versions(a.definition_id))
        elif c in ("approve", "reject"):
            b.load_proxy(a.definition_id)
            b.review(a.definition_id, a.reviewer, "approved" if c == "approve" else "rejected", a.comments)
            print(f"proxy definition {a.definition_id} {c}d by {a.reviewer}")
        elif c == "execute":
            b.load_proxy(a.definition_id)
            _print(b.execute(a.definition_id, a.user, a.allow_draft))
        elif c == "results":
            _print(b.proxy_results(a.definition_id, generation_id=a.generation))
        elif c == "compare":
            _print(b.compare_generations(a.generation_ids, a.user))
        elif c == "load-reference":
            with a.csv.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            from datetime import date as _date

            labels = [
                (
                    int(r["person_id"]),
                    r.get("label") if r.get("label") not in (None, "") else r.get("is_case"),
                    _date.fromisoformat(r["reference_date"]) if r.get("reference_date") else None,
                )
                for r in rows
            ]
            _print(b.load_reference(a.name, labels, a.source, a.user))
        elif c == "validate-reference":
            from .proxy import AcceptanceCriteria, load_yaml_strict
            from .proxy_evaluation import EligibilityRules

            tiers = [t.strip() for t in a.tiers.split(",")] if a.tiers else None
            crit = AcceptanceCriteria.model_validate(load_yaml_strict(a.criteria.read_text())) if a.criteria else None
            rules = EligibilityRules(
                min_observation_days=a.min_observation_days,
                require_data=[x.strip() for x in a.require_data.split(",") if x.strip()],
            )
            _print(
                b.evaluate_against_reference(
                    a.definition_id, a.reference, a.user, None, a.generation, tiers, a.intended_use, crit, rules
                )
            )
        elif c == "status":
            _print(b.proxy_status(a.definition_id))
        elif c == "review-evaluation":
            _print(b.review_evaluation(a.definition_id, a.validation_id, a.reviewer, a.decision, a.rationale))
        elif c == "explain-patient":
            # the CLI runs with direct database access (operator = admin); the policy flag still applies
            _print(b.patient_explanation(a.definition_id, a.subject_id, a.user, True, None, a.generation))
    except (GovernanceError, VersionConflict, ValueError, KeyError) as exc:
        sys.exit(f"refused: {str(exc).strip(chr(39))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
