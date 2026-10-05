"""Command-line interface: `cohort-builder <command>`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import Settings
from .db import connect


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _attrition_table(rows: list[dict]) -> str:
    width = max((len(r["name"]) for r in rows), default=10)
    return "\n".join(f"  {r['sequence']:>2}  {r['name']:<{width}}  {r['remaining']:>8}" for r in rows)


def _builder(settings: Settings):
    from .orchestrator import CohortBuilder

    if not settings.db_path.exists():
        sys.exit(f"Database {settings.db_path} not found. Run `cohort-builder init-demo` (or load-athena) first.")
    return CohortBuilder(settings)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cohort-builder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-demo", help="create the demo vocabulary and synthetic patients")
    s.add_argument("--persons", type=int, default=5000)
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
    s.add_argument("--allow-draft", action="store_true")
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
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    settings = Settings.from_env()

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

        uvicorn.run(create_app(), host=a.host, port=a.port)
        return 0
    if a.cmd == "mcp":
        from .mcp_server import create_server, run_http

        if not settings.db_path.exists():
            sys.exit(f"Database {settings.db_path} not found. Run `cohort-builder init-demo` (or load-athena) first.")
        if a.http:
            run_http(a.host, a.port)
        else:
            create_server().run("stdio")
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
        print(b.explainer.explain(ir) + "\n")
        _print({k: v for k, v in row.items() if k != "ir"})
    elif a.cmd in ("approve", "reject"):
        b.review(a.definition_id, a.reviewer, "approved" if a.cmd == "approve" else "rejected", a.comments)
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
        print(b.compile_sql(a.definition_id))
    elif a.cmd == "execute":
        r = b.execute(a.definition_id, a.user, a.allow_draft)
        print(f"generation {r['generation_id']}: {r['person_count']} people written to results.cohort\n")
        print(_attrition_table(r["attrition"]))
    elif a.cmd == "run":
        _print(b.store.get_run(a.run_id))
    elif a.cmd == "replay":
        _print(b.replay(a.run_id))
    elif a.cmd == "search":
        _print(b.vocab.search_concepts(a.text, a.domain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
