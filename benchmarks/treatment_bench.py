"""Reproducible SYNTHETIC benchmark for the treatment-event engine (no real data, ever).

Two steps, run in separate processes so that the peak memory of a run is not inflated by data generation:

    python benchmarks/treatment_bench.py generate --db /tmp/b.duckdb --patients 100000 --fills 10 --seed 1
    python benchmarks/treatment_bench.py run --db /tmp/b.duckdb --out results.jsonl [--engine sql]

`generate` writes a seeded, deterministic workload with DuckDB SQL (hash-based pseudo-randomness):
  * patients with 0-3 observation periods (overlapping, touching or separated), long treatment histories
  * fills drawn from metformin / two SGLT2 brands / dapagliflozin / combination products, i.e. multiple
    drugs per patient, brand changes, switches and add-ons
  * refill intervals that create overlapping supplies (early refills), same-day fills and gaps
  * ~1% missing and ~0.5% implausible days supply, ~2% rejected, ~2% reversed (with a reversal row), ~1%
    adjusted (with an adjustment row), ~1% duplicated claim lines
  * `--hf-patients` patients with `--hf-fills` fills every 1-3 days (high-frequency, cap-binding stockpiling)

`run` executes one treatment run in a fresh process and prints ONE JSON line: workload sizes, engine,
phase timings, rows read/written, throughput, peak RSS of the process (DuckDB runs in-process, so this
includes the database engine), the largest DuckDB temp-directory size seen and the database file growth.
Output contains counts and timings only - no identifiers or values.

It also runs the pre-v2 engine (`--legacy-path <checkout>/src`) for before/after comparisons.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

RULES = {
    "rule_set_version": 1,
    "name": "bench_rules",
    "version": "1.0.0",
    "drug_groups": {"metformin": {"concepts": [1503297]}, "sglt2": {"concepts": [900], "max_gap_days": 60}},
    "include_claim_status": ["paid", "not_applicable"],
    "include_evidence_types": ["dispensing"],
    "days_supply": {"missing": "exclude", "min_valid": 1, "max_valid": 365},
    "episode": {
        "max_gap_days": 45,
        "overlap": "shift_forward",
        "max_carryover_days": 90,
        "same_day_same_group": "max_supply",
        "duplicates": "same_product_same_day",
    },
    "censoring": {"observation": "dataset", "washout_days": 180},
    "transitions": {"add_on_min_overlap_days": 30, "switch_window_days": 60},
}
SOURCES = {
    "dataset": "omop_demo",
    "sources": [
        {
            "name": "pharmacy",
            "table": "bench.pharmacy",
            "person_column": "member",
            "record_id_column": "claim_id",
            "start_column": "fill_dt",
            "start_date_source": "fill_date",
            "days_supply_column": "dsup",
            "quantity_column": "qty",
            "evidence_type": "dispensing",
            "code_column": "ndc",
            "vocabulary": "NDC",
            "status": {
                "column": "status",
                "values": {"P": "paid", "R": "rejected", "X": "reversal", "A": "adjustment"},
                "reference_column": "ref_claim",
            },
        }
    ],
}
CONCEPTS = [
    (1503297, "metformin", "Drug", "RxNorm", "Ingredient", "S", "6809"),
    (45775965, "empagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1545653"),
    (44785829, "dapagliflozin", "Drug", "RxNorm", "Ingredient", "S", "1488564"),
    (900, "SGLT2 class", "Drug", "ATC", "ATC 4th", "C", "A10BK"),
    (801, "metformin 500", "Drug", "RxNorm", "Clinical Drug", "S", "P801"),
    (802, "empagliflozin [Brand X]", "Drug", "RxNorm", "Branded Drug", "S", "P802"),
    (803, "empagliflozin [Brand Y]", "Drug", "RxNorm", "Branded Drug", "S", "P803"),
    (804, "empagliflozin / metformin", "Drug", "RxNorm", "Clinical Drug", "S", "P804"),
    (805, "dapagliflozin 10", "Drug", "RxNorm", "Clinical Drug", "S", "P805"),
    (701, "ndc 801", "Drug", "NDC", "11-digit NDC", None, "11111000101"),
    (702, "ndc 802", "Drug", "NDC", "11-digit NDC", None, "22222000101"),
    (703, "ndc 803", "Drug", "NDC", "11-digit NDC", None, "33333000101"),
    (704, "ndc 804", "Drug", "NDC", "11-digit NDC", None, "44444000101"),
    (705, "ndc 805", "Drug", "NDC", "11-digit NDC", None, "55555000101"),
]
MAPS = [(701, 801), (702, 802), (703, 803), (704, 804), (705, 805)]
ANCESTORS = [
    (1503297, 801), (45775965, 802), (45775965, 803), (45775965, 804), (1503297, 804), (44785829, 805),
    (900, 45775965), (900, 44785829), (900, 802), (900, 803), (900, 804), (900, 805),
]  # fmt: skip


def _h(seed: int, *parts: str) -> str:
    """SQL: deterministic pseudo-random BIGINT in [0, 2^31) from the seed and expressions."""
    return f"CAST(hash({int(seed)}, {', '.join(parts)}) % 2147483647 AS BIGINT)"


def generate(args: argparse.Namespace) -> None:
    # --schema-src: create the schema with another checkout (the pre-v2 engine inserts by position)
    sys.path.insert(0, str(Path(args.schema_src) if args.schema_src else Path(__file__).resolve().parents[1] / "src"))
    from cohort_builder.db import connect, init_schemas

    path = Path(args.db)
    for p in (path, Path(f"{path}.wal")):
        p.unlink(missing_ok=True)
    con = connect(path)
    init_schemas(con)
    t0 = time.perf_counter()
    con.executemany("INSERT INTO vocab.concept VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)", [list(c) for c in CONCEPTS])
    con.executemany("INSERT INTO vocab.concept_relationship VALUES (?,?,'Maps to',NULL,NULL,NULL)", MAPS)
    con.executemany(
        "INSERT INTO vocab.concept_relationship VALUES (?,?,'Has brand name',NULL,NULL,NULL)", [[802, 851], [803, 852]]
    )
    con.executemany("INSERT INTO vocab.concept_ancestor VALUES (?,?,1,1)", ANCESTORS)
    con.execute("INSERT INTO cdm.cdm_source VALUES ('synthetic benchmark', DATE '2026-01-01', '5.4', 'bench')")
    s, n, f = int(args.seed), int(args.patients), int(args.fills)
    # observation periods: 1-3 per patient (10% none); overlapping, touching or separated
    con.execute(f"""
INSERT INTO cdm.observation_period
WITH p AS (SELECT range AS person_id FROM range(1, {n + int(args.hf_patients) + 1})),
k AS (SELECT person_id, CASE WHEN {_h(s, "person_id", "1")} % 10 = 0 THEN 0
                          ELSE 1 + {_h(s, "person_id", "2")} % 3 END AS periods FROM p),
r AS (SELECT person_id, unnest(range(periods)) AS i FROM k WHERE periods > 0)
SELECT row_number() OVER (), person_id,
       DATE '2015-01-01' + CAST(i * 900 + {_h(s, "person_id", "3")} % 200 - CASE WHEN i > 0 THEN
         [0, 90, 899, 1000][1 + {_h(s, "person_id", "i", "4")} % 4] ELSE 0 END AS INTEGER),
       DATE '2015-01-01' + CAST(i * 900 + 800 + {_h(s, "person_id", "i", "5")} % 100 AS INTEGER),
       32817
FROM r""")
    # fills: per patient a long history; interval pattern creates early refills, same-day fills and gaps
    con.execute("CREATE SCHEMA IF NOT EXISTS bench")
    con.execute(f"""
CREATE OR REPLACE TABLE bench.pharmacy AS
WITH p AS (SELECT range AS member FROM range(1, {n + 1})),
c AS (SELECT member, 1 + {_h(s, "member", "10")} % {max(1, 2 * f - 1)} AS fills FROM p
      UNION ALL SELECT {n} + range, {int(args.hf_fills)} FROM range(1, {int(args.hf_patients) + 1})),
r AS (SELECT member, unnest(range(fills)) AS i, member > {n} AS hf FROM c),
g AS (SELECT member, i, hf, {_h(s, "member", "i", "11")} AS x FROM r),
d AS (SELECT member, i, hf, x,
        CASE WHEN hf THEN 1 + x % 3
             ELSE [0, 20, 25, 27, 28, 30, 30, 31, 35, 45, 70, 120, 400][1 + x % 13] END AS step FROM g),
e AS (SELECT member, i, hf, x,
        DATE '2015-02-01' + CAST({_h(s, "member", "12")} % 300 + sum(step) OVER (PARTITION BY member ORDER BY i)
          AS INTEGER) AS fill_dt FROM d)
SELECT 'c' || member || '_' || i AS claim_id, member, fill_dt,
       ['11111000101', '11111000101', '22222000101', '33333000101', '44444000101', '55555000101', '00000000000']
         [1 + CASE WHEN x % 50 = 0 THEN 6 ELSE (x // 7 + i // 15) % 6 END] AS ndc,
       CASE WHEN x % 100 = 1 THEN NULL WHEN x % 200 = 2 THEN 999 ELSE [30, 30, 30, 90, 7, 60][1 + (x // 11) % 6] END
         AS dsup,
       30.0 AS qty,
       CASE WHEN x % 50 = 3 THEN 'R' ELSE 'P' END AS status, CAST(NULL AS VARCHAR) AS ref_claim
FROM e""")
    # reversals (~2%) and adjustments (~1%) of paid claims, and duplicated claim lines (~1%)
    con.execute(f"""
INSERT INTO bench.pharmacy
SELECT claim_id || 'X', member, fill_dt + 2, ndc, dsup, qty, 'X', claim_id FROM bench.pharmacy
WHERE status = 'P' AND {_h(s, "claim_id", "20")} % 50 = 0
UNION ALL
SELECT claim_id || 'A', member, fill_dt, ndc, CASE WHEN dsup = 30 THEN 90 ELSE dsup END, qty, 'A', claim_id
FROM bench.pharmacy WHERE status = 'P' AND {_h(s, "claim_id", "21")} % 100 = 1
UNION ALL
SELECT * FROM bench.pharmacy WHERE {_h(s, "claim_id", "22")} % 100 = 2""")
    rows = con.execute("SELECT count(*), count(DISTINCT member) FROM bench.pharmacy").fetchone()
    con.execute("CHECKPOINT")
    con.close()
    print(
        json.dumps(
            {
                "step": "generate",
                "patients": n + int(args.hf_patients),
                "pharmacy_rows": int(rows[0]),
                "patients_with_fills": int(rows[1]),
                "seconds": round(time.perf_counter() - t0, 2),
                "db_bytes": path.stat().st_size,
            }
        )
    )


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def run(args: argparse.Namespace) -> None:
    src = Path(args.legacy_path) if args.legacy_path else Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))
    import cohort_builder.treatment_service as ts
    from cohort_builder.config import Settings
    from cohort_builder.orchestrator import CohortBuilder
    from cohort_builder.treatment_config import TreatmentRuleSet, TreatmentSourceConfig

    db = Path(args.db)
    tmp_dir = Path(f"{db}.tmp")
    settings = dataclasses.replace(
        Settings(),
        db_path=db,
        dataset="omop_demo",
        local_datasets_dir=db.parent / "none",
        duckdb_memory_limit=args.memory_limit,
        duckdb_threads=args.threads,
    )
    extra = {}
    if hasattr(Settings(), "treatment_batch_size"):
        extra = {"treatment_batch_size": args.batch_size, "treatment_timeout_seconds": float(args.timeout)}
    settings = dataclasses.replace(settings, **extra)
    b = CohortBuilder(settings)  # the LLM is never called by treatment runs
    rules = TreatmentRuleSet.model_validate(RULES)
    sources = TreatmentSourceConfig.model_validate(SOURCES)
    svc = ts.TreatmentService(b)
    peak_tmp = 0
    stop = threading.Event()

    def watch() -> None:
        nonlocal peak_tmp
        while not stop.is_set():
            peak_tmp = max(peak_tmp, _dir_size(tmp_dir))
            stop.wait(0.25)

    phases: dict[str, float] = {}
    if args.legacy_path:  # the pre-v2 engine has no phase metrics: time its methods from outside
        for name in ("_included_events", "_observation", "_store", "_summary"):
            if hasattr(ts.TreatmentService, name):
                orig = getattr(ts.TreatmentService, name)

                def wrap(*a, _o=orig, _n=name, **k):  # type: ignore[no-untyped-def]
                    t = time.perf_counter()
                    try:
                        return _o(*a, **k)
                    finally:
                        phases[_n] = round(phases.get(_n, 0.0) + time.perf_counter() - t, 3)

                setattr(ts.TreatmentService, name, wrap)
        orig_build = ts.build

        def timed_build(*a, **k):  # type: ignore[no-untyped-def]
            t = time.perf_counter()
            try:
                return orig_build(*a, **k)
            finally:
                phases["build"] = round(time.perf_counter() - t, 3)

        ts.build = timed_build
    db_before = db.stat().st_size
    threading.Thread(target=watch, daemon=True).start()
    t0 = time.perf_counter()
    status, error = "completed", None
    out: dict = {}
    try:
        if args.legacy_path:
            out = svc.run(rules, "bench", "default", sources)
        else:
            out = svc.run(rules, "bench", "default", sources, date(2026, 1, 1), args.engine, args.batch_size)
    except Exception as exc:  # noqa: BLE001 - report the failure class only (never values)
        status, error = "failed", type(exc).__name__
    wall = time.perf_counter() - t0
    stop.set()
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    counts = {}
    if out:
        for name in ("event", "episode", "transition"):
            counts[name] = int(
                b.con.execute(
                    f"SELECT count(*) FROM results.treatment_{name} WHERE run_id = ?", [out["run_id"]]
                ).fetchone()[0]
            )
    source_rows = int(b.con.execute("SELECT count(*) FROM bench.pharmacy").fetchone()[0])
    failure = None
    if status == "failed" and not args.legacy_path:  # failed stage + redacted error class/message (no values)
        row = b.con.execute(
            "SELECT failed_stage, errors_json FROM meta.treatment_run ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        failure = {"stage": row[0], "error": json.loads(row[1] or "[]")} if row else None
    b.con.close()
    metrics = (out or {}).get("metrics") or {}
    print(
        json.dumps(
            {
                "step": "run",
                "engine": "legacy" if args.legacy_path else args.engine,
                "status": status,
                "error": error,
                "failure": failure,
                "source_rows": source_rows,
                "rows_written": counts,
                "seconds_total": round(wall, 2),
                "seconds_by_phase": metrics.get("seconds") or phases,
                "engine_rows": metrics.get("rows"),
                "events_per_second": round(source_rows / wall) if wall else None,
                "peak_rss_mb": round(rss_kb / 1024, 1),
                "peak_duckdb_temp_mb": round(peak_tmp / 1048576, 1),
                "db_growth_mb": round((db.stat().st_size - db_before) / 1048576, 1),
                "batch_size": args.batch_size,
                "memory_limit": args.memory_limit,
                "threads": args.threads,
                "output_hash": (out or {}).get("output_hash"),
                "env": {
                    "cpus": os.cpu_count(),
                    "python": platform.python_version(),
                    "duckdb": __import__("duckdb").__version__,
                    "platform": platform.platform(),
                    "commit": subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=src.parent
                    ).stdout.strip(),
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--db", required=True)
    g.add_argument("--patients", type=int, required=True)
    g.add_argument("--fills", type=int, default=10, help="average fills per patient")
    g.add_argument("--hf-patients", type=int, default=0)
    g.add_argument("--hf-fills", type=int, default=0)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--schema-src", help="src directory whose schema to create (for --legacy-path runs)")
    r = sub.add_parser("run")
    r.add_argument("--db", required=True)
    r.add_argument("--engine", choices=["sql", "reference"], default="sql")
    r.add_argument("--legacy-path", help="src directory of a pre-v2 checkout to benchmark instead")
    r.add_argument("--batch-size", type=int, default=100_000)
    r.add_argument("--memory-limit", default=None, help="DuckDB memory_limit, e.g. 4GB")
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()
    generate(a) if a.cmd == "generate" else run(a)


if __name__ == "__main__":
    main()
