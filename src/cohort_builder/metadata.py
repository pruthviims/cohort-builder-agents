"""Metadata store: runs, steps, LLM/tool calls, definitions, generations, reviews, evals.

Everything needed to audit or replay a run lives here (schema `meta`).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb

META_DDL = """
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SEQUENCE IF NOT EXISTS meta.cohort_definition_seq START 1;
CREATE TABLE IF NOT EXISTS meta.component_version (
  component_type VARCHAR, name VARCHAR, version VARCHAR, content_hash VARCHAR, recorded_at TIMESTAMP,
  PRIMARY KEY (component_type, name, version, content_hash));
CREATE TABLE IF NOT EXISTS meta.agent_run (
  run_id VARCHAR PRIMARY KEY, user_id VARCHAR, user_query VARCHAR, submitted_at TIMESTAMP,
  finished_at TIMESTAMP, status VARCHAR, retry_count INTEGER, output_definition_id BIGINT, manifest VARCHAR);
CREATE TABLE IF NOT EXISTS meta.agent_step (
  step_id VARCHAR PRIMARY KEY, run_id VARCHAR, step_seq INTEGER, attempt INTEGER, agent_name VARCHAR,
  input_json VARCHAR, output_json VARCHAR, status VARCHAR, started_at TIMESTAMP, duration_ms INTEGER);
CREATE TABLE IF NOT EXISTS meta.llm_call (
  call_id VARCHAR PRIMARY KEY, run_id VARCHAR, step_id VARCHAR, model VARCHAR, temperature DOUBLE,
  prompt_name VARCHAR, prompt_version VARCHAR, prompt_hash VARCHAR, request_hash VARCHAR,
  response_json VARCHAR, input_tokens INTEGER, output_tokens INTEGER, cache_hit BOOLEAN, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta.llm_cache (
  request_hash VARCHAR PRIMARY KEY, model VARCHAR, response_json VARCHAR, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta.tool_call (
  call_id VARCHAR PRIMARY KEY, run_id VARCHAR, step_id VARCHAR, tool_name VARCHAR, args_json VARCHAR,
  result_json VARCHAR, result_hash VARCHAR, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta.cohort_definition (
  cohort_definition_id BIGINT PRIMARY KEY, name VARCHAR, ir_json VARCHAR, content_hash VARCHAR,
  semantic_hash VARCHAR, schema_version VARCHAR, ontology_version VARCHAR, vocabulary_version VARCHAR,
  status VARCHAR, created_by VARCHAR, created_at TIMESTAMP, approved_by VARCHAR, approved_at TIMESTAMP,
  parent_definition_id BIGINT, run_id VARCHAR, issues_json VARCHAR);
CREATE TABLE IF NOT EXISTS meta.cohort_generation (
  generation_id VARCHAR PRIMARY KEY, cohort_definition_id BIGINT, data_snapshot VARCHAR,
  compiler_version VARCHAR, sql_hash VARCHAR, person_count BIGINT, executed_by VARCHAR, executed_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta.review (
  review_id VARCHAR PRIMARY KEY, cohort_definition_id BIGINT, reviewer VARCHAR, decision VARCHAR,
  comments VARCHAR, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS meta.eval_run (
  eval_run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, component_versions VARCHAR, summary_json VARCHAR);
CREATE TABLE IF NOT EXISTS meta.eval_result (
  eval_run_id VARCHAR, case_id VARCHAR, repeat INTEGER, metrics_json VARCHAR);
"""


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256((obj if isinstance(obj, str) else dumps(obj)).encode()).hexdigest()


def _rows(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class MetadataStore:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    # ---- versions --------------------------------------------------------
    def record_component(self, component_type: str, name: str, version: str, content_hash: str = "") -> None:
        self.con.execute("INSERT OR IGNORE INTO meta.component_version VALUES (?,?,?,?,?)",
                         [component_type, name, version, content_hash, now()])

    # ---- runs ------------------------------------------------------------
    def start_run(self, user_query: str, user_id: str) -> str:
        run_id = str(uuid.uuid4())
        self.con.execute("INSERT INTO meta.agent_run VALUES (?,?,?,?,NULL,'running',0,NULL,NULL)",
                         [run_id, user_id, user_query, now()])
        return run_id

    def finish_run(self, run_id: str, status: str, retry_count: int, definition_id: int | None,
                   manifest: dict) -> None:
        self.con.execute("UPDATE meta.agent_run SET finished_at=?, status=?, retry_count=?, "
                         "output_definition_id=?, manifest=? WHERE run_id=?",
                         [now(), status, retry_count, definition_id, dumps(manifest), run_id])

    def record_step(self, run_id: str, step_seq: int, attempt: int, agent_name: str, input_obj: Any,
                    output_obj: Any, status: str, started_at: datetime, duration_ms: int,
                    step_id: str | None = None) -> str:
        step_id = step_id or str(uuid.uuid4())
        self.con.execute("INSERT INTO meta.agent_step VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [step_id, run_id, step_seq, attempt, agent_name, dumps(input_obj), dumps(output_obj),
                          status, started_at, duration_ms])
        return step_id

    def record_tool_call(self, run_id: str | None, step_id: str | None, tool: str, args: Any, result: Any) -> None:
        payload = dumps(result)
        self.con.execute("INSERT INTO meta.tool_call VALUES (?,?,?,?,?,?,?,?)",
                         [str(uuid.uuid4()), run_id, step_id, tool, dumps(args), payload, sha(payload), now()])

    def record_llm_call(self, **kw: Any) -> None:
        self.con.execute(
            "INSERT INTO meta.llm_call VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [str(uuid.uuid4()), kw.get("run_id"), kw.get("step_id"), kw["model"], kw.get("temperature"),
             kw["prompt_name"], kw["prompt_version"], kw["prompt_hash"], kw["request_hash"],
             dumps(kw["response"]), kw.get("input_tokens"), kw.get("output_tokens"), kw["cache_hit"], now()])

    # ---- LLM cache -------------------------------------------------------
    def cache_get(self, request_hash: str) -> dict | None:
        row = self.con.execute("SELECT response_json FROM meta.llm_cache WHERE request_hash=?",
                               [request_hash]).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, request_hash: str, model: str, response: dict) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta.llm_cache VALUES (?,?,?,?)",
                         [request_hash, model, dumps(response), now()])

    # ---- definitions -----------------------------------------------------
    def save_definition(self, ir, status: str, created_by: str, run_id: str | None = None,
                        parent_id: int | None = None, issues: list | None = None) -> int:
        """Definitions are immutable; an edit is a new row with parent_definition_id."""
        def_id = self.con.execute("SELECT nextval('meta.cohort_definition_seq')").fetchone()[0]
        self.con.execute(
            "INSERT INTO meta.cohort_definition VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?,?)",
            [def_id, ir.name, ir.canonical_json(), ir.content_hash(), ir.semantic_hash(), ir.schema_version,
             ir.ontology_version, ir.vocabulary_version, status, created_by, now(), parent_id, run_id,
             dumps(issues or [])])
        return int(def_id)

    def get_definition(self, def_id: int) -> dict | None:
        rows = _rows(self.con.execute("SELECT * FROM meta.cohort_definition WHERE cohort_definition_id=?",
                                      [def_id]))
        if not rows:
            return None
        row = rows[0]
        row["ir"] = json.loads(row.pop("ir_json"))
        row["issues"] = json.loads(row.pop("issues_json") or "[]")
        return row

    def list_definitions(self, limit: int = 50) -> list[dict]:
        return _rows(self.con.execute(
            "SELECT cohort_definition_id, name, status, semantic_hash, created_by, created_at "
            "FROM meta.cohort_definition ORDER BY cohort_definition_id DESC LIMIT ?", [limit]))

    def review(self, def_id: int, reviewer: str, decision: str, comments: str = "") -> None:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        self.con.execute("INSERT INTO meta.review VALUES (?,?,?,?,?,?)",
                         [str(uuid.uuid4()), def_id, reviewer, decision, comments, now()])
        if decision == "approved":
            self.con.execute("UPDATE meta.cohort_definition SET status='approved', approved_by=?, approved_at=? "
                             "WHERE cohort_definition_id=?", [reviewer, now(), def_id])
        else:
            self.con.execute("UPDATE meta.cohort_definition SET status='rejected' WHERE cohort_definition_id=?",
                             [def_id])

    def record_generation(self, generation_id: str, def_id: int, data_snapshot: str, compiler_version: str,
                          sql_hash: str, person_count: int, executed_by: str) -> None:
        self.con.execute("INSERT INTO meta.cohort_generation VALUES (?,?,?,?,?,?,?,?)",
                         [generation_id, def_id, data_snapshot, compiler_version, sql_hash, person_count,
                          executed_by, now()])

    # ---- reads -----------------------------------------------------------
    def get_run(self, run_id: str) -> dict | None:
        rows = _rows(self.con.execute("SELECT * FROM meta.agent_run WHERE run_id=?", [run_id]))
        if not rows:
            return None
        run = rows[0]
        run["manifest"] = json.loads(run["manifest"]) if run["manifest"] else None
        run["steps"] = _rows(self.con.execute(
            "SELECT step_seq, attempt, agent_name, status, duration_ms FROM meta.agent_step WHERE run_id=? "
            "ORDER BY step_seq", [run_id]))
        return run

    def tool_calls(self, run_id: str) -> list[dict]:
        return _rows(self.con.execute(
            "SELECT tool_name, args_json, result_hash FROM meta.tool_call WHERE run_id=? ORDER BY created_at",
            [run_id]))

    def data_snapshot(self) -> str:
        try:
            row = self.con.execute("SELECT cdm_source_name, cdm_release_date FROM cdm.cdm_source LIMIT 1").fetchone()
            return f"{row[0]} @ {row[1]}" if row else "unknown"
        except duckdb.Error:
            return "unknown"
