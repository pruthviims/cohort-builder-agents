"""Small-cell suppression as a privacy control, plus execution limits and least-privilege hardening."""
from __future__ import annotations

import copy
import json
import random

import duckdb
import pytest
from mcp import Client

from cohort_builder.config import REPO_ROOT
from cohort_builder.executor import COMPLEMENTARY, Executor, QueryTimeout, suppress_count, suppress_series
from cohort_builder.ir import CohortDefinition
from fake_llm import T2DM_QUERY

K = 10
BASE = json.loads((REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json").read_text())


def small_ir(age_min: int, age_max: int) -> CohortDefinition:
    data = copy.deepcopy(BASE)
    data["demographics"] = {"age_min": age_min, "age_max": age_max, "gender_concept_ids": []}
    return CohortDefinition.model_validate(data)


# ---- the suppression rule ----------------------------------------------------------------
def test_primary_suppression_thresholds():
    assert suppress_count(0, K) == 0          # zero describes nobody
    assert suppress_count(1, K) == "<10"
    assert suppress_count(9, K) == "<10"
    assert suppress_count(10, K) == 10


def test_complementary_suppression_hides_small_differences():
    assert suppress_series([500, 497, 300, 296, 296], K) == [500, COMPLEMENTARY, 300, COMPLEMENTARY, COMPLEMENTARY]
    assert suppress_series([30, 30, 21, 21], K) == [30, 30, COMPLEMENTARY, COMPLEMENTARY]
    assert suppress_series([25, 12, 9, 0], K) == [25, 12, "<10", 0]
    assert suppress_series([100, 90, 80], K) == [100, 90, 80]           # differences of exactly k are fine


def _disclosed(shown: list) -> list[int]:
    return [v for v in shown if isinstance(v, int)]


def test_no_small_count_is_derivable_from_disclosed_values():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(0, 400)
        series = [n]
        for _ in range(rng.randint(1, 8)):
            series.append(max(0, series[-1] - rng.choice([0, 0, 1, 2, 5, 9, 10, 15, 40, 100])))
        shown = _disclosed(suppress_series(series, K))
        assert all(v == 0 or v >= K for v in shown), series
        for i, a in enumerate(shown):
            for b in shown[i + 1:]:
                assert not (0 < abs(a - b) < K), (series, shown)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        Executor(duckdb.connect(), min_cell_count=0)


# ---- consistent across every output path ----------------------------------------------------
def test_execute_returns_suppressed_counts(builder):
    def_id, _ = builder.submit_ir(small_ir(18, 25), "alice")
    builder.review(def_id, "bob", "approved")
    out = builder.execute(def_id, "carol")
    raw = builder.con.execute("SELECT person_count FROM meta.cohort_generation").fetchone()[0]
    assert 0 < raw < K                                   # the raw count exists only inside the database
    assert out["person_count"] == "<10"
    assert out["attrition"][-1]["remaining"] == "<10"
    assert str(raw) not in json.dumps([r["remaining"] for r in out["attrition"][-2:]])


def test_final_count_follows_complementary_suppression(builder):
    def_id, _ = builder.submit_ir(small_ir(30, 40), "alice")
    builder.review(def_id, "bob", "approved")
    out = builder.execute(def_id, "carol")
    assert out["person_count"] == COMPLEMENTARY          # would otherwise reveal a removed group of < 10


def test_validation_dry_run_and_ask_are_suppressed(builder):
    from cohort_builder.agents.validator import validate

    _, attrition = validate(small_ir(18, 25), builder.ontology, builder.vocab, builder.executor)
    assert attrition.suppressed(K)[-1]["remaining"] == "<10"
    r = builder.ask(T2DM_QUERY)
    assert all(isinstance(x["remaining"], (int, str)) for x in r.attrition)
    assert r.manifest["dry_run_count"] == r.attrition[-1]["remaining"]


def test_cli_output_is_suppressed(builder, settings, tmp_path, monkeypatch, capsys):
    from cohort_builder import cli

    path = tmp_path / "small.json"
    path.write_text(small_ir(18, 25).model_dump_json())
    builder.con.close()
    monkeypatch.setenv("CB_DB_PATH", str(settings.db_path))
    assert cli.main(["dry-run", str(path)]) == 0
    out = capsys.readouterr().out
    final_line = [ln for ln in out.splitlines() if "Exclusion: Type 1 diabetes" in ln][-1]
    assert final_line.rstrip().endswith("<10")


def test_mcp_execution_is_suppressed(builder):
    import asyncio

    from cohort_builder.mcp_server import create_server

    def_id, _ = builder.submit_ir(small_ir(18, 25), "alice")
    builder.review(def_id, "bob", "approved")

    async def go():
        async with Client(create_server(builder, "tester")) as c:
            r = await c.call_tool("execute_approved_cohort", {"definition_id": def_id})
            return json.loads(r.content[0].text)

    out = asyncio.run(go())
    assert out["person_count"] == "<10" and "subject_id" not in json.dumps(out)


def test_api_execution_is_suppressed(builder, tmp_path):
    import yaml
    from fastapi.testclient import TestClient

    from cohort_builder.api import create_app
    from cohort_builder.security import SecurityConfig, issue_token

    tok, entry = issue_token("carol", ["executor"])
    tf = tmp_path / "t.yaml"
    tf.write_text(yaml.safe_dump({"tokens": [entry]}))
    def_id, _ = builder.submit_ir(small_ir(18, 25), "alice")
    builder.review(def_id, "bob", "approved")
    client = TestClient(create_app(builder, SecurityConfig(tokens_file=tf)))
    r = client.post(f"/cohorts/{def_id}/execute", headers={"Authorization": f"Bearer {tok}"})
    assert r.json()["person_count"] == "<10"


def test_eval_report_suppresses_counts(builder):
    from cohort_builder.evaluation import compare

    m = compare(builder, small_ir(18, 25), small_ir(18, 25))
    assert m["gold_count"] == "<10" and m["got_count"] == "<10" and m["patient_jaccard"] == 1.0


# ---- execution limits and least privilege -------------------------------------------------------
def test_query_timeout_cancels_long_queries(builder):
    ex = Executor(builder.con, timeout_seconds=0.2)
    with pytest.raises(QueryTimeout, match="time limit"):
        ex._run("SELECT count(*) FROM range(100000000000) a")
    assert builder.con.execute("SELECT 1").fetchone() == (1,)   # connection still usable afterwards


def test_sql_cannot_reach_files_or_extensions_after_startup(builder):
    for sql in ("SELECT * FROM read_csv('/etc/passwd')", "ATTACH '/tmp/x.duckdb' AS x", "INSTALL httpfs",
                "COPY (SELECT 1) TO '/tmp/x.csv'"):
        with pytest.raises(duckdb.PermissionException):
            builder.con.execute(sql)
    with pytest.raises(duckdb.Error):
        builder.con.execute("SET enable_external_access = true")


def test_memory_limit_setting_is_validated(settings, fake_llm):
    import dataclasses

    from cohort_builder.orchestrator import CohortBuilder

    with pytest.raises(ValueError, match="CB_DUCKDB_MEMORY_LIMIT"):
        CohortBuilder(dataclasses.replace(settings, duckdb_memory_limit="4GB'; ATTACH 'x"), backend=fake_llm)
    b = CohortBuilder(dataclasses.replace(settings, duckdb_memory_limit="1GB", duckdb_threads=2), backend=fake_llm)
    assert b.con.execute("SELECT current_setting('threads')").fetchone()[0] == 2
