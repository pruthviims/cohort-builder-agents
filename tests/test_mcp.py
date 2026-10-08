"""MCP server tests through a real in-process MCP client."""

from __future__ import annotations

import asyncio
import json

from mcp import Client

from cohort_builder.mcp_server import BearerTokenMiddleware, create_server
from fake_llm import T2DM_QUERY


def run(coro):
    return asyncio.run(coro)


async def call(client: Client, name: str, args: dict | None = None):
    r = await client.call_tool(name, args or {})
    assert not r.is_error, r.content
    data = r.structured_content
    if data is None:  # tools returning plain dicts send JSON text
        return json.loads(r.content[0].text)
    return data["result"] if isinstance(data, dict) and set(data) == {"result"} else data


def example_ir(path) -> dict:
    data = json.loads(path.read_text())
    data.pop("ontology_version")  # the server fills current versions in
    data.pop("vocabulary_version")
    return data


def test_tools_exposed_and_approval_is_not(builder):
    async def go():
        async with Client(create_server(builder, "tester")) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {
                "build_cohort",
                "search_concepts",
                "validate_cohort",
                "save_cohort_definition",
                "execute_approved_cohort",
                "replay_run",
            } <= names
            assert not any(n.startswith(("approve", "review", "reject")) for n in names)
            schema = next(t for t in (await c.list_tools()).tools if t.name == "validate_cohort").input_schema
            assert "index_event" in json.dumps(schema)  # clients see the full cohort definition schema
            read_only = {t.name for t in (await c.list_tools()).tools if t.annotations and t.annotations.read_only_hint}
            assert {"search_concepts", "validate_cohort", "get_cohort_sql"} <= read_only
            assert "execute_approved_cohort" not in read_only

    run(go())


def test_client_driven_flow_with_human_approval(builder, example_ir_path):
    """Option B: the client's LLM searches, validates and saves; a human approves; then MCP executes."""

    async def go():
        async with Client(create_server(builder, "tester")) as c:
            curated = await call(c, "search_curated_concept_sets", {"query": "type 2 diabetes"})
            assert curated[0]["curated_key"] == "t2dm"
            found = await call(c, "search_concepts", {"query": "metformin", "domain": "Drug"})
            assert found[0]["concept_id"] == 1503297

            v = await call(c, "validate_cohort", {"definition": example_ir(example_ir_path)})
            assert v["valid"] and v["attrition"][-1]["remaining"] > 0 and "metformin" in v["explanation"]

            saved = await call(c, "save_cohort_definition", {"definition": example_ir(example_ir_path)})
            def_id = saved["cohort_definition_id"]
            assert saved["status"] == "draft" and saved["semantic_hash"] == v["semantic_hash"]

            blocked = await call(c, "execute_approved_cohort", {"definition_id": def_id})
            assert "approve" in blocked["error"]

            builder.review(def_id, "dr_reviewer", "approved")  # human step, outside MCP

            out = await call(c, "execute_approved_cohort", {"definition_id": def_id})
            assert out["person_count"] > 0 and "subject_id" not in json.dumps(out)
            sql = await call(c, "get_cohort_sql", {"definition_id": def_id})
            assert sql["sql"].startswith("WITH cs_expanded")
            return def_id

    def_id = run(go())
    row = builder.store.get_definition(def_id)
    assert row["created_by"] == "mcp:tester"
    assert (
        builder.con.execute(
            "SELECT executed_by FROM meta.cohort_generation WHERE cohort_definition_id=?", [def_id]
        ).fetchone()[0]
        == "mcp:tester"
    )


def test_server_rejects_invalid_definitions(builder, example_ir_path):
    async def go():
        async with Client(create_server(builder)) as c:
            bad = example_ir(example_ir_path)
            bad["concept_sets"][1]["items"][0]["concept_id"] = 2000003099  # deprecated
            v = await call(c, "validate_cohort", {"definition": bad})
            assert not v["valid"] and any("deprecated" in i["message"] for i in v["issues"])
            broken = await c.call_tool("validate_cohort", {"definition": {"name": "x"}})
            assert broken.is_error  # rejected by the tool's input schema before any code runs
            saved = await call(c, "save_cohort_definition", {"definition": bad})
            assert saved["status"] == "needs_review"

    run(go())


def test_reproducible_pipeline_and_replay(builder):
    """Option A: the server's own agents build the cohort; replay reproduces it exactly."""

    async def go():
        async with Client(create_server(builder, "tester")) as c:
            r = await call(c, "build_cohort", {"request": T2DM_QUERY})
            assert r["status"] == "draft" and "human reviewer" in r["next_step"]
            run_info = await call(c, "get_run", {"run_id": r["run_id"]})
            assert run_info["manifest"]["ir_semantic_hash"] == r["manifest"]["ir_semantic_hash"]
            rep = await call(c, "replay_run", {"run_id": r["run_id"]})
            assert rep["identical"]

    run(go())


def test_resources_and_prompt(builder):
    async def go():
        async with Client(create_server(builder)) as c:
            uris = {str(r.uri) for r in (await c.list_resources()).resources}
            assert {"ontology://domain", "ontology://curated-concept-sets", "cohort://ir-schema"} <= uris
            schema = json.loads((await c.read_resource("cohort://ir-schema")).contents[0].text)
            assert "index_event" in schema["properties"]
            domain = (await c.read_resource("ontology://domain")).contents[0].text
            assert "entities:" in domain
            p = await c.get_prompt("build_cohort_interactively", {"request": "adults with CKD"})
            assert "adults with CKD" in p.messages[0].content.text

    run(go())


def test_http_transport_requires_token(builder):
    from starlette.testclient import TestClient

    app = BearerTokenMiddleware(create_server(builder).streamable_http_app(), "s3cret")
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    }
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    with TestClient(app, base_url="http://127.0.0.1:8765") as http:
        assert http.post("/mcp", json=init, headers=headers).status_code == 401
        assert http.post("/mcp", json=init, headers={**headers, "authorization": "Bearer wrong"}).status_code == 401
        ok = http.post("/mcp", json=init, headers={**headers, "authorization": "Bearer s3cret"})
        assert ok.status_code == 200, ok.text
