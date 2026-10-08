"""MCP server: exposes the cohort builder to any MCP client (Claude Desktop, Claude Code, connectors, agents).

Two ways to use it:

* Reproducible pipeline (option A): `build_cohort` runs this server's own agents
  (pinned model, versioned prompts, cache/replay) and returns a draft.
* Client-driven (option B): the client's own LLM uses the ontology and vocabulary
  tools, then `validate_cohort` / `save_cohort_definition`. The server still
  enforces every ontology rule, so invalid definitions are rejected whoever wrote them.

Governance:
* Approval is deliberately NOT exposed. An AI must not approve its own work; a
  human approves through the CLI or HTTP API.
* `execute_approved_cohort` only runs approved definitions and returns aggregate,
  small-cell-suppressed counts. No tool returns patient-level rows.
* The acting identity comes from server config (CB_MCP_USER), not from tool arguments.

Run:  cohort-builder mcp                      (stdio, for a local client)
      cohort-builder mcp --http --port 8765   (shared server; set CB_MCP_TOKEN)
"""
from __future__ import annotations

import hmac
import json
import os
import threading
from typing import Any

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from .agents.validator import validate
from .executor import ExecutionError
from .ir import CohortDefinition
from .orchestrator import CohortBuilder
from .security import DEFAULT_TENANT

INSTRUCTIONS = """\
Cohort builder for OMOP CDM patient data, grounded in a semantic ontology.

For an auditable, reproducible cohort, call `build_cohort` with the researcher's request.
To build interactively, use the vocabulary tools (search_curated_concept_sets first, then
search_concepts / get_concept / get_descendants / lookup_code), write a definition that
follows the `cohort://ir-schema` resource, check it with `validate_cohort`, then save it
with `save_cohort_definition`. Only use concept IDs returned by the tools.

Saved definitions are drafts. A human reviewer must approve them outside this tool
before `execute_approved_cohort` will run them. No tool returns patient-level data.
"""

class CohortDefinitionInput(CohortDefinition):
    """Cohort definition as submitted by an MCP client; versions are filled in by the server."""
    # pydantic allows widening a field in a subclass; the server fills these before validation
    ontology_version: str | None = Field(  # type: ignore[assignment]
        default=None, description="Leave empty; the server sets it")
    vocabulary_version: str | None = Field(  # type: ignore[assignment]
        default=None, description="Leave empty; the server sets it")


READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def _j(obj: Any) -> Any:
    """Make results JSON-safe (dates, decimals) for structured tool output."""
    return json.loads(json.dumps(obj, default=str))


def create_server(builder: CohortBuilder | None = None, acting_user: str | None = None,
                  acting_tenant: str | None = None) -> MCPServer:
    state: dict[str, Any] = {"builder": builder}
    lock = threading.RLock()  # a single DuckDB connection is shared: serialize access
    user = f"mcp:{acting_user or os.environ.get('CB_MCP_USER', 'anonymous')}"
    tenant = acting_tenant or os.environ.get("CB_MCP_TENANT") or DEFAULT_TENANT

    def b() -> CohortBuilder:
        if state["builder"] is None:
            state["builder"] = CohortBuilder()
        return state["builder"]

    def parse_ir(definition: CohortDefinitionInput | dict) -> CohortDefinition:
        data = definition.model_dump() if isinstance(definition, CohortDefinition) else dict(definition)
        data["ontology_version"] = data.get("ontology_version") or b().ontology.version
        data["vocabulary_version"] = data.get("vocabulary_version") or b().vocab.version()
        return CohortDefinition.model_validate(data)

    mcp = MCPServer(name="cohort-builder", title="Cohort Builder", instructions=INSTRUCTIONS, version="0.1.0")

    # ---- option A: the reproducible agent pipeline --------------------------------
    @mcp.tool(annotations=WRITES)
    def build_cohort(request: str) -> dict:
        """Build a cohort definition from a natural-language request using the server's
        reproducible agent pipeline. Returns a DRAFT (plain-language explanation, dry-run
        attrition, issues, run manifest) that a human must approve before execution."""
        with lock:
            r = b().ask(request, user_id=user, tenant=tenant)
            out = r.as_dict()
            out["next_step"] = ("A human reviewer must approve definition "
                                f"{r.cohort_definition_id} outside MCP before it can be executed."
                                if r.status == "draft" else "Resolve the issues, then try again or edit the IR.")
            return _j(out)

    @mcp.tool(annotations=READ_ONLY)
    def get_run(run_id: str) -> dict:
        """Get a past run's steps, status and reproducibility manifest."""
        with lock:
            run = b().get_run(run_id, tenant)
            return _j(run) if run else {"error": "run not found"}

    @mcp.tool(annotations=WRITES)
    def replay_run(run_id: str) -> dict:
        """Re-run a past request from recorded LLM responses only and report whether the
        resulting cohort logic (semantic hash) is identical."""
        with lock:
            try:
                return _j(b().replay(run_id, tenant, actor=user))
            except KeyError:
                return {"error": "run not found"}

    # ---- option B: ontology and vocabulary tools ---------------------------------
    @mcp.tool(annotations=READ_ONLY)
    def describe_ontology() -> dict:
        """Entities, filterable attributes, operators, temporal rules, units and versions
        that cohort definitions must follow."""
        with lock:
            ont = b().ontology
            return _j({"ontology_version": ont.version, "ontology_hash": ont.content_hash,
                       "vocabulary_version": b().vocab.version(),
                       "summary": yaml.safe_load(ont.summary_for_prompt())})

    @mcp.tool(annotations=READ_ONLY)
    def search_curated_concept_sets(query: str, domain: str | None = None) -> list[dict]:
        """Search organization-approved concept sets. Prefer these over raw vocabulary search.
        domain: Condition, Drug, Measurement, Procedure or Visit."""
        with lock:
            return _j(b().ontology.search_curated(query, domain=domain))

    @mcp.tool(annotations=READ_ONLY)
    def search_concepts(query: str, domain: str | None = None, include_non_standard: bool = False,
                        limit: int = 10) -> list[dict]:
        """Search the OMOP vocabulary by name, synonym or code. Standard concepts only by default."""
        with lock:
            return _j(b().vocab.search_concepts(query, domain, not include_non_standard, limit))

    @mcp.tool(annotations=READ_ONLY)
    def get_concept(concept_id: int) -> dict:
        """Concept details: parents, descendant count, and Maps-to targets if non-standard."""
        with lock:
            return _j(b().vocab.get_concept(concept_id) or {"error": "concept not found"})

    @mcp.tool(annotations=READ_ONLY)
    def get_descendants(concept_id: int, limit: int = 25) -> list[dict]:
        """Concepts covered when include_descendants is true."""
        with lock:
            return _j(b().vocab.get_descendants(concept_id, limit))

    @mcp.tool(annotations=READ_ONLY)
    def lookup_code(code: str, vocabulary_id: str | None = None) -> list[dict]:
        """Find a concept by source code (e.g. ICD-10 'E11.9', LOINC '4548-4') and its standard mapping."""
        with lock:
            return _j(b().vocab.lookup_code(code, vocabulary_id))

    @mcp.tool(annotations=READ_ONLY)
    def validate_cohort(definition: CohortDefinitionInput) -> dict:
        """Check a cohort definition (see resource cohort://ir-schema) against the ontology and
        vocabulary, and dry-run it. Returns issues, suppressed attrition counts, a plain-language
        explanation and hashes. Nothing is saved."""
        with lock:
            try:
                ir = parse_ir(definition)
            except ValidationError as exc:
                return {"valid": False, "schema_errors": _j(exc.errors(include_url=False))}
            bb = b()
            issues, attrition = validate(ir, bb.ontology, bb.vocab, bb.executor)
            return _j({
                "valid": not any(i.severity == "error" for i in issues),
                "issues": [i.as_dict() for i in issues],
                "attrition": attrition.suppressed(bb.executor.min_cell) if attrition else None,
                "explanation": bb.explainer.explain(ir),
                "content_hash": ir.content_hash(),
                "semantic_hash": ir.semantic_hash(),
            })

    @mcp.tool(annotations=WRITES)
    def save_cohort_definition(definition: CohortDefinitionInput, parent_definition_id: int | None = None) -> dict:
        """Save a cohort definition as a new immutable DRAFT version (use parent_definition_id
        when editing an existing one). It must be approved by a human before execution."""
        with lock:
            try:
                ir = parse_ir(definition)
            except ValidationError as exc:
                return {"saved": False, "schema_errors": _j(exc.errors(include_url=False))}
            try:
                def_id, issues = b().submit_ir(ir, user, parent_definition_id, tenant)
            except KeyError as exc:
                return {"saved": False, "error": str(exc).strip("'\"")}
            saved_row = b().store.get_definition(def_id) or {}
            status = saved_row.get("status")
            return _j({"saved": True, "cohort_definition_id": def_id, "status": status, "issues": issues,
                       "semantic_hash": ir.semantic_hash()})

    # ---- definitions & execution ---------------------------------------------------
    @mcp.tool(annotations=READ_ONLY)
    def list_cohort_definitions(limit: int = 20) -> list[dict]:
        """Most recent cohort definitions with status (draft, needs_review, approved, rejected)."""
        with lock:
            return _j(b().store.list_definitions(max(1, min(limit, 200)), tenant=tenant))

    @mcp.tool(annotations=READ_ONLY)
    def get_cohort_definition(definition_id: int) -> dict:
        """A saved definition: IR, status, hashes, versions and plain-language explanation."""
        with lock:
            try:
                row, ir = b().load_definition(definition_id, tenant)
            except KeyError as exc:
                return {"error": str(exc)}
            return _j({**row, "explanation": b().explainer.explain(ir)})

    @mcp.tool(annotations=READ_ONLY)
    def get_cohort_sql(definition_id: int) -> dict:
        """The deterministic SQL the compiler generates for a saved definition."""
        with lock:
            try:
                return {"sql": b().compile_sql(definition_id, tenant)}
            except KeyError as exc:
                return {"error": str(exc)}

    @mcp.tool(annotations=WRITES)
    def execute_approved_cohort(definition_id: int) -> dict:
        """Materialize an APPROVED definition into the results tables. Returns the generation id
        and suppressed counts only; patient-level rows are never returned."""
        with lock:
            try:
                out = b().execute(definition_id, user, allow_draft=False, tenant=tenant)
            except KeyError as exc:
                return {"error": str(exc).strip("'\"")}
            except PermissionError as exc:
                return {"error": str(exc), "hint": "A human reviewer must approve this definition first."}
            except (ValueError, ExecutionError) as exc:
                return {"error": str(exc)}
            return _j(out)  # counts are already small-cell suppressed by the core

    # ---- resources ------------------------------------------------------------------
    def _file(name: str) -> str:
        return (b().settings.ontology_dir / name).read_text()

    @mcp.resource("ontology://domain", name="Ontology domain model", mime_type="application/yaml")
    def ontology_domain() -> str:
        return _file("domain.yaml")

    @mcp.resource("ontology://curated-concept-sets", name="Curated concept sets", mime_type="application/yaml")
    def ontology_curated() -> str:
        return _file("curated_concept_sets.yaml")

    @mcp.resource("ontology://unit-conversions", name="Unit conversions", mime_type="application/yaml")
    def ontology_units() -> str:
        return _file("unit_conversions.yaml")

    @mcp.resource("cohort://ir-schema", name="Cohort definition JSON schema", mime_type="application/json")
    def ir_schema() -> str:
        return json.dumps(CohortDefinition.model_json_schema(), indent=2)

    @mcp.resource("cohort://definitions/{definition_id}", name="Cohort definition", mime_type="application/json")
    def definition_resource(definition_id: str) -> str:
        with lock:
            try:
                row, _ = b().load_definition(int(definition_id), tenant)
            except (KeyError, ValueError):
                row = {"error": "not found"}
            return json.dumps(row, default=str, indent=2)

    # ---- prompts --------------------------------------------------------------------
    @mcp.prompt(title="Build a cohort interactively")
    def build_cohort_interactively(request: str) -> str:
        """Guide the assistant through building a cohort with the ontology tools."""
        return (
            f"Build an OMOP cohort definition for this request:\n\n{request}\n\n"
            "1. Call describe_ontology and read resource cohort://ir-schema.\n"
            "2. Identify the index event, inclusion and exclusion criteria, time windows (days relative to "
            "index, negative = before), value thresholds with units, age/gender limits.\n"
            "3. For each clinical idea call search_curated_concept_sets first; otherwise search_concepts, "
            "get_concept and get_descendants. Use ingredients for drugs, include_descendants=true, and only "
            "concept IDs returned by the tools.\n"
            "4. Call validate_cohort and fix every error. Show me the explanation and attrition, and list "
            "every assumption you made.\n"
            "5. When I confirm, call save_cohort_definition. Remind me that a human reviewer must approve it.")

    return mcp


def run_http(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve over Streamable HTTP at /mcp. Requires CB_MCP_TOKEN when binding beyond localhost."""
    import uvicorn

    token = os.environ.get("CB_MCP_TOKEN")
    local = host in ("127.0.0.1", "localhost")
    if not local and not token:
        raise SystemExit("Refusing to serve on a non-local interface without CB_MCP_TOKEN set.")
    # DNS-rebinding protection: only accept requests addressed to these hostnames.
    allowed = [h.strip() for h in os.environ.get("CB_MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = None
    if allowed:
        security = TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                             allowed_hosts=allowed + ["127.0.0.1:*", "localhost:*"],
                                             allowed_origins=[f"https://{h.split(':')[0]}" for h in allowed])
    elif not local:
        raise SystemExit("Set CB_MCP_ALLOWED_HOSTS (e.g. cohorts.example.org:*) when serving beyond localhost.")
    app = create_server().streamable_http_app(host=host, transport_security=security)
    uvicorn.run(BearerTokenMiddleware(app, token) if token else app, host=host, port=port)


class BearerTokenMiddleware:
    """Minimal shared-secret auth for a team server. Use OAuth (MCPServer auth settings) in production."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            if not hmac.compare_digest(headers.get(b"authorization", b""), b"Bearer " + self.token):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)
