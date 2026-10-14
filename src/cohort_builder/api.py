"""HTTP API (FastAPI). `cohort-builder serve` or `uvicorn cohort_builder.api:app`.

Every endpoint except /health requires a bearer token (see security.py). Identity
(who authored, reviewed or executed) always comes from the authenticated
principal, never from the request body. Resources are scoped to the principal's
tenant; other tenants' resources are reported as not found.

Roles per endpoint (admin implies all):
  GET  /health                       public (liveness only)
  GET  /me, /versions                any role
  GET  /cohorts, /cohorts/{id}, /cohorts/{id}/sql, /concepts/search   any role
  GET  /cohorts/{id}/versions, /cohorts/{id}/reviews, /concepts/{id}  any role
  GET  /cohorts/{id}/diff/{other_id}                                  any role (field-level IR differences)
  GET  /status                                                        any role (checks run now; no secrets)
  POST /cohorts/ask, /cohorts, /cohorts/validate                      author
  POST /cohorts/{id}/refine                                           author (new version via the same pipeline)
  GET  /cohorts/{id}/generations, POST /cohorts/compare               executor or reviewer (suppressed counts)
  GET  /runs                                                          author (own runs), reviewer (tenant), admin
  POST /cohorts/{id}/review                                           reviewer (not own work, unless policy allows)
  POST /cohorts/{id}/execute                                          executor (approved definitions only)
  GET  /runs/{id}, POST /runs/{id}/replay                             author (own runs), reviewer (tenant), admin
  GET  /audit                                                         admin

Proxy (indirect) cohort algorithms, same governance (see README "Proxy cohorts"):
  POST /proxy-cohorts, /proxy-cohorts/ask, /proxy-cohorts/validate      author (drafts only; nothing executes)
  GET  /proxy-cohorts, /proxy-cohorts/{id}, .../versions, .../review-packet   any role
  POST /proxy-cohorts/{id}/compile                                      any role (SQL preview only)
  POST /proxy-cohorts/{id}/review                                       reviewer (not own work, unless policy allows)
  POST /proxy-cohorts/{id}/execute                                      executor (approved definitions only)
  GET  /proxy-cohorts/{id}/results, .../evidence-summary                executor or reviewer (suppressed aggregates)
  POST /proxy-cohorts/compare                                           executor or reviewer (suppressed overlaps)
  POST /proxy-cohorts/{id}/reference-validation                         reviewer (evaluation + automatic criteria check)
  GET  /proxy-cohorts/{id}/status                                       any role (all statuses, kept separate)
  GET  /proxy-cohorts/{id}/evaluations                                  executor or reviewer
  POST /proxy-cohorts/{id}/evaluations/{validation_id}/review           reviewer (human acceptance; not own work)
  POST /proxy-references                                                admin (external reference labels)
  GET  /proxy-cohorts/{id}/patients/{subject_id}/explanation            admin AND CB_ALLOW_PATIENT_LEVEL=true

Treatment events and episodes (aggregate + suppressed unless stated):
  POST /treatment-runs                                                  executor (rule set by name or inline YAML)
  GET  /treatment-runs/{run_id}                                         executor or reviewer (own tenant)
  POST /treatment-runs/{run_id}/cancel                                  executor (own tenant)
  POST /treatment-runs/{run_id}/hold, /release                          admin (legal hold against retention)
  GET  /treatment-runs/{run_id}/episodes/{episode_id}/explanation       admin AND CB_ALLOW_PATIENT_LEVEL=true

Imported datasets (CSV; content is sent in the request body, never a server path; size-capped):
  GET  /datasets, /datasets/{name}, /datasets/{name}/versions/{version}  any role (metadata, no row values)
  POST /datasets/csv/profile                                            author (in-memory preview, nothing stored)
  POST /datasets/{name}/ingest                                          admin (new immutable version)
  POST /datasets/reconcile                                              admin (repair profile files after a failure)
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .agents.composer import format_validation_error
from .agents.validator import validate
from .db import redact_error
from .executor import ExecutionError, QueryTimeout
from .ir import CohortDefinition
from .orchestrator import CohortBuilder, GovernanceError
from .proxy import AcceptanceCriteria, DuplicateKeyError, ProxyDefinition
from .proxy_evaluation import EligibilityRules, ReferenceRecord, normalize_label
from .proxy_service import VersionConflict
from .security import (
    ADMIN,
    AUTHOR,
    EXECUTOR,
    REVIEWER,
    ROLES,
    VIEWER,
    AuthError,
    GovernancePolicy,
    Principal,
    SecurityConfig,
    TokenAuthenticator,
)

log = logging.getLogger(__name__)


class BodySizeLimit:
    """Reject request bodies above `limit` bytes (413) before they are buffered: FastAPI reads the body
    before authentication runs, so without this an unauthenticated client could make the server hold an
    arbitrarily large body in memory. Checks Content-Length, and counts streamed bytes when it is absent."""

    def __init__(self, app: Any, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.limit):
            await self._reject(send)
            return
        seen = 0
        state = {"started": False, "rejected": False}

        async def limited_receive() -> dict:
            nonlocal seen
            if state["rejected"]:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    # answer now and make the app stop reading (it sees a disconnect); its own response,
                    # if any, is dropped below
                    if not state["started"]:
                        state["rejected"] = True
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: dict) -> None:
            if state["rejected"]:
                return
            if message["type"] == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not state["rejected"]:
                raise

    async def _reject(self, send: Callable) -> None:
        body = b'{"detail":"request body too large (CB_API_MAX_BODY_BYTES)"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


ANY_ROLE = tuple(sorted(ROLES))


class _Strict(BaseModel):
    # Unknown fields (e.g. a legacy `reviewer` or `user_id`) are rejected, not silently trusted.
    model_config = ConfigDict(extra="forbid")


API_MAX_CSV_CHARS = 5_000_000  # per request body file; larger imports go through the CLI


class CsvContent(_Strict):
    file_name: str = Field(min_length=1, max_length=128, description="bare file name, e.g. patients.csv")
    content: str = Field(min_length=1, max_length=API_MAX_CSV_CHARS)
    table: str | None = Field(default=None, max_length=48)
    column_types: dict[str, Literal["VARCHAR", "BIGINT", "DOUBLE", "DATE", "BOOLEAN"]] | None = Field(
        default=None, max_length=500, description="explicit column types; every value must convert exactly"
    )


class CsvProfileRequest(CsvContent):
    null_values: list[str] = Field(default_factory=list, max_length=10)


class CsvIngestRequest(_Strict):
    files: list[CsvContent] = Field(min_length=1, max_length=20)
    mapping_yaml: str | None = Field(default=None, max_length=200_000)
    new_version: bool = False
    dry_run: bool = False
    null_values: list[str] = Field(default_factory=list, max_length=10)


class HoldRequest(_Strict):
    reason: str = Field(min_length=1, max_length=500)


class TreatmentRunRequest(_Strict):
    rule_set: str | None = Field(default=None, max_length=48, pattern=r"^[a-z][a-z0-9_]{0,47}$")
    rules_yaml: str | None = Field(default=None, max_length=200_000)
    reference_date: date | None = Field(
        default=None, description="fixed 'as of' date for date validation (default: today, recorded with the run)"
    )
    reuse: bool = Field(
        default=False, description="return a completed run with the same input fingerprint instead of recomputing"
    )

    @model_validator(mode="after")
    def _one(self) -> TreatmentRunRequest:
        if (self.rule_set is None) == (self.rules_yaml is None):
            raise ValueError("give rule_set (a configured rule set name) or rules_yaml, not both")
        return self


class AskRequest(_Strict):
    query: str = Field(min_length=1, max_length=4000)


class RefineRequest(_Strict):
    instruction: str = Field(min_length=1, max_length=2000, description="the change, in plain language")


class CompareCohortsRequest(_Strict):
    generation_ids: list[str] = Field(min_length=2, max_length=6)


class ReviewRequest(_Strict):
    decision: Literal["approved", "rejected"]
    comments: str = Field(default="", max_length=4000)


class ExecuteRequest(_Strict):
    # Requesting draft execution is only honored when the server policy allows it (development only).
    allow_draft: bool = False


class SubmitIRRequest(_Strict):
    ir: CohortDefinition
    parent_definition_id: int | None = None


class ValidateIRRequest(_Strict):
    ir: CohortDefinition


class ProxyPayload(_Strict):
    """A proxy definition as JSON (`definition`) or as YAML text (`yaml`); exactly one."""

    definition: dict[str, Any] | None = None
    yaml: str | None = Field(default=None, max_length=500_000)


class SubmitProxyRequest(ProxyPayload):
    parent_definition_id: int | None = None


class CompareRequest(_Strict):
    generation_ids: list[str] = Field(min_length=2, max_length=6)


class ReferenceValidationRequest(_Strict):
    reference_name: str = Field(min_length=1, max_length=128)
    generation_id: str | None = None
    positive_tiers: list[str] | None = None
    intended_use: str | None = Field(default=None, max_length=64)
    criteria: AcceptanceCriteria | None = None  # only when not prespecified in the definition
    eligibility: EligibilityRules | None = None


class EvaluationReviewRequest(_Strict):
    decision: Literal["accepted", "rejected"]
    rationale: str = Field(min_length=10, max_length=4000)


class ReferenceLabel(_Strict):
    """One reference record. `label` (case / non_case / unknown) or the older boolean `is_case`;
    neither = unknown (indeterminate). Several records per patient are allowed and collapsed."""

    person_id: int
    is_case: bool | None = None
    label: Literal["case", "non_case", "unknown"] | None = None
    reference_date: date | None = None

    @model_validator(mode="after")
    def _one_label(self) -> ReferenceLabel:
        if self.label is not None and self.is_case is not None:
            raise ValueError("give either label or is_case, not both")
        return self

    def as_record(self) -> ReferenceRecord:
        label = self.label or normalize_label(self.is_case)
        return ReferenceRecord(self.person_id, label, self.reference_date)


class ReferenceLoadRequest(_Strict):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.\-]+$")
    source: str = Field(min_length=3, max_length=1000, description="the external reference standard")
    labels: list[ReferenceLabel] = Field(min_length=1, max_length=1_000_000)


def create_app(
    builder: CohortBuilder | None = None,
    security: SecurityConfig | None = None,
    authenticator: TokenAuthenticator | None = None,
) -> FastAPI:
    security = security or SecurityConfig.from_env()  # raises ConfigError on unsafe settings
    security.check()
    authenticator = authenticator if authenticator is not None else security.authenticator()
    policy = GovernancePolicy.from_security(security)
    if security.dev_bypass:
        log.warning(
            "CB_AUTH_DEV_BYPASS is ON (development): unauthenticated requests act as %r with roles %s",
            security.dev_subject,
            sorted(security.dev_roles),
        )
    elif len(authenticator) == 0:
        log.warning("no API tokens configured (CB_AUTH_TOKENS_FILE): all protected endpoints will return 401")

    app = FastAPI(title="Cohort Builder Agents", version="0.2.0")
    app.add_middleware(BodySizeLimit, limit=security.max_body_bytes)
    state: dict[str, Any] = {"builder": builder}
    if builder is not None:
        builder.policy = policy
    lock = threading.RLock()  # one DuckDB connection: serialize access
    bearer = HTTPBearer(auto_error=False)

    def b() -> CohortBuilder:
        if state["builder"] is None:
            state["builder"] = CohortBuilder(policy=policy)
        return state["builder"]

    # ---- authentication / authorization ------------------------------------------
    def current_principal(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
        if creds is not None:
            if creds.scheme.lower() != "bearer":
                raise HTTPException(401, "invalid authorization scheme", headers={"WWW-Authenticate": "Bearer"})
            try:
                return authenticator.authenticate(creds.credentials)
            except AuthError as exc:
                log.warning("authentication failed for %s %s: %s", request.method, request.url.path, exc)
                raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from None
        dev = security.dev_principal()
        if dev is not None:
            return dev
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    def require(*roles: str) -> Callable[..., Principal]:
        def dep(request: Request, principal: Principal = Depends(current_principal)) -> Principal:
            if not principal.has_any(*roles):
                with lock:
                    b().store.audit(
                        principal.subject,
                        "authz.denied",
                        "endpoint",
                        request.url.path,
                        "denied",
                        principal.tenant,
                        {"method": request.method, "required_any": sorted(roles)},
                    )
                raise HTTPException(403, f"requires one of the roles: {', '.join(sorted(roles))}")
            return principal

        return dep

    any_role = require(*ANY_ROLE)
    need_author = require(AUTHOR)
    need_reviewer = require(REVIEWER)
    need_executor = require(EXECUTOR)
    need_admin = require(ADMIN)
    need_trace_reader = require(AUTHOR, REVIEWER)

    def scope(p: Principal) -> str | None:
        """Tenant filter: admins see all tenants, everyone else only their own."""
        return None if p.is_admin else p.tenant

    # ---- error mapping (no internals, SQL or secrets in responses) -----------------
    @app.exception_handler(GovernanceError)
    def _gov(_: Request, exc: GovernanceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(QueryTimeout)
    def _timeout(_: Request, exc: QueryTimeout) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "error_id": exc.error_id}, status_code=504)

    @app.exception_handler(ExecutionError)
    def _exec(_: Request, exc: ExecutionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "error_id": exc.error_id}, status_code=500)

    @app.exception_handler(Exception)
    def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        error_id = str(uuid.uuid4())[:8]
        # no log.exception: exception messages can quote data values; log the redacted message + frames
        frames = "".join(traceback.format_tb(exc.__traceback__))
        log.error("unhandled error [%s]: %s\n%s", error_id, redact_error(exc), frames)
        return JSONResponse({"detail": "internal error", "error_id": error_id}, status_code=500)

    # ---- endpoints ---------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/me")
    def me(p: Principal = Depends(any_role)) -> dict:
        return {"subject": p.subject, "roles": sorted(p.roles), "tenant": p.tenant, "auth_method": p.auth_method}

    @app.get("/versions")
    def versions(_: Principal = Depends(any_role)) -> dict:
        with lock:
            return b().component_versions()

    @app.post("/cohorts/ask")
    def ask(req: AskRequest, p: Principal = Depends(need_author)) -> dict:
        with lock:
            return b().ask(req.query, p.subject, p.tenant).as_dict()

    @app.post("/cohorts/validate")
    def validate_ir(req: ValidateIRRequest, _: Principal = Depends(need_author)) -> dict:
        with lock:
            bb = b()
            issues, attrition = validate(req.ir, bb.ontology, bb.vocab, bb.executor)
            return {
                "valid": not any(i.severity == "error" for i in issues),
                "issues": [i.as_dict() for i in issues],
                "attrition": attrition.suppressed(bb.executor.min_cell) if attrition else None,
                "explanation": bb.explainer.explain(req.ir),
                "semantic_hash": req.ir.semantic_hash(),
            }

    @app.post("/cohorts")
    def submit(req: SubmitIRRequest, p: Principal = Depends(need_author)) -> dict:
        with lock:
            try:
                def_id, issues = b().submit_ir(req.ir, p.subject, req.parent_definition_id, p.tenant)
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"")) from exc
            return {"cohort_definition_id": def_id, "issues": issues}

    @app.get("/cohorts")
    def list_cohorts(limit: int = 50, p: Principal = Depends(any_role)) -> list[dict]:
        with lock:
            return b().store.list_definitions(max(1, min(limit, 500)), tenant=scope(p))

    @app.get("/cohorts/{def_id}")
    def get_cohort(def_id: int, p: Principal = Depends(any_role)) -> dict:
        with lock:
            try:
                row, ir = b().load_definition(def_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            return {**row, "explanation": b().explain(ir)}

    @app.get("/cohorts/{def_id}/sql")
    def get_sql(def_id: int, p: Principal = Depends(any_role)) -> dict:
        with lock:
            try:
                return {"sql": b().compile_sql(def_id, scope(p))}
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            except ValueError as exc:  # e.g. the definition needs data the active dataset does not have
                raise HTTPException(409, str(exc)) from exc

    @app.post("/cohorts/{def_id}/review")
    def review(def_id: int, req: ReviewRequest, p: Principal = Depends(need_reviewer)) -> dict:
        with lock:
            try:
                b().review(def_id, p.subject, req.decision, req.comments, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {"cohort_definition_id": def_id, "decision": req.decision, "reviewer": p.subject}

    @app.post("/cohorts/{def_id}/execute")
    def execute(def_id: int, req: ExecuteRequest | None = None, p: Principal = Depends(need_executor)) -> dict:
        req = req or ExecuteRequest()
        with lock:
            try:
                return b().execute(def_id, p.subject, req.allow_draft, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    def _visible_run(run_id: str, p: Principal) -> dict:
        run = b().get_run(run_id, scope(p))
        # run traces contain the original prompt: authors see their own runs, reviewers/admins the tenant's
        if run is None or not (p.has(REVIEWER) or (p.has(AUTHOR) and run.get("user_id") == p.subject)):
            raise HTTPException(404, "run not found")
        return run

    @app.get("/runs")
    def list_runs(limit: int = 50, p: Principal = Depends(need_trace_reader)) -> list[dict]:
        own = None if p.has(REVIEWER) else p.subject  # same visibility as GET /runs/{id}
        with lock:
            return b().store.list_runs(max(1, min(limit, 500)), scope(p), own)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, p: Principal = Depends(need_trace_reader)) -> dict:
        with lock:
            return _visible_run(run_id, p)

    @app.post("/runs/{run_id}/replay")
    def replay(run_id: str, p: Principal = Depends(need_author)) -> dict:
        with lock:
            run = _visible_run(run_id, p)
            if run.get("user_id") != p.subject and not p.is_admin:
                raise HTTPException(404, "run not found")
            return b().replay(run_id, scope(p), actor=p.subject)

    @app.get("/concepts/search")
    def search(q: str, domain: str | None = None, limit: int = 10, _: Principal = Depends(any_role)) -> list[dict]:
        with lock:
            return b().vocab.search_concepts(q[:200], domain, limit=max(1, min(limit, 25)))

    @app.get("/concepts/{concept_id}")
    def concept(concept_id: int, _: Principal = Depends(any_role)) -> dict:
        with lock:
            c = b().vocab.get_concept(concept_id)
        if c is None:
            raise HTTPException(404, f"concept {concept_id} not found in the vocabulary")
        return c

    @app.get("/status")
    def status(_: Principal = Depends(any_role)) -> dict:
        with lock:
            return b().system_status()

    @app.post("/cohorts/{def_id}/refine")
    def refine(def_id: int, req: RefineRequest, p: Principal = Depends(need_author)) -> dict:
        with lock:
            try:
                return b().refine(def_id, req.instruction, p.subject, scope(p))
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"")) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/cohorts/{def_id}/versions")
    def versions_of(def_id: int, p: Principal = Depends(any_role)) -> list[dict]:
        with lock:
            try:
                b().load_definition(def_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            rows = b().store.lineage(def_id)
        t = scope(p)
        return [r for r in rows if t is None or (r.get("tenant") or "default") == t]

    @app.get("/cohorts/{def_id}/reviews")
    def reviews_of(def_id: int, p: Principal = Depends(any_role)) -> list[dict]:
        with lock:
            try:
                b().load_definition(def_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            return b().store.reviews(def_id)

    @app.get("/cohorts/{def_id}/diff/{other_id}")
    def diff_cohorts(def_id: int, other_id: int, p: Principal = Depends(any_role)) -> dict:
        from .ir_diff import ir_changes

        with lock:
            try:
                ra, a = b().load_cohort(def_id, scope(p))
                rb, bdef = b().load_cohort(other_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"")) from exc
        return {
            "from": def_id,
            "to": other_id,
            "same_dataset": ra.get("dataset") == rb.get("dataset"),
            "same_semantics": ra["semantic_hash"] == rb["semantic_hash"],
            "changes": ir_changes(a, bdef),
        }

    need_counts = require(EXECUTOR, REVIEWER)

    @app.get("/cohorts/{def_id}/generations")
    def generations_of(def_id: int, p: Principal = Depends(need_counts)) -> list[dict]:
        from .executor import suppress_count

        with lock:
            try:
                b().load_definition(def_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            k = b().executor.min_cell
            gens = b().store.generations(def_id)
        keep = ("generation_id", "cohort_definition_id", "dataset", "data_snapshot", "compiler_version", "sql_hash",
                "executed_by", "executed_at", "caveats")  # fmt: skip
        return [
            {**{x: g.get(x) for x in keep}, "person_count": suppress_count(int(g["person_count"]), k)} for g in gens
        ]

    @app.post("/cohorts/compare")
    def compare_cohorts(req: CompareCohortsRequest, p: Principal = Depends(need_counts)) -> dict:
        with lock:
            try:
                return b().compare_generations(req.generation_ids, p.subject, scope(p))
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"")) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/audit")
    def audit(limit: int = 100, _: Principal = Depends(need_admin)) -> list[dict]:
        with lock:
            return b().store.audit_events(max(1, min(limit, 1000)))

    # ---- proxy cohorts --------------------------------------------------------------------
    need_results = require(EXECUTOR, REVIEWER)

    def proxy_from(req: ProxyPayload) -> ProxyDefinition:
        if (req.definition is None) == (req.yaml is None):
            raise HTTPException(422, "provide exactly one of `definition` (JSON) or `yaml`")
        try:
            with lock:
                return b().parse_proxy_payload(req.yaml if req.yaml is not None else req.definition or {})
        except ValidationError as exc:
            raise HTTPException(422, {"errors": format_validation_error(exc)}) from exc
        except DuplicateKeyError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # malformed YAML
            raise HTTPException(422, f"invalid proxy definition: {type(exc).__name__}") from exc

    def proxy_call(fn: Callable[[], Any], def_id: int | None = None) -> Any:
        with lock:
            try:
                return fn()
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"") or f"proxy definition {def_id} not found") from exc
            except VersionConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/proxy-cohorts/validate")
    def proxy_validate(req: ProxyPayload, _: Principal = Depends(need_author)) -> dict:
        p = proxy_from(req)
        return proxy_call(lambda: b().validate_proxy_definition(p))

    @app.post("/proxy-cohorts")
    def proxy_submit(req: SubmitProxyRequest, p: Principal = Depends(need_author)) -> dict:
        defn = proxy_from(req)
        return proxy_call(lambda: b().submit_proxy(defn, p.subject, p.tenant, req.parent_definition_id))

    @app.post("/proxy-cohorts/ask")
    def proxy_ask(req: AskRequest, p: Principal = Depends(need_author)) -> dict:
        return proxy_call(lambda: b().ask_proxy(req.query, p.subject, p.tenant).as_dict())

    @app.get("/proxy-cohorts")
    def proxy_list(limit: int = 50, p: Principal = Depends(any_role)) -> list[dict]:
        with lock:
            return b().store.list_definitions(max(1, min(limit, 500)), tenant=scope(p), kind="proxy")

    @app.get("/proxy-cohorts/{def_id}")
    def proxy_get(def_id: int, p: Principal = Depends(any_role)) -> dict:
        def run() -> dict:
            row, defn = b().load_proxy(def_id, scope(p))
            return {**row, "explanation": b().explain(defn)}

        return proxy_call(run, def_id)

    @app.get("/proxy-cohorts/{def_id}/versions")
    def proxy_versions(def_id: int, p: Principal = Depends(any_role)) -> list[dict]:
        return proxy_call(lambda: b().proxy_versions(def_id, scope(p)), def_id)

    @app.get("/proxy-cohorts/{def_id}/review-packet")
    def proxy_packet(def_id: int, p: Principal = Depends(any_role)) -> dict:
        return proxy_call(lambda: b().proxy_review_packet(def_id, scope(p)), def_id)

    @app.post("/proxy-cohorts/{def_id}/review")
    def proxy_review(def_id: int, req: ReviewRequest, p: Principal = Depends(need_reviewer)) -> dict:
        def run() -> dict:
            b().load_proxy(def_id, scope(p))
            b().review(def_id, p.subject, req.decision, req.comments, scope(p))
            return {"cohort_definition_id": def_id, "decision": req.decision, "reviewer": p.subject}

        return proxy_call(run, def_id)

    @app.post("/proxy-cohorts/{def_id}/compile")
    def proxy_compile(def_id: int, p: Principal = Depends(any_role)) -> dict:
        return proxy_call(lambda: b().compile_proxy(def_id, scope(p)), def_id)

    @app.post("/proxy-cohorts/{def_id}/execute")
    def proxy_execute(def_id: int, req: ExecuteRequest | None = None, p: Principal = Depends(need_executor)) -> dict:
        allow_draft = (req or ExecuteRequest()).allow_draft

        def run() -> dict:
            b().load_proxy(def_id, scope(p))
            return b().execute(def_id, p.subject, allow_draft, scope(p))

        return proxy_call(run, def_id)

    @app.get("/proxy-cohorts/{def_id}/results")
    def proxy_results(def_id: int, generation_id: str | None = None, p: Principal = Depends(need_results)) -> dict:
        return proxy_call(lambda: b().proxy_results(def_id, scope(p), generation_id), def_id)

    @app.get("/proxy-cohorts/{def_id}/evidence-summary")
    def proxy_evidence(def_id: int, generation_id: str | None = None, p: Principal = Depends(need_results)) -> dict:
        def run() -> dict:
            r = b().proxy_results(def_id, scope(p), generation_id)
            return {
                k: r[k]
                for k in (
                    "generation_id",
                    "cohort_definition_id",
                    "algorithm",
                    "evidence_summary",
                    "min_cell_count",
                    "governance",
                )
            }

        return proxy_call(run, def_id)

    @app.post("/proxy-cohorts/compare")
    def proxy_compare(req: CompareRequest, p: Principal = Depends(need_results)) -> dict:
        return proxy_call(lambda: b().compare_generations(req.generation_ids, p.subject, scope(p)))

    @app.post("/proxy-cohorts/{def_id}/reference-validation")
    def proxy_reference_validation(
        def_id: int, req: ReferenceValidationRequest, p: Principal = Depends(need_reviewer)
    ) -> dict:
        return proxy_call(
            lambda: b().evaluate_against_reference(
                def_id,
                req.reference_name,
                p.subject,
                scope(p),
                req.generation_id,
                req.positive_tiers,
                req.intended_use,
                req.criteria,
                req.eligibility,
            ),
            def_id,
        )

    @app.get("/proxy-cohorts/{def_id}/status")
    def proxy_status(def_id: int, p: Principal = Depends(any_role)) -> dict:
        return proxy_call(lambda: b().proxy_status(def_id, scope(p)), def_id)

    @app.get("/proxy-cohorts/{def_id}/evaluations")
    def proxy_evaluations(def_id: int, p: Principal = Depends(need_results)) -> list[dict]:
        return proxy_call(lambda: b().list_evaluations(def_id, scope(p)), def_id)

    @app.post("/proxy-cohorts/{def_id}/evaluations/{validation_id}/review")
    def proxy_evaluation_review(
        def_id: int, validation_id: str, req: EvaluationReviewRequest, p: Principal = Depends(need_reviewer)
    ) -> dict:
        return proxy_call(
            lambda: b().review_evaluation(def_id, validation_id, p.subject, req.decision, req.rationale, scope(p)),
            def_id,
        )

    @app.post("/proxy-references")
    def proxy_reference_load(req: ReferenceLoadRequest, p: Principal = Depends(need_admin)) -> dict:
        labels = [lab.as_record() for lab in req.labels]
        return proxy_call(lambda: b().load_reference(req.name, labels, req.source, p.subject, p.tenant))

    @app.get("/proxy-cohorts/{def_id}/patients/{subject_id}/explanation")
    def proxy_patient(
        def_id: int, subject_id: int, generation_id: str | None = None, p: Principal = Depends(need_admin)
    ) -> dict:
        return proxy_call(
            lambda: b().patient_explanation(def_id, subject_id, p.subject, p.is_admin, scope(p), generation_id),
            def_id,
        )

    # ---- treatment events / episodes ------------------------------------------------------------
    def treatment_errors(exc: Exception) -> HTTPException:
        from .treatment_service import TreatmentRunFailed

        if isinstance(exc, TreatmentRunFailed):
            code = {"cancelled": 409, "error": 500}.get(exc.kind, 503)  # resources / timeout: 503
            return HTTPException(
                code, {"message": str(exc), "run_id": exc.run_id, "status": exc.status, "failure_kind": exc.kind}
            )
        if isinstance(exc, KeyError):
            return HTTPException(404, str(exc).strip("'\""))
        if isinstance(exc, ValidationError):
            return HTTPException(422, f"invalid rule set: {exc.error_count()} problem(s)")
        return HTTPException(422, str(exc))

    def treatment_call(fn: Callable[[Any], Any]) -> Any:
        """Run fn(service) on a treatment service with its own database connection. Only creating the
        connection holds the API lock, so a long run does not block other requests (or its own cancel)."""
        from .treatment_service import TreatmentService

        with lock:
            svc = TreatmentService(b())
        try:
            return fn(svc)
        except (KeyError, ValueError) as exc:
            raise treatment_errors(exc) from exc
        finally:
            svc.close()

    @app.post("/treatment-runs")
    def treatment_run(req: TreatmentRunRequest, p: Principal = Depends(need_executor)) -> dict:
        from .treatment_config import load_rules
        from .treatment_service import resolve_rules

        def go(svc: Any) -> Any:
            rules = (
                load_rules(req.rules_yaml)
                if req.rules_yaml
                else resolve_rules(svc.b.settings.ontology_dir, req.rule_set or "")
            )
            return svc.run(rules, p.subject, p.tenant, reference_date=req.reference_date, reuse=req.reuse)

        return treatment_call(go)

    @app.get("/treatment-runs/{run_id}")
    def treatment_summary(run_id: str, p: Principal = Depends(need_results)) -> dict:
        return treatment_call(lambda svc: svc.summary(run_id, scope(p)))

    @app.post("/treatment-runs/{run_id}/cancel")
    def treatment_cancel(run_id: str, p: Principal = Depends(need_executor)) -> dict:
        return treatment_call(lambda svc: svc.cancel(run_id, p.subject, scope(p)))

    @app.post("/treatment-runs/{run_id}/hold")
    def treatment_hold(run_id: str, req: HoldRequest, p: Principal = Depends(need_admin)) -> dict:
        from .treatment_retention import set_hold

        return treatment_call(lambda svc: set_hold(svc, run_id, p.subject, True, req.reason, scope(p)))

    @app.post("/treatment-runs/{run_id}/release")
    def treatment_release(run_id: str, p: Principal = Depends(need_admin)) -> dict:
        from .treatment_retention import set_hold

        return treatment_call(lambda svc: set_hold(svc, run_id, p.subject, False, "", scope(p)))

    @app.get("/treatment-runs/{run_id}/episodes/{episode_id}/explanation")
    def treatment_explain(run_id: str, episode_id: str, p: Principal = Depends(need_admin)) -> dict:
        return treatment_call(lambda svc: svc.explain_episode(run_id, episode_id, p.subject, p.is_admin, scope(p)))

    # ---- imported datasets (CSV) --------------------------------------------------------------
    def registry() -> Any:
        from .dataset_registry import DatasetRegistry

        bb = b()
        return DatasetRegistry(bb.con, bb.settings.ontology_dir, bb.settings.local_datasets_dir, bb.store)

    def ingest_errors(exc: Exception) -> HTTPException:
        from .csv_ingest import IngestError

        if isinstance(exc, IngestError):
            return HTTPException(422, {"message": str(exc), "problems": exc.problems})
        return HTTPException(422, {"message": f"invalid input: {exc}"})

    @app.get("/datasets")
    def list_datasets(_: Principal = Depends(any_role)) -> dict:
        with lock:
            reg = registry()
            return {
                "active": b().ontology.dataset_name,
                "built_in": sorted(reg.builtin_names()),
                "imported": reg.list_datasets(),
            }

    @app.get("/datasets/{name}")
    def dataset_info(name: str, _: Principal = Depends(any_role)) -> dict:
        with lock:
            try:
                return registry().info(name)
            except KeyError as exc:
                raise HTTPException(404, "dataset not found") from exc

    @app.get("/datasets/{name}/versions/{version}")
    def dataset_version(name: str, version: int, _: Principal = Depends(any_role)) -> dict:
        with lock:
            try:
                info = registry().info(name)
            except KeyError as exc:
                raise HTTPException(404, "dataset not found") from exc
            rows = [v for v in info["versions"] if v["version"] == version]
            if not rows:
                raise HTTPException(404, "dataset version not found")
            return {"dataset": name, **rows[0]}

    @app.post("/datasets/csv/profile")
    def profile_csv_content(req: CsvProfileRequest, _: Principal = Depends(need_author)) -> dict:
        from .csv_ingest import profile_text

        return profile_text(
            req.file_name,
            req.content,
            req.table,
            tuple(req.null_values),
            API_MAX_CSV_CHARS * 4,
            dict(req.column_types) if req.column_types else None,
        )

    @app.post("/datasets/reconcile")
    def reconcile_datasets(p: Principal = Depends(need_admin)) -> dict:
        with lock:
            return registry().reconcile(p.subject)

    @app.post("/datasets/{name}/ingest")
    def ingest_csv(name: str, req: CsvIngestRequest, p: Principal = Depends(need_admin)) -> dict:
        from .csv_ingest import default_table_name
        from .dataset_registry import Upload, load_mapping

        with lock:
            try:
                mapping = load_mapping(req.mapping_yaml) if req.mapping_yaml else None
                files: list[tuple[str | None, Any]] = [(f.table, Upload(f.file_name, f.content)) for f in req.files]
                types: dict[str, dict[str, str]] = {}
                for f in req.files:
                    if f.column_types:
                        table = f.table or default_table_name(Path(Path(f.file_name).name))
                        types[table] = dict(f.column_types)
                return registry().ingest(
                    name,
                    files,
                    p.subject,
                    mapping=mapping,
                    new_version=req.new_version,
                    dry_run=req.dry_run,
                    null_values=tuple(req.null_values),
                    max_bytes=API_MAX_CSV_CHARS * 4,
                    column_types=types or None,
                )
            except (ValueError, ValidationError) as exc:
                raise ingest_errors(exc) from exc

    _ = VIEWER  # every role can read; VIEWER is the role for read-only users
    return app


def __getattr__(name: str) -> Any:
    # `uvicorn cohort_builder.api:app` builds the app on first access, so importing this
    # module never reads configuration or opens the database.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
