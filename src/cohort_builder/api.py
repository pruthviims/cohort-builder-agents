"""HTTP API (FastAPI). `cohort-builder serve` or `uvicorn cohort_builder.api:app`.

Every endpoint except /health requires a bearer token (see security.py). Identity
(who authored, reviewed or executed) always comes from the authenticated
principal, never from the request body. Resources are scoped to the principal's
tenant; other tenants' resources are reported as not found.

Roles per endpoint (admin implies all):
  GET  /health                       public (liveness only)
  GET  /me, /versions                any role
  GET  /cohorts, /cohorts/{id}, /cohorts/{id}/sql, /concepts/search   any role
  POST /cohorts/ask, /cohorts, /cohorts/validate                      author
  POST /cohorts/{id}/review                                           reviewer (not own work, unless policy allows)
  POST /cohorts/{id}/execute                                          executor (approved definitions only)
  GET  /runs/{id}, POST /runs/{id}/replay                             author (own runs), reviewer (tenant), admin
  GET  /audit                                                         admin
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .agents.validator import validate
from .executor import ExecutionError, QueryTimeout
from .ir import CohortDefinition
from .orchestrator import CohortBuilder, GovernanceError
from .security import (ADMIN, AUTHOR, EXECUTOR, REVIEWER, ROLES, VIEWER, AuthError, GovernancePolicy, Principal,
                       SecurityConfig, TokenAuthenticator)

log = logging.getLogger(__name__)
ANY_ROLE = tuple(sorted(ROLES))


class _Strict(BaseModel):
    # Unknown fields (e.g. a legacy `reviewer` or `user_id`) are rejected, not silently trusted.
    model_config = ConfigDict(extra="forbid")


class AskRequest(_Strict):
    query: str = Field(min_length=1, max_length=4000)


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


def create_app(builder: CohortBuilder | None = None, security: SecurityConfig | None = None,
               authenticator: TokenAuthenticator | None = None) -> FastAPI:
    security = security or SecurityConfig.from_env()   # raises ConfigError on unsafe settings
    security.check()
    authenticator = authenticator if authenticator is not None else security.authenticator()
    policy = GovernancePolicy.from_security(security)
    if security.dev_bypass:
        log.warning("CB_AUTH_DEV_BYPASS is ON (development): unauthenticated requests act as %r with roles %s",
                    security.dev_subject, sorted(security.dev_roles))
    elif len(authenticator) == 0:
        log.warning("no API tokens configured (CB_AUTH_TOKENS_FILE): all protected endpoints will return 401")

    app = FastAPI(title="Cohort Builder Agents", version="0.2.0")
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
    def current_principal(request: Request,
                          creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
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
                    b().store.audit(principal.subject, "authz.denied", "endpoint", request.url.path, "denied",
                                    principal.tenant, {"method": request.method, "required_any": sorted(roles)})
                raise HTTPException(403, f"requires one of the roles: {', '.join(sorted(roles))}")
            return principal
        return dep

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
        log.exception("unhandled error [%s]", error_id)
        return JSONResponse({"detail": "internal error", "error_id": error_id}, status_code=500)

    # ---- endpoints ---------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/me")
    def me(p: Principal = Depends(require(*ANY_ROLE))) -> dict:
        return {"subject": p.subject, "roles": sorted(p.roles), "tenant": p.tenant, "auth_method": p.auth_method}

    @app.get("/versions")
    def versions(_: Principal = Depends(require(*ANY_ROLE))) -> dict:
        with lock:
            return b().component_versions()

    @app.post("/cohorts/ask")
    def ask(req: AskRequest, p: Principal = Depends(require(AUTHOR))) -> dict:
        with lock:
            return b().ask(req.query, p.subject, p.tenant).as_dict()

    @app.post("/cohorts/validate")
    def validate_ir(req: ValidateIRRequest, _: Principal = Depends(require(AUTHOR))) -> dict:
        with lock:
            bb = b()
            issues, attrition = validate(req.ir, bb.ontology, bb.vocab, bb.executor)
            return {"valid": not any(i.severity == "error" for i in issues),
                    "issues": [i.as_dict() for i in issues],
                    "attrition": attrition.suppressed(bb.executor.min_cell) if attrition else None,
                    "explanation": bb.explainer.explain(req.ir), "semantic_hash": req.ir.semantic_hash()}

    @app.post("/cohorts")
    def submit(req: SubmitIRRequest, p: Principal = Depends(require(AUTHOR))) -> dict:
        with lock:
            try:
                def_id, issues = b().submit_ir(req.ir, p.subject, req.parent_definition_id, p.tenant)
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'\"")) from exc
            return {"cohort_definition_id": def_id, "issues": issues}

    @app.get("/cohorts")
    def list_cohorts(limit: int = 50, p: Principal = Depends(require(*ANY_ROLE))) -> list[dict]:
        with lock:
            return b().store.list_definitions(max(1, min(limit, 500)), tenant=scope(p))

    @app.get("/cohorts/{def_id}")
    def get_cohort(def_id: int, p: Principal = Depends(require(*ANY_ROLE))) -> dict:
        with lock:
            try:
                row, ir = b().load_definition(def_id, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            return {**row, "explanation": b().explainer.explain(ir)}

    @app.get("/cohorts/{def_id}/sql")
    def get_sql(def_id: int, p: Principal = Depends(require(*ANY_ROLE))) -> dict:
        with lock:
            try:
                return {"sql": b().compile_sql(def_id, scope(p))}
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc

    @app.post("/cohorts/{def_id}/review")
    def review(def_id: int, req: ReviewRequest, p: Principal = Depends(require(REVIEWER))) -> dict:
        with lock:
            try:
                b().review(def_id, p.subject, req.decision, req.comments, scope(p))
            except KeyError as exc:
                raise HTTPException(404, f"cohort definition {def_id} not found") from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {"cohort_definition_id": def_id, "decision": req.decision, "reviewer": p.subject}

    @app.post("/cohorts/{def_id}/execute")
    def execute(def_id: int, req: ExecuteRequest | None = None, p: Principal = Depends(require(EXECUTOR))) -> dict:
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

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, p: Principal = Depends(require(AUTHOR, REVIEWER))) -> dict:
        with lock:
            return _visible_run(run_id, p)

    @app.post("/runs/{run_id}/replay")
    def replay(run_id: str, p: Principal = Depends(require(AUTHOR))) -> dict:
        with lock:
            run = _visible_run(run_id, p)
            if run.get("user_id") != p.subject and not p.is_admin:
                raise HTTPException(404, "run not found")
            return b().replay(run_id, scope(p), actor=p.subject)

    @app.get("/concepts/search")
    def search(q: str, domain: str | None = None, limit: int = 10,
               _: Principal = Depends(require(*ANY_ROLE))) -> list[dict]:
        with lock:
            return b().vocab.search_concepts(q[:200], domain, limit=max(1, min(limit, 25)))

    @app.get("/audit")
    def audit(limit: int = 100, _: Principal = Depends(require(ADMIN))) -> list[dict]:
        with lock:
            return b().store.audit_events(max(1, min(limit, 1000)))

    _ = VIEWER  # every role can read; VIEWER is the role for read-only users
    return app


def __getattr__(name: str) -> Any:
    # `uvicorn cohort_builder.api:app` builds the app on first access, so importing this
    # module never reads configuration or opens the database.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
