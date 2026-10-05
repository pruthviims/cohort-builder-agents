"""HTTP API (FastAPI). `cohort-builder serve` or `uvicorn cohort_builder.api:app`."""
from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ir import CohortDefinition
from .orchestrator import CohortBuilder


class AskRequest(BaseModel):
    query: str
    user_id: str = "api"


class ReviewRequest(BaseModel):
    reviewer: str
    decision: str  # approved | rejected
    comments: str = ""


class ExecuteRequest(BaseModel):
    user_id: str = "api"
    allow_draft: bool = False


class SubmitIRRequest(BaseModel):
    ir: CohortDefinition
    user_id: str = "api"
    parent_definition_id: int | None = None


def create_app(builder: CohortBuilder | None = None) -> FastAPI:
    app = FastAPI(title="Cohort Builder Agents", version="0.1.0")
    state: dict[str, Any] = {"builder": builder}
    lock = threading.Lock()  # one DuckDB connection: serialize access

    def b() -> CohortBuilder:
        if state["builder"] is None:
            state["builder"] = CohortBuilder()
        return state["builder"]

    @app.get("/health")
    def health() -> dict:
        with lock:
            return {"status": "ok", "versions": b().component_versions()}

    @app.post("/cohorts/ask")
    def ask(req: AskRequest) -> dict:
        with lock:
            return b().ask(req.query, req.user_id).as_dict()

    @app.post("/cohorts")
    def submit(req: SubmitIRRequest) -> dict:
        with lock:
            def_id, issues = b().submit_ir(req.ir, req.user_id, req.parent_definition_id)
            return {"cohort_definition_id": def_id, "issues": issues}

    @app.get("/cohorts")
    def list_cohorts(limit: int = 50) -> list[dict]:
        with lock:
            return b().store.list_definitions(limit)

    @app.get("/cohorts/{def_id}")
    def get_cohort(def_id: int) -> dict:
        with lock:
            try:
                row, ir = b().load_definition(def_id)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {**row, "explanation": b().explainer.explain(ir)}

    @app.get("/cohorts/{def_id}/sql")
    def get_sql(def_id: int) -> dict:
        with lock:
            try:
                return {"sql": b().compile_sql(def_id)}
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc

    @app.post("/cohorts/{def_id}/review")
    def review(def_id: int, req: ReviewRequest) -> dict:
        with lock:
            try:
                b().review(def_id, req.reviewer, req.decision, req.comments)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {"cohort_definition_id": def_id, "decision": req.decision}

    @app.post("/cohorts/{def_id}/execute")
    def execute(def_id: int, req: ExecuteRequest) -> dict:
        with lock:
            try:
                return b().execute(def_id, req.user_id, req.allow_draft)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with lock:
            run = b().store.get_run(run_id)
            if run is None:
                raise HTTPException(404, "run not found")
            return run

    @app.post("/runs/{run_id}/replay")
    def replay(run_id: str) -> dict:
        with lock:
            try:
                return b().replay(run_id)
            except KeyError as exc:
                raise HTTPException(404, "run not found") from exc

    @app.get("/concepts/search")
    def search(q: str, domain: str | None = None, limit: int = 10) -> list[dict]:
        with lock:
            return b().vocab.search_concepts(q, domain, limit=limit)

    return app


app = create_app()
