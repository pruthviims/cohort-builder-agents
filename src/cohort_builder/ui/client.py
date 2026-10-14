"""Typed HTTP client for the Cohort Builder API, used by the web workspace.

The workspace never talks to the database or the agents directly: every read and every action goes through
the authenticated API, so authorization, approval gates, suppression and auditing stay in one place.

The bearer token lives only in this object (server side, in the Streamlit session); it is never written to
logs, URLs or error messages. Errors are mapped to `ApiError` with a `kind` the UI can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

ErrorKind = Literal["unauthorized", "forbidden", "not_found", "conflict", "invalid", "unavailable", "server"]
DEFAULT_TIMEOUT = 300.0  # agent runs (LLM + dry run) can take minutes


@dataclass
class ApiError(Exception):
    kind: ErrorKind
    status: int | None
    message: str
    detail: Any = None

    def __str__(self) -> str:
        return self.message


def _message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("detail") or detail)
    if isinstance(detail, list):  # FastAPI validation errors
        parts = []
        for d in detail[:5]:
            loc = ".".join(str(x) for x in d.get("loc", []) if x != "body")
            parts.append(f"{loc}: {d.get('msg')}" if loc else str(d.get("msg")))
        return "; ".join(parts) or "invalid request"
    return "request failed"


class ApiClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._token = token
        # an injected client (tests: FastAPI's TestClient, which is an httpx.Client) or a real one
        self._http = session or httpx.Client(base_url=self.base_url, timeout=timeout)

    def __repr__(self) -> str:  # never show the token
        return f"ApiClient({self.base_url!r}, token={'set' if self._token else 'none'})"

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    # ---- transport ------------------------------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            r = self._http.request(method, path, headers=headers, **kw)
        except httpx.TimeoutException as exc:
            raise ApiError(
                "unavailable", None, "the API did not answer in time (it may still finish the request)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                "unavailable", None, f"cannot reach the API at {self.base_url}: is `cohort-builder serve` running?"
            ) from exc
        if r.status_code < 400:
            return r.json() if r.content else None
        try:
            body = r.json()
        except ValueError:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        kinds: dict[int, ErrorKind] = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            422: "invalid",
            502: "unavailable",
            503: "unavailable",
            504: "unavailable",
        }
        kind: ErrorKind = kinds.get(r.status_code, "server" if r.status_code >= 500 else "invalid")
        msg = _message(detail)
        if kind == "unauthorized":
            msg = "not signed in, or the token is unknown or expired"
        elif kind == "server":
            eid = body.get("error_id") if isinstance(body, dict) else None
            msg = f"the server hit an internal error{f' (error id {eid})' if eid else ''}"
        raise ApiError(kind, r.status_code, msg, detail)

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: Any = None) -> Any:
        return self._request("POST", path, json=body)

    # ---- identity / system ------------------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def me(self) -> dict:
        return self.get("/me")

    def status(self) -> dict:
        return self.get("/status")

    def versions(self) -> dict:
        return self.get("/versions")

    def audit(self, limit: int = 50) -> list[dict]:
        return self.get("/audit", limit=limit)

    # ---- cohorts ----------------------------------------------------------------------------------
    def list_cohorts(self, limit: int = 200) -> list[dict]:
        return self.get("/cohorts", limit=limit)

    def cohort(self, def_id: int) -> dict:
        return self.get(f"/cohorts/{int(def_id)}")

    def cohort_sql(self, def_id: int) -> str:
        return str(self.get(f"/cohorts/{int(def_id)}/sql")["sql"])

    def versions_of(self, def_id: int) -> list[dict]:
        return self.get(f"/cohorts/{int(def_id)}/versions")

    def reviews_of(self, def_id: int) -> list[dict]:
        return self.get(f"/cohorts/{int(def_id)}/reviews")

    def generations_of(self, def_id: int) -> list[dict]:
        return self.get(f"/cohorts/{int(def_id)}/generations")

    def diff(self, def_id: int, other_id: int) -> dict:
        return self.get(f"/cohorts/{int(def_id)}/diff/{int(other_id)}")

    def ask(self, query: str) -> dict:
        return self.post("/cohorts/ask", {"query": query})

    def refine(self, def_id: int, instruction: str) -> dict:
        return self.post(f"/cohorts/{int(def_id)}/refine", {"instruction": instruction})

    def validate(self, ir: dict) -> dict:
        return self.post("/cohorts/validate", {"ir": ir})

    def submit(self, ir: dict, parent_definition_id: int | None) -> dict:
        return self.post("/cohorts", {"ir": ir, "parent_definition_id": parent_definition_id})

    def review(self, def_id: int, decision: str, comments: str) -> dict:
        return self.post(f"/cohorts/{int(def_id)}/review", {"decision": decision, "comments": comments})

    def execute(self, def_id: int) -> dict:
        return self.post(f"/cohorts/{int(def_id)}/execute", {"allow_draft": False})

    def compare(self, generation_ids: list[str]) -> dict:
        return self.post("/cohorts/compare", {"generation_ids": generation_ids})

    # ---- runs -------------------------------------------------------------------------------------
    def runs(self, limit: int = 100) -> list[dict]:
        return self.get("/runs", limit=limit)

    def run(self, run_id: str) -> dict:
        return self.get(f"/runs/{run_id}")

    def replay(self, run_id: str) -> dict:
        return self.post(f"/runs/{run_id}/replay")

    # ---- vocabulary -------------------------------------------------------------------------------
    def search_concepts(self, q: str, domain: str | None = None, limit: int = 15) -> list[dict]:
        return self.get("/concepts/search", q=q, domain=domain, limit=limit)

    def concept(self, concept_id: int) -> dict:
        return self.get(f"/concepts/{int(concept_id)}")

    # ---- datasets ---------------------------------------------------------------------------------
    def datasets(self) -> dict:
        return self.get("/datasets")

    def dataset(self, name: str) -> dict:
        return self.get(f"/datasets/{name}")

    def profile_csv(self, file_name: str, content: str) -> dict:
        return self.post("/datasets/csv/profile", {"file_name": file_name, "content": content})

    def ingest_csv(
        self, name: str, files: list[dict], mapping_yaml: str | None, new_version: bool, dry_run: bool
    ) -> dict:
        body = {"files": files, "mapping_yaml": mapping_yaml, "new_version": new_version, "dry_run": dry_run}
        return self.post(f"/datasets/{name}/ingest", body)
