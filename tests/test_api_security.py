"""HTTP API authentication, role-based authorization, governance policy and tenant isolation."""

from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from cohort_builder.api import create_app
from cohort_builder.config import REPO_ROOT
from cohort_builder.security import ConfigError, SecurityConfig, TokenAuthenticator, issue_token
from fake_llm import T2DM_QUERY

EXAMPLE_IR = json.loads((REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json").read_text())

USERS = {
    "alice": (["author", "viewer"], "research"),
    "bob": (["reviewer", "viewer"], "research"),
    "carol": (["executor", "viewer"], "research"),
    "victor": (["viewer"], "research"),
    "dana": (["author", "reviewer"], "research"),  # can author and review: self-approval test
    "eve": (["author", "reviewer", "executor"], "other"),  # different tenant
    "root": (["admin"], "ops"),
}


@pytest.fixture
def tokens(tmp_path):
    entries, out = [], {}
    for name, (roles, tenant) in USERS.items():
        token, entry = issue_token(name, roles, tenant, expires_days=30, token_id=f"{name}-test")
        entries.append(entry)
        out[name] = token
    expired_token, expired = issue_token("old", ["admin"], "ops", expires_days=30)
    expired["expires_at"] = "2020-01-01T00:00:00Z"
    entries.append(expired)
    out["expired"] = expired_token
    path = tmp_path / "tokens.yaml"
    path.write_text(yaml.safe_dump({"tokens": entries}))
    return path, out


def make_client(builder, tokens_path, **cfg) -> TestClient:
    security = SecurityConfig(environment=cfg.pop("environment", "production"), tokens_file=tokens_path, **cfg)
    return TestClient(create_app(builder, security), raise_server_exceptions=False)


def h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(builder, tokens):
    path, toks = tokens
    return make_client(builder, path), toks


def _draft(client, toks, who="alice") -> int:
    r = client.post("/cohorts", json={"ir": EXAMPLE_IR}, headers=h(toks[who]))
    assert r.status_code == 200, r.text
    return r.json()["cohort_definition_id"]


# ---- authentication ------------------------------------------------------------------
def test_health_is_public_and_minimal(api):
    client, _ = api
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/cohorts"),
        ("post", "/cohorts/ask"),
        ("get", "/me"),
        ("post", "/cohorts/1/review"),
        ("post", "/cohorts/1/execute"),
        ("get", "/audit"),
        ("get", "/versions"),
    ],
)
def test_unauthenticated_requests_are_rejected(api, method, path):
    client, _ = api
    r = getattr(client, method)(path)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize(
    "header", ["Bearer not-a-real-token-but-long-enough-0123456789", "Bearer short", "Basic YWxpY2U6cGFzcw=="]
)
def test_invalid_credentials_are_rejected(api, header):
    client, _ = api
    assert client.get("/me", headers={"Authorization": header}).status_code == 401


def test_expired_token_is_rejected(api):
    client, toks = api
    r = client.get("/me", headers=h(toks["expired"]))
    assert r.status_code == 401 and "expired" in r.json()["detail"]


def test_identity_comes_from_token(api):
    client, toks = api
    assert client.get("/me", headers=h(toks["alice"])).json() == {
        "subject": "alice",
        "roles": ["author", "viewer"],
        "tenant": "research",
        "auth_method": "token",
    }


def test_identity_fields_in_body_are_rejected(api):
    client, toks = api
    r = client.post("/cohorts/ask", json={"query": T2DM_QUERY, "user_id": "mallory"}, headers=h(toks["alice"]))
    assert r.status_code == 422
    def_id = _draft(client, toks)
    r = client.post(
        f"/cohorts/{def_id}/review", json={"decision": "approved", "reviewer": "mallory"}, headers=h(toks["bob"])
    )
    assert r.status_code == 422


# ---- role-based authorization ----------------------------------------------------------
def test_viewer_cannot_approve(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    r = client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["victor"]))
    assert r.status_code == 403
    assert builder.store.get_definition(def_id)["status"] == "draft"
    denied = [e for e in builder.store.audit_events() if e["action"] == "authz.denied"]
    assert denied and denied[0]["actor"] == "victor"


def test_viewer_cannot_author(api):
    client, toks = api
    assert client.post("/cohorts", json={"ir": EXAMPLE_IR}, headers=h(toks["victor"])).status_code == 403


def test_author_cannot_execute(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["bob"]))
    r = client.post(f"/cohorts/{def_id}/execute", json={}, headers=h(toks["alice"]))
    assert r.status_code == 403
    assert builder.con.execute("SELECT count(*) FROM meta.cohort_generation").fetchone()[0] == 0


def test_reviewer_approval_records_authenticated_reviewer(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    r = client.post(
        f"/cohorts/{def_id}/review", json={"decision": "approved", "comments": "ok"}, headers=h(toks["bob"])
    )
    assert r.status_code == 200 and r.json()["reviewer"] == "bob"
    row = builder.store.get_definition(def_id)
    assert row["status"] == "approved" and row["approved_by"] == "bob" and row["created_by"] == "alice"
    events = [e for e in builder.store.audit_events() if e["action"] == "definition.review"]
    assert events[0]["actor"] == "bob" and events[0]["outcome"] == "success"
    assert events[0]["details"]["content_hash"] == row["content_hash"]


def test_self_approval_is_blocked_by_default(api, builder):
    client, toks = api
    def_id = _draft(client, toks, who="dana")
    r = client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["dana"]))
    assert r.status_code == 403 and "self-approval" in r.json()["detail"]
    assert builder.store.get_definition(def_id)["status"] == "draft"
    assert any(e["details"].get("reason") == "self-approval" for e in builder.store.audit_events())


def test_self_approval_allowed_only_by_explicit_policy(builder, tokens):
    path, toks = tokens
    client = make_client(builder, path, allow_self_approval=True)
    def_id = _draft(client, toks, who="dana")
    r = client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["dana"]))
    assert r.status_code == 200


def test_approved_definition_executes_with_suppressed_counts(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["bob"]))
    r = client.post(f"/cohorts/{def_id}/execute", headers=h(toks["carol"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["min_cell_count"] == 10 and isinstance(body["person_count"], int | str)
    assert "subject_id" not in r.text
    ex = [e for e in builder.store.audit_events() if e["action"] == "definition.execute"]
    assert ex[0]["actor"] == "carol" and ex[0]["outcome"] == "success" and ex[0]["details"]["generation_id"]
    gen = builder.con.execute("SELECT executed_by, tenant FROM meta.cohort_generation").fetchall()
    assert gen == [("carol", "research")]


def test_draft_execution_cannot_be_enabled_by_request(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    r = client.post(f"/cohorts/{def_id}/execute", json={"allow_draft": True}, headers=h(toks["carol"]))
    assert r.status_code == 403 and "disabled" in r.json()["detail"]
    r = client.post(f"/cohorts/{def_id}/execute", json={}, headers=h(toks["carol"]))
    assert r.status_code == 403 and "approve" in r.json()["detail"]
    assert builder.con.execute("SELECT count(*) FROM results.cohort").fetchone()[0] == 0


def test_draft_execution_in_development_when_configured(builder, tokens):
    path, toks = tokens
    client = make_client(builder, path, environment="development", allow_draft_execution=True)
    def_id = _draft(client, toks)
    r = client.post(f"/cohorts/{def_id}/execute", json={"allow_draft": True}, headers=h(toks["carol"]))
    assert r.status_code == 200
    ev = [e for e in builder.store.audit_events() if e["action"] == "definition.execute"][0]
    assert ev["details"]["draft"] is True


def test_admin_only_audit_log(api):
    client, toks = api
    assert client.get("/audit", headers=h(toks["bob"])).status_code == 403
    assert client.get("/audit", headers=h(toks["root"])).status_code == 200


# ---- tenancy and run traces ------------------------------------------------------------
def test_other_tenant_resources_are_not_visible(api):
    client, toks = api
    def_id = _draft(client, toks)
    assert client.get(f"/cohorts/{def_id}", headers=h(toks["eve"])).status_code == 404
    assert client.get(f"/cohorts/{def_id}/sql", headers=h(toks["eve"])).status_code == 404
    assert def_id not in [d["cohort_definition_id"] for d in client.get("/cohorts", headers=h(toks["eve"])).json()]
    assert (
        client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["eve"])).status_code
        == 404
    )
    assert client.post(f"/cohorts/{def_id}/execute", headers=h(toks["eve"])).status_code == 404
    assert (
        client.post(
            "/cohorts", json={"ir": EXAMPLE_IR, "parent_definition_id": def_id}, headers=h(toks["eve"])
        ).status_code
        == 404
    )
    # same tenant and admin can see it
    assert client.get(f"/cohorts/{def_id}", headers=h(toks["victor"])).status_code == 200
    assert client.get(f"/cohorts/{def_id}", headers=h(toks["root"])).status_code == 200


def test_run_traces_visible_to_owner_and_reviewers_only(api):
    client, toks = api
    r = client.post("/cohorts/ask", json={"query": T2DM_QUERY}, headers=h(toks["alice"]))
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=h(toks["alice"])).status_code == 200
    assert client.get(f"/runs/{run_id}", headers=h(toks["bob"])).status_code == 200  # reviewer, same tenant
    assert client.get(f"/runs/{run_id}", headers=h(toks["dana"])).status_code == 200  # reviewer role
    assert client.get(f"/runs/{run_id}", headers=h(toks["victor"])).status_code == 403  # viewer: no traces
    assert client.get(f"/runs/{run_id}", headers=h(toks["eve"])).status_code == 404  # other tenant
    assert client.post(f"/runs/{run_id}/replay", headers=h(toks["alice"])).json()["identical"] is True
    assert client.post(f"/runs/{run_id}/replay", headers=h(toks["eve"])).status_code == 404


# ---- configuration safety -----------------------------------------------------------------
def test_production_refuses_dev_bypass():
    env = {
        "CB_ENV": "production",
        "CB_AUTH_DEV_BYPASS": "true",
        "CB_AUTH_DEV_SUBJECT": "dev",
        "CB_AUTH_DEV_ROLES": "admin",
    }
    with pytest.raises(ConfigError, match="not allowed when CB_ENV=production"):
        SecurityConfig.from_env(env)


def test_default_environment_is_production():
    assert SecurityConfig.from_env({}).is_production
    with pytest.raises(ConfigError):
        SecurityConfig.from_env({"CB_AUTH_DEV_BYPASS": "1", "CB_AUTH_DEV_SUBJECT": "d", "CB_AUTH_DEV_ROLES": "admin"})


def test_production_refuses_draft_execution_and_bad_values():
    with pytest.raises(ConfigError):
        SecurityConfig.from_env({"CB_ENV": "production", "CB_ALLOW_DRAFT_EXECUTION": "true"})
    with pytest.raises(ConfigError):
        SecurityConfig.from_env({"CB_ENV": "staging"})
    with pytest.raises(ConfigError):
        SecurityConfig.from_env({"CB_ALLOW_SELF_APPROVAL": "maybe"})


def test_create_app_refuses_unsafe_config(builder):
    with pytest.raises(ConfigError):
        create_app(
            builder,
            SecurityConfig(environment="production", dev_bypass=True, dev_subject="d", dev_roles=frozenset({"admin"})),
        )


def test_dev_bypass_requires_explicit_identity():
    with pytest.raises(ConfigError, match="CB_AUTH_DEV_SUBJECT"):
        SecurityConfig.from_env({"CB_ENV": "development", "CB_AUTH_DEV_BYPASS": "true"})


def test_dev_bypass_in_development(builder, tokens):
    path, toks = tokens
    client = make_client(
        builder,
        path,
        environment="development",
        dev_bypass=True,
        dev_subject="devuser",
        dev_roles=frozenset({"viewer"}),
    )
    assert client.get("/me").json()["auth_method"] == "dev-bypass"
    assert client.post("/cohorts", json={"ir": EXAMPLE_IR}).status_code == 403  # bypass roles still apply
    assert client.get("/me", headers=h("x" * 40)).status_code == 401  # bad tokens still rejected


def test_token_file_must_hold_hashes_and_known_roles(tmp_path):
    raw = tmp_path / "raw.yaml"
    raw.write_text(yaml.safe_dump({"tokens": [{"sha256": "cbk_plaintext-token", "subject": "a", "roles": ["viewer"]}]}))
    with pytest.raises(ConfigError, match="sha256"):
        TokenAuthenticator.from_file(raw)
    _, entry = issue_token("a", ["viewer"])
    entry["roles"] = ["superuser"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tokens": [entry]}))
    with pytest.raises(ConfigError, match="roles"):
        TokenAuthenticator.from_file(bad)


def test_no_tokens_configured_fails_closed(builder):
    client = TestClient(create_app(builder, SecurityConfig(environment="production")))
    assert client.get("/cohorts", headers=h("y" * 40)).status_code == 401


def test_secrets_are_not_persisted(api, builder):
    client, toks = api
    def_id = _draft(client, toks)
    client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["victor"]))
    client.get("/me", headers=h("z" * 40))
    dump = json.dumps(builder.con.execute("SELECT * FROM meta.audit_event").fetchall(), default=str)
    assert not any(t in dump for t in toks.values())


def test_database_errors_return_safe_message(api, builder, monkeypatch):
    client, toks = api
    def_id = _draft(client, toks)
    client.post(f"/cohorts/{def_id}/review", json={"decision": "approved"}, headers=h(toks["bob"]))

    original = builder.executor._run

    def failing_run(sql, params=None, fetch="all"):
        if sql.lstrip().upper().startswith("INSERT"):  # make the materialization step hit a DB error
            return original("SELECT * FROM cdm.secret_table_xyz", None, fetch)
        return original(sql, params, fetch)

    monkeypatch.setattr(builder.executor, "_run", failing_run)
    r = client.post(f"/cohorts/{def_id}/execute", headers=h(toks["carol"]))
    assert r.status_code == 500
    assert "secret_table_xyz" not in r.text and r.json()["error_id"]
    failed = [e for e in builder.store.audit_events() if e["outcome"] == "failed"]
    assert failed and failed[0]["action"] == "definition.execute"
