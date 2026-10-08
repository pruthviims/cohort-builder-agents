"""Authentication, roles and governance configuration.

Tokens: opaque random bearer tokens. Only their SHA-256 hashes are stored (in a
JSON/YAML token file referenced by CB_AUTH_TOKENS_FILE), so a leaked config file
does not leak credentials. Issue tokens with `cohort-builder auth issue-token`.
This is a deliberately small, dependency-free mechanism; for enterprise SSO put
an OIDC-aware gateway in front of the API, or replace `TokenAuthenticator` with an
OIDC/JWT verifier implementing the same `authenticate(token) -> Principal` call.

Environments (CB_ENV):
  production  (default) fail closed; development-only switches are refused at startup.
  development allows CB_AUTH_DEV_BYPASS (a fixed, explicitly configured local identity)
              and CB_ALLOW_DRAFT_EXECUTION.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

VIEWER, AUTHOR, REVIEWER, EXECUTOR, ADMIN = "viewer", "author", "reviewer", "executor", "admin"
ROLES = frozenset({VIEWER, AUTHOR, REVIEWER, EXECUTOR, ADMIN})
ROLE_DESCRIPTIONS = {
    VIEWER: "read definitions, explanations, SQL and vocabulary in own tenant",
    AUTHOR: "create and validate definitions (ask, submit), read own run traces",
    REVIEWER: "approve or reject definitions; read run traces in own tenant",
    EXECUTOR: "execute approved definitions; read aggregate (suppressed) results",
    ADMIN: "all of the above across tenants, plus the audit log",
}
DEFAULT_TENANT = "default"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9_.@:+\-]{1,128}$")
MIN_TOKEN_LENGTH = 32


class ConfigError(RuntimeError):
    """Unsafe or invalid security configuration (raised at startup, never at request time)."""


class AuthError(Exception):
    """Missing, malformed, unknown or expired credentials."""


def _bool(env: dict[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be true/false, got {raw!r}")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    tenant: str = DEFAULT_TENANT
    auth_method: str = "token"
    token_id: str | None = None  # label of the token entry, for audit (never the secret)

    def has(self, role: str) -> bool:
        return role in self.roles or ADMIN in self.roles

    def has_any(self, *roles: str) -> bool:
        return any(self.has(r) for r in roles)

    @property
    def is_admin(self) -> bool:
        return ADMIN in self.roles

    def can_access_tenant(self, tenant: str | None) -> bool:
        return self.is_admin or (tenant or DEFAULT_TENANT) == self.tenant


@dataclass(frozen=True)
class TokenEntry:
    token_id: str
    sha256: str
    subject: str
    roles: frozenset[str]
    tenant: str = DEFAULT_TENANT
    expires_at: datetime | None = None


class TokenAuthenticator:
    """Validates bearer tokens against hashed entries from a token file."""

    def __init__(self, entries: list[TokenEntry]):
        self._by_hash = {e.sha256: e for e in entries}

    @classmethod
    def from_file(cls, path: Path) -> "TokenAuthenticator":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"token file {path} does not exist")
        raw = path.read_text()
        doc = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
        items = (doc or {}).get("tokens", [])
        entries, seen = [], set()
        for i, item in enumerate(items):
            where = f"{path.name} entry {i}"
            digest = str(item.get("sha256", "")).lower()
            if not _HEX64.match(digest):
                raise ConfigError(f"{where}: sha256 must be 64 hex characters (store hashes, never raw tokens)")
            if digest in seen:
                raise ConfigError(f"{where}: duplicate token hash")
            seen.add(digest)
            subject, tenant = str(item.get("subject", "")), str(item.get("tenant", DEFAULT_TENANT))
            if not _NAME.match(subject) or not _NAME.match(tenant):
                raise ConfigError(f"{where}: subject/tenant must match {_NAME.pattern}")
            roles = frozenset(item.get("roles") or [])
            if not roles or not roles <= ROLES:
                raise ConfigError(f"{where}: roles must be a non-empty subset of {sorted(ROLES)}")
            expires = item.get("expires_at")
            if expires is not None and not isinstance(expires, datetime):
                expires = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            if expires is not None and expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            entries.append(TokenEntry(str(item.get("id") or f"token-{i}"), digest, subject, roles, tenant, expires))
        return cls(entries)

    def authenticate(self, token: str | None) -> Principal:
        if not token:
            raise AuthError("missing bearer token")
        if len(token) < MIN_TOKEN_LENGTH or len(token) > 512:
            raise AuthError("invalid bearer token")
        digest = hash_token(token)
        entry = self._by_hash.get(digest)
        if entry is None or not hmac.compare_digest(entry.sha256, digest):
            raise AuthError("invalid bearer token")
        if entry.expires_at is not None and entry.expires_at <= datetime.now(timezone.utc):
            raise AuthError("expired bearer token")
        return Principal(entry.subject, entry.roles, entry.tenant, "token", entry.token_id)

    def __len__(self) -> int:
        return len(self._by_hash)


def issue_token(
    subject: str,
    roles: list[str],
    tenant: str = DEFAULT_TENANT,
    expires_days: int | None = 90,
    token_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a new random token. Returns (token, file entry). Show the token once; store only the entry."""
    roles_set = frozenset(roles)
    if not roles_set or not roles_set <= ROLES:
        raise ValueError(f"roles must be a non-empty subset of {sorted(ROLES)}")
    if not _NAME.match(subject) or not _NAME.match(tenant):
        raise ValueError(f"subject/tenant must match {_NAME.pattern}")
    token = "cbk_" + secrets.token_urlsafe(32)
    entry: dict[str, Any] = {
        "id": token_id or f"{subject}-{secrets.token_hex(3)}",
        "sha256": hash_token(token),
        "subject": subject,
        "roles": sorted(roles_set),
        "tenant": tenant,
    }
    if expires_days:
        entry["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=expires_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return token, entry


@dataclass(frozen=True)
class SecurityConfig:
    environment: str = "production"
    tokens_file: Path | None = None
    dev_bypass: bool = False
    dev_subject: str | None = None
    dev_roles: frozenset[str] = field(default_factory=frozenset)
    dev_tenant: str = DEFAULT_TENANT
    allow_self_approval: bool = False
    allow_draft_execution: bool = False

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SecurityConfig":
        env = dict(os.environ if env is None else env)
        environment = (env.get("CB_ENV") or "production").strip().lower()
        if environment not in ("production", "development"):
            raise ConfigError(f"CB_ENV must be 'production' or 'development', got {environment!r}")
        tokens = env.get("CB_AUTH_TOKENS_FILE")
        roles = frozenset(r.strip() for r in (env.get("CB_AUTH_DEV_ROLES") or "").split(",") if r.strip())
        cfg = cls(
            environment=environment,
            tokens_file=Path(tokens) if tokens else None,
            dev_bypass=_bool(env, "CB_AUTH_DEV_BYPASS"),
            dev_subject=env.get("CB_AUTH_DEV_SUBJECT") or None,
            dev_roles=roles,
            dev_tenant=env.get("CB_AUTH_DEV_TENANT") or DEFAULT_TENANT,
            allow_self_approval=_bool(env, "CB_ALLOW_SELF_APPROVAL"),
            allow_draft_execution=_bool(env, "CB_ALLOW_DRAFT_EXECUTION"),
        )
        cfg.check()
        return cfg

    def check(self) -> None:
        if self.is_production and self.dev_bypass:
            raise ConfigError("CB_AUTH_DEV_BYPASS is not allowed when CB_ENV=production")
        if self.is_production and self.allow_draft_execution:
            raise ConfigError("CB_ALLOW_DRAFT_EXECUTION is not allowed when CB_ENV=production")
        if self.dev_bypass:
            if not self.dev_subject or not _NAME.match(self.dev_subject):
                raise ConfigError("CB_AUTH_DEV_BYPASS requires an explicit CB_AUTH_DEV_SUBJECT")
            if not self.dev_roles or not self.dev_roles <= ROLES:
                raise ConfigError(f"CB_AUTH_DEV_BYPASS requires CB_AUTH_DEV_ROLES (subset of {sorted(ROLES)})")

    def dev_principal(self) -> Principal | None:
        if not self.dev_bypass:
            return None
        return Principal(self.dev_subject or "", self.dev_roles, self.dev_tenant, "dev-bypass")

    def authenticator(self) -> TokenAuthenticator:
        return TokenAuthenticator.from_file(self.tokens_file) if self.tokens_file else TokenAuthenticator([])


@dataclass(frozen=True)
class GovernancePolicy:
    """Rules every interface (CLI, API, MCP) shares; enforced inside CohortBuilder."""

    allow_self_approval: bool = False
    allow_draft_execution: bool = False

    @classmethod
    def from_security(cls, cfg: SecurityConfig) -> "GovernancePolicy":
        return cls(cfg.allow_self_approval, cfg.allow_draft_execution)
