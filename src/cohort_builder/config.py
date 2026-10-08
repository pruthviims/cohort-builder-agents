"""Runtime settings, read from environment variables (CB_*)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _temperature(raw: str | None) -> float | None:
    if raw is None:
        return 0.0
    if raw.strip().lower() in ("", "none", "null"):
        return None  # omit the parameter (for models that do not accept it)
    return float(raw)


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "cohort_builder.duckdb")
    ontology_dir: Path = REPO_ROOT / "ontology"
    prompts_dir: Path = REPO_ROOT / "prompts"
    dataset: str | None = None  # dataset profile in ontology/datasets/ (None = ontology default)
    model: str = "claude-sonnet-5-5"
    temperature: float | None = 0.0
    max_tokens: int = 4096
    # live   = always call the API (still records the call)
    # cached = reuse a recorded response for an identical request, else call the API
    # replay = recorded responses only; fail on a cache miss (exact reproduction)
    llm_mode: str = "cached"
    max_retries: int = 2
    max_resolver_turns: int = 8
    # execution limits and DuckDB hardening
    query_timeout_seconds: float = 300.0
    duckdb_memory_limit: str | None = None      # e.g. "4GB"
    duckdb_threads: int | None = None
    lock_external_access: bool = True           # block file/network access from SQL after setup
    prompt_versions: dict[str, str] = field(
        default_factory=lambda: {"intent_parser": "v2", "concept_resolver": "v1", "critic": "v2"}
    )

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        base = cls()
        return cls(
            db_path=Path(env.get("CB_DB_PATH", base.db_path)),
            ontology_dir=Path(env.get("CB_ONTOLOGY_DIR", base.ontology_dir)),
            prompts_dir=Path(env.get("CB_PROMPTS_DIR", base.prompts_dir)),
            dataset=env.get("CB_DATASET") or None,
            model=env.get("CB_MODEL", base.model),
            temperature=_temperature(env.get("CB_TEMPERATURE")),
            max_tokens=int(env.get("CB_MAX_TOKENS", base.max_tokens)),
            llm_mode=env.get("CB_LLM_MODE", base.llm_mode),
            max_retries=int(env.get("CB_MAX_RETRIES", base.max_retries)),
            max_resolver_turns=int(env.get("CB_MAX_RESOLVER_TURNS", base.max_resolver_turns)),
            query_timeout_seconds=float(env.get("CB_QUERY_TIMEOUT_SECONDS", base.query_timeout_seconds)),
            duckdb_memory_limit=env.get("CB_DUCKDB_MEMORY_LIMIT") or None,
            duckdb_threads=int(env["CB_DUCKDB_THREADS"]) if env.get("CB_DUCKDB_THREADS") else None,
            lock_external_access=(env.get("CB_DUCKDB_LOCK_EXTERNAL_ACCESS", "true").strip().lower()
                                  not in ("0", "false", "no", "off")),
        )
