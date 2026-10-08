"""Runs compiled cohorts. Plain code, read-only on the CDM, writes only to the results schema.

Privacy: counts leave this module only through `suppress_series` / `suppress_count`.
Small-cell suppression here is a disclosure-risk control, not a guarantee of
anonymity (see README "Small-cell suppression").
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import dataclass, field

import duckdb

from .compiler import CompiledCohort

log = logging.getLogger(__name__)

PRIMARY = "<{k}"        # count below the threshold
COMPLEMENTARY = "suppressed"  # count hidden because it would reveal a small difference


class ExecutionError(RuntimeError):
    """A cohort query failed. The message is safe to show; details stay in the server log."""

    def __init__(self, message: str, error_id: str | None = None):
        super().__init__(message)
        self.error_id = error_id


class QueryTimeout(ExecutionError):
    pass


def suppress_count(n: int, min_cell: int) -> int | str:
    """Primary suppression of one count. Zero is disclosed (no individual is described)."""
    return PRIMARY.format(k=min_cell) if 0 < n < min_cell else n


def suppress_series(counts: list[int], min_cell: int) -> list[int | str]:
    """Suppress a monotone attrition series so no small count can be derived by subtraction.

    Primary: any count in 1..min_cell-1 is shown as "<k".
    Complementary: a count is hidden ("suppressed") if the difference to the last *disclosed*
    count is in 1..min_cell-1, because subtracting the two would reveal a small group.
    """
    out: list[int | str] = []
    last: int | None = None
    for n in counts:
        if 0 < n < min_cell:
            out.append(PRIMARY.format(k=min_cell))
        elif last is not None and 0 < last - n < min_cell:
            out.append(COMPLEMENTARY)
        else:
            out.append(n)
            last = n
    return out


@dataclass
class Attrition:
    rules: list[dict] = field(default_factory=list)  # {sequence, name, remaining}

    @property
    def final_count(self) -> int:
        return self.rules[-1]["remaining"] if self.rules else 0

    def suppressed(self, min_cell: int) -> list[dict]:
        shown = suppress_series([r["remaining"] for r in self.rules], min_cell)
        return [{**r, "remaining": v} for r, v in zip(self.rules, shown)]

    def suppressed_final(self, min_cell: int) -> int | str:
        """The final count, consistent with the suppressed attrition table."""
        shown = self.suppressed(min_cell)
        return shown[-1]["remaining"] if shown else 0


_FORBIDDEN = re.compile(r"\b(ATTACH|DETACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|PRAGMA|SET|CALL|DROP|DELETE|UPDATE|"
                        r"INSERT|ALTER|CREATE|GRANT|REVOKE|TRUNCATE)\b", re.IGNORECASE)


def guard_select(sql: str) -> str:
    """Defense in depth: compiled cohort SQL must be one read-only query."""
    body = sql.strip()
    if ";" in body:
        raise ExecutionError("refusing to run SQL containing ';'")
    if not re.match(r"^(WITH|SELECT)\b", body, re.IGNORECASE):
        raise ExecutionError("refusing to run SQL that is not a SELECT query")
    if _FORBIDDEN.search(re.sub(r"'[^']*'", "''", body)):  # ignore quoted literals (e.g. concept set ids)
        raise ExecutionError("refusing to run SQL containing a write or administrative keyword")
    return body


class Executor:
    def __init__(self, con: duckdb.DuckDBPyConnection, min_cell_count: int = 10, timeout_seconds: float = 300.0):
        if min_cell_count < 1:
            raise ValueError("min_cell_count must be >= 1")
        self.con = con
        self.min_cell = min_cell_count
        self.timeout = timeout_seconds

    def _run(self, sql: str, params: list | None = None, fetch: str = "all"):
        """Execute with a timeout (DuckDB interrupt) and map database errors to safe messages."""
        timer = threading.Timer(self.timeout, self.con.interrupt) if self.timeout and self.timeout > 0 else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            cur = self.con.execute(sql, params or [])
            return cur.fetchone() if fetch == "one" else cur.fetchall()
        except duckdb.InterruptException as exc:
            raise QueryTimeout(f"cohort query exceeded the {self.timeout:g}s time limit and was cancelled") from exc
        except duckdb.Error as exc:
            error_id = str(uuid.uuid4())[:8]
            # full detail (may include SQL fragments) goes to the server log only
            log.error("cohort query failed [%s]: %s: %s", error_id, type(exc).__name__, exc)
            raise ExecutionError(f"cohort query failed (error id {error_id})", error_id) from exc
        finally:
            if timer:
                timer.cancel()

    def attrition(self, compiled: CompiledCohort) -> Attrition:
        row = self._run(guard_select(compiled.attrition_sql), fetch="one")
        return Attrition([{"sequence": i, "name": name, "remaining": int(row[i])}
                          for i, name in enumerate(compiled.rule_names)])

    def generate(self, compiled: CompiledCohort, cohort_definition_id: int) -> tuple[str, Attrition]:
        """Materialize the cohort into results.cohort under a new generation_id (all-or-nothing)."""
        generation_id = str(uuid.uuid4())
        cohort_sql = guard_select(compiled.cohort_sql)
        attrition = self.attrition(compiled)
        self.con.execute("BEGIN")
        try:
            self._run(f"INSERT INTO results.cohort SELECT ?, subject_id, cohort_start_date, cohort_end_date, ? "
                      f"FROM ({cohort_sql})", [cohort_definition_id, generation_id])
            for r in attrition.rules:
                self.con.execute("INSERT INTO results.cohort_inclusion_stats VALUES (?,?,?,?,?)",
                                 [generation_id, cohort_definition_id, r["sequence"], r["name"], r["remaining"]])
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return generation_id, attrition

    def person_ids(self, compiled: CompiledCohort) -> set[int]:
        """Patient-level ids: for internal evaluation/tests only, never returned by an interface."""
        return {r[0] for r in self._run(f"SELECT subject_id FROM ({guard_select(compiled.cohort_sql)})")}
