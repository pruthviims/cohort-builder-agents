"""Runs compiled cohorts. Plain code, read-only on the CDM, writes to the results schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import duckdb

from .compiler import CompiledCohort


def suppress(n: int, min_cell: int) -> int | str:
    """Small-cell suppression for anything shown to agents or users."""
    return f"<{min_cell}" if 0 < n < min_cell else n


@dataclass
class Attrition:
    rules: list[dict] = field(default_factory=list)  # {sequence, name, remaining}

    @property
    def final_count(self) -> int:
        return self.rules[-1]["remaining"] if self.rules else 0

    def suppressed(self, min_cell: int) -> list[dict]:
        return [{**r, "remaining": suppress(r["remaining"], min_cell)} for r in self.rules]


class Executor:
    def __init__(self, con: duckdb.DuckDBPyConnection, min_cell_count: int = 10):
        self.con = con
        self.min_cell = min_cell_count

    def attrition(self, compiled: CompiledCohort) -> Attrition:
        row = self.con.execute(compiled.attrition_sql).fetchone()
        return Attrition([{"sequence": i, "name": name, "remaining": int(row[i])}
                          for i, name in enumerate(compiled.rule_names)])

    def generate(self, compiled: CompiledCohort, cohort_definition_id: int) -> tuple[str, Attrition]:
        """Materialize the cohort into results.cohort under a new generation_id."""
        generation_id = str(uuid.uuid4())
        attrition = self.attrition(compiled)
        self.con.execute("BEGIN")
        try:
            self.con.execute(
                f"INSERT INTO results.cohort SELECT ?, subject_id, cohort_start_date, cohort_end_date, ? "
                f"FROM ({compiled.cohort_sql})", [cohort_definition_id, generation_id])
            for r in attrition.rules:
                self.con.execute("INSERT INTO results.cohort_inclusion_stats VALUES (?,?,?,?,?)",
                                 [generation_id, cohort_definition_id, r["sequence"], r["name"], r["remaining"]])
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return generation_id, attrition

    def person_ids(self, compiled: CompiledCohort) -> set[int]:
        return {r[0] for r in self.con.execute(f"SELECT subject_id FROM ({compiled.cohort_sql})").fetchall()}
