"""Deterministic IR -> SQL compiler. No LLM involved: same IR + ontology => same SQL.

The SQL uses only portable constructs (date + integer, date - date,
COUNT(*) FILTER, window functions), so it runs on DuckDB and PostgreSQL.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from . import COMPILER_VERSION
from .ir import CohortDefinition, ConceptSet, Criterion, Demographics, IndexEvent, ValueFilter
from .ontology import Ontology

_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True)
class CompiledCohort:
    cohort_sql: str
    attrition_sql: str
    rule_names: list[str]
    compiler_version: str = COMPILER_VERSION

    @property
    def sql_hash(self) -> str:
        return "sha256:" + hashlib.sha256((self.cohort_sql + "\n--\n" + self.attrition_sql).encode()).hexdigest()


def _lit(cs_id: str) -> str:
    if not _ID.match(cs_id):
        raise ValueError(f"invalid concept set id: {cs_id!r}")
    return f"'{cs_id}'"


def _num(x: float | int) -> str:
    return repr(float(x)) if isinstance(x, float) else str(int(x))


def _id_list(ids: list[int]) -> str:
    return ", ".join(str(int(i)) for i in sorted(set(ids)))


class Compiler:
    def __init__(self, ontology: Ontology):
        self.ont = ontology
        self.vocab = ontology.vocab_schema()

    # ---- pieces -------------------------------------------------------------
    def _concept_set_sql(self, cs: ConceptSet) -> str:
        def part(items: list) -> str | None:
            desc = [i.concept_id for i in items if i.include_descendants]
            exact = [i.concept_id for i in items if not i.include_descendants]
            parts = []
            if desc:
                parts.append(
                    f"SELECT descendant_concept_id AS concept_id FROM {self.vocab}.concept_ancestor "
                    f"WHERE ancestor_concept_id IN ({_id_list(desc)})"
                )
            if exact:
                parts.append(f"SELECT concept_id FROM {self.vocab}.concept WHERE concept_id IN ({_id_list(exact)})")
            return "\n      UNION ".join(parts) if parts else None

        incl = part([i for i in cs.items if not i.is_excluded])
        excl = part([i for i in cs.items if i.is_excluded])
        sql = f"  SELECT DISTINCT {_lit(cs.id)} AS cs_id, x.concept_id FROM (\n      {incl}\n  ) x"
        if excl:
            sql += f"\n  WHERE x.concept_id NOT IN (\n      {excl}\n  )"
        return sql

    def _unit_norm_sql(self) -> str | None:
        rows = []
        for analyte_id, a in sorted(self.ont.analytes.items()):
            for conv in sorted(a.get("conversions", []), key=lambda c: c["from_unit"]):
                rows.append(
                    f"({analyte_id}, {int(conv['from_unit'])}, {int(a['canonical_unit'])}, "
                    f"{float(conv['factor'])!r}, {float(conv['offset'])!r})"
                )
        if not rows:
            return None
        return (
            f"  SELECT ca.descendant_concept_id AS measurement_concept_id, v.from_unit, v.to_unit, v.factor, "
            f"v.offset_\n  FROM (VALUES {', '.join(rows)}) AS v(analyte_id, from_unit, to_unit, factor, offset_)\n"
            f"  JOIN {self.vocab}.concept_ancestor ca ON ca.ancestor_concept_id = v.analyte_id"
        )

    def _value_sql(self, vf: ValueFilter, has_unit_norm: bool) -> tuple[str, str]:
        """Returns (extra join, condition) for a measurement value filter, normalizing units."""
        m = self.ont.table_mapping("Measurement")
        val, unit = f"e.{m['value_col']}", f"e.{m['unit_col']}"
        u = int(vf.unit_concept_id)
        if has_unit_norm:
            join = (
                f"\n      LEFT JOIN unit_norm u ON u.measurement_concept_id = e.{m['concept_col']} "
                f"AND u.from_unit = {unit} AND u.to_unit = {u}"
            )
            expr = f"(CASE WHEN {unit} = {u} THEN {val} ELSE {val} * u.factor + u.offset_ END)"
        else:
            join, expr = "", f"(CASE WHEN {unit} = {u} THEN {val} END)"
        if vf.op == "between":
            assert vf.value_high is not None  # guaranteed by the ValueFilter model
            cond = f"{expr} BETWEEN {_num(vf.value)} AND {_num(vf.value_high)}"
        else:
            cond = f"{expr} {vf.op} {_num(vf.value)}"
        return join, cond

    def _claims_conditions(self, entity: str, claim_status: Sequence[str] | None, dx_position: str | None) -> str:
        """Claims filters (adjudication status, diagnosis position) from the dataset mapping."""
        m = self.ont.table_mapping(entity)
        conds = []
        status = claim_status
        if status is None and "status_col" in m:
            status = self.ont.default_claim_status()  # dataset default, e.g. paid claims only
        if status is not None:
            if "status_col" not in m:
                raise ValueError(f"dataset {self.ont.dataset_name} has no claim status for {entity}")
            allowed = set(self.ont.claim_status_values())
            vals = sorted(set(status))
            if not set(vals) <= allowed:
                raise ValueError(f"invalid claim_status {vals}")
            conds.append(f"e.{m['status_col']} IN ({', '.join(repr(v) for v in vals)})")
        if dx_position == "primary":
            if "position_col" not in m:
                raise ValueError(f"dataset {self.ont.dataset_name} has no diagnosis position for {entity}")
            conds.append(f"e.{m['position_col']} = 1")
        return "".join(f" AND {c}" for c in conds)

    def event_parts(self, c: Criterion, has_unit_norm: bool) -> dict[str, str]:
        """SQL fragments selecting a criterion's qualifying events for base row `b` (shared with proxies)."""
        m = self.ont.table_mapping(c.entity)
        start = f"e.{m['start_col']}"
        lo = f"b.index_date + ({c.window.start_days})" if c.window.start_days is not None else "b.op_start"
        hi = f"b.index_date + ({c.window.end_days})" if c.window.end_days is not None else "b.op_end"
        join, cond = ("", "")
        if c.value_filter:
            join, cond = self._value_sql(c.value_filter, has_unit_norm)
            cond = f"\n        AND {cond}"
        cond += self._claims_conditions(c.entity, c.claim_status, c.dx_position)
        return {
            "start": start,
            "table": m["table"],
            "concept_join": f"JOIN cs_expanded c ON c.cs_id = {_lit(c.concept_set_id)} "
            f"AND c.concept_id = e.{m['concept_col']}",
            "extra_join": join,
            "window": f"{start} >= {lo} AND {start} <= {hi}",
            "filters": cond,
        }

    def _criterion_sql(self, c: Criterion, has_unit_norm: bool) -> str:
        """Boolean SQL: does the person meet the criterion's occurrence rule?"""
        p = self.event_parts(c, has_unit_norm)
        start = p["start"]
        frm = (
            f"FROM {p['table']} e\n"
            f"      {p['concept_join']}"
            f"{p['extra_join']}\n"
            f"      WHERE e.person_id = b.person_id AND {p['window']}{p['filters']}"
        )
        counted = f"COUNT(DISTINCT {start})" if c.count_by == "dates" else "COUNT(*)"
        if c.min_span_days is not None:
            if c.occurrence != "at_least":
                raise ValueError("min_span_days requires occurrence at_least")
            # at least N qualifying events whose first and last dates are >= min_span_days apart
            return (
                f"((SELECT CASE WHEN {counted} >= {int(c.count)} AND MAX({start}) - MIN({start}) >= "
                f"{int(c.min_span_days)} THEN 1 ELSE 0 END {frm}) = 1)"
            )
        op = {"at_least": ">=", "at_most": "<=", "exactly": "="}[c.occurrence]
        return f"((SELECT {counted} {frm}) {op} {int(c.count)})"

    def head_ctes(
        self, concept_sets: Sequence[ConceptSet], ie: IndexEvent, needs_value: bool
    ) -> tuple[list[str], bool]:
        """cs_expanded, unit_norm, index_candidates, index_events and base CTEs (shared with proxies)."""
        unit_norm = self._unit_norm_sql() if needs_value else None
        ctes = [
            "cs_expanded AS (\n"
            + "\n  UNION ALL\n".join(self._concept_set_sql(cs) for cs in sorted(concept_sets, key=lambda s: s.id))
            + "\n)"
        ]
        if unit_norm:
            ctes.append(f"unit_norm AS (\n{unit_norm}\n)")

        m = self.ont.table_mapping(ie.entity)
        join, conds = "", []
        if ie.value_filter:
            join, vcond = self._value_sql(ie.value_filter, unit_norm is not None)
            conds.append(vcond)
        claims = self._claims_conditions(ie.entity, ie.claim_status, ie.dx_position)
        conds += [c for c in claims.split(" AND ") if c]
        cond = ("\n  WHERE " + " AND ".join(conds)) if conds else ""
        ctes.append(
            f"index_candidates AS (\n"
            f"  SELECT e.person_id, e.{m['start_col']} AS index_date,\n"
            f"         ROW_NUMBER() OVER (PARTITION BY e.person_id "
            f"ORDER BY e.{m['start_col']}, e.{m['concept_col']}) AS rn\n"
            f"  FROM {m['table']} e\n"
            f"  JOIN cs_expanded c ON c.cs_id = {_lit(ie.concept_set_id)} AND c.concept_id = e.{m['concept_col']}"
            f"{join}{cond}\n)"
        )
        where = "WHERE rn = 1" if ie.first_occurrence_only else ""
        ctes.append(f"index_events AS (\n  SELECT DISTINCT person_id, index_date FROM index_candidates {where}\n)")

        op, pm = self.ont.mappings["observation_period"], self.ont.mappings["person"]
        ctes.append(
            f"base AS (\n"
            f"  SELECT ie.person_id, ie.index_date, op.{op['start_col']} AS op_start, op.{op['end_col']} AS op_end,\n"
            f"         p.{pm['gender_col']} AS gender_concept_id,\n"
            f"         EXTRACT(YEAR FROM ie.index_date) - p.{pm['year_of_birth_col']} AS age_at_index\n"
            f"  FROM index_events ie\n"
            f"  JOIN {op['table']} op ON op.person_id = ie.person_id\n"
            f"   AND ie.index_date BETWEEN op.{op['start_col']} AND op.{op['end_col']}\n"
            f"  JOIN {pm['table']} p ON p.{pm['person_key']} = ie.person_id\n)"
        )
        return ctes, unit_norm is not None

    @staticmethod
    def population_rules(prior: int, post: int, d: Demographics) -> list[tuple[str, str]]:
        """Observation and demographic rules over base row `b`: (label, boolean SQL meaning 'passes')."""
        rules: list[tuple[str, str]] = []
        if prior:
            rules.append(
                (f"At least {prior} days of observation before index", f"(b.index_date - b.op_start) >= {int(prior)}")
            )
        if post:
            rules.append(
                (f"At least {post} days of observation after index", f"(b.op_end - b.index_date) >= {int(post)}")
            )
        if d.age_min is not None or d.age_max is not None:
            lo = d.age_min if d.age_min is not None else 0
            hi = d.age_max if d.age_max is not None else 200
            label = (
                f"Age at index >= {lo}"
                if d.age_max is None
                else f"Age at index <= {hi}"
                if d.age_min is None
                else f"Age at index {lo}-{hi}"
            )
            rules.append((label, f"b.age_at_index BETWEEN {int(lo)} AND {int(hi)}"))
        if d.gender_concept_ids:
            rules.append(("Gender", f"b.gender_concept_id IN ({_id_list(d.gender_concept_ids)})"))
        return rules

    # ---- main ---------------------------------------------------------------
    def compile(self, ir: CohortDefinition) -> CompiledCohort:
        needs_value = bool(any(x.value_filter for x in [*ir.inclusion, *ir.exclusion]) or ir.index_event.value_filter)
        ctes, has_unit_norm = self.head_ctes(ir.concept_sets, ir.index_event, needs_value)
        unit_norm = "unit_norm" if has_unit_norm else None
        rules = self.population_rules(ir.prior_observation_days, ir.post_observation_days, ir.demographics)
        for c in ir.inclusion:
            rules.append((f"Inclusion: {c.name}", self._criterion_sql(c, unit_norm is not None)))
        for c in ir.exclusion:
            rules.append((f"Exclusion: {c.name}", f"NOT {self._criterion_sql(c, unit_norm is not None)}"))

        flag_cols = ",\n".join(f"    {expr} AS r{i + 1}" for i, (_, expr) in enumerate(rules))
        flags = (
            "flags AS (\n  SELECT b.person_id, b.index_date, b.op_start, b.op_end"
            + (f",\n{flag_cols}" if rules else "")
            + "\n  FROM base b\n)"
        )
        ctes.append(flags)
        prefix = "WITH " + ",\n".join(ctes) + "\n"

        all_pass = " AND ".join(f"r{i + 1}" for i in range(len(rules))) or "TRUE"
        if ir.exit.type == "fixed_days":
            end_expr = f"LEAST(index_date + ({int(ir.exit.days or 0)}), op_end)"
        else:
            end_expr = "op_end"
        cohort_sql = (
            prefix + ", ranked AS (\n"
            f"  SELECT person_id, index_date, op_end,\n"
            # deterministic: earliest qualifying index; if observation periods overlap, the one
            # starting earliest (longest history), then the one ending latest
            f"         ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY index_date, op_start, "
            f"op_end DESC) AS rn\n"
            f"  FROM flags WHERE {all_pass}\n)\n"
            f"SELECT person_id AS subject_id, index_date AS cohort_start_date, {end_expr} AS cohort_end_date\n"
            f"FROM ranked WHERE rn = 1\nORDER BY subject_id"
        )

        # rule 0 counts people with a qualifying index event *before* the observation-period join, so people
        # lost for missing/non-covering observation periods (or missing person records) are visible
        cols = [
            "  (SELECT COUNT(DISTINCT person_id) FROM index_events) AS rule_0",
            "  COUNT(DISTINCT person_id) AS rule_1",
        ]
        for i in range(len(rules)):
            cond = " AND ".join(f"r{j + 1}" for j in range(i + 1))
            cols.append(f"  COUNT(DISTINCT person_id) FILTER (WHERE {cond}) AS rule_{i + 2}")
        attrition_sql = prefix + "SELECT\n" + ",\n".join(cols) + "\nFROM flags"

        names = ["Persons with a qualifying index event", "Index event within an observation period"] + [
            r[0] for r in rules
        ]
        return CompiledCohort(cohort_sql=cohort_sql, attrition_sql=attrition_sql, rule_names=names)
