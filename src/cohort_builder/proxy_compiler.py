"""Deterministic ProxyDefinition -> SQL compiler (no LLM). Reuses the cohort compiler's front end.

SQL layout (every block is a named CTE so a reviewer can read it top to bottom):

  cs_expanded, unit_norm, index_candidates, index_events, base   -- shared with cohort definitions
  pbase       one row per (person, index, observation period) with a stable row id `bid`
  ev_<id>     qualifying events of each evidence item for each base row (bid, d)
  facts       per base row: one boolean per evidence (ev_<id>) and temporal rule (tr_<id>)
  scored      + evidence_score (deterministic rule score)
  classified  + population rules, entry/exclusion/conflict flags, funnel steps, tier
  ranked/members   the chosen row per qualifying person (earliest index; deterministic ties)

Joins never multiply patients: evidence is aggregated per `bid` in subqueries, and each person
contributes exactly one member row. Only validated identifiers and numbers are interpolated;
free text (names, notes, descriptions) never reaches SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compiler import CompiledCohort, Compiler
from .ontology import Ontology
from .proxy import EvidenceCriterion, Expr, ProxyDefinition, provenance_summary


def evidence_support(ev: EvidenceCriterion, ont: Ontology) -> list[str]:
    """Reasons the active dataset cannot provide this evidence (empty list = supported)."""
    reasons: list[str] = []
    ds = ont.dataset_name
    caps = ont.capabilities
    if not ont.supports_entity(ev.entity):
        reasons.append(f"{ev.entity} data is not available in dataset {ds!r}")
        return reasons
    req = (ont.domain.get("evidence_categories") or {}).get(ev.category, {})
    if req.get("requires_entity") and not ont.supports_entity(req["requires_entity"]):
        reasons.append(f"{ev.category} evidence needs {req['requires_entity']} data, unavailable in {ds!r}")
    if req.get("requires_attribute") and not ont.supports_attribute(req["requires_attribute"]):
        reasons.append(f"{ev.category} evidence needs {req['requires_attribute']}, unavailable in {ds!r}")
    if req.get("requires_data") and req["requires_data"] not in caps.get("extra_data", []):
        reasons.append(
            f"{ev.category} evidence needs {req['requires_data']} data, which dataset {ds!r} does not provide"
        )
    m = ont.mappings["entities"].get(ev.entity, {})
    if ev.place_of_service and not (ont.supports_attribute("place_of_service") and "pos_col" in m):
        reasons.append(f"place of service is not available for {ev.entity} in {ds!r}")
    if ev.provider_specialty and not (ont.supports_attribute("provider_specialty") and "specialty_col" in m):
        reasons.append(f"provider specialty is not available for {ev.entity} in {ds!r}")
    if ev.claim_status is not None and not (ont.supports_attribute("claim_status") and "status_col" in m):
        reasons.append(f"claim status is not available for {ev.entity} in {ds!r}")
    if ev.dx_position not in (None, "any") and not (ont.supports_attribute("dx_position") and "position_col" in m):
        reasons.append(f"diagnosis position is not available for {ev.entity} in {ds!r}")
    return reasons


@dataclass(frozen=True)
class CompiledProxy(CompiledCohort):
    assignment_sql: str = ""
    evidence_sql: str = ""
    summary_sql: str = ""
    summary_columns: tuple[str, ...] = field(default_factory=tuple)
    unavailable_evidence: tuple[str, ...] = field(default_factory=tuple)
    # provenance and roles carried with the SQL for review/reporting; not part of sql_hash
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def sql_hash(self) -> str:
        import hashlib

        blob = "\n--\n".join(
            [self.cohort_sql, self.attrition_sql, self.assignment_sql, self.evidence_sql, self.summary_sql]
        )
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def _case(cond: str) -> str:
    return f"(CASE WHEN {cond} THEN 1 ELSE 0 END)"


class ProxyCompiler(Compiler):
    def compile_proxy(self, p: ProxyDefinition) -> CompiledProxy:
        self._p = p
        unsupported = {e.id for e in p.evidence if evidence_support(e, self.ont)}
        if required := sorted(e.id for e in p.evidence if e.id in unsupported and e.required):
            raise ValueError(f"required evidence not available in dataset {self.ont.dataset_name!r}: {required}")
        self._unsupported = unsupported
        needs_value = bool(p.index_event.value_filter or any(e.value_filter for e in p.evidence))
        ctes, has_un = self.head_ctes(p.concept_sets, p.index_event, needs_value)
        ctes.append(
            "pbase AS (\n  SELECT ROW_NUMBER() OVER (ORDER BY person_id, index_date, op_start, op_end) AS bid,"
            "\n         base.*\n  FROM base\n)"
        )
        for ev in p.evidence:
            if ev.id not in unsupported:
                ctes.append(self._event_cte(ev, has_un))

        fact_cols = [f"    {self._evidence_bool(ev)} AS ev_{ev.id}" for ev in p.evidence]
        fact_cols += [f"    {self._temporal_bool(t)} AS tr_{t.id}" for t in p.temporal_rules]
        ctes.append(
            "facts AS (\n  SELECT b.bid, b.person_id, b.index_date, b.op_start, b.op_end, "
            "b.gender_concept_id, b.age_at_index,\n" + ",\n".join(fact_cols) + "\n  FROM pbase b\n)"
        )
        score = "0"
        if p.scoring:
            score = " + ".join(
                f"(CASE WHEN {self.expr(w.ref)} THEN {int(w.points)} ELSE 0 END)" for w in p.scoring.weights
            )
        ctes.append(f"scored AS (\n  SELECT b.*, ({score}) AS evidence_score\n  FROM facts b\n)")

        rules = self.population_rules(p.prior_observation_days, p.post_observation_days, p.demographics)
        if p.entry:
            rules.append(("Entry rule", self.expr(p.entry)))
        rules += [(f"Temporal: {t.name}", f"b.tr_{t.id}") for t in p.temporal_rules if t.required]
        if p.exclusion:
            rules.append(("Not excluded by the exclusion rule", f"(NOT {self.expr(p.exclusion)})"))
        rules += [
            (f"Conflict exclusion: {c.label or c.name}", f"(NOT b.cf_{c.name})")
            for c in p.conflicts
            if c.action == "exclude"
        ]
        tier_case = (
            "CASE\n" + "\n".join(f"      WHEN {self._tier_cond(t)} THEN '{t.name}'" for t in p.tiers) + "\n    END"
        )
        rules.append(("Assigned an evidence tier", "b.tier IS NOT NULL"))

        cls_cols = [f"    {self.expr(c.rule)} AS cf_{c.name}" for c in p.conflicts]
        cls_cols += [f"    {self.expr(f.rule)} AS fn_{i + 1}" for i, f in enumerate(p.funnel)]
        cls_cols.append(f"    {tier_case} AS tier")
        ctes.append("tiered AS (\n  SELECT b.*,\n" + ",\n".join(cls_cols) + "\n  FROM scored b\n)")
        rule_cols = ",\n".join(f"    {expr} AS r{i + 1}" for i, (_, expr) in enumerate(rules))
        ctes.append(f"classified AS (\n  SELECT b.*,\n{rule_cols}\n  FROM tiered b\n)")
        prefix = "WITH " + ",\n".join(ctes) + "\n"

        n = len(rules)
        all_pass = " AND ".join(f"r{i + 1}" for i in range(n))
        cand_pass = " AND ".join(f"r{i + 1}" for i in range(n - 1)) or "TRUE"
        order = "ORDER BY index_date, op_start, op_end DESC"
        members = (
            f", ranked AS (\n  SELECT b.*, ROW_NUMBER() OVER (PARTITION BY person_id {order}) AS rn\n"
            f"  FROM classified b WHERE {all_pass}\n),\nmembers AS (SELECT * FROM ranked WHERE rn = 1)\n"
        )
        end_expr = f"LEAST(index_date + ({int(p.exit.days or 0)}), op_end)" if p.exit.type == "fixed_days" else "op_end"
        cohort_sql = (
            prefix + members + f"SELECT person_id AS subject_id, index_date AS cohort_start_date, "
            f"{end_expr} AS cohort_end_date\nFROM members\nORDER BY subject_id"
        )
        assignment_sql = (
            prefix + members + "SELECT person_id AS subject_id, tier, evidence_score\nFROM members\nORDER BY subject_id"
        )
        keys = (
            [(f"ev_{e.id}", "evidence") for e in p.evidence]
            + [(f"tr_{t.id}", "temporal") for t in p.temporal_rules]
            + [(f"cf_{c.name}", "conflict") for c in p.conflicts]
        )
        evidence_sql = (
            prefix
            + members
            + "\nUNION ALL\n".join(
                f"SELECT person_id AS subject_id, '{k}' AS evidence_key, '{kind}' AS kind, {k} AS present FROM members"
                for k, kind in keys
            )
            + "\nORDER BY subject_id, evidence_key"
        )

        attr_cols = [
            "  (SELECT COUNT(DISTINCT person_id) FROM index_events) AS rule_0",
            "  COUNT(DISTINCT person_id) AS rule_1",
        ]
        for i in range(n):
            cond = " AND ".join(f"r{j + 1}" for j in range(i + 1))
            attr_cols.append(f"  COUNT(DISTINCT person_id) FILTER (WHERE {cond}) AS rule_{i + 2}")
        attrition_sql = prefix + "SELECT\n" + ",\n".join(attr_cols) + "\nFROM classified"

        summary: list[tuple[str, str]] = [("candidates", "COUNT(*)")]
        summary += [(f"ev_{e.id}", f"COUNT(*) FILTER (WHERE ev_{e.id})") for e in p.evidence]
        summary += [(f"tr_{t.id}", f"COUNT(*) FILTER (WHERE tr_{t.id})") for t in p.temporal_rules]
        summary += [(f"cf_{c.name}", f"COUNT(*) FILTER (WHERE cf_{c.name})") for c in p.conflicts]
        summary += [(f"tier_{t.name}", f"COUNT(*) FILTER (WHERE tier = '{t.name}')") for t in p.tiers]
        summary.append(("tier_none", "COUNT(*) FILTER (WHERE tier IS NULL)"))
        for i in range(len(p.funnel)):
            cond = " AND ".join(f"fn_{j + 1}" for j in range(i + 1))
            summary.append((f"funnel_{i + 1}", f"COUNT(*) FILTER (WHERE {cond})"))
        # one candidate row per person; a row that received a tier wins, so tier counts equal the members
        cand_order = "ORDER BY (CASE WHEN tier IS NULL THEN 1 ELSE 0 END), index_date, op_start, op_end DESC"
        summary_sql = (
            prefix + f", cand AS (\n  SELECT b.*, ROW_NUMBER() OVER (PARTITION BY person_id {cand_order}) "
            f"AS rn\n  FROM classified b WHERE {cand_pass}\n)\nSELECT\n"
            + ",\n".join(f"  {expr} AS {name}" for name, expr in summary)
            + "\nFROM cand WHERE rn = 1"
        )
        names = ["Persons with a qualifying index event", "Index event within an observation period"] + [
            r[0] for r in rules
        ]
        return CompiledProxy(
            cohort_sql=cohort_sql,
            attrition_sql=attrition_sql,
            rule_names=names,
            assignment_sql=assignment_sql,
            evidence_sql=evidence_sql,
            summary_sql=summary_sql,
            summary_columns=tuple(name for name, _ in summary),
            unavailable_evidence=tuple(sorted(unsupported)),
            metadata={"semantic_hash": p.semantic_hash(), "provenance": provenance_summary(p)},
        )

    # ---- evidence events -----------------------------------------------------------------------
    def _event_cte(self, ev: EvidenceCriterion, has_un: bool) -> str:
        p = self.event_parts(ev, has_un)
        m = self.ont.table_mapping(ev.entity)
        attrs = ""
        for values, col in ((ev.place_of_service, "pos_col"), (ev.provider_specialty, "specialty_col")):
            if values:
                attrs += f" AND e.{m[col]} IN ({', '.join(repr(v) for v in values)})"
        return (
            f"ev_{ev.id} AS (\n  SELECT b.bid, {p['start']} AS d\n  FROM pbase b\n"
            f"  JOIN {p['table']} e ON e.person_id = b.person_id\n  {p['concept_join']}{p['extra_join']}\n"
            f"  WHERE {p['window']}{p['filters']}{attrs}\n)"
        )

    def _evidence_bool(self, ev: EvidenceCriterion) -> str:
        if ev.id in self._unsupported:
            return "FALSE"  # optional evidence the dataset cannot provide: treated as absent (validator warns)
        src = f"ev_{ev.id}"
        counted = "COUNT(DISTINCT {a}.d)" if ev.count_by == "dates" else "COUNT(*)"
        if ev.max_span_days is not None:
            inner = counted.format(a="x2")
            return (
                f"EXISTS (SELECT 1 FROM {src} x1 WHERE x1.bid = b.bid AND (SELECT {inner} FROM {src} x2 "
                f"WHERE x2.bid = x1.bid AND x2.d >= x1.d AND x2.d <= x1.d + ({int(ev.max_span_days)})) "
                f">= {int(ev.count)})"
            )
        c = counted.format(a="x")
        if ev.min_span_days is not None:
            return (
                f"((SELECT CASE WHEN {c} >= {int(ev.count)} AND MAX(x.d) - MIN(x.d) >= {int(ev.min_span_days)} "
                f"THEN 1 ELSE 0 END FROM {src} x WHERE x.bid = b.bid) = 1)"
            )
        op = {"at_least": ">=", "at_most": "<=", "exactly": "="}[ev.occurrence]
        return f"((SELECT {c} FROM {src} x WHERE x.bid = b.bid) {op} {int(ev.count)})"

    def _temporal_bool(self, t) -> str:
        if t.a in self._unsupported or t.b in self._unsupported:
            return "FALSE"
        lo, hi = t.bounds()
        conds = ["x.bid = b.bid"]
        if lo is not None:
            conds.append(f"y.d - x.d >= {int(lo)}")
        if hi is not None:
            conds.append(f"y.d - x.d <= {int(hi)}")
        return f"EXISTS (SELECT 1 FROM ev_{t.a} x JOIN ev_{t.b} y ON y.bid = x.bid WHERE {' AND '.join(conds)})"

    # ---- expressions ---------------------------------------------------------------------------
    def expr(self, e: Expr, _stack: tuple[str, ...] = ()) -> str:
        k = e.kind
        if k == "evidence":
            return f"b.ev_{e.evidence}"
        if k == "temporal":
            return f"b.tr_{e.temporal}"
        if k == "group":
            assert e.group is not None and e.group not in _stack  # cycles are rejected by the model
            return self.expr(self._p.groups[e.group], (*_stack, e.group))
        if k in ("all", "any"):
            joiner = " AND " if k == "all" else " OR "
            return "(" + joiner.join(self.expr(c, _stack) for c in getattr(e, k)) + ")"
        if k == "not_":
            assert e.not_ is not None
            return f"(NOT {self.expr(e.not_, _stack)})"
        nof = getattr(e, k)
        op = {"at_least": ">=", "at_most": "<=", "exactly": "="}[k]
        if nof.within_days is not None and nof.n > 0:
            return self._within(nof.n, [c.evidence for c in nof.of], nof.within_days)
        total = " + ".join(_case(self.expr(c, _stack)) for c in nof.of)
        return f"(({total}) {op} {int(nof.n)})"

    def _within(self, n: int, ids: list[str], days: int) -> str:
        """At least n of the evidence hold AND have a qualifying event inside one [d, d + days] window."""
        live = [i for i in ids if i not in self._unsupported]
        if len(live) < n:
            return "FALSE"
        anchors = " UNION ".join(f"SELECT x.d FROM ev_{i} x WHERE x.bid = b.bid" for i in live)
        hits = " + ".join(
            _case(
                f"b.ev_{i} AND EXISTS (SELECT 1 FROM ev_{i} z WHERE z.bid = b.bid AND z.d >= w.d "
                f"AND z.d <= w.d + ({int(days)}))"
            )
            for i in live
        )
        return f"EXISTS (SELECT 1 FROM ({anchors}) w WHERE ({hits}) >= {int(n)})"

    def _tier_cond(self, t) -> str:
        conds = []
        if t.rule is not None:
            conds.append(self.expr(t.rule))
        if t.min_score is not None:
            conds.append(f"b.evidence_score >= {int(t.min_score)}")
        return " AND ".join(conds)
