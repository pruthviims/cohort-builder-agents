"""Explainer (deterministic): renders the IR as plain language for reviewers and the critic.

Generated from the IR itself, so the explanation can never disagree with
what will actually run.
"""

from __future__ import annotations

from ..ir import CohortDefinition, Criterion, ValueFilter, Window
from ..ontology import Ontology
from ..vocab import Vocabulary

ENTITY_NOUN = {
    "ConditionOccurrence": "diagnosis of",
    "DrugExposure": "exposure to",
    "Measurement": "measurement of",
    "ProcedureOccurrence": "procedure:",
    "VisitOccurrence": "visit:",
}


def _window(w: Window) -> str:
    s, e = w.start_days, w.end_days
    if s is None and e is None:
        return "at any time during observation"
    if s is None:
        assert e is not None
        return (
            "any time before index"
            if e == 0
            else (f"any time up to {abs(e)} days before index" if e < 0 else f"any time up to {e} days after index")
        )
    if e is None:
        return "any time after index" if s == 0 else f"from {s:+d} days relative to index onward"
    if s == e == 0:
        return "on the index date"
    if e == 0 and s < 0:
        return f"in the {abs(s)} days before index (inclusive)"
    if s == 0 and e > 0:
        return f"in the {e} days after index (inclusive)"
    return f"between day {s:+d} and day {e:+d} relative to index"


class Explainer:
    def __init__(self, ont: Ontology, vocab: Vocabulary):
        self.ont, self.vocab = ont, vocab

    def _value(self, vf: ValueFilter | None) -> str:
        if vf is None:
            return ""
        unit = self.ont.unit_symbol(vf.unit_concept_id)
        txt = (
            f" with value between {vf.value:g} and {vf.value_high:g} {unit}"
            if vf.op == "between"
            else f" with value {vf.op} {vf.value:g} {unit}"
        )
        if vf.original_text:
            txt += f" (user specified {vf.original_text})"
        return txt

    def _concepts(self, ir: CohortDefinition, cs_id: str) -> str:
        cs = ir.concept_set(cs_id)
        info = self.vocab.concepts([i.concept_id for i in cs.items])
        parts = []
        for it in cs.items:
            name = info.get(it.concept_id, {}).get("concept_name", "?")
            s = f"{name} [{it.concept_id}]" + (" + descendants" if it.include_descendants else "")
            parts.append(("NOT " if it.is_excluded else "") + s)
        src = f"; {cs.source}" if cs.source != "resolved" else ""
        return f'"{cs.name}" ({", ".join(parts)}{src})'

    def _claims(self, entity: str, claim_status, dx_position) -> str:
        parts = []
        status = claim_status
        if status is None and self.ont.supports_entity(entity) and "status_col" in self.ont.table_mapping(entity):
            status = self.ont.default_claim_status()
        if status:
            parts.append(" or ".join(sorted(set(status))) + " claims only")
        if dx_position == "primary":
            parts.append("primary diagnosis only")
        return f" [{'; '.join(parts)}]" if parts else ""

    @staticmethod
    def _noun(entity: str, claim_status) -> str:
        if entity == "DrugExposure" and claim_status and "paid" not in claim_status:
            return "pharmacy claim for"  # a rejected / reversed claim is not a drug exposure
        return ENTITY_NOUN[entity]

    def _criterion(self, ir: CohortDefinition, c: Criterion) -> str:
        occ = {"at_least": "at least", "at_most": "at most", "exactly": "exactly"}[c.occurrence]
        times = (
            ("day" if c.count == 1 else "distinct days")
            if c.count_by == "dates"
            else ("time" if c.count == 1 else "times")
        )
        span = f", with the first and last at least {c.min_span_days} days apart" if c.min_span_days else ""
        return (
            f"{occ} {c.count} {times}: {self._noun(c.entity, c.claim_status)} {self._concepts(ir, c.concept_set_id)}"
            f"{self._value(c.value_filter)}{self._claims(c.entity, c.claim_status, c.dx_position)}, "
            f"{_window(c.window)}{span}"
        )

    def explain(self, ir: CohortDefinition) -> str:
        ie = ir.index_event
        lines = [f"### {ir.name}", ""]
        first = "the first-ever" if ie.first_occurrence_only else "any"
        lines.append(
            f"**Entry (index date):** {first} {self._noun(ie.entity, ie.claim_status)} "
            f"{self._concepts(ir, ie.concept_set_id)}{self._value(ie.value_filter)}"
            f"{self._claims(ie.entity, ie.claim_status, ie.dx_position)}."
        )
        lines.append(
            f"**Data source:** {self.ont.dataset_name} (observation: "
            f"{self.ont.capabilities.get('observation', 'observation_period').replace('_', ' ')})."
        )
        req = []
        if ir.prior_observation_days:
            req.append(f"{ir.prior_observation_days} days of observation before index")
        if ir.post_observation_days:
            req.append(f"{ir.post_observation_days} days of observation after index")
        d = ir.demographics
        if d.age_min is not None or d.age_max is not None:
            req.append(
                f"age at index {d.age_min if d.age_min is not None else 0}"
                f"–{d.age_max if d.age_max is not None else 'any'}"
            )
        if d.gender_concept_ids:
            req.append(
                "gender: " + ", ".join({8507: "male", 8532: "female"}.get(g, str(g)) for g in d.gender_concept_ids)
            )
        if req:
            lines.append("**Requirements:** " + "; ".join(req) + ".")
        if ir.inclusion:
            lines.append("**Include people with:**")
            lines += [f"- {self._criterion(ir, c)}" for c in ir.inclusion]
        if ir.exclusion:
            lines.append("**Exclude people with:**")
            lines += [f"- {self._criterion(ir, c)}" for c in ir.exclusion]
        lines.append(
            "**Exit:** "
            + (
                "end of the observation period."
                if ir.exit.type == "end_of_observation"
                else f"{ir.exit.days} days after index (or end of observation)."
            )
        )
        if ir.assumptions:
            lines.append("**Assumptions made:**")
            lines += [f"- {a}" for a in ir.assumptions]
        return "\n".join(lines)
