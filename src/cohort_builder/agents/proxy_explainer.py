"""Proxy explainer and review packet (deterministic, generated from the definition itself).

Wording rules: the output never says a patient "has" the target condition. It describes *evidence*
that a data-based rule found, names the classification (direct / proxy / exploratory / validated
against a reference standard), and calls the score a rule score.
"""

from __future__ import annotations

from typing import Any

from ..ontology import Ontology
from ..proxy import CLASSIFICATION_LABELS, Expr, ProxyDefinition, TemporalRule
from ..vocab import Vocabulary
from .explainer import Explainer, _window

SCORE_NOTE = (
    "The evidence score is a deterministic rule score (sum of configured points). It is not a "
    "probability, sensitivity, specificity, PPV or measure of clinical certainty."
)
COHORT_NOTE = (
    "Patients in this cohort have data patterns that match the algorithm's rules. Membership does "
    "not establish that a patient has {target}."
)
NOT_VALIDATED = (
    "This algorithm has not been validated against an external reference standard; its accuracy "
    "(sensitivity, PPV, ...) is unknown."
)


def temporal_text(t: TemporalRule, names: dict[str, str]) -> str:
    a, b = names.get(t.a, t.a), names.get(t.b, t.b)
    lo, hi = t.bounds()
    if t.relation == "same_day":
        return f"{a} and {b} on the same day"
    if t.relation == "before":
        rng = f"up to {hi} days before" if hi is not None else "any time before"
        return f"{a} {rng} {b}" + (" (same day allowed)" if t.allow_same_day else " (strictly earlier)")
    if t.relation == "after":
        rng = f"up to {-lo} days after" if lo is not None else "any time after"
        return f"{a} {rng} {b}" + (" (same day allowed)" if t.allow_same_day else " (strictly later)")
    if t.relation == "within":
        return f"{a} and {b} within {hi} days of each other (either order, inclusive)"
    lo_s = "-inf" if lo is None else f"{lo:+d}"
    hi_s = "+inf" if hi is None else f"{hi:+d}"
    return f"{b} between {lo_s} and {hi_s} days relative to {a} (inclusive)"


class ProxyExplainer(Explainer):
    def __init__(self, ont: Ontology, vocab: Vocabulary):
        super().__init__(ont, vocab)

    # ---- expressions ---------------------------------------------------------------------------
    def expr_text(self, e: Expr, p: ProxyDefinition) -> str:
        names = {ev.id: ev.name for ev in p.evidence}
        k = e.kind
        if k == "evidence":
            return names.get(e.evidence or "", e.evidence or "")
        if k == "group":
            return f"[{e.group}]"
        if k == "temporal":
            return next((f"<{t.name}>" for t in p.temporal_rules if t.id == e.temporal), str(e.temporal))
        if k in ("all", "any"):
            joiner = " AND " if k == "all" else " OR "
            return "(" + joiner.join(self.expr_text(c, p) for c in e.children()) + ")"
        if k == "not_":
            return "NOT " + self.expr_text(e.children()[0], p)
        nof = getattr(e, k)
        word = {"at_least": "at least", "at_most": "at most", "exactly": "exactly"}[k]
        within = f", within {nof.within_days} days of each other" if nof.within_days is not None else ""
        return f"{word} {nof.n} of ({'; '.join(self.expr_text(c, p) for c in nof.of)}){within}"

    def evidence_text(self, p: ProxyDefinition, eid: str) -> str:
        ev = p.evidence_by_id(eid)
        txt = self._criterion(p, ev)
        if ev.max_span_days is not None:
            txt += f", all within one span of at most {ev.max_span_days} days"
        if ev.place_of_service:
            txt += f"; place of service in {ev.place_of_service}"
        if ev.provider_specialty:
            txt += f"; provider specialty in {ev.provider_specialty}"
        return txt

    # ---- plain-language explanation ----------------------------------------------------------------
    def explain(self, p: ProxyDefinition) -> str:
        names = {ev.id: ev.name for ev in p.evidence}
        ie = p.index_event
        lines = [f"### {p.algorithm_name} v{p.version}: {p.label}", ""]
        lines.append(
            f"**Classification:** {CLASSIFICATION_LABELS[p.classification]}."
            + ("" if p.classification == "clinically_validated" else f" {NOT_VALIDATED}")
        )
        lines.append(f"**Target:** {p.target.name}. " + COHORT_NOTE.format(target=p.target.name))
        first = "the first-ever" if ie.first_occurrence_only else "any"
        lines.append(
            f"**Index date:** {first} {self._noun(ie.entity, ie.claim_status)} {self._concepts(p, ie.concept_set_id)}."
        )
        lines.append(
            f"**Data source:** {self.ont.dataset_name}"
            + (f" (designed for {p.dataset_profile})" if p.dataset_profile not in (None, self.ont.dataset_name) else "")
            + "."
        )
        req = []
        if p.prior_observation_days:
            req.append(f"{p.prior_observation_days} days of observation before index")
        if p.post_observation_days:
            req.append(f"{p.post_observation_days} days of observation after index")
        if req:
            lines.append("**Requirements:** " + "; ".join(req) + ".")
        lines.append("**Evidence:**")
        for ev in p.evidence:
            opt = "" if ev.required else " (optional: treated as absent if the dataset cannot provide it)"
            lines.append(f"- `{ev.id}` {ev.name} [{ev.category}]{opt}: {self.evidence_text(p, ev.id)}")
        if p.groups:
            lines.append("**Evidence groups:**")
            lines += [f"- [{g}] = {self.expr_text(e, p)}" for g, e in p.groups.items()]
        if p.temporal_rules:
            lines.append("**Temporal rules:**")
            lines += [
                f"- <{t.name}>: {temporal_text(t, names)}" + (" (required for entry)" if t.required else "")
                for t in p.temporal_rules
            ]
        if p.entry:
            lines.append(f"**Every member must have:** {self.expr_text(p.entry, p)}.")
        if p.exclusion:
            lines.append(f"**Excluded if:** {self.expr_text(p.exclusion, p)}.")
        for c in p.conflicts:
            act = "flagged (kept)" if c.action == "flag" else "excluded"
            lines.append(f"**Conflicting evidence ({act}):** {c.label or c.name}: {self.expr_text(c.rule, p)}.")
        if p.scoring:
            lines.append(
                f"**{p.scoring.name.capitalize()}:** "
                + ", ".join(f"{w.points:+d} for {self.expr_text(w.ref, p)}" for w in p.scoring.weights)
                + f". {SCORE_NOTE}"
            )
        lines.append("**Evidence tiers (first match wins; no match = not in the cohort):**")
        for t in p.tiers:
            cond = [self.expr_text(t.rule, p)] if t.rule else []
            if t.min_score is not None:
                cond.append(f"score >= {t.min_score}")
            lines.append(f"- {t.label or t.name}: {' AND '.join(cond)}")
        if p.funnel:
            lines.append("**Evidence funnel:** " + " -> ".join(f.name for f in p.funnel) + ".")
        lines.append(
            "**Exit:** "
            + (
                "end of the observation period."
                if p.exit.type == "end_of_observation"
                else f"{p.exit.days} days after index (or end of observation)."
            )
        )
        if p.assumptions:
            lines.append("**Assumptions made:**")
            lines += [f"- {a}" for a in p.assumptions]
        return "\n".join(lines)

    # ---- human review packet ---------------------------------------------------------------------
    def review_packet(
        self,
        p: ProxyDefinition,
        issues: list[dict],
        row: dict | None = None,
        sql_hash: str | None = None,
        attrition: list[dict] | None = None,
    ) -> dict[str, Any]:
        names = {ev.id: ev.name for ev in p.evidence}
        direct = [ev for ev in p.evidence if ev.category == "direct"]
        unavailable = [i["message"] for i in issues if "unavailable" in i["message"] or i["stage"] == "dataset"]
        meta = {
            k: (row or {}).get(k)
            for k in (
                "cohort_definition_id",
                "status",
                "created_by",
                "created_at",
                "updated_at",
                "approved_by",
                "approved_at",
                "tenant",
            )
        }
        return {
            "algorithm": {
                "algorithm_name": p.algorithm_name,
                "version": p.version,
                "classification": p.classification,
                "classification_label": CLASSIFICATION_LABELS[p.classification],
                "validation_reference": p.validation_reference,
                **meta,
                "content_hash": p.content_hash(),
                "semantic_hash": p.semantic_hash(),
                "sql_hash": sql_hash,
            },
            "target": {"name": p.target.name, "description": p.target.description},
            "clinical_notes": p.clinical_notes,
            "why_no_direct_code": (
                "The definition contains direct-diagnosis evidence: " + ", ".join(e.name for e in direct)
                if direct
                else "No evidence is marked as a direct diagnosis code; membership rests on combined proxy evidence."
            ),
            "index_event": f"{'first' if p.index_event.first_occurrence_only else 'any'} "
            f"{self._concepts(p, p.index_event.concept_set_id)}",
            "evidence": [
                {
                    "id": ev.id,
                    "name": ev.name,
                    "category": ev.category,
                    "required": ev.required,
                    "rule": self.evidence_text(p, ev.id),
                    "window": _window(ev.window),
                    "concept_set": ev.concept_set_id,
                }
                for ev in p.evidence
            ],
            "logic": {
                "groups": {g: self.expr_text(e, p) for g, e in p.groups.items()},
                "entry": self.expr_text(p.entry, p) if p.entry else None,
                "exclusion": self.expr_text(p.exclusion, p) if p.exclusion else None,
                "conflicts": [
                    {"name": c.name, "label": c.label, "action": c.action, "rule": self.expr_text(c.rule, p)}
                    for c in p.conflicts
                ],
            },
            "temporal_rules": [
                {
                    "id": t.id,
                    "name": t.name,
                    "rule": temporal_text(t, names),
                    "bounds_days": list(t.bounds()),
                    "required": t.required,
                }
                for t in p.temporal_rules
            ],
            "scoring": None
            if not p.scoring
            else {
                "name": p.scoring.name,
                "note": SCORE_NOTE,
                "weights": [{"rule": self.expr_text(w.ref, p), "points": w.points} for w in p.scoring.weights],
            },
            "expected_tiers": [
                {
                    "name": t.name,
                    "label": t.label,
                    "description": t.description,
                    "rule": self.expr_text(t.rule, p) if t.rule else None,
                    "min_score": t.min_score,
                }
                for t in p.tiers
            ],
            "funnel": [f.name for f in p.funnel],
            "dataset": {
                "active": self.ont.dataset_name,
                "designed_for": p.dataset_profile,
                "observation": self.ont.capabilities.get("observation"),
                "absence_inference": self.ont.capabilities.get("absence_inference"),
                "limitations": unavailable,
            },
            "versions": {
                "ontology": p.ontology_version,
                "vocabulary": p.vocabulary_version,
                "concept_sets": p.concept_set_versions(),
            },
            "issues": issues,
            "dry_run_attrition": attrition,
            "assumptions": p.assumptions,
            "explanation": self.explain(p),
            "governance": [
                COHORT_NOTE.format(target=p.target.name),
                SCORE_NOTE,
                *([] if p.classification == "clinically_validated" else [NOT_VALIDATED]),
                "Approval records that a reviewer accepted these rules; it is not clinical validation.",
            ],
            "reviewer_checklist": [
                "Concept sets are clinically appropriate and complete for the target",
                "Evidence windows and temporal rules match the clinical pathway",
                "Tier rules and score weights reflect the intended evidence strength",
                "Dataset limitations (missing evidence, absence semantics) are acceptable for the study",
                "Classification wording is accurate (no claim of validation without a reference standard)",
            ],
        }
