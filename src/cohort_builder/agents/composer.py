"""Composer (deterministic): intent + resolved concept sets -> CohortDefinition IR.

Also normalizes value thresholds to the analyte's canonical unit using the
ontology, so the stored IR is unit-consistent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..ir import (CohortDefinition, CohortExit, ConceptSet, Criterion, Demographics, IndexEvent, ValueFilter,
                  Window)
from ..ontology import Ontology
from ..vocab import Vocabulary
from .intent import CohortIntent, ValueSpec
from .resolver import ResolvedConceptSet

GENDERS = {"male": 8507, "female": 8532}


@dataclass
class Issue:
    severity: str  # error | warning
    stage: str     # intent | concepts | data
    message: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "stage": self.stage, "message": self.message}


def _cs_id(key: str) -> str:
    return "cs_" + (re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_").lower() or "x")


def _analyte_for(ont: Ontology, vocab: Vocabulary, cs: ResolvedConceptSet) -> int | None:
    for analyte_id in sorted(ont.analytes):
        if all(vocab.is_descendant_or_self(i.concept_id, analyte_id) for i in cs.items if not i.is_excluded):
            return analyte_id
    return None


def _value_filter(spec: ValueSpec, cs: ResolvedConceptSet, ont: Ontology, vocab: Vocabulary,
                  where: str, issues: list[Issue]) -> ValueFilter | None:
    unit = ont.find_unit(spec.unit_text)
    if unit is None:
        issues.append(Issue("error", "intent", f"{where}: unknown unit {spec.unit_text!r}; known units: "
                            f"{sorted(u['symbol'] for u in ont.units.values())}"))
        return None
    analyte = _analyte_for(ont, vocab, cs)
    value, high, original = spec.value, spec.value_high, None
    if analyte is not None:
        a = ont.analytes[analyte]
        canonical = int(a["canonical_unit"])
        if unit != canonical:
            conv = next((c for c in a.get("conversions", []) if int(c["from_unit"]) == unit), None)
            if conv is None:
                issues.append(Issue("error", "intent", f"{where}: cannot convert {spec.unit_text} to "
                                    f"{ont.unit_symbol(canonical)} for {a['name']}"))
                return None
            original = f"{spec.op} {spec.value}{'' if high is None else ' and ' + str(high)} {spec.unit_text}"
            value = round(spec.value * conv["factor"] + conv["offset"], 3)
            high = None if high is None else round(high * conv["factor"] + conv["offset"], 3)
            unit = canonical
    return ValueFilter(op=spec.op, value=value, value_high=high, unit_concept_id=unit, original_text=original)


def _claim_status(entity: str, requested: list[str] | None, ont: Ontology) -> list[str] | None:
    """Make the dataset's default claim status explicit in the IR (e.g. paid claims only)."""
    if requested is not None:
        return sorted(set(requested))
    if ont.supports_entity(entity) and "status_col" in ont.table_mapping(entity):
        default = ont.default_claim_status()
        return sorted(default) if default else None
    return None


def compose(intent: CohortIntent, resolved: dict[str, ResolvedConceptSet], ont: Ontology, vocab: Vocabulary,
            ) -> tuple[CohortDefinition | None, list[Issue]]:
    issues: list[Issue] = []
    mentions = {m.key: m for m in intent.mentions}
    concept_sets: dict[str, ConceptSet] = {}
    for key, rcs in sorted(resolved.items()):
        concept_sets[key] = ConceptSet(id=_cs_id(key), name=rcs.name, domain=rcs.domain, items=rcs.items,
                                       source=rcs.source)

    def cs_for(key: str, where: str) -> ConceptSet | None:
        if key not in mentions:
            issues.append(Issue("error", "intent", f"{where}: mention_key {key!r} is not declared in mentions"))
            return None
        if key not in concept_sets:
            issues.append(Issue("error", "concepts", f"{where}: mention {key!r} was not resolved"))
            return None
        return concept_sets[key]

    idx_cs = cs_for(intent.index_mention_key, "index event")
    index_vf = None
    if idx_cs and intent.index_value:
        index_vf = _value_filter(intent.index_value, resolved[intent.index_mention_key], ont, vocab,
                                 "index event", issues)

    inclusion, exclusion = [], []
    for c in intent.criteria:
        cs = cs_for(c.mention_key, f"criterion {c.name!r}")
        if cs is None:
            continue
        vf = None
        if c.value:
            vf = _value_filter(c.value, resolved[c.mention_key], ont, vocab, f"criterion {c.name!r}", issues)
        try:
            window = Window(start_days=c.window_start_days, end_days=c.window_end_days)
        except ValueError as exc:
            issues.append(Issue("error", "intent", f"criterion {c.name!r}: {exc}"))
            continue
        target = inclusion if c.role == "inclusion" else exclusion
        prefix = "inc" if c.role == "inclusion" else "exc"
        entity = mentions[c.mention_key].entity
        target.append(Criterion(id=f"{prefix}_{len(target) + 1}", name=c.name, entity=entity,
                                concept_set_id=cs.id, window=window, occurrence=c.occurrence, count=c.count,
                                value_filter=vf, claim_status=_claim_status(entity, c.claim_status, ont),
                                dx_position=c.dx_position if c.dx_position == "primary" else None,
                                min_span_days=c.min_span_days))

    if idx_cs is None or any(i.severity == "error" for i in issues):
        return None, issues

    used = {idx_cs.id} | {c.concept_set_id for c in inclusion + exclusion}
    ir = CohortDefinition(
        ontology_version=ont.version,
        vocabulary_version=vocab.version(),
        name=intent.name,
        description=intent.description,
        concept_sets=sorted((cs for cs in concept_sets.values() if cs.id in used), key=lambda s: s.id),
        index_event=IndexEvent(entity=mentions[intent.index_mention_key].entity, concept_set_id=idx_cs.id,
                               first_occurrence_only=intent.index_first_occurrence_only, value_filter=index_vf,
                               claim_status=_claim_status(mentions[intent.index_mention_key].entity,
                                                          intent.index_claim_status, ont),
                               dx_position=intent.index_dx_position if intent.index_dx_position == "primary"
                               else None),
        prior_observation_days=intent.prior_observation_days,
        post_observation_days=intent.post_observation_days,
        demographics=Demographics(age_min=intent.age_min, age_max=intent.age_max,
                                  gender_concept_ids=sorted(GENDERS[g] for g in intent.genders)),
        inclusion=inclusion,
        exclusion=exclusion,
        exit=CohortExit(type=intent.exit_type, days=intent.exit_days),
        assumptions=intent.assumptions,
    )
    return ir, issues
