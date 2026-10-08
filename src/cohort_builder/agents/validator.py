"""Validator (deterministic): ontology rules, vocabulary checks, data-coverage caveats and a dry-run count.

Structural rules (ranges, required pairs, references) live in the IR models (ir.py).
This module checks what needs the ontology, the vocabulary, the active dataset or the data:

  errors    (severity "error")   the definition cannot be run correctly as written
  warnings  (severity "warning") it can run, but a reviewer must understand a limitation;
            stage "data" warnings are carried into execution metadata as caveats
"""
from __future__ import annotations

from typing import Any

from ..compiler import Compiler
from ..executor import Attrition, Executor
from ..ir import CohortDefinition, Criterion, ValueFilter
from ..ontology import Ontology
from ..vocab import Vocabulary
from .composer import Issue


def _absence_based(role: str, c: Criterion) -> bool:
    """True when the rule is satisfied by NOT finding records ('no prior X', 'at most 0 X')."""
    if role == "exclusion":
        return True
    return c.occurrence in ("at_most", "exactly") and c.count == 0


def validate(ir: CohortDefinition, ont: Ontology, vocab: Vocabulary, executor: Executor | None = None,
             ) -> tuple[list[Issue], Attrition | None]:
    issues: list[Issue] = []

    def err(stage: str, msg: str) -> None:
        issues.append(Issue("error", stage, msg))

    def warn(stage: str, msg: str) -> None:
        issues.append(Issue("warning", stage, msg))

    ds = ont.dataset_name
    caps = ont.capabilities
    if ir.ontology_version != ont.version:
        warn("intent", f"IR built with ontology {ir.ontology_version}, current is {ont.version}")

    # ---- dataset prerequisites ------------------------------------------------------
    has_observation = bool(ont.mappings.get("observation_period")) and caps.get("observation") not in (None, "none")
    if not has_observation:
        err("dataset", f"dataset {ds!r} defines no observation periods: prior/post observation, open-ended windows "
                       "and 'no record' criteria cannot be evaluated. Add an observation definition to the profile.")

    # ---- concept sets (vocabulary) ------------------------------------------------------
    ids = {cs.id for cs in ir.concept_sets}
    info = vocab.concepts([i.concept_id for cs in ir.concept_sets for i in cs.items])
    for cs in ir.concept_sets:
        for it in cs.items:
            c = info.get(it.concept_id)
            where = f"concept set {cs.id!r}, concept {it.concept_id}"
            if c is None:
                err("concepts", f"{where}: not in the loaded vocabulary ({vocab.version()})")
            elif c["invalid_reason"]:
                err("concepts", f"{where}: deprecated ({c['concept_name']}); use its replacement / Maps-to target")
            elif c["standard_concept"] not in ("S", "C"):
                err("concepts", f"{where}: non-standard ({c['concept_name']}); use the standard concept it maps to")
            elif c["domain_id"] != cs.domain:
                err("concepts", f"{where}: domain {c['domain_id']} != concept set domain {cs.domain}")
            elif c["standard_concept"] == "C" and not it.include_descendants and not it.is_excluded:
                err("concepts", f"{where}: classification concept ({c['concept_name']}) has no records of its own; "
                                "set include_descendants=true")

    # ---- criteria -------------------------------------------------------------------
    def check_ref(entity: str, cs_id: str, where: str) -> None:
        cs = ir.concept_set(cs_id)  # existence is guaranteed by the IR model
        if ont.entity_domain(entity) != cs.domain:
            err("concepts", f"{where}: entity {entity} needs domain {ont.entity_domain(entity)}, "
                            f"concept set {cs_id!r} is {cs.domain}")

    def check_value(entity: str, cs_id: str, vf: ValueFilter | None, where: str) -> None:
        if vf is None:
            return
        if not ont.entity_has_value(entity):
            err("intent", f"{where}: value filters are only allowed on entities with values, not {entity}")
            return
        if vf.op not in ont.value_operators():
            err("intent", f"{where}: operator {vf.op} not allowed (allowed: {ont.value_operators()})")
        if vf.unit_concept_id not in ont.units:
            err("intent", f"{where}: unit {vf.unit_concept_id} not declared in ontology/unit_conversions.yaml")
            return
        check_units(cs_id, vf, where)

    def check_units(cs_id: str, vf: ValueFilter, where: str) -> None:
        """Never compare values in incompatible units silently."""
        items = [i.concept_id for i in ir.concept_set(cs_id).items if not i.is_excluded]
        analytes: dict[int, set[int]] = {}
        unknown = []
        for cid in items:
            found = [a for a in sorted(ont.analytes) if vocab.is_descendant_or_self(cid, a)]
            unknown += [] if found else [cid]
            for a in found:
                analytes.setdefault(a, set()).add(cid)
        canon = {int(ont.analytes[a]["canonical_unit"]) for a in analytes}
        if len(canon) > 1:
            names = ", ".join(ont.analytes[a]["name"] for a in sorted(analytes))
            err("intent", f"{where}: one value filter spans analytes with different units ({names}); "
                          "split it into one criterion per analyte")
            return
        if analytes:
            meta = ont.analytes[sorted(analytes)[0]]
            canonical = int(meta["canonical_unit"])
            if vf.unit_concept_id != canonical:
                conv = next((c for c in meta.get("conversions", []) if int(c["from_unit"]) == vf.unit_concept_id),
                            None)
                hint = ""
                if conv:
                    hint = f" ({vf.value:g} {ont.unit_symbol(vf.unit_concept_id)} = " \
                           f"{vf.value * conv['factor'] + conv['offset']:.3f} {ont.unit_symbol(canonical)})"
                err("intent", f"{where}: {meta['name']} thresholds must be in the canonical unit "
                              f"{ont.unit_symbol(canonical)}, got {ont.unit_symbol(vf.unit_concept_id)}{hint}. "
                              "Results in other units are converted to the canonical unit before comparison.")
            if unknown:
                warn("data", f"{where}: concepts {unknown} have no unit metadata; only their results recorded in "
                             f"{ont.unit_symbol(vf.unit_concept_id)} are compared, other units are excluded")
        else:
            warn("data", f"{where}: no unit-conversion metadata for these measurements; only results recorded in "
                         f"{ont.unit_symbol(vf.unit_concept_id)} are compared, results in other units are excluded "
                         "(not converted)")

    def check_dataset(entity: str, x: Any, where: str) -> None:
        """Can the active dataset answer this? (stage 'dataset' = not fixable by rephrasing)"""
        if not ont.supports_entity(entity):
            err("dataset", f"{where}: {entity} data ({ont.entity_domain(entity).lower()} records"
                           f"{', e.g. lab values' if ont.entity_has_value(entity) else ''}) is not available in "
                           f"dataset {ds!r}; available: {', '.join(caps['entities'])}")
            return
        if x.claim_status is not None and entity != "DrugExposure":
            err("intent", f"{where}: claim_status only applies to drug (pharmacy claim) criteria")
        elif x.claim_status is not None:
            if not ont.supports_attribute("claim_status") or "status_col" not in ont.table_mapping(entity):
                err("dataset", f"{where}: claim status filters are not available for {entity} in dataset {ds!r}")
        if x.dx_position not in (None, "any") and entity != "ConditionOccurrence":
            err("intent", f"{where}: dx_position only applies to diagnosis criteria")
        elif x.dx_position not in (None, "any"):
            if not ont.supports_attribute("dx_position") or "position_col" not in ont.table_mapping(entity):
                err("dataset", f"{where}: diagnosis position is not available for {entity} in dataset {ds!r}")
        if entity in caps.get("partial_entities", []):
            warn("data", f"{where}: {entity} records are only partially captured in dataset {ds!r}; "
                         "a missing record is not evidence of a normal result or of absence")

    def check_coverage(role: str, c: Criterion, where: str) -> None:
        for m in _coverage_messages(role, c, ir.prior_observation_days, ir.post_observation_days,
                                    caps.get("observation", ""), ds):
            warn("data", f"{where}: {m}")

    check_ref(ir.index_event.entity, ir.index_event.concept_set_id, "index event")
    check_value(ir.index_event.entity, ir.index_event.concept_set_id, ir.index_event.value_filter, "index event")
    check_dataset(ir.index_event.entity, ir.index_event, "index event")
    crits: list[tuple[str, Criterion]] = [("inclusion", c) for c in ir.inclusion] + \
                                         [("exclusion", c) for c in ir.exclusion]
    for role, c in crits:
        where = f"{role} {c.name!r}"
        check_ref(c.entity, c.concept_set_id, where)
        check_value(c.entity, c.concept_set_id, c.value_filter, where)
        check_dataset(c.entity, c, where)
        if role == "exclusion" and c.occurrence != "at_least":
            warn("intent", f"{where}: exclusion with occurrence {c.occurrence} is unusual (excludes people who "
                           f"satisfy '{c.occurrence} {c.count}')")
        if has_observation:
            check_coverage(role, c, where)
    if len(ir.inclusion) > ont.rules["max_inclusion_criteria"]:
        err("intent", f"more than {ont.rules['max_inclusion_criteria']} inclusion criteria")
    used = {ir.index_event.concept_set_id} | {c.concept_set_id for _, c in crits}
    for unused in sorted(ids - used):
        warn("intent", f"concept set {unused!r} is not used by any rule")
    allowed_genders = {int(k) for k in ont.domain["attributes"]["Person.gender"]["allowed_concepts"]}
    for g in ir.demographics.gender_concept_ids:
        if g not in allowed_genders:
            err("intent", f"demographics: gender concept {g} is not one of the ontology's allowed values "
                          f"{sorted(allowed_genders)}")

    if any(i.severity == "error" for i in issues) or executor is None:
        return issues, None

    # ---- dry run (aggregate counts only) --------------------------------------
    attrition = executor.attrition(Compiler(ont).compile(ir))
    rules = attrition.rules
    if rules[0]["remaining"] == 0:
        err("concepts", "no index events found in the data: check the index concept set")
    elif rules[1]["remaining"] == 0:
        err("data", "index events were found, but none fall inside an observation period (or the person record "
                    "is missing): check the dataset's observation data")
    elif attrition.final_count == 0:
        err("intent", "the cohort is empty after applying all rules")
    ratio = ont.rules["attrition_warning_ratio"]
    for prev, cur in zip(rules, rules[1:], strict=False):
        if prev["remaining"] and 1 - cur["remaining"] / prev["remaining"] > ratio:
            warn("data", f"rule {cur['name']!r} removes more than {ratio:.1%} of remaining people")
    return issues, attrition


def _coverage_messages(role: str, c: Criterion, prior: int, post: int, observation: str, ds: str) -> list[str]:
    msgs = []
    absence = _absence_based(role, c)
    s, e = c.window.start_days, c.window.end_days
    if absence:
        basis = ("observation is inferred from claim activity (no enrollment data), so a missing claim is weak "
                 "evidence that the event did not happen" if observation == "activity_based" else
                 "events recorded outside this data source (other providers or payers, before data capture began) "
                 "are not seen")
        msgs.append(f"'no record' is treated as 'did not happen' in dataset {ds!r}; {basis}")
    if s is None and absence:
        msgs.append(f"'any time before' only covers observed history, which is guaranteed to be at least "
                    f"{prior} days" + (" (none required: history may be empty)" if prior == 0 else ""))
    elif s is not None and s < 0 and -s > prior:
        effect = "people with shorter history may be wrongly kept" if absence else "events may be under-counted"
        msgs.append(f"looks back {-s} days but only {prior} days of prior observation are required: for people "
                    f"with shorter history the lookback is truncated ({effect}); consider prior_observation_days "
                    f">= {-s}")
    if e is not None and e > 0 and e > post:
        effect = "people with short follow-up may be wrongly kept" if absence else "events may be under-counted"
        msgs.append(f"looks {e} days after index but only {post} days of follow-up are required ({effect}); "
                    f"consider post_observation_days >= {e}")
    return msgs

