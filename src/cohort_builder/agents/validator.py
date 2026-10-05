"""Validator (deterministic): ontology rules, vocabulary checks and a dry-run count."""
from __future__ import annotations

from ..compiler import Compiler
from ..executor import Attrition, Executor
from ..ir import CohortDefinition, Criterion
from ..ontology import Ontology
from ..vocab import Vocabulary
from .composer import Issue


def validate(ir: CohortDefinition, ont: Ontology, vocab: Vocabulary, executor: Executor | None = None,
             ) -> tuple[list[Issue], Attrition | None]:
    issues: list[Issue] = []
    err = lambda stage, msg: issues.append(Issue("error", stage, msg))  # noqa: E731
    warn = lambda stage, msg: issues.append(Issue("warning", stage, msg))  # noqa: E731

    if ir.ontology_version != ont.version:
        warn("intent", f"IR built with ontology {ir.ontology_version}, current is {ont.version}")

    # ---- concept sets -------------------------------------------------------
    ids = {cs.id for cs in ir.concept_sets}
    if len(ids) != len(ir.concept_sets):
        err("concepts", "duplicate concept set ids")
    info = vocab.concepts([i.concept_id for cs in ir.concept_sets for i in cs.items])
    for cs in ir.concept_sets:
        for it in cs.items:
            c = info.get(it.concept_id)
            where = f"concept set {cs.id!r}, concept {it.concept_id}"
            if c is None:
                err("concepts", f"{where}: not in vocabulary")
            elif c["invalid_reason"]:
                err("concepts", f"{where}: deprecated ({c['concept_name']})")
            elif c["standard_concept"] not in ("S", "C"):
                err("concepts", f"{where}: non-standard ({c['concept_name']})")
            elif c["domain_id"] != cs.domain:
                err("concepts", f"{where}: domain {c['domain_id']} != concept set domain {cs.domain}")
        if all(i.is_excluded for i in cs.items):
            err("concepts", f"concept set {cs.id!r} only has excluded items")

    # ---- criteria -------------------------------------------------------------
    def check_ref(entity: str, cs_id: str, where: str) -> None:
        if cs_id not in ids:
            err("intent", f"{where}: unknown concept set {cs_id!r}")
            return
        cs = ir.concept_set(cs_id)
        if ont.entity_domain(entity) != cs.domain:
            err("concepts", f"{where}: entity {entity} needs domain {ont.entity_domain(entity)}, "
                            f"concept set {cs_id!r} is {cs.domain}")

    def check_value(entity: str, vf, where: str) -> None:
        if vf is None:
            return
        if not ont.entity_has_value(entity):
            err("intent", f"{where}: value filters are only allowed on entities with values, not {entity}")
        if vf.op not in ont.value_operators():
            err("intent", f"{where}: operator {vf.op} not allowed")
        if vf.unit_concept_id not in ont.units:
            err("intent", f"{where}: unit {vf.unit_concept_id} not declared in ontology")

    check_ref(ir.index_event.entity, ir.index_event.concept_set_id, "index event")
    check_value(ir.index_event.entity, ir.index_event.value_filter, "index event")
    crits: list[tuple[str, Criterion]] = [("inclusion", c) for c in ir.inclusion] + \
                                         [("exclusion", c) for c in ir.exclusion]
    for role, c in crits:
        where = f"{role} {c.name!r}"
        check_ref(c.entity, c.concept_set_id, where)
        check_value(c.entity, c.value_filter, where)
        if role == "exclusion" and c.occurrence != "at_least":
            warn("intent", f"{where}: exclusion with occurrence {c.occurrence} is unusual")
    if len(ir.inclusion) > ont.rules["max_inclusion_criteria"]:
        err("intent", f"more than {ont.rules['max_inclusion_criteria']} inclusion criteria")
    used = {ir.index_event.concept_set_id} | {c.concept_set_id for _, c in crits}
    for unused in sorted(ids - used):
        warn("intent", f"concept set {unused!r} is not used by any rule")

    d = ir.demographics
    if d.age_min is not None and d.age_max is not None and d.age_min > d.age_max:
        err("intent", "age_min > age_max")
    if ir.exit.type == "fixed_days" and not ir.exit.days:
        err("intent", "exit type fixed_days needs days > 0")
    if ir.prior_observation_days < 0 or ir.post_observation_days < 0:
        err("intent", "observation days must be >= 0")

    if any(i.severity == "error" for i in issues) or executor is None:
        return issues, None

    # ---- dry run (aggregate counts only) --------------------------------------
    attrition = executor.attrition(Compiler(ont).compile(ir))
    rules = attrition.rules
    if rules[0]["remaining"] == 0:
        err("concepts", "no index events found in the data: check the index concept set")
    elif attrition.final_count == 0:
        err("intent", "the cohort is empty after applying all rules")
    ratio = ont.rules["attrition_warning_ratio"]
    for prev, cur in zip(rules, rules[1:]):
        if prev["remaining"] and 1 - cur["remaining"] / prev["remaining"] > ratio:
            warn("data", f"rule {cur['name']!r} removes more than {ratio:.1%} of remaining people")
    return issues, attrition
