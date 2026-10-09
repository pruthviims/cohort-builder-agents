"""Validator for proxy definitions (deterministic).

Reuses every cohort check (concepts, domains, units, dataset entities/attributes, data coverage,
dry-run attrition) by treating each evidence item as a criterion, then adds proxy-specific checks:

* Dataset capability per evidence item. Required evidence the dataset cannot provide is an ERROR
  ("Proxy rule requires pathology evidence, but the selected dataset does not provide pathology
  data"). Optional evidence (required: false) is a WARNING and is compiled as absent; the algorithm
  then operates on the remaining evidence.
* Reachability. With unavailable evidence treated as absent, a tier, entry rule or required temporal
  rule that can never be satisfied is reported (three-valued evaluation, no data needed).
* Absence semantics. Every NOT / "at most / exactly N" / exclusion rule gets a warning whose
  strength follows the dataset's `absence_inference` capability (weak | observed_period | supported).
* Governance wording. The score is a rule score; classification 'clinically_validated' needs a
  recorded reference validation (checked by the orchestrator, which owns the database).
"""

from __future__ import annotations

from ..executor import Attrition, Executor
from ..ontology import Ontology
from ..proxy import Expr, ProxyDefinition
from ..proxy_compiler import CompiledProxy, ProxyCompiler, evidence_support
from ..vocab import Vocabulary
from .composer import Issue
from .validator import validate

ABSENCE_WEAK = (
    "Absence of a claim does not establish absence of disease. This rule is limited by observation and claims coverage"
)


def absence_inference(ont: Ontology) -> str:
    caps = ont.capabilities
    if caps.get("absence_inference") in ("weak", "observed_period", "supported"):
        return str(caps["absence_inference"])
    return "weak" if caps.get("observation") == "activity_based" else "observed_period"


def truth(expr: Expr, p: ProxyDefinition, unavailable: set[str]) -> bool | None:
    """Three-valued evaluation: False/True when decided by unavailable (=absent) evidence alone, else None."""
    k = expr.kind
    if k == "evidence":
        return False if expr.evidence in unavailable else None
    if k == "temporal":
        t = next(t for t in p.temporal_rules if t.id == expr.temporal)
        return False if (t.a in unavailable or t.b in unavailable) else None
    if k == "group":
        return truth(p.groups[expr.group or ""], p, unavailable)
    vals = [truth(c, p, unavailable) for c in expr.children()]
    if k == "all":
        return False if False in vals else (True if all(v is True for v in vals) else None)
    if k == "any":
        return True if True in vals else (False if all(v is False for v in vals) else None)
    if k == "not_":
        return None if vals[0] is None else not vals[0]
    n = getattr(expr, k).n
    yes, u = vals.count(True), vals.count(None)
    if k == "at_least":
        return True if yes >= n else (False if yes + u < n else None)
    if k == "at_most":
        return False if yes > n else (True if yes + u <= n else None)
    return False if (yes > n or yes + u < n) else (True if u == 0 and yes == n else None)


def validate_proxy(
    p: ProxyDefinition, ont: Ontology, vocab: Vocabulary, executor: Executor | None = None
) -> tuple[list[Issue], Attrition | None, CompiledProxy | None]:
    issues: list[Issue] = []

    def err(stage: str, msg: str) -> None:
        issues.append(Issue("error", stage, msg))

    def warn(stage: str, msg: str) -> None:
        issues.append(Issue("warning", stage, msg))

    ds = ont.dataset_name
    if p.dataset_profile and p.dataset_profile != ds:
        warn(
            "data",
            f"algorithm was designed for dataset {p.dataset_profile!r} but is validated on {ds!r}; "
            "evidence availability and coding may differ",
        )

    # ---- dataset capability per evidence ------------------------------------------------
    unavailable: set[str] = set()
    for ev in p.evidence:
        reasons = evidence_support(ev, ont)
        if not reasons:
            continue
        unavailable.add(ev.id)
        cat = ev.category
        if ev.required:
            err(
                "dataset",
                f"Proxy rule requires {cat} evidence {ev.name!r}, but the selected dataset {ds!r} "
                f"does not provide it: {'; '.join(reasons)}. Mark it required: false to run without it.",
            )
        else:
            warn(
                "data",
                f"{cat.capitalize()} evidence {ev.name!r} is unavailable in dataset {ds!r} "
                f"({'; '.join(reasons)}). The proxy algorithm will operate using the remaining evidence "
                "(this evidence is treated as absent).",
            )
    for t in p.temporal_rules:
        if t.a in unavailable or t.b in unavailable:
            (err if t.required else warn)(
                "dataset" if t.required else "data",
                f"temporal rule {t.name!r} uses unavailable evidence and can never hold on dataset {ds!r}",
            )

    # ---- reachability with unavailable evidence treated as absent --------------------------
    if p.entry is not None and truth(p.entry, p, unavailable) is False:
        err("dataset", f"the entry rule can never be satisfied on dataset {ds!r} (it depends on unavailable evidence)")
    reachable = []
    for tier in p.tiers:
        if tier.rule is not None and truth(tier.rule, p, unavailable) is False:
            warn(
                "data", f"tier {tier.name!r} can never be assigned on dataset {ds!r}: it requires unavailable evidence"
            )
        else:
            reachable.append(tier.name)
    if not reachable:
        err("dataset", f"no evidence tier can be assigned on dataset {ds!r}")

    # ---- absence semantics -------------------------------------------------------------------
    level = absence_inference(ont)
    negated = [w for w, e in p.expressions() if e.has_negation()]
    if p.exclusion is not None:
        negated.append("exclusion rule (patients WITHOUT a matching record are kept)")
    negated += [
        f"evidence {e.name!r} (at most/exactly {e.count})"
        for e in p.evidence
        if e.occurrence in ("at_most", "exactly") and e.id not in unavailable
    ]
    if negated and level != "supported":
        basis = (
            ABSENCE_WEAK + "; open/activity-based claims have no enrollment, so missing claims are expected"
            if level == "weak"
            else "'No record' only means no record during observed time; events recorded elsewhere are not seen"
        )
        for where in negated:
            warn("data", f"{where}: {basis}.")

    # ---- shared checks (concepts, units, dataset entities, coverage, dry run) ----------------
    criteria = [("evidence", e) for e in p.evidence if e.id not in unavailable]
    exempt = {e.concept_set_id for e in p.evidence if e.id in unavailable}
    pre_errors = any(i.severity == "error" for i in issues)
    compiled = None
    if not pre_errors:
        compiled = ProxyCompiler(ont).compile_proxy(p)
    base_issues, attrition = validate(
        p,
        ont,
        vocab,
        executor if not pre_errors else None,
        criteria=criteria,
        compiled=compiled,
        exempt_concept_sets=exempt,
    )
    issues += base_issues
    if any(i.severity == "error" for i in issues):
        attrition = None
    return issues, attrition, compiled
