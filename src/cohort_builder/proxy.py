"""Proxy (indirect) cohort definitions: identify patients when no single reliable code exists.

A proxy definition combines named *evidence* (diagnoses, procedures, treatments, labs,
pathology, provider/care-setting signals) with nested boolean logic, N-of-M rules,
temporal relationships between evidence, optional deterministic scoring, and ordered
evidence *tiers*. It is typed IR, like `CohortDefinition`: agents may draft it, but only
the deterministic compiler (proxy_compiler.py) turns it into SQL.

Nothing here is disease-specific: histology-defined cancers, rare metabolic diseases or molecular subtypes are
expressed purely as configuration and concept sets.

Wording rules (governance): a proxy definition is a *claims/data-based proxy* or an
*exploratory algorithm*. Claims-based proxies infer that a patient's data match a pattern;
they do not independently establish a clinical diagnosis. A definition may only be labelled
`clinically_validated` when it references a recorded evaluation against an external reference
standard that met prespecified acceptance criteria AND was accepted by a human reviewer, for
the same logic (semantic hash) - see proxy_service.py. The evidence score is a rule score, not
a probability, sensitivity, specificity or PPV.

Identifiers are strict: concept-set ids, evidence/group/temporal ids, tier and conflict names
must each be unique, and YAML/JSON input with duplicate mapping keys is rejected instead of
silently keeping the last value.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from . import IR_SCHEMA_VERSION
from .ir import (
    MAX_DAYS,
    CohortExit,
    ConceptSet,
    Criterion,
    Demographics,
    IndexEvent,
    _IRModel,
)

PROXY_SCHEMA_VERSION = "1.0"
# bump when the semantic form changes, so hashes from different contracts never compare equal
SEMANTIC_HASH_VERSION = "proxy-semantic-v2"
INTENDED_USE_PATTERN = r"^[a-z][a-z0-9_\-]{0,63}$"
# names that become SQL identifiers/literals: lower-case, no hyphens
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,47}$"
VERSION_PATTERN = r"^[0-9]+(\.[0-9]+){0,2}$"
Category = Literal[
    "direct",
    "supporting",
    "treatment",
    "procedure",
    "laboratory",
    "pathology",
    "provider",
    "care_setting",
    "conflicting",
    "other",
]
Classification = Literal["direct", "proxy", "exploratory", "clinically_validated"]
CLASSIFICATION_LABELS = {
    "direct": "Direct-diagnosis definition",
    "proxy": "Data-based proxy definition",
    "exploratory": "Exploratory identification algorithm",
    "clinically_validated": "Proxy definition validated against a reference standard",
}


class Provenance(_IRModel):
    """Where a rule or code set comes from. Free text for reviewers; never reaches SQL.

    Leave a field empty rather than guessing: missing provenance is reported by the validator,
    invented provenance is not detectable."""

    source_reference: str | None = Field(default=None, max_length=1000, description="citation, guideline or SOP")
    clinical_rationale: str | None = Field(default=None, max_length=4000)
    limitations: list[str] = Field(default_factory=list, max_length=50)


class ProxyConceptSet(ConceptSet):
    """Concept set with optional code-system provenance (metadata only: it does not filter events)."""

    code_system: str | None = Field(default=None, max_length=200, description="e.g. 'ICD-10-CM' / 'SNOMED CT'")
    code_system_version: str | None = Field(default=None, max_length=100)
    version: str | None = Field(default=None, max_length=100, description="version of this concept set")
    effective_from: date | None = None
    effective_to: date | None = None
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _dates(self) -> ProxyConceptSet:
        if self.effective_from and self.effective_to and self.effective_from > self.effective_to:
            raise ValueError(f"concept set {self.id!r}: effective_from is after effective_to")
        return self


class EvidenceCriterion(Criterion):
    """One named piece of evidence: qualifying events of a concept set in a window around the index.

    Inherits every Criterion rule (window, occurrence/count, count_by, value filter, claim status,
    diagnosis position, min_span_days). Adds:
      * max_span_days: at least `count` qualifying events falling within ONE span of at most this
        many days (e.g. "at least 2 diagnoses within 180 days"); inclusive.
      * place_of_service / provider_specialty: dataset attribute filters (capability-checked).
      * required: if the dataset cannot provide this evidence, validation fails (True, default) or
        the evidence is treated as absent with a warning (False).
    """

    id: str = Field(pattern=NAME_PATTERN)  # becomes a SQL column name
    category: Category = "other"
    description: str = Field(default="", max_length=1000)
    required: bool = True
    max_span_days: int | None = Field(default=None, ge=0, le=MAX_DAYS)
    place_of_service: list[str] | None = Field(default=None, min_length=1, max_length=50)
    provider_specialty: list[str] | None = Field(default=None, min_length=1, max_length=50)
    provenance: Provenance | None = None

    @field_validator("place_of_service", "provider_specialty")
    @classmethod
    def _codes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        for code in v:
            if not code or len(code) > 40 or not all(ch.isalnum() or ch in " _-./" for ch in code):
                raise ValueError(f"invalid attribute code {code!r} (letters, digits, space, _ - . / only)")
        return sorted(set(v))

    @model_validator(mode="after")
    def _spans(self) -> EvidenceCriterion:
        if self.max_span_days is not None:
            if self.occurrence != "at_least" or self.count < 2:
                raise ValueError(f"evidence {self.id!r}: max_span_days needs occurrence 'at_least' with count >= 2")
            if self.min_span_days is not None:
                raise ValueError(f"evidence {self.id!r}: use either min_span_days or max_span_days, not both")
        return self


class NOf(_IRModel):
    """N-of-M: count how many of `of` hold. Optional within_days: the counted evidence must all
    have a qualifying event inside one window of `within_days` days (inclusive); only for
    evidence references."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    n: int = Field(ge=0, le=100)
    of: list[Expr] = Field(min_length=1, max_length=50)
    within_days: int | None = Field(default=None, ge=0, le=MAX_DAYS)


class Expr(_IRModel):
    """Boolean expression over evidence. Exactly one key: evidence | group | temporal | all | any |
    not | at_least | at_most | exactly. Nests arbitrarily (bounded depth)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid", populate_by_name=True)
    evidence: str | None = Field(default=None, pattern=NAME_PATTERN)
    group: str | None = Field(default=None, pattern=NAME_PATTERN)
    temporal: str | None = Field(default=None, pattern=NAME_PATTERN)
    all: list[Expr] | None = Field(default=None, min_length=1, max_length=50)
    any: list[Expr] | None = Field(default=None, min_length=1, max_length=50)
    not_: Expr | None = Field(default=None, alias="not")
    at_least: NOf | None = None
    at_most: NOf | None = None
    exactly: NOf | None = None

    _KEYS = ("evidence", "group", "temporal", "all", "any", "not_", "at_least", "at_most", "exactly")

    @model_validator(mode="after")
    def _one(self) -> Expr:
        set_keys = [k for k in self._KEYS if getattr(self, k) is not None]
        if len(set_keys) != 1:
            shown = [("not" if k == "not_" else k) for k in set_keys]
            raise ValueError(
                f"an expression needs exactly one of evidence/group/temporal/all/any/not/at_least/"
                f"at_most/exactly, got {shown or 'none'}"
            )
        return self

    @property
    def kind(self) -> str:
        return next(k for k in self._KEYS if getattr(self, k) is not None)

    def children(self) -> list[Expr]:
        k = self.kind
        if k in ("all", "any"):
            return list(getattr(self, k))
        if k == "not_":
            return [self.not_] if self.not_ else []
        if k in ("at_least", "at_most", "exactly"):
            return list(getattr(self, k).of)
        return []

    def refs(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for k in ("evidence", "group", "temporal"):
            if getattr(self, k) is not None:
                out.add((k, getattr(self, k)))
        for ch in self.children():
            out |= ch.refs()
        return out

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children()), default=0)

    def has_negation(self) -> bool:
        if self.kind == "not_":
            return True
        nof = self.at_most or self.exactly
        if nof is not None and nof.n < len(nof.of):  # "at most/exactly n" can be met by absent evidence
            return True
        return any(c.has_negation() for c in self.children())


NOf.model_rebuild()
Expr.model_rebuild()


class TemporalRule(_IRModel):
    """Relationship between the qualifying events of two evidence items A and B.

    Holds when SOME pair (event a of A, event b of B) satisfies min_days <= (b_date - a_date) <= max_days
    (calendar days, inclusive; null = unbounded). Relationships are shorthand for the bounds:
      before   : A strictly before B, at most `days` earlier   -> [1, days]   ([0, days] if allow_same_day)
      after    : A strictly after B, at most `days` later      -> [-days, -1] ([-days, 0] if allow_same_day)
      within   : A and B within `days` of each other, any order -> [-days, days]
      same_day : same calendar day                              -> [0, 0]
      between  : explicit signed bounds min_days..max_days
    `days` omitted for before/after = no upper limit. required=True makes it a cohort entry step;
    otherwise it can be referenced from expressions as {temporal: id}.
    """

    id: str = Field(pattern=NAME_PATTERN)
    name: str = Field(min_length=1, max_length=300)
    a: str = Field(pattern=NAME_PATTERN, description="evidence id A")
    b: str = Field(pattern=NAME_PATTERN, description="evidence id B")
    relation: Literal["before", "after", "within", "same_day", "between"]
    days: int | None = Field(default=None, ge=0, le=MAX_DAYS)
    min_days: int | None = Field(default=None, ge=-MAX_DAYS, le=MAX_DAYS)
    max_days: int | None = Field(default=None, ge=-MAX_DAYS, le=MAX_DAYS)
    allow_same_day: bool = False
    required: bool = True

    @model_validator(mode="after")
    def _bounds(self) -> TemporalRule:
        if self.a == self.b:
            raise ValueError(f"temporal rule {self.id!r}: a and b must differ (use max_span_days for repeats)")
        r = self.relation
        if r == "between":
            if self.min_days is None and self.max_days is None:
                raise ValueError(f"temporal rule {self.id!r}: 'between' needs min_days and/or max_days")
            if self.days is not None:
                raise ValueError(f"temporal rule {self.id!r}: 'between' uses min_days/max_days, not days")
        else:
            if self.min_days is not None or self.max_days is not None:
                raise ValueError(f"temporal rule {self.id!r}: min_days/max_days are only for 'between'")
            if r in ("within",) and self.days is None:
                raise ValueError(f"temporal rule {self.id!r}: 'within' needs days")
            if r == "same_day" and self.days is not None:
                raise ValueError(f"temporal rule {self.id!r}: 'same_day' takes no days")
        lo, hi = self.bounds()
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"temporal rule {self.id!r}: empty range [{lo}, {hi}]")
        return self

    def bounds(self) -> tuple[int | None, int | None]:
        d, same = self.days, (0 if self.allow_same_day else 1)
        if self.relation == "before":
            return same, d
        if self.relation == "after":
            return (None if d is None else -d), -same
        if self.relation == "within":
            return (None if d is None else -d), d
        if self.relation == "same_day":
            return 0, 0
        return self.min_days, self.max_days


class Weight(_IRModel):
    ref: Expr
    points: int = Field(ge=-100, le=100)
    label: str = Field(default="", max_length=200)


class Scoring(_IRModel):
    """Deterministic additive rule score: sum of points for each weight whose expression holds.
    It is NOT a probability, sensitivity, specificity, PPV or measure of clinical certainty."""

    name: str = Field(default="evidence score", max_length=100)
    weights: list[Weight] = Field(min_length=1, max_length=100)


class Tier(_IRModel):
    """Evidence tier. Tiers are evaluated in order; a patient gets the FIRST tier whose `rule`
    holds AND whose `min_score` (if any) is reached. Patients matching no tier are not in the cohort."""

    name: str = Field(pattern=NAME_PATTERN)
    label: str = Field(default="", max_length=100)
    description: str = Field(default="", max_length=1000)
    rule: Expr | None = None
    min_score: int | None = Field(default=None, ge=-10_000, le=10_000)

    @model_validator(mode="after")
    def _something(self) -> Tier:
        if self.rule is None and self.min_score is None:
            raise ValueError(f"tier {self.name!r} needs a rule and/or a min_score")
        return self


class Conflict(_IRModel):
    """Evidence that argues against the target. action=flag keeps the patient but marks them;
    action=exclude removes them (shown as an attrition step)."""

    name: str = Field(pattern=NAME_PATTERN)
    label: str = Field(default="", max_length=200)
    rule: Expr
    action: Literal["flag", "exclude"] = "flag"


class FunnelStep(_IRModel):
    name: str = Field(min_length=1, max_length=200)
    rule: Expr


class Target(_IRModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class AcceptanceCriteria(_IRModel):
    """Prespecified performance criteria for ONE intended use of an algorithm.

    There are deliberately no default thresholds: what is acceptable depends on the disease, the
    reference standard and the intended use. Unset criteria are not checked; at least one must be set.
    With use_confidence_lower_bound, a metric passes only if the lower bound of its Wilson interval
    at confidence_level reaches the threshold (stricter for small samples)."""

    description: str = Field(default="", max_length=2000)
    min_sensitivity: float | None = Field(default=None, ge=0, le=1)
    min_ppv: float | None = Field(default=None, ge=0, le=1)
    min_specificity: float | None = Field(default=None, ge=0, le=1)
    min_npv: float | None = Field(default=None, ge=0, le=1)
    min_evaluated: int | None = Field(default=None, ge=1, description="minimum eligible, evaluated patients")
    min_reference_positive: int | None = Field(default=None, ge=1)
    min_reference_negative: int | None = Field(default=None, ge=1)
    use_confidence_lower_bound: bool = False
    confidence_level: float = Field(default=0.95, description="0.90, 0.95 or 0.99")

    @field_validator("confidence_level")
    @classmethod
    def _level(cls, v: float) -> float:
        if v not in (0.9, 0.95, 0.99):
            raise ValueError("confidence_level must be 0.90, 0.95 or 0.99")
        return v

    @model_validator(mode="after")
    def _something(self) -> AcceptanceCriteria:
        keys = (
            "min_sensitivity",
            "min_ppv",
            "min_specificity",
            "min_npv",
            "min_evaluated",
            "min_reference_positive",
            "min_reference_negative",
        )
        if all(getattr(self, k) is None for k in keys):
            raise ValueError("acceptance criteria need at least one threshold or minimum sample size")
        return self


class ProxyDefinition(_IRModel):
    """A versioned proxy identification algorithm (immutable once saved; changes need a new version)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    proxy_schema_version: str = PROXY_SCHEMA_VERSION
    ir_schema_version: str = IR_SCHEMA_VERSION
    algorithm_name: str = Field(pattern=NAME_PATTERN)
    version: str = Field(pattern=VERSION_PATTERN)
    classification: Classification = "proxy"
    validation_reference: str | None = Field(
        default=None,
        description="validation_id of a recorded reference-standard validation (required "
        "for classification 'clinically_validated')",
    )
    target: Target
    clinical_notes: str = Field(default="", max_length=10_000)
    dataset_profile: str | None = Field(default=None, description="dataset the algorithm was designed for")
    ontology_version: str
    vocabulary_version: str
    concept_sets: list[ProxyConceptSet] = Field(min_length=1)
    index_event: IndexEvent
    prior_observation_days: int = Field(default=0, ge=0, le=MAX_DAYS)
    post_observation_days: int = Field(default=0, ge=0, le=MAX_DAYS)
    demographics: Demographics = Field(default_factory=Demographics)
    evidence: list[EvidenceCriterion] = Field(min_length=1, max_length=100)
    groups: dict[str, Expr] = Field(default_factory=dict)
    temporal_rules: list[TemporalRule] = Field(default_factory=list, max_length=50)
    entry: Expr | None = Field(default=None, description="additional mandatory rule for every cohort member")
    exclusion: Expr | None = Field(default=None, description="patients for whom this holds are excluded")
    conflicts: list[Conflict] = Field(default_factory=list, max_length=20)
    scoring: Scoring | None = None
    tiers: list[Tier] = Field(min_length=1, max_length=10)
    funnel: list[FunnelStep] = Field(default_factory=list, max_length=20)
    exit: CohortExit = Field(default_factory=CohortExit)
    assumptions: list[str] = Field(default_factory=list)
    acceptance_criteria: dict[str, AcceptanceCriteria] = Field(
        default_factory=dict,
        description="prespecified performance criteria per intended use (not part of the selection logic)",
    )

    @field_validator("acceptance_criteria")
    @classmethod
    def _uses(cls, v: dict[str, AcceptanceCriteria]) -> dict[str, AcceptanceCriteria]:
        for name in v:
            if not re.match(INTENDED_USE_PATTERN, name):
                raise ValueError(f"intended use {name!r} must match {INTENDED_USE_PATTERN}")
        return v

    @field_validator("groups")
    @classmethod
    def _group_names(cls, v: dict[str, Expr]) -> dict[str, Expr]:
        for name in v:
            if not re.match(NAME_PATTERN, name):
                raise ValueError(f"group name {name!r} must match {NAME_PATTERN}")
        return v

    @model_validator(mode="after")
    def _integrity(self) -> ProxyDefinition:
        self._check_concept_set_ids()
        names: list[str] = [e.id for e in self.evidence] + list(self.groups) + [t.id for t in self.temporal_rules]
        if dupes := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"evidence, group and temporal rule ids must be unique: {dupes}")
        for kind, items in (("tier", [t.name for t in self.tiers]), ("conflict", [c.name for c in self.conflicts])):
            if dupes := sorted({n for n in items if items.count(n) > 1}):
                raise ValueError(f"duplicate {kind} names: {dupes}")
        cs_ids = {cs.id for cs in self.concept_sets}
        if self.index_event.concept_set_id not in cs_ids:
            raise ValueError(f"index_event references unknown concept set {self.index_event.concept_set_id!r}")
        evidence = {e.id for e in self.evidence}
        for e in self.evidence:
            if e.concept_set_id not in cs_ids:
                raise ValueError(f"evidence {e.id!r} references unknown concept set {e.concept_set_id!r}")
        temporal = {t.id for t in self.temporal_rules}
        for t in self.temporal_rules:
            for side in (t.a, t.b):
                if side not in evidence:
                    raise ValueError(f"temporal rule {t.id!r} references unknown evidence {side!r}")
        for where, expr in self.expressions():
            if expr.depth() > 12:
                raise ValueError(f"{where}: expression nesting is deeper than 12 levels")
            for kind, ref in expr.refs():
                pool = {"evidence": evidence, "group": set(self.groups), "temporal": temporal}[kind]
                if ref not in pool:
                    raise ValueError(f"{where}: unknown {kind} {ref!r}")
            self._check_nof(expr, where)
        self._check_group_cycles()
        if self.classification == "clinically_validated" and not self.validation_reference:
            raise ValueError(
                "classification 'clinically_validated' requires validation_reference "
                "(a recorded validation against an external reference standard)"
            )
        if any(t.min_score is not None for t in self.tiers) and self.scoring is None:
            raise ValueError("tiers use min_score but no scoring is defined")
        return self

    def _check_concept_set_ids(self) -> None:
        """Concept-set ids must be unique: the compiler expands concept sets by id, so a duplicate would
        merge two code lists into one (or depend on list order). Rejected even if the copies are identical."""
        seen: dict[str, ProxyConceptSet] = {}
        identical: set[str] = set()
        conflicting: set[str] = set()
        for cs in self.concept_sets:
            if cs.id in seen:
                (identical if seen[cs.id] == cs else conflicting).add(cs.id)
            seen.setdefault(cs.id, cs)
        problems = []
        if conflicting:
            problems.append(f"conflicting definitions for {sorted(conflicting)}")
        if identical:
            problems.append(f"repeated identical definitions for {sorted(identical)} (remove the copies)")
        if problems:
            raise ValueError(
                "duplicate concept set ids: " + "; ".join(problems) + ". Each concept set id must "
                "be unique so every reference resolves to exactly one code list."
            )

    def _check_nof(self, expr: Expr, where: str) -> None:
        for kind in ("at_least", "at_most", "exactly"):
            nof: NOf | None = getattr(expr, kind)
            if nof is None:
                continue
            m = len(nof.of)
            if kind in ("at_least", "exactly") and nof.n > m:
                raise ValueError(f"{where}: {kind} {nof.n} of {m} items can never be satisfied")
            if nof.within_days is not None:
                if kind != "at_least":
                    raise ValueError(f"{where}: within_days is only supported with at_least")
                if any(ch.kind != "evidence" for ch in nof.of):
                    raise ValueError(f"{where}: within_days needs every item to be an evidence reference")
        for ch in expr.children():
            self._check_nof(ch, where)

    def _check_group_cycles(self) -> None:
        def visit(name: str, stack: tuple[str, ...]) -> None:
            if name in stack:
                raise ValueError(f"evidence groups form a cycle: {' -> '.join((*stack, name))}")
            for kind, ref in self.groups[name].refs():
                if kind == "group":
                    visit(ref, (*stack, name))

        for g in self.groups:
            visit(g, ())

    # ---- helpers ------------------------------------------------------------------------
    def expressions(self) -> list[tuple[str, Expr]]:
        out: list[tuple[str, Expr]] = [(f"group {g!r}", e) for g, e in self.groups.items()]
        if self.entry:
            out.append(("entry", self.entry))
        if self.exclusion:
            out.append(("exclusion", self.exclusion))
        out += [(f"conflict {c.name!r}", c.rule) for c in self.conflicts]
        out += [(f"tier {t.name!r}", t.rule) for t in self.tiers if t.rule is not None]
        if self.scoring:
            out += [(f"score weight {i + 1}", w.ref) for i, w in enumerate(self.scoring.weights)]
        out += [(f"funnel step {f.name!r}", f.rule) for f in self.funnel]
        return out

    def evidence_by_id(self, eid: str) -> EvidenceCriterion:
        return next(e for e in self.evidence if e.id == eid)

    def concept_set(self, cs_id: str) -> ProxyConceptSet:
        matches = [cs for cs in self.concept_sets if cs.id == cs_id]
        if len(matches) != 1:  # unreachable for a validated definition; kept as a guard
            raise KeyError(f"concept set {cs_id!r} resolves to {len(matches)} definitions")
        return matches[0]

    def concept_set_versions(self) -> dict[str, str]:
        """Version provenance of every concept set: its declared version, else curated key@version,
        else 'resolved@<vocabulary version>'."""
        out = {}
        for cs in sorted(self.concept_sets, key=lambda c: c.id):
            base = cs.source if cs.source != "resolved" else f"resolved@{self.vocabulary_version}"
            out[cs.id] = f"{cs.version} ({base})" if cs.version else base
        return out

    def evidence_roles(self) -> dict[str, list[str]]:
        """Where each evidence item is used: entry (mandatory for every member), exclusion, conflict,
        tier, score, temporal (required = mandatory), funnel. Evidence used only in tiers/scores is
        supporting evidence; it is never treated as mandatory unless the definition says so."""
        roles: dict[str, set[str]] = {e.id: set() for e in self.evidence}

        def mark(expr: Expr | None, role: str, stack: tuple[str, ...] = ()) -> None:
            if expr is None:
                return
            for kind, ref in expr.refs():
                if kind == "evidence":
                    roles[ref].add(role)
                elif kind == "group" and ref not in stack:
                    mark(self.groups[ref], role, (*stack, ref))
                elif kind == "temporal":
                    t = next(t for t in self.temporal_rules if t.id == ref)
                    roles[t.a].add(role)
                    roles[t.b].add(role)

        mark(self.entry, "entry")
        mark(self.exclusion, "exclusion")
        for c in self.conflicts:
            mark(c.rule, f"conflict_{c.action}")
        for t in self.tiers:
            mark(t.rule, "tier")
        if self.scoring:
            for w in self.scoring.weights:
                mark(w.ref, "score")
        for f in self.funnel:
            mark(f.rule, "funnel")
        for t in self.temporal_rules:
            for side in (t.a, t.b):
                roles[side].add("temporal_required" if t.required else "temporal")
        return {k: sorted(v) for k, v in roles.items()}

    def mandatory_evidence(self) -> set[str]:
        """Evidence every member must have (referenced from entry or a required temporal rule)."""
        return {e for e, r in self.evidence_roles().items() if "entry" in r or "temporal_required" in r}

    @property
    def label(self) -> str:
        return f"{CLASSIFICATION_LABELS[self.classification]} for {self.target.name}"

    # ---- hashing ---------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def semantic_form(self) -> dict[str, Any]:
        """Canonical selection logic (the semantic-hash contract, version SEMANTIC_HASH_VERSION).

        Included: everything that decides who is selected, which tier/score they get and which output
        keys exist - index event, observation, demographics, every evidence rule (entity, inlined concept
        items, window, counts, spans, filters, category, required), evidence/group/temporal ids (they name
        output columns), temporal bounds, entry/exclusion, conflicts (name + action), score weights, tier
        names/rules/min_score IN ORDER (first match wins), funnel rules IN ORDER, exit.

        Excluded: names, labels, descriptions, notes, target, classification, versions, dataset_profile,
        assumptions, provenance, acceptance criteria, concept-set ids/names/sources (concepts are inlined).

        Canonicalized (order or spelling carries no meaning): concept items (sorted, de-duplicated),
        concept sets, evidence, temporal rules, conflicts and score weights (sorted), AND/OR children
        (sorted, de-duplicated), N-of-M items (sorted, duplicates kept because they count), attribute
        code lists, gender ids, and temporal relations (reduced to their [min, max] day bounds, so
        'before 30 days, same day allowed' equals 'between 0 and 30')."""

        def canon(x: Any) -> str:
            return json.dumps(x, sort_keys=True, separators=(",", ":"))

        def items(cs_id: str) -> list[list[Any]]:
            uniq = {(i.concept_id, i.include_descendants, i.is_excluded) for i in self.concept_set(cs_id).items}
            return [list(t) for t in sorted(uniq)]

        def expr(e: Expr | None) -> Any:
            if e is None:
                return None
            k = e.kind
            if k in ("evidence", "group", "temporal"):
                return {k: getattr(e, k)}
            if k in ("all", "any"):
                kids = sorted({canon(expr(c)) for c in e.children()})
                return {k: [json.loads(c) for c in kids]}
            if k == "not_":
                return {"not": expr(e.not_)}
            nof = getattr(e, k)
            return {k: {"n": nof.n, "within_days": nof.within_days, "of": sorted((expr(c) for c in nof.of), key=canon)}}

        def claims(x: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            if x.claim_status is not None:
                out["claim_status"] = sorted(set(x.claim_status))
            if x.dx_position is not None:
                out["dx_position"] = x.dx_position
            return out

        def value(x: Any) -> Any:
            return x.value_filter.model_dump(mode="json", exclude={"original_text"}) if x.value_filter else None

        def evidence(e: EvidenceCriterion) -> dict[str, Any]:
            return {
                "id": e.id,
                "category": e.category,
                "entity": e.entity,
                "concepts": items(e.concept_set_id),
                "window": e.window.model_dump(mode="json"),
                "occurrence": e.occurrence,
                "count": e.count,
                "count_by": e.count_by,
                "min_span_days": e.min_span_days,
                "max_span_days": e.max_span_days,
                "required": e.required,
                "value": value(e),
                "place_of_service": sorted(e.place_of_service or []),
                "provider_specialty": sorted(e.provider_specialty or []),
                **claims(e),
            }

        ie = self.index_event
        return {
            "index": {
                "entity": ie.entity,
                "concepts": items(ie.concept_set_id),
                "first_only": ie.first_occurrence_only,
                "value": value(ie),
                **claims(ie),
            },
            "prior_obs": self.prior_observation_days,
            "post_obs": self.post_observation_days,
            "demographics": {
                **self.demographics.model_dump(mode="json"),
                "gender_concept_ids": sorted(self.demographics.gender_concept_ids),
            },
            "evidence": sorted((evidence(e) for e in self.evidence), key=lambda d: d["id"]),
            "groups": {g: expr(e) for g, e in sorted(self.groups.items())},
            "temporal": sorted(
                (
                    {"id": t.id, "a": t.a, "b": t.b, "bounds": list(t.bounds()), "required": t.required}
                    for t in self.temporal_rules
                ),
                key=lambda d: d["id"],
            ),
            "entry": expr(self.entry),
            "exclusion": expr(self.exclusion),
            "conflicts": sorted(
                ({"name": c.name, "action": c.action, "rule": expr(c.rule)} for c in self.conflicts),
                key=lambda d: d["name"],
            ),
            "weights": sorted(({"rule": expr(w.ref), "points": w.points} for w in self.scoring.weights), key=canon)
            if self.scoring
            else None,
            "tiers": [{"name": t.name, "rule": expr(t.rule), "min_score": t.min_score} for t in self.tiers],
            "funnel": [expr(f.rule) for f in self.funnel],
            "exit": self.exit.model_dump(mode="json"),
        }

    def semantic_hash(self) -> str:
        """Hash of `semantic_form()` (logic only; see its docstring for what counts). Pure: never mutates."""
        payload = {"contract": SEMANTIC_HASH_VERSION, "form": self.semantic_form()}
        return (
            "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        )

    # ---- (de)serialization ---------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, text: str, **defaults: Any) -> ProxyDefinition:
        data = load_yaml_strict(text) or {}
        if not isinstance(data, dict):
            raise ValueError("a proxy definition must be a YAML mapping")
        for k, v in defaults.items():
            data.setdefault(k, v)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str, **defaults: Any) -> ProxyDefinition:
        data = load_json_strict(text)
        if not isinstance(data, dict):
            raise ValueError("a proxy definition must be a JSON object")
        for k, v in defaults.items():
            data.setdefault(k, v)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Path, **defaults: Any) -> ProxyDefinition:
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path.read_text(), **defaults)
        return cls.from_json(path.read_text(), **defaults)


def is_proxy_payload(data: dict[str, Any]) -> bool:
    return "algorithm_name" in data and "tiers" in data


def provenance_summary(p: ProxyDefinition) -> dict[str, Any]:
    """Provenance and role of every evidence item and concept set (for packets, compile metadata, reports)."""
    roles = p.evidence_roles()
    return {
        "evidence": [
            {
                "id": e.id,
                "category": e.category,
                "dataset_required": e.required,
                "roles": roles[e.id],
                "mandatory_for_membership": e.id in p.mandatory_evidence(),
                "provenance": e.provenance.model_dump(mode="json") if e.provenance else None,
            }
            for e in p.evidence
        ],
        "concept_sets": [
            {
                "id": cs.id,
                "code_system": cs.code_system,
                "code_system_version": cs.code_system_version,
                "version": cs.version,
                "source": cs.source,
                "effective_from": cs.effective_from.isoformat() if cs.effective_from else None,
                "effective_to": cs.effective_to.isoformat() if cs.effective_to else None,
                "provenance": cs.provenance.model_dump(mode="json") if cs.provenance else None,
            }
            for cs in p.concept_sets
        ],
    }


class DuplicateKeyError(ValueError):
    """YAML/JSON input repeats a mapping key (the second value would silently replace the first)."""


class _StrictLoader(yaml.SafeLoader):
    pass


def _strict_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    out: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in out:
            line = key_node.start_mark.line + 1
            raise DuplicateKeyError(f"duplicate key {key!r} at line {line}; every key must appear once")
        out[key] = loader.construct_object(value_node, deep=deep)
    return out


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping)


def load_yaml_strict(text: str) -> Any:
    """yaml.safe_load that rejects duplicate mapping keys at any depth."""
    return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def load_json_strict(text: str) -> Any:
    """json.loads that rejects duplicate object keys at any depth."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise DuplicateKeyError(f"duplicate key {k!r}; every key must appear once")
            out[k] = v
        return out

    return json.loads(text, object_pairs_hook=pairs)
