"""Proxy (indirect) cohort definitions: identify patients when no single reliable code exists.

A proxy definition combines named *evidence* (diagnoses, procedures, treatments, labs,
pathology, provider/care-setting signals) with nested boolean logic, N-of-M rules,
temporal relationships between evidence, optional deterministic scoring, and ordered
evidence *tiers*. It is typed IR, like `CohortDefinition`: agents may draft it, but only
the deterministic compiler (proxy_compiler.py) turns it into SQL.

Nothing here is disease-specific: histology-defined cancers, rare metabolic diseases or molecular subtypes are
expressed purely as configuration and concept sets.

Wording rules (governance): a proxy definition is a *claims/data-based proxy* or an
*exploratory algorithm*. It is only labelled clinically validated when it references a
recorded validation against an external reference standard (see orchestrator).
The evidence score is a rule score, not a probability, sensitivity, specificity or PPV.
"""

from __future__ import annotations

import hashlib
import json
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
    concept_sets: list[ConceptSet] = Field(min_length=1)
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

    @field_validator("groups")
    @classmethod
    def _group_names(cls, v: dict[str, Expr]) -> dict[str, Expr]:
        import re

        for name in v:
            if not re.match(NAME_PATTERN, name):
                raise ValueError(f"group name {name!r} must match {NAME_PATTERN}")
        return v

    @model_validator(mode="after")
    def _integrity(self) -> ProxyDefinition:
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

    def concept_set(self, cs_id: str) -> ConceptSet:
        return next(cs for cs in self.concept_sets if cs.id == cs_id)

    def concept_set_versions(self) -> dict[str, str]:
        """Version provenance of every concept set (curated key@version, or 'resolved' + vocabulary)."""
        return {
            cs.id: (cs.source if cs.source != "resolved" else f"resolved@{self.vocabulary_version}")
            for cs in sorted(self.concept_sets, key=lambda c: c.id)
        }

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

    def semantic_hash(self) -> str:
        """Logic only: names, labels, notes and descriptions are ignored."""
        data = self.to_dict()
        for k in (
            "algorithm_name",
            "version",
            "classification",
            "validation_reference",
            "target",
            "clinical_notes",
            "dataset_profile",
            "assumptions",
            "ontology_version",
            "vocabulary_version",
        ):
            data.pop(k, None)

        def strip(node: Any) -> Any:
            if isinstance(node, dict):
                return {
                    k: strip(v)
                    for k, v in node.items()
                    if k not in ("name", "label", "description") or not isinstance(v, str)
                }
            if isinstance(node, list):
                return [strip(v) for v in node]
            return node

        return "sha256:" + hashlib.sha256(json.dumps(strip(data), sort_keys=True).encode()).hexdigest()

    # ---- (de)serialization ---------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, text: str, **defaults: Any) -> ProxyDefinition:
        data = yaml.safe_load(text) or {}
        for k, v in defaults.items():
            data.setdefault(k, v)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Path, **defaults: Any) -> ProxyDefinition:
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path.read_text(), **defaults)
        data = json.loads(path.read_text())
        for k, v in defaults.items():
            data.setdefault(k, v)
        return cls.model_validate(data)


def is_proxy_payload(data: dict[str, Any]) -> bool:
    return "algorithm_name" in data and "tiers" in data
