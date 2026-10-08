"""Cohort definition intermediate representation (IR).

The IR is the artifact of record: it is stored, reviewed, diffed, hashed and
compiled. Agents produce it; the deterministic compiler consumes it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import IR_SCHEMA_VERSION

Entity = Literal["ConditionOccurrence", "DrugExposure", "Measurement", "ProcedureOccurrence", "VisitOccurrence"]
ClaimStatus = Literal["paid", "rejected", "reversed"]
Domain = Literal["Condition", "Drug", "Measurement", "Procedure", "Visit"]

MAX_DAYS = 36_600          # ~100 years: bound for windows and observation requirements
MAX_AGE = 150
MAX_COUNT = 10_000
ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"   # concept set / criterion ids end up in SQL literals


class _IRModel(BaseModel):
    # validate on assignment too, so in-place edits cannot produce an invalid IR
    model_config = ConfigDict(validate_assignment=True)


class ConceptSetItem(_IRModel):
    concept_id: int = Field(gt=0, description="Concept id (0 = 'no matching concept' is not allowed)")
    include_descendants: bool = True
    is_excluded: bool = False


class ConceptSet(_IRModel):
    id: str = Field(pattern=ID_PATTERN, description="Short identifier referenced by criteria, e.g. 'cs_t2dm'")
    name: str = Field(min_length=1, max_length=300)
    domain: Domain
    items: list[ConceptSetItem] = Field(min_length=1, max_length=5000)
    source: str = Field(default="resolved", description="'curated:<key>@v<version>' or 'resolved'")

    @model_validator(mode="after")
    def _items(self) -> "ConceptSet":
        if all(i.is_excluded for i in self.items):
            raise ValueError(f"concept set {self.id!r} has only excluded items; add at least one included concept")
        included = {i.concept_id for i in self.items if not i.is_excluded}
        excluded = {i.concept_id for i in self.items if i.is_excluded}
        if both := sorted(included & excluded):
            raise ValueError(f"concept set {self.id!r}: concept(s) {both} are both included and excluded")
        return self


class ValueFilter(_IRModel):
    """Numeric filter on a measurement value, in `unit_concept_id`. Bounds of `between` are inclusive."""
    model_config = ConfigDict(validate_assignment=True, allow_inf_nan=False)
    op: Literal[">", ">=", "<", "<=", "=", "between"]
    value: float
    value_high: float | None = None
    unit_concept_id: int = Field(gt=0)
    original_text: str | None = Field(default=None, description="Threshold as the user wrote it, if converted")

    @model_validator(mode="after")
    def _bounds(self) -> "ValueFilter":
        if self.op == "between":
            if self.value_high is None:
                raise ValueError("op 'between' requires value_high (the upper bound)")
            if self.value > self.value_high:
                raise ValueError(f"between: lower bound {self.value} is greater than upper bound {self.value_high}")
        elif self.value_high is not None:
            raise ValueError(f"value_high is only used with op 'between', not {self.op!r}")
        return self


class Window(_IRModel):
    """Days relative to the index date, BOTH ENDS INCLUSIVE: start_days <= event_day - index_day <= end_days.

    null start/end = bounded by the person's observation period that contains the index date.
    Example: {start_days: -365, end_days: 0} = the 365 days before index plus the index day itself.
    """
    start_days: int | None = Field(default=None, ge=-MAX_DAYS, le=MAX_DAYS)
    end_days: int | None = Field(default=None, ge=-MAX_DAYS, le=MAX_DAYS)

    @model_validator(mode="after")
    def _order(self) -> "Window":
        if self.start_days is not None and self.end_days is not None and self.start_days > self.end_days:
            raise ValueError(f"window start_days ({self.start_days}) must be <= end_days ({self.end_days})")
        return self


def _claim_status_list(v: list[str] | None) -> list[str] | None:
    if v is None:
        return None
    if not v:
        raise ValueError("claim_status must list at least one status, or be null for the dataset default")
    return sorted(set(v))


class Criterion(_IRModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=300)
    entity: Entity
    concept_set_id: str = Field(pattern=ID_PATTERN)
    window: Window = Field(default_factory=Window)
    occurrence: Literal["at_least", "at_most", "exactly"] = "at_least"
    count: int = Field(default=1, ge=0, le=MAX_COUNT,
                       description="Number of qualifying events; at_most/exactly 0 = 'none'")
    count_by: Literal["records", "dates"] = Field(
        default="records", description="What `count` counts: every qualifying record (default, OHDSI-style), or "
                                       "distinct event dates (several records on one day count once)")
    value_filter: ValueFilter | None = None
    # claims attributes (only on datasets whose profile supports them)
    claim_status: list[ClaimStatus] | None = Field(
        default=None, description="DrugExposure on claims data: which adjudication outcomes count")
    dx_position: Literal["primary", "any"] | None = Field(
        default=None, description="ConditionOccurrence on claims data: primary = first-listed diagnosis")
    min_span_days: int | None = Field(
        default=None, ge=1, le=MAX_DAYS, description="With at_least N: qualifying events must span >= this many days")

    @field_validator("claim_status")
    @classmethod
    def _status(cls, v: list[str] | None) -> list[str] | None:
        return _claim_status_list(v)

    @model_validator(mode="after")
    def _occurrence(self) -> "Criterion":
        where = f"criterion {self.id!r} ({self.name})"
        if self.occurrence == "at_least" and self.count == 0:
            raise ValueError(f"{where}: 'at_least 0' is always true; use count >= 1, or at_most/exactly 0 for 'none'")
        if self.min_span_days is not None and (self.occurrence != "at_least" or self.count < 2):
            raise ValueError(f"{where}: min_span_days needs occurrence 'at_least' with count >= 2")
        return self


class IndexEvent(_IRModel):
    entity: Entity
    concept_set_id: str = Field(pattern=ID_PATTERN)
    first_occurrence_only: bool = True
    value_filter: ValueFilter | None = None
    claim_status: list[ClaimStatus] | None = None
    dx_position: Literal["primary", "any"] | None = None

    @field_validator("claim_status")
    @classmethod
    def _status(cls, v: list[str] | None) -> list[str] | None:
        return _claim_status_list(v)


class Demographics(_IRModel):
    """Age at index = calendar year of index minus year of birth. Bounds are inclusive; null = open-ended."""
    age_min: int | None = Field(default=None, ge=0, le=MAX_AGE)
    age_max: int | None = Field(default=None, ge=0, le=MAX_AGE)
    gender_concept_ids: list[int] = Field(default_factory=list)

    @field_validator("gender_concept_ids")
    @classmethod
    def _genders(cls, v: list[int]) -> list[int]:
        if any(g <= 0 for g in v):
            raise ValueError("gender_concept_ids must be positive concept ids")
        return sorted(set(v))

    @model_validator(mode="after")
    def _ages(self) -> "Demographics":
        if self.age_min is not None and self.age_max is not None and self.age_min > self.age_max:
            raise ValueError(f"age_min ({self.age_min}) must be <= age_max ({self.age_max})")
        return self


class CohortExit(_IRModel):
    type: Literal["end_of_observation", "fixed_days"] = "end_of_observation"
    days: int | None = Field(default=None, ge=1, le=MAX_DAYS)

    @model_validator(mode="after")
    def _days(self) -> "CohortExit":
        if self.type == "fixed_days" and self.days is None:
            raise ValueError("exit type 'fixed_days' requires days >= 1")
        if self.type == "end_of_observation" and self.days is not None:
            raise ValueError("exit 'days' is only used with type 'fixed_days'")
        return self


class CohortDefinition(_IRModel):
    schema_version: str = IR_SCHEMA_VERSION
    ontology_version: str
    vocabulary_version: str
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=5000)
    concept_sets: list[ConceptSet] = Field(min_length=1)
    index_event: IndexEvent
    prior_observation_days: int = Field(default=0, ge=0, le=MAX_DAYS)
    post_observation_days: int = Field(default=0, ge=0, le=MAX_DAYS)
    demographics: Demographics = Field(default_factory=Demographics)
    inclusion: list[Criterion] = Field(default_factory=list, max_length=50)
    exclusion: list[Criterion] = Field(default_factory=list, max_length=50)
    exit: CohortExit = Field(default_factory=CohortExit)
    assumptions: list[str] = Field(default_factory=list,
                                   description="Interpretation choices made while building, for reviewers")

    @model_validator(mode="after")
    def _references(self) -> "CohortDefinition":
        ids = [cs.id for cs in self.concept_sets]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate concept set ids: {dupes}")
        crit_ids = [c.id for c in self.inclusion + self.exclusion]
        if dupes := sorted({i for i in crit_ids if crit_ids.count(i) > 1}):
            raise ValueError(f"duplicate criterion ids: {dupes}")
        known = set(ids)
        if self.index_event.concept_set_id not in known:
            raise ValueError(f"index_event references unknown concept set {self.index_event.concept_set_id!r}")
        for role, crits in (("inclusion", self.inclusion), ("exclusion", self.exclusion)):
            for c in crits:
                if c.concept_set_id not in known:
                    raise ValueError(f"{role} criterion {c.id!r} references unknown concept set {c.concept_set_id!r}")
        return self

    def concept_set(self, cs_id: str) -> ConceptSet:
        for cs in self.concept_sets:
            if cs.id == cs_id:
                return cs
        raise KeyError(cs_id)

    # ---- hashing ----------------------------------------------------------
    def canonical_json(self) -> str:
        """Full canonical form: sorted keys, sorted concept-set items."""
        data = self.model_dump(mode="json")
        data["concept_sets"] = sorted(
            [{**cs, "items": sorted(cs["items"], key=lambda i: (i["concept_id"], i["is_excluded"]))}
             for cs in data["concept_sets"]], key=lambda c: c["id"])
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def semantic_form(self) -> dict:
        """Logic only: labels, ids, assumptions and criterion order removed; concept sets inlined.
        Two definitions with the same semantic hash select the same patients."""
        def items(cs_id: str) -> list:
            cs = self.concept_set(cs_id)
            return sorted([[i.concept_id, i.include_descendants, i.is_excluded] for i in cs.items])

        def crit(c: Criterion) -> dict:
            return {
                "entity": c.entity, "concepts": items(c.concept_set_id), "window": c.window.model_dump(),
                "occurrence": c.occurrence, "count": c.count,
                "value": c.value_filter.model_dump(exclude={"original_text"}) if c.value_filter else None,
                **claims(c),
            }

        def claims(x) -> dict:
            out = {}
            if x.claim_status is not None:
                out["claim_status"] = sorted(set(x.claim_status))
            if x.dx_position is not None:
                out["dx_position"] = x.dx_position
            if getattr(x, "min_span_days", None) is not None:
                out["min_span_days"] = x.min_span_days
            if getattr(x, "count_by", "records") != "records":
                out["count_by"] = x.count_by
            return out

        def key(d: dict) -> str:
            return json.dumps(d, sort_keys=True)

        return {
            "index": {
                "entity": self.index_event.entity, "concepts": items(self.index_event.concept_set_id),
                "first_only": self.index_event.first_occurrence_only,
                "value": self.index_event.value_filter.model_dump(exclude={"original_text"})
                if self.index_event.value_filter else None,
                **claims(self.index_event),
            },
            "prior_obs": self.prior_observation_days,
            "post_obs": self.post_observation_days,
            "demographics": {**self.demographics.model_dump(),
                             "gender_concept_ids": sorted(self.demographics.gender_concept_ids)},
            "inclusion": sorted((crit(c) for c in self.inclusion), key=key),
            "exclusion": sorted((crit(c) for c in self.exclusion), key=key),
            "exit": self.exit.model_dump(),
        }

    def semantic_hash(self) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(self.semantic_form(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
