"""Cohort definition intermediate representation (IR).

The IR is the artifact of record: it is stored, reviewed, diffed, hashed and
compiled. Agents produce it; the deterministic compiler consumes it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from . import IR_SCHEMA_VERSION

Entity = Literal["ConditionOccurrence", "DrugExposure", "Measurement", "ProcedureOccurrence", "VisitOccurrence"]
ClaimStatus = Literal["paid", "rejected", "reversed"]
Domain = Literal["Condition", "Drug", "Measurement", "Procedure", "Visit"]


class ConceptSetItem(BaseModel):
    concept_id: int
    include_descendants: bool = True
    is_excluded: bool = False


class ConceptSet(BaseModel):
    id: str = Field(description="Short identifier referenced by criteria, e.g. 'cs_t2dm'")
    name: str
    domain: Domain
    items: list[ConceptSetItem] = Field(min_length=1)
    source: str = Field(default="resolved", description="'curated:<key>@v<version>' or 'resolved'")


class ValueFilter(BaseModel):
    op: Literal[">", ">=", "<", "<=", "=", "between"]
    value: float
    value_high: float | None = None
    unit_concept_id: int
    original_text: str | None = Field(default=None, description="Threshold as the user wrote it, if converted")

    @model_validator(mode="after")
    def _between(self) -> "ValueFilter":
        if self.op == "between" and self.value_high is None:
            raise ValueError("op 'between' requires value_high")
        return self


class Window(BaseModel):
    """Days relative to the index date; null = bounded by the observation period."""
    start_days: int | None = None
    end_days: int | None = None

    @model_validator(mode="after")
    def _order(self) -> "Window":
        if self.start_days is not None and self.end_days is not None and self.start_days > self.end_days:
            raise ValueError("window start_days must be <= end_days")
        return self


class Criterion(BaseModel):
    id: str
    name: str
    entity: Entity
    concept_set_id: str
    window: Window = Field(default_factory=Window)
    occurrence: Literal["at_least", "at_most", "exactly"] = "at_least"
    count: int = Field(default=1, ge=0)
    value_filter: ValueFilter | None = None
    # claims attributes (only on datasets whose profile supports them)
    claim_status: list[ClaimStatus] | None = Field(
        default=None, description="DrugExposure on claims data: which adjudication outcomes count")
    dx_position: Literal["primary", "any"] | None = Field(
        default=None, description="ConditionOccurrence on claims data: primary = first-listed diagnosis")
    min_span_days: int | None = Field(
        default=None, ge=1, description="With at_least N: qualifying events must span >= this many days")


class IndexEvent(BaseModel):
    entity: Entity
    concept_set_id: str
    first_occurrence_only: bool = True
    value_filter: ValueFilter | None = None
    claim_status: list[ClaimStatus] | None = None
    dx_position: Literal["primary", "any"] | None = None


class Demographics(BaseModel):
    age_min: int | None = None
    age_max: int | None = None
    gender_concept_ids: list[int] = Field(default_factory=list)


class CohortExit(BaseModel):
    type: Literal["end_of_observation", "fixed_days"] = "end_of_observation"
    days: int | None = None


class CohortDefinition(BaseModel):
    schema_version: str = IR_SCHEMA_VERSION
    ontology_version: str
    vocabulary_version: str
    name: str
    description: str = ""
    concept_sets: list[ConceptSet]
    index_event: IndexEvent
    prior_observation_days: int = 0
    post_observation_days: int = 0
    demographics: Demographics = Field(default_factory=Demographics)
    inclusion: list[Criterion] = Field(default_factory=list)
    exclusion: list[Criterion] = Field(default_factory=list)
    exit: CohortExit = Field(default_factory=CohortExit)
    assumptions: list[str] = Field(default_factory=list,
                                   description="Interpretation choices made while building, for reviewers")

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
