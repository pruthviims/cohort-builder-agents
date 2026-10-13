"""Validated, versioned configuration for the treatment-event engine.

Two kinds of file, both strict YAML (duplicate keys, anchors/aliases and unknown fields refused) and both
content-hashed so every run records exactly which configuration produced it:

* **Treatment source config** (one per dataset, `ontology/treatment/sources/<dataset>.yaml`): which tables
  hold MEDICATION evidence and how their columns map to the canonical treatment event. Dataset-specific
  knowledge (claim status codes, reversal references, code systems, which date a row carries) lives here,
  never in engine code. Evidence type is declared per source (or per row through a value map), never
  guessed: a pharmacy claim is evidence of DISPENSING, not of ingestion; a medical claim with a drug code is
  evidence of ADMINISTRATION billing.
* **Treatment rule set** (one per analytics use case, `ontology/treatment/rules/<name>.yaml`): drug groups
  and every clinical/analytical assumption (gaps, overlap handling, missing days supply, duplicates,
  censoring, transition windows). Nothing has a hidden global default: thresholds must be written out.

Only validated identifiers, integers and enumerated literals ever reach SQL.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .dataset_mapping import CodeNormalization, normalization_for

EVENT_SCHEMA_VERSION = "treatment_event/2"  # /2: observation partitioning, end_status, product_change
EvidenceType = Literal["dispensing", "administration", "prescription", "unknown"]
StatusCategory = Literal["paid", "rejected", "reversal", "adjustment"]
DateSource = Literal["fill_date", "service_date", "administration_date", "prescription_date", "start_date"]

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_TABLE = re.compile(r"^[a-z_][a-z0-9_]{0,62}\.[a-z_][a-z0-9_]{0,62}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_VOCAB = re.compile(r"^[A-Za-z0-9_]{1,20}$")
_VALUE = re.compile(r"^[A-Za-z0-9 _\-.]{1,40}$")  # source code values in status / type maps
FORBIDDEN_SCHEMAS = {"meta", "results", "information_schema", "pg_catalog", "temp", "main"}
MAX_CONFIG_CHARS = 200_000


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ident(v: str | None) -> str | None:
    if v is not None and not _IDENT.match(v):
        raise ValueError(f"{v!r} is not a valid column name")
    return v


def content_hash(model: BaseModel) -> str:
    data = model.model_dump(mode="json")
    if data.get("provenance", "absent") is None:
        data.pop("provenance")  # configs without a provenance block keep the hash they always had
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


ReviewStatus = Literal["illustrative", "unverified", "validated"]


class Provenance(_M):
    """Where a configuration's codes, mappings and thresholds come from, and whether anyone reviewed them.

    illustrative  examples for the synthetic demo data; must never be used for real analyses
    unverified    meant for real use, not yet reviewed
    validated     reviewed: names the reviewer, the date and the review record (evidence)
    """

    review_status: ReviewStatus
    source: str = Field(min_length=1, max_length=500, description="origin of the codes / mappings / thresholds")
    reviewed_by: str | None = Field(default=None, max_length=200)
    reviewed_on: date | None = None
    evidence: str | None = Field(default=None, max_length=500, description="review record / ticket / document")

    @model_validator(mode="after")
    def _review(self) -> Provenance:
        if self.review_status == "validated" and not (self.reviewed_by and self.reviewed_on and self.evidence):
            raise ValueError("review_status 'validated' needs reviewed_by, reviewed_on and evidence")
        return self


# ---- sources ---------------------------------------------------------------------------------------------
class StatusMapping(_M):
    """How a source encodes claim adjudication. Values not listed are 'unknown' status."""

    column: str
    values: dict[str, StatusCategory] = Field(min_length=1)
    reference_column: str | None = Field(
        default=None, description="on reversal / adjustment rows: the record id they reverse or replace"
    )
    reference_scope: Literal["record", "record_and_person"] = Field(
        default="record",
        description="'record': a reversal/adjustment acts on the referenced record id (claim ids unique per "
        "source); 'record_and_person': only when it belongs to the same patient (claim ids unique per patient)",
    )

    @field_validator("column", "reference_column")
    @classmethod
    def _cols(cls, v: str | None) -> str | None:
        return _ident(v)

    @field_validator("values")
    @classmethod
    def _vals(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not _VALUE.match(k):
                raise ValueError(f"unsupported status value {k!r}")
        return v

    @model_validator(mode="after")
    def _ref(self) -> StatusMapping:
        if {"reversal", "adjustment"} & set(self.values.values()) and not self.reference_column:
            raise ValueError("reversal/adjustment statuses need reference_column (the record they act on)")
        return self


class TreatmentSource(_M):
    name: str
    table: str = Field(description="schema.table")
    person_column: str
    record_id_column: str = Field(description="claim / record id: the stable reference every event keeps")
    start_column: str
    start_date_source: DateSource
    end_column: str | None = None
    days_supply_column: str | None = None
    quantity_column: str | None = None
    evidence_type: EvidenceType | None = Field(default=None, description="fixed for every row of this source")
    evidence_type_column: str | None = None
    evidence_type_values: dict[str, EvidenceType] | None = None
    concept_column: str | None = Field(default=None, description="standard drug concept ids")
    code_column: str | None = Field(default=None, description="source codes mapped through the vocabulary")
    vocabulary: str | None = None
    code_normalization: CodeNormalization | None = None
    code_type_column: str | None = Field(default=None, description="per-row code system (e.g. CPT / HCPCS)")
    code_type_values: dict[str, str] | None = Field(default=None, description="code type value -> vocabulary_id")
    status: StatusMapping | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError(f"{v!r} is not a valid source name")
        return v

    @field_validator("table")
    @classmethod
    def _table(cls, v: str) -> str:
        if not _TABLE.match(v):
            raise ValueError(f"{v!r} must be schema.table (lower-case identifiers)")
        if v.split(".")[0] in FORBIDDEN_SCHEMAS:
            raise ValueError(f"{v!r}: internal schemas cannot be treatment sources")
        return v

    @field_validator(
        "person_column",
        "record_id_column",
        "start_column",
        "end_column",
        "days_supply_column",
        "quantity_column",
        "evidence_type_column",
        "concept_column",
        "code_column",
        "code_type_column",
    )
    @classmethod
    def _cols(cls, v: str | None) -> str | None:
        return _ident(v)

    @model_validator(mode="after")
    def _check(self) -> TreatmentSource:
        if (self.evidence_type is None) == (self.evidence_type_column is None):
            raise ValueError(f"{self.name}: give evidence_type, or evidence_type_column + evidence_type_values")
        if self.evidence_type_column and not self.evidence_type_values:
            raise ValueError(f"{self.name}: evidence_type_column needs evidence_type_values")
        for k in self.evidence_type_values or {}:
            if not _VALUE.match(k):
                raise ValueError(f"{self.name}: unsupported evidence type value {k!r}")
        if (self.concept_column is None) == (self.code_column is None):
            raise ValueError(f"{self.name}: give exactly one of concept_column or code_column")
        if self.code_column:
            if (self.vocabulary is None) == (self.code_type_column is None):
                raise ValueError(f"{self.name}: code_column needs vocabulary, or code_type_column + code_type_values")
            vocabs = [self.vocabulary] if self.vocabulary else list((self.code_type_values or {}).values())
            if self.code_type_column and not self.code_type_values:
                raise ValueError(f"{self.name}: code_type_column needs code_type_values")
            for k in self.code_type_values or {}:
                if not _VALUE.match(k):
                    raise ValueError(f"{self.name}: unsupported code type value {k!r}")
            for v in vocabs:
                if not v or not _VOCAB.match(v):
                    raise ValueError(f"{self.name}: invalid vocabulary id {v!r}")
        elif self.vocabulary or self.code_type_column or self.code_normalization:
            raise ValueError(f"{self.name}: vocabulary / code settings only apply to code_column")
        return self

    def vocabularies(self) -> list[tuple[str, str]]:
        """(vocabulary, normalization) pairs this source looks codes up in."""
        vocabs = [self.vocabulary] if self.vocabulary else sorted(set((self.code_type_values or {}).values()))
        return [(v, normalization_for(v, self.code_normalization)) for v in vocabs if v]


class TreatmentSourceConfig(_M):
    config_version: Literal[1] = 1
    dataset: str
    description: str = Field(default="", max_length=2000)
    sources: list[TreatmentSource] = Field(min_length=1, max_length=20)
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _unique(self) -> TreatmentSourceConfig:
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        return self

    def content_hash(self) -> str:
        return content_hash(self)


# ---- rules ------------------------------------------------------------------------------------------------
class DrugGroup(_M):
    """Members are the listed concepts and all their descendants that are ingredients (e.g. an ATC class,
    or ingredients directly). Grouping is by ingredient; brand / product changes stay within a group."""

    concepts: list[int] = Field(min_length=1, max_length=500)
    max_gap_days: int | None = Field(default=None, ge=0, le=3650, description="overrides episode.max_gap_days")
    justification: str = Field(default="", max_length=500)

    @field_validator("concepts")
    @classmethod
    def _ids(cls, v: list[int]) -> list[int]:
        if any(c <= 0 for c in v):
            raise ValueError("concept ids are positive integers")
        return sorted(set(v))


class DaysSupplyRule(_M):
    missing: Literal["exclude", "assume"]
    assume_days: dict[EvidenceType, int] = Field(default_factory=dict)
    min_valid: int = Field(ge=1, le=3650)
    max_valid: int = Field(ge=1, le=3650)

    @model_validator(mode="after")
    def _check(self) -> DaysSupplyRule:
        if self.min_valid > self.max_valid:
            raise ValueError("days_supply.min_valid must be <= max_valid")
        if self.missing == "assume" and not self.assume_days:
            raise ValueError("days_supply.missing 'assume' needs assume_days per evidence type")
        if any(d < 1 or d > 3650 for d in self.assume_days.values()):
            raise ValueError("assume_days must be 1..3650")
        return self


class EpisodeRule(_M):
    max_gap_days: int = Field(ge=0, le=3650, description="uncovered days allowed inside one episode")
    overlap: Literal["shift_forward", "ignore"]
    max_carryover_days: int = Field(ge=0, le=3650, description="cap on supply carried forward (shift_forward)")
    same_day_same_group: Literal["max_supply", "sum_supply"]
    duplicates: Literal["same_record", "same_product_same_day"]


class CensoringRule(_M):
    observation: Literal["dataset", "none"]
    washout_days: int = Field(
        ge=0, le=3650, description="episodes starting sooner after observation start are left-censored"
    )


class TransitionRule(_M):
    add_on_min_overlap_days: int = Field(ge=1, le=3650)
    switch_window_days: int = Field(ge=0, le=3650, description="max days from one group's end to the next start")


class TreatmentRuleSet(_M):
    rule_set_version: Literal[1] = 1
    name: str
    version: str = Field(pattern=r"^[0-9]+(\.[0-9]+){0,2}$")
    description: str = Field(default="", max_length=2000)
    drug_groups: dict[str, DrugGroup] = Field(min_length=1, max_length=100)
    include_claim_status: list[Literal["paid", "not_applicable", "unknown"]] = Field(min_length=1)
    include_evidence_types: list[EvidenceType] = Field(min_length=1)
    days_supply: DaysSupplyRule
    episode: EpisodeRule
    censoring: CensoringRule
    transitions: TransitionRule
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError(f"{v!r} is not a valid rule set name")
        return v

    @field_validator("drug_groups")
    @classmethod
    def _groups(cls, v: dict[str, DrugGroup]) -> dict[str, DrugGroup]:
        for k in v:
            if not _NAME.match(k):
                raise ValueError(f"{k!r} is not a valid drug group name")
        return v

    def gap_for(self, group: str) -> int:
        g = self.drug_groups[group].max_gap_days
        return self.episode.max_gap_days if g is None else g

    def content_hash(self) -> str:
        return content_hash(self)


# ---- loading ---------------------------------------------------------------------------------------------
def _load_yaml(text: str, what: str) -> dict[str, Any]:
    from .proxy import load_yaml_strict

    if len(text) > MAX_CONFIG_CHARS:
        raise ValueError(f"{what} is larger than {MAX_CONFIG_CHARS} characters")
    try:
        events = list(yaml.parse(text))
    except yaml.YAMLError as exc:
        raise ValueError(f"{what} is not valid YAML") from exc
    if any(isinstance(ev, yaml.AliasEvent) or getattr(ev, "anchor", None) for ev in events):
        raise ValueError(f"{what}: YAML anchors and aliases are not allowed")
    data = load_yaml_strict(text)
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be a YAML mapping")
    return data


def load_sources(text: str) -> TreatmentSourceConfig:
    return TreatmentSourceConfig.model_validate(_load_yaml(text, "treatment source config"))


def load_rules(text: str) -> TreatmentRuleSet:
    return TreatmentRuleSet.model_validate(_load_yaml(text, "treatment rule set"))


def sources_path(ontology_dir: Path, dataset: str) -> Path:
    return Path(ontology_dir) / "treatment" / "sources" / f"{dataset}.yaml"


def rules_path(ontology_dir: Path, name: str) -> Path:
    if not _NAME.match(name):
        raise ValueError(f"{name!r} is not a valid rule set name")
    return Path(ontology_dir) / "treatment" / "rules" / f"{name}.yaml"
