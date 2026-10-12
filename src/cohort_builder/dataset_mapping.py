"""Source-to-canonical mapping for imported (CSV) datasets, and the dataset profile it produces.

An imported table is NEVER treated as OMOP just because of its file or column names. A mapping file
(YAML, versioned, validated) states explicitly which staged table and column play which role:

    mapping_version: 1
    description: ...
    data_type: ehr | claims | registry | other
    person:       {table, person_column, gender_column?, gender_values? | gender_is_concept_id?,
                   year_of_birth_column? | birth_date_column?}
    observation:  {mode: table | activity | none, absence_inference: weak | observed_period | supported,
                   table/person_column/start_column/end_column (table mode), max_gap_days (activity mode)}
    entities:     {<ontology entity>: {table, person_column, start_column, end_column?, date_format?,
                   concept_id_column | (code_column + vocabulary), value_column?, unit_concept_id_column |
                   (unit_column + unit_values)}}
    relationships: [{from_table, from_column, to_table, to_column}]

The mapping is checked against the ACTUAL staged schema (tables, columns, types), and it is compiled
into semantic views (like the IQVIA LAAD profile) that give the existing compiler the column names it
expects. Only validated identifiers, integers and strictly-filtered literals reach SQL; the mapping is
never executed as SQL.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .csv_ingest import COLUMN_PATTERN, NAME_PATTERN, qi
from .ir import Entity
from .ontology import Ontology

MAPPING_VERSION = 1
_VOCAB = re.compile(r"^[A-Za-z0-9_]{1,20}$")
_LITERAL = re.compile(r"^[A-Za-z0-9 %/\-_.^*()\[\]µ]{1,40}$")  # gender/unit source values
_DATE_FORMAT = re.compile(r"^(%[YmdyHMSbB]|[-/ .:T])+$")
ALLOWED_GENDERS = {8507, 8532, 0}


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _col(v: str | None) -> str | None:
    if v is not None and not COLUMN_PATTERN.match(v):
        raise ValueError(f"{v!r} is not a valid (normalized) column name")
    return v


def _tab(v: str) -> str:
    if not NAME_PATTERN.match(v):
        raise ValueError(f"{v!r} is not a valid table name")
    return v


class PersonMapping(_M):
    table: str
    person_column: str
    gender_column: str | None = None
    gender_values: dict[str, int] | None = Field(default=None, description="source value -> 8507 / 8532 / 0")
    gender_is_concept_id: bool = False
    year_of_birth_column: str | None = None
    birth_date_column: str | None = None

    @field_validator("table")
    @classmethod
    def _table(cls, v: str) -> str:
        return _tab(v)

    @field_validator("person_column", "gender_column", "year_of_birth_column", "birth_date_column")
    @classmethod
    def _columns(cls, v: str | None) -> str | None:
        return _col(v)

    @model_validator(mode="after")
    def _check(self) -> PersonMapping:
        if self.year_of_birth_column and self.birth_date_column:
            raise ValueError("person: give year_of_birth_column or birth_date_column, not both")
        if self.gender_column and not (self.gender_values or self.gender_is_concept_id):
            raise ValueError(
                "person: gender_column needs gender_values (e.g. {M: 8507, F: 8532}) or gender_is_concept_id: true"
            )
        if self.gender_values and self.gender_is_concept_id:
            raise ValueError("person: use gender_values or gender_is_concept_id, not both")
        for k, v in (self.gender_values or {}).items():
            if not _LITERAL.match(k):
                raise ValueError(f"person.gender_values: unsupported source value {k!r}")
            if v not in ALLOWED_GENDERS:
                raise ValueError(
                    f"person.gender_values: {v} is not a gender concept the ontology allows (8507, 8532,"
                    " or 0 for unknown)"
                )
        return self


class ObservationMapping(_M):
    mode: Literal["table", "activity", "none"]
    absence_inference: Literal["weak", "observed_period", "supported"]
    table: str | None = None
    person_column: str | None = None
    start_column: str | None = None
    end_column: str | None = None
    date_format: str | None = None
    max_gap_days: int | None = Field(default=None, ge=1, le=3650)

    @field_validator("person_column", "start_column", "end_column")
    @classmethod
    def _columns(cls, v: str | None) -> str | None:
        return _col(v)

    @model_validator(mode="after")
    def _check(self) -> ObservationMapping:
        if self.mode == "table" and not all((self.table, self.person_column, self.start_column, self.end_column)):
            raise ValueError("observation mode 'table' needs table, person_column, start_column and end_column")
        if self.mode == "activity" and self.max_gap_days is None:
            raise ValueError("observation mode 'activity' needs max_gap_days (state it explicitly)")
        if self.table:
            _tab(self.table)
        if self.date_format and not _DATE_FORMAT.match(self.date_format):
            raise ValueError(f"unsupported date_format {self.date_format!r}")
        return self


class EntityMapping(_M):
    table: str
    person_column: str
    start_column: str
    end_column: str | None = None
    date_format: str | None = Field(default=None, description="strptime format for text dates, e.g. %m/%d/%Y")
    concept_id_column: str | None = Field(default=None, description="column already holding standard concept ids")
    code_column: str | None = Field(default=None, description="source codes mapped through the vocabulary")
    vocabulary: str | None = Field(default=None, description="vocabulary_id of code_column, e.g. ICD10CM, NDC")
    value_column: str | None = None
    unit_concept_id_column: str | None = None
    unit_column: str | None = None
    unit_values: dict[str, int] | None = None

    @field_validator("table")
    @classmethod
    def _table(cls, v: str) -> str:
        return _tab(v)

    @field_validator(
        "person_column",
        "start_column",
        "end_column",
        "concept_id_column",
        "code_column",
        "value_column",
        "unit_concept_id_column",
        "unit_column",
    )
    @classmethod
    def _columns(cls, v: str | None) -> str | None:
        return _col(v)

    @model_validator(mode="after")
    def _check(self) -> EntityMapping:
        if (self.concept_id_column is None) == (self.code_column is None):
            raise ValueError("give exactly one of concept_id_column or code_column (+ vocabulary)")
        if self.code_column and not (self.vocabulary and _VOCAB.match(self.vocabulary)):
            raise ValueError("code_column needs a vocabulary id (letters, digits, underscore), e.g. ICD10CM")
        if self.date_format and not _DATE_FORMAT.match(self.date_format):
            raise ValueError(f"unsupported date_format {self.date_format!r}")
        if self.unit_column and not self.unit_values:
            raise ValueError("unit_column needs unit_values (source unit text -> unit concept id)")
        if self.unit_column and self.unit_concept_id_column:
            raise ValueError("use unit_column or unit_concept_id_column, not both")
        for k, v in (self.unit_values or {}).items():
            if not _LITERAL.match(k) or v <= 0:
                raise ValueError(f"unit_values: unsupported entry {k!r}: {v}")
        return self


class Relationship(_M):
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    @field_validator("from_table", "to_table")
    @classmethod
    def _tables(cls, v: str) -> str:
        return _tab(v)

    @field_validator("from_column", "to_column")
    @classmethod
    def _columns(cls, v: str) -> str:
        return _col(v) or v


class DatasetMapping(_M):
    mapping_version: Literal[1] = 1
    description: str = Field(default="", max_length=2000)
    data_type: Literal["ehr", "claims", "registry", "other"] = "other"
    person: PersonMapping
    observation: ObservationMapping
    entities: dict[Entity, EntityMapping] = Field(min_length=1)
    relationships: list[Relationship] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


# ---- validation against the staged schema ---------------------------------------------------------
Schema = dict[str, dict[str, str]]  # table -> column -> type


def validate_mapping(m: DatasetMapping, schema: Schema, ont: Ontology) -> tuple[list[str], list[str]]:
    """Structural checks against the staged tables. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    def col(where: str, table: str | None, column: str | None, allowed: tuple[str, ...] | None = None) -> str | None:
        if table is None or column is None:
            return None
        if table not in schema:
            errors.append(f"{where}: table {table!r} was not imported (available: {sorted(schema)})")
            return None
        if column not in schema[table]:
            errors.append(f"{where}: column {column!r} does not exist in table {table!r}")
            return None
        t = schema[table][column]
        if allowed and t not in allowed:
            errors.append(f"{where}: column {table}.{column} is {t}, expected {' or '.join(allowed)}")
        return t

    def date_col(where: str, table: str, column: str | None, fmt: str | None) -> None:
        t = col(where, table, column)
        if t is None:
            return
        if fmt and t != "VARCHAR":
            errors.append(f"{where}: date_format only applies to text columns ({table}.{column} is {t})")
        elif not fmt and t != "DATE":
            errors.append(f"{where}: {table}.{column} is {t}, not an ISO date; give date_format for text dates")

    person_types: dict[str, str] = {}
    pm = m.person
    t = col("person.person_column", pm.table, pm.person_column, ("BIGINT", "VARCHAR"))
    if t:
        person_types["person"] = t
    if pm.gender_column:
        col(
            "person.gender_column",
            pm.table,
            pm.gender_column,
            ("BIGINT",) if pm.gender_is_concept_id else ("VARCHAR", "BIGINT", "BOOLEAN"),
        )
    else:
        warnings.append("person: no gender column mapped; gender criteria will never match")
    if pm.year_of_birth_column:
        col("person.year_of_birth_column", pm.table, pm.year_of_birth_column, ("BIGINT",))
    elif pm.birth_date_column:
        col("person.birth_date_column", pm.table, pm.birth_date_column, ("DATE",))
    else:
        warnings.append("person: no year of birth mapped; age criteria will never match")

    ob = m.observation
    if ob.mode == "table":
        assert ob.table is not None
        t = col("observation.person_column", ob.table, ob.person_column, ("BIGINT", "VARCHAR"))
        if t:
            person_types["observation"] = t
        date_col("observation.start_column", ob.table, ob.start_column, ob.date_format)
        date_col("observation.end_column", ob.table, ob.end_column, ob.date_format)
    elif ob.mode == "none":
        warnings.append(
            "observation mode 'none': prior/post observation, open-ended windows and 'no record' "
            "criteria cannot be evaluated; the validator will refuse cohort definitions"
        )
    else:
        warnings.append(
            f"observation inferred from activity in the mapped tables (gaps > {ob.max_gap_days} days "
            "split periods); absence of a record is weak evidence"
        )

    for entity, em in m.entities.items():
        where = f"entities.{entity}"
        t = col(f"{where}.person_column", em.table, em.person_column, ("BIGINT", "VARCHAR"))
        if t:
            person_types[entity] = t
        date_col(f"{where}.start_column", em.table, em.start_column, em.date_format)
        if em.end_column:
            date_col(f"{where}.end_column", em.table, em.end_column, em.date_format)
        if em.concept_id_column:
            col(f"{where}.concept_id_column", em.table, em.concept_id_column, ("BIGINT",))
        if em.code_column:
            col(f"{where}.code_column", em.table, em.code_column, ("VARCHAR", "BIGINT"))
        if ont.entity_has_value(entity):
            if not em.value_column:
                errors.append(f"{where}: {entity} needs value_column (numeric results)")
            else:
                col(f"{where}.value_column", em.table, em.value_column, ("BIGINT", "DOUBLE"))
            if em.unit_concept_id_column:
                col(f"{where}.unit_concept_id_column", em.table, em.unit_concept_id_column, ("BIGINT",))
            elif em.unit_column:
                col(f"{where}.unit_column", em.table, em.unit_column, ("VARCHAR",))
            else:
                warnings.append(f"{where}: no unit mapped; value thresholds will never match these results")
        elif em.value_column or em.unit_column or em.unit_concept_id_column:
            errors.append(f"{where}: value/unit columns only apply to entities with values (e.g. Measurement)")
        roles = [c for c in (em.person_column, em.start_column, em.concept_id_column or em.code_column) if c]
        if len(set(roles)) != len(roles):
            warnings.append(f"{where}: the same column is used for more than one role (ambiguous mapping)")

    if len(set(person_types.values())) > 1:
        warnings.append(
            "person identifiers have different types across tables "
            f"({', '.join(f'{k}: {v}' for k, v in sorted(person_types.items()))}); they are compared as "
            "text through a surrogate key"
        )
    declared = {(r.from_table, r.to_table) for r in m.relationships}
    for r in m.relationships:
        a = col(f"relationship {r.from_table}->{r.to_table}", r.from_table, r.from_column)
        b = col(f"relationship {r.from_table}->{r.to_table}", r.to_table, r.to_column)
        if a and b and (a in ("BIGINT", "DOUBLE")) != (b in ("BIGINT", "DOUBLE")):
            warnings.append(
                f"relationship {r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}: "
                f"incompatible types {a} / {b} (compared as text)"
            )
    for entity, em in m.entities.items():
        if em.table != pm.table and (em.table, pm.table) not in declared:
            warnings.append(
                f"entities.{entity}: no relationship declared from {em.table!r} to the person table "
                f"{pm.table!r}; orphan patients are still counted during registration"
            )
    unmapped = sorted(set(ont.entities) - set(m.entities))
    if unmapped:
        warnings.append(f"not mapped (cohort criteria on these will be refused): {', '.join(unmapped)}")
    return errors, warnings


# ---- semantic layer (SQL generated from the validated mapping) ---------------------------------------
def needs_person_map(m: DatasetMapping, schema: Schema) -> bool:
    cols = [(m.person.table, m.person.person_column)] + [(e.table, e.person_column) for e in m.entities.values()]
    if m.observation.mode == "table":
        cols.append((m.observation.table or "", m.observation.person_column or ""))
    return any(schema.get(t, {}).get(c) != "BIGINT" for t, c in cols)


def person_map_sql(m: DatasetMapping, stg: str) -> str:
    """Deterministic surrogate person_id for text identifiers (materialized in the staging schema)."""
    sources = [(m.person.table, m.person.person_column)] + [(e.table, e.person_column) for e in m.entities.values()]
    if m.observation.mode == "table":
        sources.append((m.observation.table or "", m.observation.person_column or ""))
    union = " UNION ".join(
        f"SELECT CAST({qi(c)} AS VARCHAR) AS source_id FROM {qi(stg)}.{qi(t)} WHERE {qi(c)} IS NOT NULL"
        for t, c in sorted(set(sources))
    )
    return (
        f"CREATE TABLE {qi(stg)}.person_map AS SELECT source_id, ROW_NUMBER() OVER (ORDER BY source_id) "
        f"AS person_id FROM ({union})"
    )


def _lit(v: str) -> str:
    if not _LITERAL.match(v):  # already enforced by the model; never interpolate anything else
        raise ValueError("unsafe literal")
    return "'" + v.upper().strip() + "'"


def _date(expr: str, fmt: str | None) -> str:
    return f"CAST(try_strptime({expr}, '{fmt}') AS DATE)" if fmt else expr


def semantic_sql(m: DatasetMapping, stg: str, sem: str, use_person_map: bool, vocab: str = "vocab") -> list[str]:
    """CREATE statements for the semantic views (idempotent; also run when the profile is activated)."""

    def pid(alias: str, column: str) -> tuple[str, str]:
        if use_person_map:
            return (
                "pm.person_id",
                f" JOIN {qi(stg)}.person_map pm ON pm.source_id = CAST({alias}.{qi(column)} AS VARCHAR)",
            )
        return f"{alias}.{qi(column)}", ""

    out = [f"CREATE SCHEMA IF NOT EXISTS {qi(sem)}"]
    vocabs = sorted({e.vocabulary for e in m.entities.values() if e.code_column and e.vocabulary})
    if vocabs:
        in_list = ", ".join(f"'{v}'" for v in vocabs)
        out.append(
            f"CREATE OR REPLACE TABLE {qi(sem)}.code_map AS "
            f"SELECT s.vocabulary_id, upper(replace(s.concept_code, '.', '')) AS code, r.concept_id_2 AS concept_id "
            f"FROM {vocab}.concept s JOIN {vocab}.concept_relationship r ON r.concept_id_1 = s.concept_id "
            f"AND r.relationship_id = 'Maps to' AND r.invalid_reason IS NULL WHERE s.vocabulary_id IN ({in_list}) "
            f"UNION SELECT vocabulary_id, upper(replace(concept_code, '.', '')), concept_id FROM {vocab}.concept "
            f"WHERE standard_concept = 'S' AND invalid_reason IS NULL AND vocabulary_id IN ({in_list})"
        )
    p = m.person
    person_id, join = pid("t", p.person_column)
    if p.gender_is_concept_id and p.gender_column:
        gender = f"CAST(t.{qi(p.gender_column)} AS BIGINT)"
    elif p.gender_values and p.gender_column:
        cases = " ".join(f"WHEN {_lit(k)} THEN {int(v)}" for k, v in sorted(p.gender_values.items()))
        gender = f"CASE upper(trim(CAST(t.{qi(p.gender_column)} AS VARCHAR))) {cases} ELSE 0 END"
    else:
        gender = "0"
    if p.year_of_birth_column:
        yob = f"CAST(t.{qi(p.year_of_birth_column)} AS BIGINT)"
    elif p.birth_date_column:
        yob = f"EXTRACT(YEAR FROM t.{qi(p.birth_date_column)})"
    else:
        yob = "CAST(NULL AS BIGINT)"
    out.append(
        f"CREATE OR REPLACE VIEW {qi(sem)}.person AS SELECT DISTINCT {person_id} AS person_id, {gender} AS "
        f"gender_concept_id, {yob} AS year_of_birth FROM {qi(stg)}.{qi(p.table)} t{join} "
        f"WHERE t.{qi(p.person_column)} IS NOT NULL"
    )
    for entity, e in sorted(m.entities.items()):
        person_id, join = pid("t", e.person_column)
        start = _date(f"t.{qi(e.start_column)}", e.date_format)
        end = _date(f"t.{qi(e.end_column)}", e.date_format) if e.end_column else start
        cmap = ""
        if e.concept_id_column:
            concept = f"coalesce(CAST(t.{qi(e.concept_id_column)} AS BIGINT), 0)"
            source = f"CAST(t.{qi(e.concept_id_column)} AS VARCHAR)"
        else:
            code = f"upper(replace(CAST(t.{qi(e.code_column or '')} AS VARCHAR), '.', ''))"
            cmap = f" LEFT JOIN {qi(sem)}.code_map c ON c.vocabulary_id = '{e.vocabulary}' AND c.code = {code}"
            concept, source = "coalesce(c.concept_id, 0)", code
        extra = ""
        if e.value_column:
            extra += f", CAST(t.{qi(e.value_column)} AS DOUBLE) AS value_as_number"
            if e.unit_concept_id_column:
                extra += f", CAST(t.{qi(e.unit_concept_id_column)} AS BIGINT) AS unit_concept_id"
            elif e.unit_column and e.unit_values:
                cases = " ".join(f"WHEN {_lit(k)} THEN {int(v)}" for k, v in sorted(e.unit_values.items()))
                extra += f", CASE upper(trim(t.{qi(e.unit_column)})) {cases} ELSE NULL END AS unit_concept_id"
            else:
                extra += ", CAST(NULL AS BIGINT) AS unit_concept_id"
        out.append(
            f"CREATE OR REPLACE VIEW {qi(sem)}.{qi(_view(entity))} AS SELECT {person_id} AS person_id, {concept} AS "
            f"concept_id, {start} AS start_date, {end} AS end_date{extra}, {source} AS source_value "
            f"FROM {qi(stg)}.{qi(e.table)} t{join}{cmap} "
            f"WHERE t.{qi(e.person_column)} IS NOT NULL AND {start} IS NOT NULL"
        )
    ob = m.observation
    if ob.mode == "table":
        person_id, join = pid("t", ob.person_column or "")
        s = _date(f"t.{qi(ob.start_column or '')}", ob.date_format)
        e_ = _date(f"t.{qi(ob.end_column or '')}", ob.date_format)
        out.append(
            f"CREATE OR REPLACE VIEW {qi(sem)}.observation_period AS SELECT {person_id} AS person_id, {s} AS "
            f"observation_period_start_date, {e_} AS observation_period_end_date FROM "
            f"{qi(stg)}.{qi(ob.table or '')} t{join} WHERE {s} IS NOT NULL AND {e_} IS NOT NULL AND {e_} >= {s}"
        )
    elif ob.mode == "activity":
        act = " UNION ".join(
            f"SELECT person_id, start_date AS d FROM {qi(sem)}.{qi(_view(ent))}" for ent in sorted(m.entities)
        )
        gap = int(ob.max_gap_days or 0)
        out.append(
            f"CREATE OR REPLACE VIEW {qi(sem)}.observation_period AS WITH act AS ({act}), "
            f"o AS (SELECT person_id, d, CASE WHEN d - lag(d) OVER (PARTITION BY person_id ORDER BY d) > {gap} "
            f"THEN 1 ELSE 0 END AS brk FROM act), "
            f"seg AS (SELECT person_id, d, sum(brk) OVER (PARTITION BY person_id ORDER BY d) AS segment FROM o) "
            f"SELECT person_id, min(d) AS observation_period_start_date, max(d) AS observation_period_end_date "
            f"FROM seg GROUP BY person_id, segment"
        )
    return out


def _view(entity: str) -> str:
    return re.sub(r"(?<!^)([A-Z])", r"_\1", entity).lower()  # ConditionOccurrence -> condition_occurrence


def build_profile(
    m: DatasetMapping,
    name: str,
    version: int,
    stg: str,
    sem: str,
    use_person_map: bool,
    ont: Ontology,
    source: dict[str, Any],
) -> dict[str, Any]:
    """A regular dataset profile (same schema as ontology/datasets/*.yaml) for the mapped import."""
    entities = sorted(m.entities)
    obs = {"table": "observation_period", "activity": "activity_based", "none": "none"}[m.observation.mode]
    mapping: dict[str, Any] = {
        "vocab_schema": ont.vocab_schema(),
        "person": {
            "table": f"{sem}.person",
            "person_key": "person_id",
            "gender_col": "gender_concept_id",
            "year_of_birth_col": "year_of_birth",
        },
        "entities": {},
    }
    if m.observation.mode != "none":
        mapping["observation_period"] = {
            "table": f"{sem}.observation_period",
            "start_col": "observation_period_start_date",
            "end_col": "observation_period_end_date",
        }
    for entity in entities:
        em: dict[str, Any] = {
            "table": f"{sem}.{_view(entity)}",
            "concept_col": "concept_id",
            "start_col": "start_date",
            "end_col": "end_date",
        }
        if m.entities[entity].value_column:
            em.update(value_col="value_as_number", unit_col="unit_concept_id")
        mapping["entities"][entity] = em
    unmapped = sorted(set(ont.entities) - set(entities))
    notes = [
        "Imported from CSV files; tables are mapped to ontology entities by an explicit, validated mapping.",
        *([f"Not available in this dataset: {', '.join(unmapped)}."] if unmapped else []),
        *m.notes,
    ]
    if m.observation.mode == "activity":
        notes.append("Observation is inferred from recorded activity, not enrollment.")
    snapshot = f"csv:{name} v{version} ({source['content_hash'][:23]})"
    return {
        "name": name,
        "version": f"csv-{version}",
        "description": m.description or f"CSV import {name} v{version}",
        "data_type": m.data_type,
        "source": source,
        "capabilities": {
            "entities": entities,
            "attributes": [],
            "observation": obs,
            "partial_entities": [e for e in entities if ont.entity_has_value(e)],
            "extra_data": [],
            "absence_inference": m.observation.absence_inference,
            "notes": notes,
        },
        "snapshot_sql": f"SELECT '{snapshot}'",
        "mapping": mapping,
        "setup_sql": semantic_sql(m, stg, sem, use_person_map, ont.vocab_schema()),
    }
