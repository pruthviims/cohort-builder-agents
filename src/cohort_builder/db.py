"""DuckDB connection and OMOP CDM v5.4 / vocabulary DDL (subset used by the builder)."""
from __future__ import annotations

from pathlib import Path

import duckdb

VOCAB_DDL = """
CREATE SCHEMA IF NOT EXISTS vocab;
CREATE TABLE IF NOT EXISTS vocab.concept (
  concept_id BIGINT PRIMARY KEY, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR,
  concept_class_id VARCHAR, standard_concept VARCHAR, concept_code VARCHAR,
  valid_start_date DATE, valid_end_date DATE, invalid_reason VARCHAR);
CREATE TABLE IF NOT EXISTS vocab.concept_relationship (
  concept_id_1 BIGINT, concept_id_2 BIGINT, relationship_id VARCHAR,
  valid_start_date DATE, valid_end_date DATE, invalid_reason VARCHAR);
CREATE TABLE IF NOT EXISTS vocab.concept_ancestor (
  ancestor_concept_id BIGINT, descendant_concept_id BIGINT,
  min_levels_of_separation INTEGER, max_levels_of_separation INTEGER);
CREATE TABLE IF NOT EXISTS vocab.concept_synonym (
  concept_id BIGINT, concept_synonym_name VARCHAR, language_concept_id BIGINT);
CREATE TABLE IF NOT EXISTS vocab.vocabulary (
  vocabulary_id VARCHAR, vocabulary_name VARCHAR, vocabulary_reference VARCHAR,
  vocabulary_version VARCHAR, vocabulary_concept_id BIGINT);
"""

CDM_DDL = """
CREATE SCHEMA IF NOT EXISTS cdm;
CREATE TABLE IF NOT EXISTS cdm.person (
  person_id BIGINT PRIMARY KEY, gender_concept_id BIGINT, year_of_birth INTEGER,
  month_of_birth INTEGER, day_of_birth INTEGER, race_concept_id BIGINT,
  ethnicity_concept_id BIGINT, person_source_value VARCHAR);
CREATE TABLE IF NOT EXISTS cdm.observation_period (
  observation_period_id BIGINT, person_id BIGINT,
  observation_period_start_date DATE, observation_period_end_date DATE,
  period_type_concept_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.visit_occurrence (
  visit_occurrence_id BIGINT, person_id BIGINT, visit_concept_id BIGINT,
  visit_start_date DATE, visit_end_date DATE, visit_type_concept_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.condition_occurrence (
  condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id BIGINT,
  condition_start_date DATE, condition_end_date DATE, condition_type_concept_id BIGINT,
  condition_source_value VARCHAR, condition_source_concept_id BIGINT, visit_occurrence_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.drug_exposure (
  drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id BIGINT,
  drug_exposure_start_date DATE, drug_exposure_end_date DATE, days_supply INTEGER,
  drug_type_concept_id BIGINT, drug_source_value VARCHAR, visit_occurrence_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.measurement (
  measurement_id BIGINT, person_id BIGINT, measurement_concept_id BIGINT,
  measurement_date DATE, value_as_number DOUBLE, unit_concept_id BIGINT,
  measurement_type_concept_id BIGINT, measurement_source_value VARCHAR, visit_occurrence_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.procedure_occurrence (
  procedure_occurrence_id BIGINT, person_id BIGINT, procedure_concept_id BIGINT,
  procedure_date DATE, procedure_type_concept_id BIGINT, procedure_source_value VARCHAR,
  visit_occurrence_id BIGINT);
CREATE TABLE IF NOT EXISTS cdm.cdm_source (
  cdm_source_name VARCHAR, cdm_release_date DATE, cdm_version VARCHAR, vocabulary_version VARCHAR);
"""

RESULTS_DDL = """
CREATE SCHEMA IF NOT EXISTS results;
CREATE TABLE IF NOT EXISTS results.cohort (
  cohort_definition_id BIGINT, subject_id BIGINT, cohort_start_date DATE,
  cohort_end_date DATE, generation_id VARCHAR);
CREATE TABLE IF NOT EXISTS results.cohort_inclusion_stats (
  generation_id VARCHAR, cohort_definition_id BIGINT, rule_sequence INTEGER,
  rule_name VARCHAR, remaining_count BIGINT);
"""


def connect(path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_schemas(con: duckdb.DuckDBPyConnection) -> None:
    from .metadata import META_DDL

    for ddl in (VOCAB_DDL, CDM_DDL, RESULTS_DDL, META_DDL):
        con.execute(ddl)
