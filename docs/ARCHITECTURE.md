# Architecture notes: data layers, provenance and the path to treatment analytics

This note records how the data layers fit together today and which foundations later analytics modules
(brand switch, line of therapy, persistence and adherence, driver analysis) can build on. It describes
what exists. It is not a design for modules that haven't been built.

## Layers

| Layer | Where | Contents | Who writes it |
|---|---|---|---|
| Raw source | `stg_<dataset>_v<N>.<table>` | Imported rows, exact values (ids and codes as text), `_source_row` = CSV line | CSV ingestion (deterministic) |
| Normalized meaning | `sem_<dataset>_v<N>.<entity view>` | `person_id`, `concept_id`, dates, values, plus `source_value`, `source_code_normalized`, `source_table`, `source_row` | SQL generated from a validated mapping |
| Vocabulary | `vocab.*` | OMOP-style concepts and relationships (demo subset or Athena), with versions | `init-demo` / `load-athena` |
| Dataset profile | `ontology/datasets/*.yaml`, `data/datasets/*.yaml` | Entity -> table/column mapping and capabilities | Hand-written (built-in) or generated (imports) |
| Definitions and results | `meta.*`, `results.*` | Cohort IR, compiled SQL, generations, audit | Deterministic compiler and executor |

Built-in datasets (`omop_demo`, and `iqvia_laad` through its own semantic views) and CSV imports all reach
the compiler through the same dataset-profile contract. Nothing assumes the source is OMOP: the semantic
views are the adapter.

**Raw data is never rewritten.** Normalization (surrogate person ids, parsed text dates, code lookup keys,
concept mapping) happens only in views. The staged value stays available next to the derived one, so any
derived record can be traced back to a file, a line and the original string.

## Treatment-event engine (implemented)

`treatment_config.py` (validated source configs and rule sets), `treatment_events.py` (config -> SQL ->
canonical `treatment_event/1` rows, every source record accounted for), `treatment_episodes.py` (pure,
deterministic episodes, censoring and transition evidence) and `treatment_service.py` (runs, provenance,
suppressed summaries, gated explanations). Dataset-specific knowledge (status codes, reversal links,
code systems, which date a row carries, evidence type) lives in `ontology/treatment/sources/<dataset>.yaml`;
use-case assumptions live in `ontology/treatment/rules/<name>.yaml`. See README "Treatment events and
episodes". The section below records the reasoning that preceded it.

## Treatment-event foundation (what exists, and why nothing more)

Planned treatment analytics need, for each drug record:
- the patient and the dates
- a source reference
- the source code and the normalized concept
- the provenance of the transformation

The `drug_exposure` semantic view of a CSV import now carries all of these: `person_id`, `start_date`,
`end_date`, `concept_id`, `source_value`, `source_code_normalized`, `source_table` and `source_row`. The
dataset, version, mapping hash and code-normalization strategy are in the profile's `source` block.

No separate "treatment event" table or interface was added. No module consumes one yet, and the existing
semantic view already holds the fields. Defining a contract before its first consumer would only freeze
guesses. The concrete gaps are listed under "Next milestone" below.

## Versioned, validated configuration (the pattern to reuse)

Rules that change results must be explicit data, not code paths or prompt text. The CSV mapping already
follows the pattern future rule sets should reuse:
1. A strict Pydantic model (`extra="forbid"`, literal enums, bounded numbers), loaded with the strict YAML
   loader (duplicate keys refused).
2. Validation against the actual data or schema, with errors and warnings kept separate.
3. A content hash (`sha256` of canonical JSON), stored with the version that used it.
4. Compilation into SQL by deterministic code; only validated identifiers and literals reach SQL.
5. Immutable versions in a `meta.*` table, and an audit event per change.

Brand groupings, episode and grace-period rules or line-of-therapy definitions should be new models of
this kind, recorded with their hash. They shouldn't be options added to the cohort IR or text in agent
prompts. LLM agents may *propose* such a configuration. The deterministic engine validates and runs it,
and a human approves it, as with cohort definitions today.

## Provenance available today

| Item | Where |
|---|---|
| Dataset and version, file hashes, row counts | `meta.dataset_version`, profile `source` |
| Mapping version and hash, code-normalization strategy, vocabulary versions | `meta.dataset_version.mapping_hash`, profile `source` |
| Data snapshot id | profile `snapshot_sql`, recorded with every cohort generation |
| Ontology, prompts, compiler and software versions | `meta.component_version`, run manifests |
| Execution time, status, actor | `meta.cohort_generation`, `meta.audit_event` |

A future analysis run should add a `rule_set_hash` and its parameters next to these, using the same
tables and audit events rather than a parallel mechanism.

## Recommended next milestone: treatment-event engine (minimal)

1. **Define the event view contract** from the first real consumer (e.g. persistence for one chronic
   therapy area): `person_id`, `concept_id`, `ingredient_concept_id`, `start_date`, `days_supply` /
   `end_date`, `source_*`. Implement it as a view per dataset profile. CSV imports need a mapping field
   for `days_supply`; the built-in profiles need equivalent columns.
2. **Add ingredient rollup** through `vocab.concept_ancestor`, so brand, NDC and clinical-drug records
   group deterministically. Record the vocabulary version used.
3. **Add a versioned `TherapyRules` model** (drug groups, grace period, episode gap, add-on window), with
   validation, a content hash and a `meta.rule_set_version` table, following the mapping pattern above.
4. **Build a deterministic episode builder** (SQL, like the observation-period gaps-and-islands already
   used for `mode: activity`), with golden tests on synthetic timelines.
5. **Return aggregate outputs only** through the existing small-cell suppression. Patient-level timelines
   stay behind the existing patient-level gate.

Brand switch and line of therapy come after this, as rule sets over the same episodes.
