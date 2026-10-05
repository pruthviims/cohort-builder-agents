# Cohort Builder Agents

This project uses LLM agents and a semantic ontology to build patient cohorts on **OMOP CDM** data, and every result can be reproduced.

A researcher asks a question, for example:

> *"Adults with type 2 diabetes who started metformin and had an HbA1c above 8% in the year before starting. Exclude anyone with type 1 diabetes before starting."*

The system then:

1. turns the request into a typed, versioned **cohort definition** (the IR),
2. explains that definition in plain language and shows dry-run attrition counts,
3. waits for a human to approve it, and
4. compiles it to SQL **without any LLM** and materializes the cohort.

No step depends on the therapeutic area. Disease knowledge comes from the vocabulary (SNOMED, RxNorm, LOINC) and from curated concept sets.

```
            ┌──────────────────────── LLM agents ────────────────────────┐
request ──► │ intent parser ──► concept resolver ──► (composer) ──► (validator) ──► (explainer) ──► critic │
            └────────▲────────────────▲───────────────────────────────────────────────────────┬─────────┘
                     └──── feedback (max 2 retries) ───────────────────────────────────────────┘
                                         │ grounded by tools
                              semantic ontology + OMOP vocabulary
                                         ▼
            cohort definition (IR, hashed) ──► human review ──► deterministic SQL compiler ──► results.cohort
```

Only **three steps call an LLM**: the intent parser, the concept resolver and the critic. Composing, validating and explaining are deterministic code, which removes as much randomness as possible.

## Quickstart

```bash
pip install -e ".[api,dev]"
cohort-builder init-demo                  # demo vocabulary + 5,000 synthetic patients (DuckDB)
pytest                                    # 22 tests, no API key needed

export ANTHROPIC_API_KEY=sk-ant-...
cohort-builder ask "Adults with type 2 diabetes who started metformin and had an HbA1c above 8% in the year before starting"
cohort-builder approve 1 --reviewer dr_rao
cohort-builder execute 1                  # writes results.cohort + attrition
cohort-builder replay <run_id>            # re-run from recorded LLM responses, compare hashes
```

You can try the pipeline without an LLM by running a hand-written definition:

```bash
cohort-builder dry-run examples/t2dm_metformin_hba1c.json
```

### HTTP API

Start it with `cohort-builder serve`. Interactive docs are at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/cohorts/ask` | Natural-language request → draft definition, explanation, attrition, manifest |
| POST | `/cohorts` | Submit a hand-written or edited IR (saved as a new immutable version) |
| GET | `/cohorts/{id}` · `/cohorts/{id}/sql` | Definition with explanation · compiled SQL |
| POST | `/cohorts/{id}/review` | `approved` / `rejected` (execution requires approval) |
| POST | `/cohorts/{id}/execute` | Materialize into `results.cohort` |
| GET / POST | `/runs/{run_id}` · `/runs/{run_id}/replay` | Run trace and manifest · exact replay |
| GET | `/concepts/search?q=` | Vocabulary search |

## How reproducibility works

**Level 1: replay is guaranteed.** The IR is the artifact of record. It is canonicalized and hashed twice:

- `content_hash` covers everything in the IR.
- `semantic_hash` covers the logic only. Labels, IDs and criterion order are ignored, and concept sets are inlined.

The compiler is deterministic and versioned. The same IR on the same data snapshot always returns the same patients. A test checks this against an independent pure-Python implementation of the example cohort.

**Level 2: regeneration consistency is measured.** The settings below make reruns as consistent as possible:

- temperature 0 and a pinned model ID
- prompts as versioned files (`prompts/<agent>/v1.md`)
- forced tool calls for structured output
- sorted, deterministic tool results

Every LLM request is canonicalized and hashed. The `CB_LLM_MODE` setting controls how recorded responses are used:

| `CB_LLM_MODE` | Behavior |
|---|---|
| `cached` (default) | An identical request reuses its recorded response |
| `live` | Always call the API, still recording the call (use this for eval repeats) |
| `replay` | Recorded responses only. A cache miss fails loudly, so changed inputs can't slip through |

Every run writes a **manifest** with:

- the model, temperature and LLM mode
- prompt versions and hashes
- the ontology version and hash
- the vocabulary version
- the compiler version and data snapshot
- LLM call, cache-hit and tool-call counts
- both IR hashes

All of it is stored in the `meta` schema: `agent_run`, `agent_step`, `llm_call`, `llm_cache`, `tool_call`, `cohort_definition`, `cohort_generation`, `review`, `eval_run` and `eval_result`.

**Guardrails built into the agent loop:**

- The concept resolver can only submit concept IDs that appeared in its own tool results. Submissions with invented IDs are rejected in code and sent back.
- Curated (organization-approved) concept sets are searched first and copied verbatim.
- Value thresholds are normalized to each analyte's canonical unit. For example, "HbA1c > 64 mmol/mol" is stored as "> 8.007 %". At query time, rows recorded in other units are converted.
- If the critic rejects a definition and the revision produces the same logic, the run stops with `needs_review` instead of looping.
- Agents never see patient rows. They only see aggregate attrition counts, with cells under 10 suppressed.

## The semantic ontology (`ontology/`)

| File | Contents |
|---|---|
| `domain.yaml` | Entities, filterable attributes and operators, temporal semantics, cohort construction rules (semver `version`) |
| `mappings.yaml` | Entity → physical OMOP table and column mapping (change this for a non-OMOP warehouse) |
| `curated_concept_sets.yaml` | Approved concept sets per therapeutic area, with synonyms |
| `unit_conversions.yaml` | Analyte-specific canonical units and conversions |

The ontology's content hash goes into every manifest. If you change the ontology, bump `version`.

### Adding a therapeutic area

No code changes are needed for most areas:

1. Add curated concept sets, approved by clinicians, to `curated_concept_sets.yaml`.
2. Add the area's key lab conversions to `unit_conversions.yaml`.
3. Add 30–50 golden cases to `eval/golden_cases.yaml`, with gold IRs in `eval/gold/`.
4. Run `cohort-builder eval --repeats 3` and fix any gaps before onboarding users.

Oncology (lines of therapy, staging, biomarkers) needs new entities and IR criterion types, built on the OMOP Oncology extension's `episode` table.

## Moving to real data

- **Vocabulary:** download the vocabularies from [OHDSI Athena](https://athena.ohdsi.org) and run `cohort-builder load-athena /path/to/athena`. This replaces the demo vocabulary.
  - Concept and unit IDs at or above 2,000,000,000 in the demo files are local placeholders (for example eGFR, LVEF and mmol/mol). Replace them with the real Athena IDs in `unit_conversions.yaml` and `curated_concept_sets.yaml`.
  - The other top-level IDs follow standard OMOP but should still be verified against your vocabulary release.
- **CDM:** the executor runs on DuckDB. You can load your CDM into the `cdm` schema, or point `mappings.yaml` at your tables.
  - The compiled SQL sticks to portable constructs (`date + int`, `COUNT(*) FILTER`, window functions), so `cohort-builder compile <id>` output is written to run on PostgreSQL.
  - Executing directly against Postgres, Databricks or Snowflake needs a small executor adapter. That has not been tested yet.
- **Search:** concept search is lexical (exact, synonym, code and Jaro-Winkler). An embedding searcher, such as pgvector, can be plugged in through `vocab.ConceptSearcher`.

## Evaluation

```bash
cohort-builder eval --repeats 3 --out eval_report.json   # exits non-zero below thresholds
```

The eval harness measures, for each golden case:

- **Validity rate:** the share of runs that end as a valid draft.
- **Consistency:** the share of repeats that produce the same semantic hash.
- **Concept-set Jaccard:** overlap of expanded concept IDs, for the index event and for the criteria.
- **Structure match:** whether windows, thresholds and occurrence counts match the gold definition.
- **Patient-level Jaccard:** overlap between the generated cohort and the gold cohort on the same data.

The golden set covers cardiometabolic, renal and mental-health cases. The `eval` GitHub workflow runs it against the live API when you trigger it.

## Configuration

| Variable | Default |
|---|---|
| `ANTHROPIC_API_KEY` | (required for `ask` in live/cached mode) |
| `CB_MODEL` | `claude-sonnet-5-5` |
| `CB_TEMPERATURE` | `0` (set `none` to omit the parameter for models that do not accept it) |
| `CB_LLM_MODE` | `cached` |
| `CB_DB_PATH` | `data/cohort_builder.duckdb` |
| `CB_MAX_RETRIES` / `CB_MAX_RESOLVER_TURNS` | `2` / `8` |

## Layout

```
ontology/            semantic ontology (YAML, versioned)
prompts/             versioned agent prompts
src/cohort_builder/
  ir.py              cohort definition IR + canonical/semantic hashing
  vocab.py           vocabulary tools (search, hierarchy, Maps-to)
  compiler.py        deterministic IR -> SQL with attrition
  executor.py        runs SQL, small-cell suppression, results tables
  llm.py             Anthropic client, request hashing, cache/replay
  agents/            intent parser, concept resolver, critic (LLM); composer, validator, explainer (code)
  orchestrator.py    fixed agent graph, review gate, manifests, replay
  metadata.py        audit/metadata store
  evaluation.py      golden-case evaluation
  synthetic.py       demo vocabulary, synthetic patients, Athena loader
  cli.py, api.py     interfaces
eval/                golden cases + gold IRs
tests/               offline tests with a scripted fake LLM
```

## Status and limitations

- **Live API untested:** the agents were built against the Anthropic Messages API (forced tool use), but the live API path hasn't been run from this repo yet. The test suite uses a scripted fake LLM.
- **Run a live baseline first:** with an API key, run `cohort-builder eval` to get a real baseline before trusting any accuracy numbers.
- **Synthetic data only:** the demo data is synthetic. Counts mean nothing clinically.
- **Simplified cohort model:** each person can enter a cohort only once (their earliest qualifying index). Era collapsing and censoring events are not implemented yet.
