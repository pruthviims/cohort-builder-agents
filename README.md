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
pip install -e ".[api,mcp,dev]"
cohort-builder init-demo                  # demo vocabulary + synthetic OMOP patients and LAAD-style claims (DuckDB)
pytest                                    # full offline test suite, no API key needed

export ANTHROPIC_API_KEY=sk-ant-...
cohort-builder ask "Adults with type 2 diabetes who started metformin and had an HbA1c above 8% in the year before starting"
cohort-builder approve 1 --reviewer dr_rao   # a different person than the author (self-approval is refused)
cohort-builder execute 1                  # writes results.cohort; prints suppressed counts + caveats
cohort-builder replay <run_id>            # re-run from recorded LLM responses, compare hashes
```

You can try the pipeline without an LLM by running a hand-written definition:

```bash
cohort-builder dry-run examples/t2dm_metformin_hba1c.json
```

### HTTP API

Every endpoint except `GET /health` requires a bearer token. The caller's identity and roles come from the
token, never from the request body. To set it up:

```bash
cohort-builder auth issue-token --subject alice@example.org --roles author,viewer --tokens-file tokens.yaml
# the token is printed once (stderr); only its hash is written to tokens.yaml (mode 0600)
export CB_ENV=production CB_AUTH_TOKENS_FILE=$PWD/tokens.yaml
cohort-builder serve                      # refuses to start on unsafe settings; docs at http://127.0.0.1:8000/docs
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/me
```

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/health` | public | Liveness only |
| GET | `/me`, `/versions` | any | Who am I · component versions |
| POST | `/cohorts/ask` | author | Natural-language request → draft definition, explanation, attrition, manifest |
| POST | `/cohorts` · `/cohorts/validate` | author | Save a hand-written or edited IR as a new immutable draft · validate without saving |
| GET | `/cohorts` · `/cohorts/{id}` · `/cohorts/{id}/sql` | any | List (own tenant) · definition with explanation · compiled SQL |
| POST | `/cohorts/{id}/review` | reviewer | `{"decision": "approved"\|"rejected"}`; you can't approve your own definition |
| POST | `/cohorts/{id}/execute` | executor | Run an **approved** definition; returns suppressed counts and caveats |
| GET / POST | `/runs/{run_id}` · `/runs/{run_id}/replay` | author (own), reviewer | Run trace and manifest · exact replay |
| GET | `/concepts/search?q=` | any | Vocabulary search |
| GET | `/audit` | admin | Audit log of submissions, reviews, executions and denials |

Roles, tenants, token rotation, the development bypass and the audit log are described in
[`docs/SECURITY.md`](docs/SECURITY.md).

### MCP server

The same backend is available as an MCP server, so you can build cohorts from Claude Desktop, Claude Code, a Claude connector or another agent.

```bash
pip install -e ".[mcp]"
cohort-builder mcp                                   # stdio, for a local client
claude mcp add cohort-builder -- cohort-builder mcp  # register it in Claude Code
```

For Claude Desktop, add this to `claude_desktop_config.json`:

```json
{"mcpServers": {"cohort-builder": {"command": "cohort-builder", "args": ["mcp"],
  "env": {"CB_DB_PATH": "/path/to/cohort_builder.duckdb", "CB_MCP_USER": "your.name",
          "ANTHROPIC_API_KEY": "sk-ant-..."}}}}
```

There are two ways to use it:

| Mode | Tools | Who reasons | Reproducibility |
|---|---|---|---|
| **Reproducible pipeline** | `build_cohort`, `get_run`, `replay_run` | This server's own agents (pinned model, versioned prompts, cache/replay). Needs `ANTHROPIC_API_KEY` on the server | Full: replay and regeneration both tracked |
| **Client-driven** | `describe_ontology`, `search_curated_concept_sets`, `search_concepts`, `get_concept`, `get_descendants`, `lookup_code`, `validate_cohort`, `save_cohort_definition` | The client's LLM, e.g. Claude in Desktop. No API key on the server | Saved definitions replay exactly. Regeneration depends on the client |

In both modes, the server enforces every ontology rule. A definition with deprecated, non-standard or wrong-domain concepts, or with invalid units, is rejected no matter which LLM wrote it.

Other tools: `list_cohort_definitions`, `get_cohort_definition`, `get_cohort_sql`, `execute_approved_cohort`.

Resources: `ontology://domain`, `ontology://curated-concept-sets`, `ontology://unit-conversions`, `cohort://ir-schema`, `cohort://definitions/{id}`.

Prompt: `build_cohort_interactively`.

**Governance over MCP:**
- **No approval tool.** An AI must not approve its own work, so approval happens only through the CLI or HTTP API (`cohort-builder approve`).
- **Execution is gated.** `execute_approved_cohort` refuses drafts and returns suppressed counts only. No tool returns patient rows.
- **Server-side identity.** The acting identity comes from `CB_MCP_USER`, not from tool arguments, and is recorded on every saved definition and execution.

**Shared team server (Streamable HTTP):**

```bash
CB_MCP_TOKEN=<long-random-secret> CB_MCP_ALLOWED_HOSTS=cohorts.example.org:* \
  cohort-builder mcp --http --host 0.0.0.0 --port 8765      # endpoint: /mcp
```

- **Startup checks:** the server won't bind beyond localhost unless both a token and a hostname allowlist (DNS-rebinding protection) are set.
- **Bearer token:** this is shared-secret auth, fine for a pilot. For production, put it behind TLS and switch to OAuth through the MCP SDK's auth settings, so each user is identified individually.

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

All of it is stored in the `meta` schema: `agent_run`, `agent_step`, `llm_call`, `llm_cache`, `tool_call`, `cohort_definition`, `cohort_generation`, `review`, `audit_event`, `eval_run` and `eval_result`.

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
| `datasets/*.yaml` | One **dataset profile** per data source: what it can answer, where each entity physically lives, how "observable" is defined, and any semantic views to install (`omop_demo`, `iqvia_laad`) |
| `curated_concept_sets.yaml` | Approved concept sets per therapeutic area, with synonyms |
| `unit_conversions.yaml` | Analyte-specific canonical units and conversions |

The ontology's content hash goes into every manifest. If you change the ontology, bump `version`.

### Data sources: OMOP is optional

The agents, IR and compiler only know the ontology. A dataset profile binds that ontology to a physical source. Pick one with `--dataset` or `CB_DATASET`:

```bash
cohort-builder datasets                                    # list profiles and what each can answer
cohort-builder --dataset iqvia_laad ask "New users of SGLT2 inhibitors with T2D on 2+ claims 30+ days apart"
```

**`iqvia_laad`** is a profile for IQVIA LAAD-style open US claims. It works on the vendor's native tables, so no OMOP conversion is needed. A layer of semantic views (`sem` schema) does the translation:

- **Codes:** raw codes are mapped to standard concepts through the OMOP *vocabularies*. That covers NDC → RxNorm, dot-less ICD-10-CM → SNOMED, and CPT. Wide `dx1..dxN` columns are unpivoted, keeping each code's position on the claim.
- **Claim status:** each pharmacy claim is classified as **paid, rejected or reversed**. Reversal transactions are dropped. Drug criteria count paid claims unless the request is about rejections.
- **Observation without enrollment:** open data has no enrollment table, so observation is inferred from **claim activity**. A period runs from first to last activity and splits on gaps longer than 365 days (configurable). The validator warns that exclusions are weaker evidence on this basis.
- **New cohort rules:**
  - `claim_status` (e.g. "first *rejected* claim, then *paid* within 90 days")
  - `dx_position` (primary diagnosis)
  - `min_span_days` (the claims case definition "≥2 claims ≥30 days apart")
- **What LAAD can't answer:** there are no lab values and no visits. A request that needs them (e.g. HbA1c > 8%) is stopped immediately with a clear message, instead of quietly returning nobody.

> **The `laad` table and column names are placeholders.** Replace them with the names from your IQVIA data dictionary in `ontology/datasets/iqvia_laad.yaml`. Only that file changes. Verify the transaction-type, reject-code and reversal conventions against your delivery too.

**One definition, several sources.** The same cohort logic can be executed on several sources. Execution re-checks what the active source can answer, and records which dataset each run used.

To add another source (Optum, Komodo, your EHR), copy a profile and edit its capabilities, mapping and views.

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
- **CDM:** the executor runs on DuckDB. You can load your CDM into the `cdm` schema, or point a dataset profile (`ontology/datasets/`) at your tables.
  - **PostgreSQL:** the compiled cohort SQL uses portable constructs (`date + int`, `COUNT(*) FILTER`, window functions). CI **executes** it on PostgreSQL 16, and the tests check that it returns the same patients as DuckDB.
  - **What isn't portable yet:** dataset-profile setup SQL (e.g. the LAAD semantic views use `CREATE OR REPLACE TABLE`) is DuckDB-specific. An executor that runs directly against PostgreSQL, Databricks or Snowflake is not implemented.
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

The golden sets cover cardiometabolic, renal, mental-health and market-access (claim rejection) cases. LAAD cases live in `eval/golden_cases_laad.yaml` (`cohort-builder --dataset iqvia_laad eval --cases eval/golden_cases_laad.yaml`). The `eval` GitHub workflow runs it against the live API when you trigger it.

## Validation results: errors and warnings

Every definition is checked twice: by the IR models, which enforce structure, and by the validator, which
checks it against the ontology, vocabulary and dataset. Results come back as `issues`, each with a
`severity` and a `stage`.

| Severity / stage | Meaning | What to do |
|---|---|---|
| error / `intent` | The logic is invalid as written: unknown unit, threshold not in the canonical unit, wrong operator, contradictory rules | Fix the definition; agents get this as feedback |
| error / `concepts` | Unknown, deprecated, non-standard or wrong-domain concept; no index events found | Pick standard concepts in the right domain |
| error / `dataset` | The selected data source cannot answer this (e.g. lab values on open claims, no observation periods) | Use another dataset, or drop the requirement knowingly |
| error / `data` | The data can't support the run (e.g. index events but none inside an observation period) | Check the data and its observation coverage |
| warning / `data` | It runs, but there is a limitation the reviewer must accept. Execution returns and stores all warnings as **caveats** | Read them before using the result |
| warning / `intent` | Unusual but allowed (unused concept set, exclusion with `at_most`) | Confirm it's intended |

Typical `data` caveats:

- **"No record" treated as "did not happen":** any exclusion or zero-count rule. The wording depends on the
  dataset: on open claims, observation is inferred from claim activity, so the evidence is weak.
- **Lookback longer than the required prior observation:** e.g. "no T1D in the 730 days before" with only
  365 days of observation required.
- **Follow-up window longer than the required post observation.**
- **Measurements only partially captured,** or results in units with no conversion metadata.

These caveats describe limits of the data. They are not clinical validation. Concept sets, definitions and
dataset profiles still need review by clinicians and data owners.

## Cohort semantics (what the compiler guarantees)

- **Windows:** inclusive at both ends, in calendar days relative to the index date. `{start_days: -365,
  end_days: 0}` means the 365 days before index plus the index day. `null` bounds mean "within the
  observation period that contains the index".
- **Index:** `first_occurrence_only` takes the person's earliest qualifying event *ever*. If that event falls
  outside an observation period, the person does not enter; there is no fallback to a later event. With
  `false`, the earliest event that satisfies all rules is used.
- **Ties:** same-day ties are resolved deterministically. If observation periods overlap, the
  earliest-starting period wins, then the latest-ending one.
- **Age:** calendar year of index minus year of birth.
- **Counting:** `count` counts qualifying records by default; `count_by: "dates"` counts distinct event days.
  `min_span_days` requires the first and last qualifying events to be at least that many days apart.
- **Labs:** thresholds must be in the analyte's canonical unit. Results in other units are converted when a
  conversion exists, and otherwise never compared. Null values never match.
- **Claims (LAAD profile):**
  - identical duplicate claims count once
  - a diagnosis code counts once per claim, at its best position
  - reversed claims are not "paid"
  - rows without a patient id are dropped
  - patients missing from the patient table show up as a drop between the first two attrition steps

Each rule above has a test on a small hand-built dataset in `tests/test_compiler_semantics.py`.

## Running tests and checks

```bash
pip install -e ".[api,mcp,dev]"
pytest                                    # synthetic data and a scripted fake LLM; no network, no API key
pytest --cov                              # with coverage (CI gate: 80%)
CB_TEST_POSTGRES_DSN=postgresql://user:pass@localhost:5432/postgres pytest   # also run compiled SQL on PostgreSQL
ruff check src tests && ruff format --check src tests && mypy
pip-audit --skip-editable                 # dependency vulnerabilities (run in a clean virtualenv)
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline   # secret scan
```

CI runs all of these on Python 3.11–3.13, against a PostgreSQL 16 service, using synthetic data and
throwaway credentials only.

## Configuration

| Variable | Default |
|---|---|
| `CB_ENV` | `production` (fails closed; `development` enables the dev-only switches) |
| `CB_AUTH_TOKENS_FILE` | (none: every protected API endpoint returns 401) |
| `CB_ALLOW_SELF_APPROVAL` | `false` |
| `CB_AUTH_DEV_BYPASS`, `CB_AUTH_DEV_SUBJECT`, `CB_AUTH_DEV_ROLES` | off; development only |
| `CB_ALLOW_DRAFT_EXECUTION` | `false`; development only |
| `ANTHROPIC_API_KEY` | (required for `ask` in live/cached mode) |
| `CB_DATASET` | `omop_demo` (or `iqvia_laad`) |
| `CB_MODEL` | `claude-sonnet-5-5` |
| `CB_TEMPERATURE` | `0` (set `none` to omit the parameter for models that do not accept it) |
| `CB_LLM_MODE` | `cached` |
| `CB_DB_PATH` | `data/cohort_builder.duckdb` |
| `CB_MAX_RETRIES` / `CB_MAX_RESOLVER_TURNS` | `2` / `8` |
| `CB_QUERY_TIMEOUT_SECONDS` | `300` |
| `CB_DUCKDB_MEMORY_LIMIT` / `CB_DUCKDB_THREADS` | DuckDB defaults |
| `CB_DUCKDB_LOCK_EXTERNAL_ACCESS` | `true` (SQL cannot read files, attach databases or load extensions) |
| `CB_MCP_USER`, `CB_MCP_TENANT`, `CB_MCP_TOKEN`, `CB_MCP_ALLOWED_HOSTS` | see MCP server |

A placeholder-only template is in `.env.example`.

## Layout

```
ontology/            semantic ontology (YAML, versioned) + datasets/ profiles
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
  synthetic.py       demo vocabulary, synthetic OMOP patients, Athena loader
  synthetic_laad.py  synthetic LAAD-style claims (native vendor-shaped tables)
  cli.py, api.py     interfaces (CLI, HTTP API)
  mcp_server.py      MCP server (stdio + Streamable HTTP)
eval/                golden cases + gold IRs
tests/               offline tests with a scripted fake LLM
```

## Status and limitations

**Experimental or untested:**
- **Live API path untested:** the agents were built against the Anthropic Messages API (forced tool use), but the live API path hasn't been run from this repo yet. The test suite uses a scripted fake LLM. Run `cohort-builder eval` with an API key to get a real accuracy baseline before trusting the agents.
- **Placeholder LAAD layout:** the LAAD profile uses placeholder table and column names. Map it to your IQVIA data dictionary and check its claim-status conventions before use.
- **Not compliance-assessed:** no clinical validation and no regulatory compliance assessment has been done. The demo data is synthetic, and its counts mean nothing clinically.

**Not implemented yet:**
- **Single entry per person:** each person enters a cohort once, at their earliest qualifying index. Era collapsing and censoring events are not implemented.
- **No warehouse executor:** a PostgreSQL or warehouse executor and OIDC/SSO are not implemented. See [`docs/SECURITY.md`](docs/SECURITY.md) for the controls that need a deployment-level assessment.
