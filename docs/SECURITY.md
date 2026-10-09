# Security, privacy and governance

This document describes the controls implemented in this repository, how to configure them, and what still
requires assessment at the deployment level.

> **No compliance claim.** These controls support a controlled analytics environment. They do not by
> themselves make a deployment HIPAA-compliant, or compliant with any other regulation. Compliance depends on
> hosting, access management, agreements (e.g. BAAs and data use agreements), policies, monitoring and review
> by your privacy and security officers.

## Environments

`CB_ENV` selects the environment. **The default is `production`**, and it fails closed.

| Setting | production | development |
|---|---|---|
| `CB_AUTH_DEV_BYPASS` (unauthenticated requests act as a fixed, configured identity) | **refused at startup** | allowed; needs explicit `CB_AUTH_DEV_SUBJECT` and `CB_AUTH_DEV_ROLES` |
| `CB_ALLOW_DRAFT_EXECUTION` (run unapproved drafts) | **refused at startup** | allowed |
| `CB_ALLOW_SELF_APPROVAL` (approve your own definition) | allowed only if explicitly set (not recommended) | allowed if set |

Invalid values, such as `CB_ENV=staging` or `CB_ALLOW_SELF_APPROVAL=maybe`, stop the server from starting.
They are never silently ignored.

## Authentication (HTTP API)

- **Bearer tokens.** Every endpoint except `GET /health` requires `Authorization: Bearer <token>`.
- **Tokens are opaque random strings.** Only their **SHA-256 hashes** are stored, in the token file named by
  `CB_AUTH_TOKENS_FILE` (YAML or JSON). A leaked token file therefore does not leak credentials.
- **Rejected requests.** Missing, malformed, unknown or expired tokens get `401` with `WWW-Authenticate: Bearer`.
  If no tokens are configured, every protected endpoint returns `401`.

Issue a token. It is printed once on stderr, and only its hash is written to the token file (created with mode
0600). Without `--tokens-file`, the entry is printed to stdout instead.

```bash
cohort-builder auth issue-token --subject alice@example.org --roles author,viewer --tenant research \
  --expires-days 90 --tokens-file /etc/cohort-builder/tokens.yaml
```

Token file format:

```yaml
tokens:
  - id: alice-laptop                     # label shown in audit records (never the secret)
    sha256: 3f1c...e9                    # sha256 hex of the token
    subject: alice@example.org           # the identity recorded as author/reviewer/executor
    roles: [author, viewer]
    tenant: research
    expires_at: "2027-01-01T00:00:00Z"   # optional
```

**Rotation.** Issue a new token, add its entry, distribute it, remove the old entry, then restart the server.
Tokens are never logged; authentication failures are logged without the credential.

**SSO.** For enterprise SSO, put an OIDC-aware gateway in front of the API. Alternatively, replace
`security.TokenAuthenticator` with an OIDC/JWT verifier that implements the same
`authenticate(token) -> Principal` method. OIDC is not implemented here.

## Roles and endpoints

| Role | Can |
|---|---|
| `viewer` | read definitions, explanations, SQL and vocabulary in their own tenant |
| `author` | create and validate definitions (`/cohorts/ask`, `/cohorts`, `/cohorts/validate`); read and replay their own run traces |
| `reviewer` | approve or reject definitions; read the tenant's run traces |
| `executor` | execute **approved** definitions; receives suppressed aggregate counts only |
| `admin` | everything above, across tenants, plus `GET /audit` |

Proxy cohort endpoints use the same roles: authors create, validate and draft (`/proxy-cohorts`,
`/proxy-cohorts/ask`, `/proxy-cohorts/validate`); any role reads definitions, versions, review packets and
the SQL preview; reviewers approve or reject and run reference validations; executors run approved
definitions; executors and reviewers read suppressed results, evidence summaries, evaluations and
comparisons; any role reads `/proxy-cohorts/{id}/status`; a reviewer records the human acceptance decision for an
evaluation (`/evaluations/{validation_id}/review`; never the algorithm's or the evaluation's author, identity from
the token, recorded once); admins
load reference-standard labels (`/proxy-references`) and, only when `CB_ALLOW_PATIENT_LEVEL=true`, read
per-patient explanations.

How the rules are enforced:

- **Identity comes from the token.** Request bodies that carry identity fields (`user_id`, `reviewer`) are
  rejected with `422`.
- **No self-approval by default.** `created_by == reviewer` is refused unless `CB_ALLOW_SELF_APPROVAL=true`.
  The same rule applies in the CLI.
- **Draft execution is off.** Sending `allow_draft=true` is refused with `403` unless the server runs with
  `CB_ENV=development` and `CB_ALLOW_DRAFT_EXECUTION=true`.
- **Tenant isolation.** A tenant's definitions and runs are invisible to other tenants. They return `404`, so
  their existence isn't revealed. Admins can see all tenants.
- **Run traces contain the original prompt.** They are visible only to the author who ran them, the tenant's
  reviewers, and admins.

## CLI and MCP

- **CLI.** The CLI is a local operator tool with direct database access. It is **not** an authentication
  boundary: `--reviewer` and `--user` are trusted. The governance rules (no self-approval, no draft execution,
  approval only from draft) still apply, because they live in the core.
- **MCP.** The MCP server acts as one configured identity (`CB_MCP_USER`) in one tenant (`CB_MCP_TENANT`). It
  has no approval tool, and it executes approved definitions only. The proxy tools follow the same rule:
  `get_proxy_review_packet` is read-only, `execute_proxy_cohort` refuses anything a human has not approved,
  and no MCP tool returns patient-level data. An AI that generated an algorithm therefore cannot run it. Its HTTP mode uses one shared bearer token
  (`CB_MCP_TOKEN`) plus a host allowlist. That is suitable for a pilot only; per-user OAuth is remaining work.

## Audit log

`meta.audit_event` is append-only from the application's point of view. Each record holds the actor, tenant,
action, resource, outcome (`success` / `denied` / `failed`) and details. For example: the content hash of the
reviewed definition, the generation id and SQL hash of an execution, or the reason for a denial.

Events recorded:
- `definition.ask`
- `definition.submit`
- `definition.review`
- `definition.execute`
- `authz.denied`
- `proxy.ask`, `proxy.submit`, `proxy.compare`, `proxy.evaluate_reference`, `proxy.evaluation_review`, `reference.load`
- `proxy.patient_explanation` (success and denied; the patient is recorded only as a hash of
  generation id and subject id, never the id itself)

No tokens or patient data are written to the audit log. Admins read it through `GET /audit`.

For tamper evidence, ship these rows to an external write-once store. That is a deployment step and is not
implemented here.

## Small-cell suppression

Every count that leaves the core passes through `executor.suppress_series` / `suppress_count`. That covers
dry-run attrition, `ask`, `validate`, `execute`, CLI output, MCP tools, the API, and evaluation reports. The
default threshold is `min_cell_count: 10`, set in `ontology/domain.yaml`.

- **Primary suppression.** A count from 1 to k−1 is shown as `"<k"`. Zero is disclosed, because it describes
  nobody.
- **Complementary suppression.** In an attrition table, a count is shown as `"suppressed"` if its difference
  from the last disclosed count is between 1 and k−1. Otherwise that small group could be recovered by
  subtraction. The final `person_count` always matches the suppressed table.
- **Counts next to a disclosed total** (proxy evidence counts among candidates, reference labels) are
  hidden if the count *or its complement* is small (`suppress_with_total`).
- **Partitions** (proxy tiers, which sum to the candidates): if exactly one cell is hidden, the smallest
  other non-zero cell is hidden too, so it cannot be recovered by subtraction (`suppress_partition`).
- **Evaluation reports** show only aggregate, suppressed counts (reference patients, labels, eligible,
  excluded by reason, confusion matrix). Excluded or labelled patients are never listed.
- **Overlaps** between two generations: small cells, and any cell recoverable from a disclosed cohort size,
  are hidden. **Confusion matrices:** if any cell is small, all non-zero cells are hidden and the metrics are
  withheld.
- **Patient-level proxy output** (per-patient evidence in `results.proxy_assignment` /
  `results.proxy_evidence`) is returned only to admins when `CB_ALLOW_PATIENT_LEVEL=true`; it is off by
  default and every access is audited.
- **Raw counts and patient rows stay in the database:** `results.cohort` and `meta.cohort_generation`. No
  interface returns patient-level rows.

**What it does not guarantee:**
- **Differencing across queries.** It does not protect against comparing results across *different*
  definitions, e.g. two age bands submitted one after the other.
- **Repeated querying.** It does not protect against many similar queries over time.
- **Linkage.** It does not protect against linkage with outside data.

Limiting those risks needs query auditing, rate limits or noise (differential privacy), and review of what
executors may submit. Treat suppression as one layer, not as anonymization.

## Execution limits and least privilege

- **Query timeout.** `CB_QUERY_TIMEOUT_SECONDS` (default 300) cancels long cohort queries through DuckDB's
  interrupt. The API returns `504`.
- **Resource limits.** `CB_DUCKDB_MEMORY_LIMIT` (e.g. `4GB`) and `CB_DUCKDB_THREADS` cap resources. Both
  values are validated before use.
- **External-access lock.** After setup, `SET enable_external_access=false` is applied
  (`CB_DUCKDB_LOCK_EXTERNAL_ACCESS`, default on). SQL can no longer read or write files, attach databases or
  load extensions, and the lock cannot be undone on that connection.
- **Statement guard.** The executor only runs a single `SELECT`/`WITH` statement with no write or
  administrative keywords. This is defense in depth: SQL is generated only from the typed IR, and free text
  never reaches it.
- **Identifier validation.** Dataset profile identifiers (table and column names) are checked against a plain
  identifier pattern when the profile is loaded.
- **Safe errors.** Database errors are logged server-side with an error id. Clients receive only a generic
  message and that id, never SQL text or table names.
- **Least privilege at deployment level.** Run the service under an OS account that can read the database file
  and nothing else it doesn't need. If you move execution to PostgreSQL or a warehouse, use a role with
  `SELECT` on CDM and vocabulary schemas and `INSERT` on `results` only. A PostgreSQL executor is not
  implemented yet; compiled SQL is tested on PostgreSQL 16 in CI.

## Sensitive data handled by the application

| Data | Where | Notes |
|---|---|---|
| Natural-language prompts | `meta.agent_run.user_query`, `meta.llm_call` | May describe sensitive study questions. Access is limited to the author, reviewers and admins. Define a retention period |
| LLM requests and responses | `meta.llm_call`, `meta.llm_cache` | Contain prompts, ontology summaries and tool results (vocabulary data, suppressed counts), never patient rows. Leave `CB_LLM_MODE` as is, but consider purging `llm_call` after the retention period |
| Cohort membership | `results.cohort` | Patient-level. Restrict database access to authorized analysts |
| Proxy tiers, scores and per-patient evidence | `results.proxy_assignment`, `results.proxy_evidence` | Patient-level. Returned only to admins with `CB_ALLOW_PATIENT_LEVEL=true` |
| Reference-standard labels | `meta.reference_label` | Patient-level, from an external source. Admin-only input, immutable per name, never returned |
| Tokens | Hashes only, in the token file | Protect the file anyway (it maps identities to roles) |

The LLM provider receives prompts and tool results. Check your organization's policy and agreements before
sending real study questions to an external API.

## Items requiring deployment-level assessment

- TLS termination, network isolation, rate limiting and request-size limits (put the API behind a reverse
  proxy or gateway)
- Enterprise SSO/OIDC, joiner/mover/leaver processes, token rotation
- Backup, retention and deletion policies for `meta` and `results`
- External, tamper-evident audit storage and monitoring or alerting on `denied` and `failed` events
- Privacy review of the suppression threshold and of differencing risks for your data
- Clinical validation of concept sets, cohort logic and dataset profiles (see README, "Validation results")
