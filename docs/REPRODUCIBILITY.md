# Treatment engine: reproducibility guarantees

This page states what the treatment-event engine guarantees about repeating a run, how each guarantee is
checked, and where it stops. Episode and transition semantics are defined in
`src/cohort_builder/treatment_episodes.py`; claims normalization in `treatment_events.py`.

## What identifies a run

| Record | Contract | What it covers |
|---|---|---|
| `input_fingerprint` | `treatment-input/1` | every input the output depends on (below) |
| `output_hash` | `treatment-output/1` | the published events (with decisions and coverage), episodes and transitions |
| `rule_set_hash`, `source_config_hash` | `sha256:` of the canonical JSON | the two configuration files, including their provenance block |

The **input fingerprint** is the SHA-256 of a key-ordered JSON document (stored in
`meta.treatment_run.input_fingerprint_json`) holding:

- each source table and the observation-period table **by content**: row count, `bit_xor` of DuckDB's row
  hash, and `sum` of the row hash. Both aggregates ignore row order. The sum also catches an exact
  duplicate row, which cancels out in an XOR. A missing table is recorded as `missing`, and the run then
  fails in its events stage;
- the vocabulary by version label, concept count and highest concept id (not by content, see limits);
- rule set and source configuration hashes, the reference date, the engine and its version, the event
  schema, the output-hash contract, the software version and the DuckDB version.

Batch size, bucket size, thread count and memory settings are deliberately left out, because they do not
change the output (tested below).

## Guarantees and the tests that check them

| Guarantee | Test |
|---|---|
| Same inputs give the same `output_hash` and the same summary | `test_treatment_scale.py`, `test_synthetic_t2d.py` (runs repeated on 1,500 and 10,000 synthetic patients) |
| Independent of batch size and person-bucket count | `test_batch_size_and_person_buckets_do_not_change_results` |
| Independent of physical row order and DuckDB thread count; the fingerprint too | `test_row_order_and_thread_count_do_not_change_output` (original, reversed and shuffled loads, 1 and 2 threads) |
| SQL engine equals the in-memory reference engine row by row, with `diff_runs` diagnostics on failure | `test_sql_engine_matches_reference*`, `test_hand_verified_t2d_fixture[sql/reference]` |
| Hand-computed outcomes on a fixed fixture | `test_t2d_fixture.py` (the arithmetic is written in the file) |
| Fingerprint changes when a row is added, duplicated or changed, or a rule changes | `test_input_fingerprint_is_order_invariant_and_content_sensitive` |
| Concurrent runs on separate connections give the same results as sequential runs | `test_concurrent_runs_on_separate_connections_match_sequential_runs` |
| A failed or cancelled run publishes nothing, so no partial output carries a hash | `test_mocked_resource_exhaustion_publishes_nothing`, `test_cancel_stops_a_running_run` |
| The synthetic generator is deterministic for a seed | `test_generator_is_deterministic_and_labeled` |

## Idempotent retries

`treatment run --reuse` (API: `"reuse": true`) looks for a `completed`, unpurged run of the same tenant
with the same input fingerprint. If it finds one, it returns that run instead of computing a new one, so
a client retry after a timeout does not create a duplicate run. Without `--reuse`, every call creates a
new run. That is useful when you want two independent runs to compare with `diff_runs`.

## Limits (what is NOT guaranteed)

- **DuckDB version.** Row hashes, and with them the fingerprint, are comparable only within one DuckDB
  version. The version is part of the fingerprint, so an upgrade changes it. The output hash is computed
  from values, not DuckDB hashes, and is expected to survive an upgrade, but this is not tested across
  versions.
- **Vocabulary content.** The fingerprint identifies the vocabulary by version label and concept
  count / max id, not by content. A vocabulary edited in place under the same label and counts would not
  change the fingerprint. The output hash would still reveal any changed mapping.
- **Reference date.** It defaults to today, and dates after it are excluded as implausible. Pass
  `--reference-date` to reproduce a run on a later day; it is recorded with every run.
- **Software changes.** A new engine version or event schema is recorded in the run and changes the
  fingerprint. Comparing outputs across versions is a review task (`diff_runs`), not a guarantee.
- **Activity-based observation (LAAD-style data).** Observation periods come from claim activity. Adding
  claims, even ones that are later excluded (for example rejected or future-dated), can move a period
  boundary and change censoring. This is intended (the data changed), and the fingerprint changes with it.
- **Purged runs** keep their hashes and fingerprint in `meta.treatment_run`. Their rows are gone, or
  only in the verified archive, so `explain` and `export` refuse them.
