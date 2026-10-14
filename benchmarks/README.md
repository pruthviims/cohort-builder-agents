# Treatment engine benchmarks (synthetic data only)

`treatment_bench.py` generates a seeded synthetic claims workload in DuckDB, then times one treatment
run in a fresh process. It never uses real data, and its output contains counts and timings only.

```bash
python benchmarks/treatment_bench.py generate --db /tmp/b.duckdb --patients 100000 --fills 10 \
    --hf-patients 100 --hf-fills 10000 --seed 7
python benchmarks/treatment_bench.py run --db /tmp/b.duckdb --memory-limit 4GB --threads 2
# the pre-v2 engine, for comparison (needs a schema created by that version):
python benchmarks/treatment_bench.py generate --db /tmp/old.duckdb ... --schema-src <old checkout>/src
python benchmarks/treatment_bench.py run --db /tmp/old.duckdb --legacy-path <old checkout>/src
```

**Workload contents:**
- 0–3 observation periods per patient (overlapping, touching or separated)
- long fill histories across metformin, two brands of empagliflozin, dapagliflozin and a combination
  product: brand changes, switches and add-ons
- early refills (overlapping supply), same-day fills, gaps and restarts
- about 1% missing and 0.5% implausible days supply, 2% rejected, 2% reversed (plus the reversal rows),
  1% adjusted (plus the adjustment rows), and 1% duplicated claim lines
- `--hf-patients` patients with `--hf-fills` fills every 1–3 days. This is high-frequency stockpiling,
  which makes the carry-over cap bind and exercises the streaming path.

## Environment of the reported runs

| | |
|---|---|
| Machine | cloud container, 2 vCPU, 7.8 GiB RAM, local SSD |
| Software | Linux 6.18 x86_64, Python 3.13.16, DuckDB 1.5.6 |
| Settings | `--memory-limit 4GB --threads 2`, batch size 100,000 (S2 also 10,000), `CB_TREATMENT_BUCKET_EVENTS` 5,000,000 |
| Code | after: `a92b083` (this branch); before: `af46a4c` (`main` before this work) |
| Seed | 7 |

Each run was alone on the machine.

## Results: after (set-based SQL engine)

"Events" are canonical treatment events: one per source record and ingredient, so combination products
count twice. Throughput is source rows divided by the total run time.

| Workload | Patients | Source rows | Events | Episodes | Transitions | Total | Rows/s | Peak RSS | Peak DuckDB temp | DB growth |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10,000 | 103,600 | 120,427 | 34,859 | 64,090 | 4.5 s | 23,141 | 485 MB | 0 | 22 MB |
| S2 (+100 × 10,000 high-frequency) | 100,100 | 2,080,500 | 2,419,594 | 350,956 | 691,130 | 36.5 s | 56,961 | 1,367 MB | 0 | 313 MB |
| S2, batch size 10,000 | same | same | same | same | same | 34.5 s | 60,348 | 1,383 MB | 0 | 107 MB |
| S3 (+100 × 10,000 high-frequency) | 1,000,100 | 11,439,076 | 13,304,223 | 3,500,110 | 6,529,990 | 274 s | 41,695 | 4,398 MB | 1,139 MB | 1,840 MB |
| S4 stress (2M × 15 fills, +500 × 10,000 high-frequency) | 2,000,500 | 36,386,049 | 42,319,211 | 8,884,899 | 18,751,825 | 1,055 s | 34,476 | 4,923 MB | 8,983 MB | 5,089 MB |

**Phase timings (seconds).** "Episodes" covers all per-bucket steps; the capped-streaming and
transition times are parts of it, shown separately.

| Workload | Events (SQL) | Episodes | of which capped streaming | of which transitions | Publish | Output hash | Summary | Buckets | Capped partitions / day rows |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.74 | 2.06 | 0.21 | 0.15 | 0.44 | 0.56 | 0.06 | 1 | 1,163 / 8,893 |
| S2 | 7.58 | 12.48 | 2.75 | 1.28 | 6.82 | 8.91 | 0.56 | 1 | 11,603 / 155,913 |
| S3 | 45.6 | 109.4 | 14.5 | 11.0 | 46.8 | 67.2 | 5.2 | 3 | 114,189 / 956,855 |
| S4 | 190.6 | 337.5 | 51.8 | 29.8 | 262.5 | 247.7 | 16.5 | 7 | 341,682 / 3,628,547 |

**Peak memory:** peak RSS is the whole process, including DuckDB's buffer pool, which was capped at
4 GB. Python itself never holds a run's events. Above that cap DuckDB spills to its temp directory,
which is why S3 and S4 show temp usage.

**Determinism:** these runs reproduced the same output hashes:
- S2 with batch sizes 100,000 and 10,000;
- S1–S3 before and after person bucketing (1 bucket vs 3 at S3).

## Reliability / claims-normalization changes (`main` `8285424` vs this branch)

Same machine and settings, the same generated database for both sides, runs alternated (before, after,
before, after), each in a fresh process. "After" includes the new claims consistency checks (reference
scope, conflicting duplicates, reversal ordering, quantity), the input fingerprint (one extra scan of the
source tables) and the per-run connection. S4 was not repeated.

| Workload | Before: total (2 runs) | After: total (2 runs) | Events phase before / after | Peak RSS before / after | Peak DuckDB temp before / after | Output hash |
|---|---|---|---|---|---|---|
| S1 | 4.12 s, 4.10 s | 4.60 s, 4.27 s | 0.77, 0.73 / 0.80, 0.76 s | 474–490 / 468–472 MB | 0 / 0 | identical |
| S2 | 34.3 s, 36.3 s | 37.4 s, 37.4 s | 7.0, 7.2 / 8.4, 8.1 s | 1,372–1,378 / 1,362–1,391 MB | 0 / 0 | identical |
| S3 | 258.3 s, 256.8 s | 271.9 s, 261.8 s | 43.9, 43.9 / 53.6, 52.4 s | 4,402–4,442 / 4,351–4,377 MB | 1,173–1,188 / 1,753–1,775 MB | identical |

**Result: no speed-up.** This change is about correctness and operability, and it costs 2–6% of wall time
(S3: +10 s, almost all in the events phase, where the new consistency CTEs run). The events phase also
spills about 0.6 GB more at S3. Output hashes are identical on this workload, because it contains none
of the patterns the new flags detect.

**10,000 synthetic T2D patients** (`init-t2d-demo --patients 10000 --seed 2026`: 292,130 pharmacy rows,
164,423 diagnosis rows; 297,240 events, 13,247 episodes, 4,972 transitions), CLI end to end with
`CB_DUCKDB_MEMORY_LIMIT=4GB`, `CB_DUCKDB_THREADS=2`:

| | Run 1 | Run 2 | Peak RSS |
|---|---|---|---|
| Before (`8285424`) | 6.7 s | 6.1 s | 524–529 MB |
| After (this branch) | 6.9 s | 6.4 s | 523–529 MB |

Generating that database took 8.3 s (peak RSS 381 MB). Before and after differ in output hash on this
data. `diff_runs` on both runs over the same database shows that episodes and transitions are identical,
and that inclusion and exclusion are identical for every event. 2,507 events differ, and only in
`quality_flags`: the new consistency flags fire on the injected artefacts (cross-patient and early
reversals, conflicting duplicates, invalid quantities).

**Bulk writes without pandas.** The full test suite took 917–982 s per Python version in the CI-like
virtual environments, against 492–600 s on `main`, and the 10k end-to-end test had not finished after
8 minutes there, while it took 21 s with the system Python. The cause was DuckDB's conversion of Python
list parameters. When pandas is not installed, it retries the pandas import for every element:
20,000 values took about 2 s, against 0.02 s with pandas. `bulk_insert` (used by the reference engine,
the synthetic generators and the tests) now sends each column as one JSON array that is parsed in C++.
After the change, the 10k test took 21.5 s in the same environment without pandas. S1 and S2 were
re-measured with unchanged hashes and times (4.15 s and 36.3 s): the SQL engine's main path does not use
`bulk_insert`.

## Before (the pre-v2 engine, `af46a4c`)

| Workload | Total | Rows/s | Peak RSS | Notes |
|---|---|---|---|---|
| S1 (10,000 patients, 103,600 rows) | 417.7 s | 248 | 541 MB | 407.8 s of it was writing rows one at a time (`executemany`, about 2,000 rows/s in DuckDB) |
| S2 (100,100 patients, 2,080,500 rows) | **did not finish in 3,600 s** (killed by `timeout`) | – | ≥ 2,508 MB | Peak RSS sampled from `/proc` while it ran. All events were loaded into Python before writing started |

S3 and S4 were not run on the old engine. At the 248 rows/s measured at S1, S2 alone would need about
2.3 hours. The two engines also produce different outputs (v2 changed the episode and transition rules),
so this is a runtime and memory comparison, not an output comparison.

## Bottlenecks and limits observed

- **Publishing and hashing** dominate at the largest scale: 262 s and 248 s at S4.
  - Publishing writes 42M event rows into `results`.
  - The output hash serializes about 70M rows into a streamed SHA-256 in Python.
  - Both are linear and bounded in memory. They are the next optimization targets: for example, an
    optional per-section hash that skips event rows for very large runs, or a narrower event table.
- **Disk:** DuckDB spilled about 9 GB at S4 under a 4 GB memory limit. `CB_DUCKDB_MAX_TEMP_DIRECTORY_SIZE`
  caps this, and a run that exceeds the cap fails cleanly, recorded as `failed` with the stage. The
  database grows about 120 bytes per event per run, and result rows are kept until `treatment retention`
  purges them (optionally after a verified archive).
- **An earlier version of the engine** failed S4 with a DuckDB out-of-memory error in the episode stage.
  Episodes are now built per person bucket (`CB_TREATMENT_BUCKET_EVENTS`), and the same workload
  completed.
- **The largest workload tested** is S4: 36.4M source rows, 42.3M events, 2M patients, on 2 vCPU and
  7.8 GiB RAM. Nothing larger was run. Production capacity has to be measured on production hardware.
