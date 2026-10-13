"""Set-based episode engine (DuckDB SQL), with bounded streaming only where SQL cannot be exact.

Semantics are defined in treatment_episodes.py (whose `build` is the in-memory reference / test oracle).
Every bulk step runs in DuckDB over run-scoped staging tables `tx_stage.<step>_<run hex>`; Python never
holds the run's events, episodes or transitions.

Coverage (step 4): with overlap 'ignore' the latest covered day is a running max - a window function.
With 'shift_forward' and no cap, coverage end E_i = max(start_i - 1, E_{i-1}) + supply_i, which unrolls to
      E_i = D_i + max_{k<=i} (start_k - 1 - D_k + supply_k)        (D_i = cumulative supply)
i.e. a running sum and a running max - also window functions. The carry-over cap turns this into a clamped
recurrence with no exact window form. It coincides with the uncapped one until the first day whose
uncapped carry exceeds the cap, so: compute every partition in SQL, find partitions where the cap binds
(carry > cap anywhere), and recompute only those by streaming their days in order (bounded fetch batches,
per-partition O(1) state, bulk writes) through the shared `coverage_partition`. Results are identical
for any batch size: partitions are never split, state is per partition.

Transitions: restart / resumed via LAG over each (person, group); product_change via LAG within an
episode; cross-group pairs via a join on (person, period) bounded by
      a.start <= b.start <= a.end + switch_window_days + 1
so only candidates that can qualify are compared (all qualifying pairs are kept). Cost is
O(sum over person-periods of candidate pairs); the worst case is one person-period with many
simultaneously overlapping episodes of different groups (quadratic in that person's episodes only).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import duckdb

from .csv_ingest import qi
from .treatment_config import TreatmentRuleSet
from .treatment_episodes import Day, coverage_partition

STAGE_SCHEMA = "tx_stage"
OUTPUT_HASH_VERSION = "treatment-output/1"
EPOCH = "DATE '1970-01-01'"


class RunCancelled(RuntimeError):
    """The run exceeded its time budget (or was cancelled)."""


@dataclass
class Stage:
    """Names of one run's staging tables (validated hex suffix)."""

    run_hex: str

    def __post_init__(self) -> None:
        if len(self.run_hex) != 32 or any(c not in "0123456789abcdef" for c in self.run_hex):
            raise ValueError("invalid run id")

    def t(self, step: str) -> str:
        if step not in self.STEPS:
            raise ValueError("invalid staging step")
        return f"{qi(STAGE_SCHEMA)}.{qi(f'{step}_{self.run_hex}')}"

    # per-run: ev0 (canonical events), obs; per person bucket: ev st day cov dec ep tr; accumulated: a*
    STEPS = ("ev0", "obs", "ev", "st", "day", "cov", "dec", "ep", "tr", "adec", "aep", "atr")


@dataclass
class Metrics:
    timings: dict[str, float] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)

    def timed(self, name: str) -> _Timer:
        return _Timer(self, name)


class _Timer:
    def __init__(self, m: Metrics, name: str) -> None:
        self.m, self.name = m, name

    def __enter__(self) -> None:
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self.m.timings[self.name] = round(self.m.timings.get(self.name, 0.0) + time.perf_counter() - self.t0, 4)


def _gap_case(rules: TreatmentRuleSet, col: str = "drug_group") -> str:
    """Per-group max gap as SQL (group names are validated identifiers)."""
    whens = " ".join(f"WHEN '{g}' THEN {int(rules.gap_for(g))}" for g in sorted(rules.drug_groups))
    return f"CASE {col} {whens} END"


def drop_stage(con: duckdb.DuckDBPyConnection, stage: Stage) -> None:
    for step in Stage.STEPS:
        con.execute(f"DROP TABLE IF EXISTS {stage.t(step)}")


# ---- steps 0-3: observation periods, record merge, duplicates, dispensing days ----------------------------
def observation_sql(stage: Stage, table: str, key: str, start_col: str, end_col: str) -> str:
    """Merged (overlapping or touching) observation periods of the run's persons, numbered per person."""
    tab = ".".join(qi(x) for x in table.split("."))
    return f"""
CREATE TABLE {stage.t("obs")} AS
WITH p AS (
  SELECT CAST(o.{qi(key)} AS BIGINT) AS person_id, CAST(o.{qi(start_col)} AS DATE) AS s,
         CAST(o.{qi(end_col)} AS DATE) AS e
  FROM {tab} o
  WHERE o.{qi(start_col)} IS NOT NULL AND o.{qi(end_col)} IS NOT NULL AND o.{qi(end_col)} >= o.{qi(start_col)}
    AND CAST(o.{qi(key)} AS BIGINT) IN (SELECT DISTINCT person_id FROM {stage.t("ev0")} WHERE included)),
q AS (
  SELECT *, max(e) OVER (PARTITION BY person_id ORDER BY s, e ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            AS prev_max FROM p),
r AS (
  SELECT *, sum(CASE WHEN prev_max IS NULL OR s > prev_max + 1 THEN 1 ELSE 0 END)
            OVER (PARTITION BY person_id ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS period_id FROM q)
SELECT person_id, CAST(period_id AS INTEGER) AS period_id, min(s) AS obs_start, max(e) AS obs_end
FROM r GROUP BY person_id, period_id
"""


def steps_sql(
    stage: Stage, rules: TreatmentRuleSet, has_observation: bool, bucket: int = 0, buckets: int = 1
) -> list[str]:
    """Steps 0-3 for one person bucket (person_id mod buckets = bucket). Every partition, episode and
    transition belongs to one person, so buckets are independent and the result does not depend on their
    number - it only bounds how much each statement has to hold."""
    period = (
        f"LEFT JOIN {stage.t('obs')} o ON o.person_id = e.person_id AND e.start_date BETWEEN o.obs_start AND o.obs_end"
        if has_observation
        else ""
    )
    pid = "o.period_id" if has_observation else "0"
    dup = "same_product_same_day" == rules.episode.duplicates
    agg = "sum" if rules.episode.same_day_same_group == "sum_supply" else "max"
    return [
        f"""
CREATE TABLE {stage.t("ev")} AS
SELECT e.event_id, e.person_id, e.drug_group, e.product_concept_id AS product, e.start_date,
       e.days_supply AS days, e.source_name || ':' || e.source_record_id AS record, {pid} AS period_id
FROM {stage.t("ev0")} e {period}
WHERE e.included AND ((e.person_id % {int(buckets)}) + {int(buckets)}) % {int(buckets)} = {int(bucket)}
""",
        # one pass of window functions (no self-joins): record merge -> duplicates -> dispensing-day order
        f"""
CREATE TABLE {stage.t("st")} AS
WITH a AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY person_id, drug_group, period_id, record ORDER BY event_id) AS rec_rn
  FROM {stage.t("ev")} WHERE period_id IS NOT NULL),
b AS (
  SELECT *, CASE WHEN rec_rn = 1 THEN {
            "ROW_NUMBER() OVER (PARTITION BY person_id, drug_group, period_id, start_date, product, days, rec_rn = 1 "
            "ORDER BY event_id)"
            if dup
            else "1"
        } END AS dup_rn
  FROM a),
c AS (
  SELECT *, max(dup_rn) OVER (PARTITION BY person_id, drug_group, period_id, record) AS record_dup_rn,
         rec_rn = 1 AND dup_rn = 1 AS kept
  FROM b),
d AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY person_id, drug_group, period_id, start_date, kept
                               ORDER BY product, days DESC, event_id) AS day_rn
  FROM c)
SELECT event_id, person_id, drug_group, period_id, start_date, product,
       CASE WHEN rec_rn > 1 AND record_dup_rn > 1 THEN 'duplicate_event'
            WHEN rec_rn > 1 THEN 'same_record_merged'
            WHEN dup_rn > 1 THEN 'duplicate_event'
            WHEN day_rn = 1 THEN 'counted'
            ELSE 'same_day_merged' END AS step
FROM d
UNION ALL
SELECT event_id, person_id, drug_group, NULL, start_date, product, 'outside_observation'
FROM {stage.t("ev")} WHERE period_id IS NULL
""",
        f"""
CREATE TABLE {stage.t("day")} AS
SELECT s.person_id, s.drug_group, s.period_id, s.start_date, {agg}(e.days) AS supply,
       array_to_string(list_sort(list_distinct(list(s.product))), ',') AS products
FROM {stage.t("st")} s JOIN {stage.t("ev")} e ON e.event_id = s.event_id
WHERE s.step IN ('counted', 'same_day_merged')
GROUP BY s.person_id, s.drug_group, s.period_id, s.start_date
""",
    ]


# ---- steps 4-5: coverage and episodes ----------------------------------------------------------------------
def coverage_sql(stage: Stage, rules: TreatmentRuleSet) -> str:
    k = "PARTITION BY person_id, drug_group, period_id ORDER BY s ROWS UNBOUNDED PRECEDING"
    if rules.episode.overlap == "shift_forward":
        end_expr = f"dsum + max(s - 1 - dsum + supply) OVER ({k})"
        carry = "greatest(0, coalesce(e_prev - s + 1, 0))"
    else:
        end_expr = f"max(s + supply - 1) OVER ({k})"
        carry = "0"
    return f"""
CREATE TABLE {stage.t("cov")} AS
WITH d AS (
  SELECT *, CAST(start_date - {EPOCH} AS BIGINT) AS s, {_gap_case(rules)} AS gap FROM {stage.t("day")}),
w AS (SELECT *, sum(supply) OVER ({k}) AS dsum FROM d),
x AS (SELECT *, {end_expr} AS e_run FROM w),
y AS (SELECT *, lag(e_run) OVER (PARTITION BY person_id, drug_group, period_id ORDER BY s) AS e_prev FROM x),
z AS (SELECT *, {carry} AS carry FROM y),
c AS (
  SELECT person_id, drug_group, period_id, start_date, supply, products, gap, carry, s,
         s + carry AS cs, s + carry + supply - 1 AS ce,
         CASE WHEN e_prev IS NULL OR s + carry - e_prev - 1 > gap THEN 1 ELSE 0 END AS is_new,
         CASE WHEN e_prev IS NULL OR s + carry - e_prev - 1 > gap THEN 0
              ELSE greatest(0, s + carry - e_prev - 1) END AS bridged
  FROM z)
SELECT person_id, drug_group, period_id, start_date, supply, products, gap,
       CAST({EPOCH} + CAST(cs AS INTEGER) AS DATE) AS coverage_start,
       CAST({EPOCH} + CAST(ce AS INTEGER) AS DATE) AS coverage_end,
       CAST(carry AS INTEGER) AS shifted_days,
       CAST(sum(is_new) OVER (PARTITION BY person_id, drug_group, period_id ORDER BY s ROWS UNBOUNDED PRECEDING)
            AS INTEGER) AS episode_no,
       CAST(bridged AS INTEGER) AS bridged
FROM c
"""


def capped_partitions_sql(stage: Stage, cap: int) -> str:
    return (
        f"SELECT count(*) FROM (SELECT person_id, drug_group, period_id FROM {stage.t('cov')} "
        f"GROUP BY ALL HAVING max(shifted_days) > {int(cap)})"
    )


def recompute_capped(
    con: duckdb.DuckDBPyConnection,
    reader: duckdb.DuckDBPyConnection,
    stage: Stage,
    rules: TreatmentRuleSet,
    batch_size: int,
    check: Callable[[], None],
) -> int:
    """Exact capped coverage for the partitions where the cap binds: stream their days in order (bounded
    fetch batches, one partition's state at a time), run the shared recurrence, write results in bulk
    batches, then update those partitions' coverage rows. Returns the number of day rows recomputed."""
    cap = int(rules.episode.max_carryover_days)
    todo, new = stage.t("dec"), stage.t("tr")  # both free at this point; re-created later
    con.execute(
        f"CREATE TABLE {todo} AS SELECT c.person_id, c.drug_group, c.period_id, c.start_date, c.supply "
        f"FROM {stage.t('cov')} c JOIN (SELECT person_id, drug_group, period_id FROM {stage.t('cov')} "
        f"GROUP BY ALL HAVING max(shifted_days) > {cap}) USING (person_id, drug_group, period_id)"
    )
    con.execute(f"CREATE TABLE {new} ({', '.join(f'{qi(n)} {t}' for n, t in COV_NEW)})")
    cur = reader.execute(
        f"SELECT person_id, drug_group, period_id, start_date, supply FROM {todo} "
        "ORDER BY person_id, drug_group, period_id, start_date"
    )
    out: list[tuple] = []
    written = 0
    key: tuple | None = None
    days: list[Day] = []

    def finish_partition() -> None:
        if key is not None and days:
            for c in coverage_partition(days, rules.episode, rules.gap_for(key[1])):
                out.append((*key, c.start, c.coverage_start, c.coverage_end, c.shifted_days, c.episode_no, c.bridged))
        days.clear()

    while rows := cur.fetchmany(batch_size):
        check()
        for person, group, period, start, supply in rows:
            if (person, group, period) != key:
                finish_partition()
                key = (person, group, period)
            days.append(Day(start, int(supply)))
        if len(out) >= batch_size:
            _bulk_insert(con, new, COV_NEW, out)
            written += len(out)
            out.clear()
    finish_partition()
    _bulk_insert(con, new, COV_NEW, out)
    written += len(out)
    con.execute(
        f"UPDATE {stage.t('cov')} c SET coverage_start = n.coverage_start, coverage_end = n.coverage_end, "
        f"shifted_days = n.shifted_days, episode_no = n.episode_no, bridged = n.bridged FROM {new} n "
        "WHERE c.person_id = n.person_id AND c.drug_group = n.drug_group AND c.period_id = n.period_id "
        "AND c.start_date = n.start_date"
    )
    con.execute(f"DROP TABLE {new}")
    con.execute(f"DROP TABLE {todo}")
    return written


COV_NEW = [
    ("person_id", "BIGINT"),
    ("drug_group", "VARCHAR"),
    ("period_id", "INTEGER"),
    ("start_date", "DATE"),
    ("coverage_start", "DATE"),
    ("coverage_end", "DATE"),
    ("shifted_days", "INTEGER"),
    ("episode_no", "INTEGER"),
    ("bridged", "INTEGER"),
]


def _bulk_insert(con: duckdb.DuckDBPyConnection, table: str, columns: Sequence[tuple[str, str]], rows: list) -> None:
    """One INSERT per call: each column is passed as one typed list parameter and unnested (no per-row
    statements; executemany runs ~2,000 rows/s in DuckDB, this ~150,000+)."""
    if not rows:
        return
    cols = list(zip(*rows, strict=True))
    select = ", ".join(f"unnest(?::{typ}[])" for _, typ in columns)
    names = ", ".join(qi(n) for n, _ in columns)
    con.execute(f"INSERT INTO {table} ({names}) SELECT {select}", [list(c) for c in cols])


def bulk_insert(con: duckdb.DuckDBPyConnection, table: str, columns: Sequence[tuple[str, str]], rows: list,
                batch_size: int) -> None:  # fmt: skip
    for i in range(0, len(rows), batch_size):
        _bulk_insert(con, table, columns, rows[i : i + batch_size])


def episodes_sql(stage: Stage, rules: TreatmentRuleSet) -> list[str]:
    washout = int(rules.censoring.washout_days)
    return [
        # event decisions (step, coverage, episode) for every event that reached the episode engine
        f"""
CREATE TABLE {stage.t("dec")} AS
SELECT s.event_id, s.person_id, s.drug_group, s.period_id, s.start_date, s.product, s.step,
       CASE WHEN s.step IN ('counted', 'same_day_merged', 'same_record_merged') THEN c.coverage_start END
         AS coverage_start,
       CASE WHEN s.step IN ('counted', 'same_day_merged', 'same_record_merged') THEN c.coverage_end END
         AS coverage_end,
       CASE WHEN s.step = 'counted' THEN c.shifted_days ELSE 0 END AS shifted_days,
       CASE WHEN s.step IN ('counted', 'same_day_merged', 'same_record_merged') THEN c.episode_no END AS episode_no
FROM {stage.t("st")} s
LEFT JOIN {stage.t("cov")} c ON c.person_id = s.person_id AND c.drug_group = s.drug_group
     AND c.period_id = s.period_id AND c.start_date = s.start_date
""",
        f"""
CREATE TABLE {stage.t("ep")} AS
WITH g AS (
  SELECT person_id, drug_group, period_id, episode_no, min(start_date) AS episode_start,
         max(coverage_end) AS episode_end, sum(supply) AS days_supplied, sum(bridged) AS gap_days_bridged,
         any_value(gap) AS max_gap_days
  FROM {stage.t("cov")} GROUP BY person_id, drug_group, period_id, episode_no),
m AS (
  SELECT person_id, drug_group, period_id, episode_no, count(*) AS n_events, count(DISTINCT product) AS n_products
  FROM {stage.t("dec")} WHERE episode_no IS NOT NULL GROUP BY ALL),
n AS (
  SELECT g.*, m.n_events, m.n_products, o.obs_start, o.obs_end,
         g.person_id || ':' || g.drug_group || ':' || CAST(ROW_NUMBER() OVER (
           PARTITION BY g.person_id, g.drug_group ORDER BY g.episode_start) AS VARCHAR) AS episode_id
  FROM g JOIN m USING (person_id, drug_group, period_id, episode_no)
  LEFT JOIN {stage.t("obs")} o ON o.person_id = g.person_id AND o.period_id = g.period_id)
SELECT episode_id, person_id, drug_group, period_id, episode_no, episode_start, episode_end,
       CAST(n_events AS INTEGER) AS n_events, CAST(days_supplied AS INTEGER) AS days_supplied,
       CAST(gap_days_bridged AS INTEGER) AS gap_days_bridged, CAST(max_gap_days AS INTEGER) AS max_gap_days,
       CAST(n_products AS INTEGER) AS n_products,
       CASE WHEN obs_start IS NULL THEN NULL ELSE episode_start - obs_start < {washout} END AS left_censored,
       CASE WHEN obs_end IS NULL THEN NULL ELSE episode_end + max_gap_days >= obs_end END AS right_censored,
       CASE WHEN obs_end IS NULL THEN 'unknown' WHEN episode_end + max_gap_days >= obs_end THEN 'observation_end'
            ELSE 'gap_exceeded' END AS end_reason,
       obs_start AS observation_start, obs_end AS observation_end,
       CASE WHEN obs_end IS NULL THEN 'unknown' WHEN episode_end + max_gap_days >= obs_end THEN 'censored'
            ELSE 'discontinued' END AS end_status,
       CASE WHEN obs_end IS NULL THEN episode_end ELSE least(episode_end, obs_end) END AS observed_end,
       coalesce(episode_end > obs_end, false) AS coverage_beyond_observation
FROM n
""",
    ]


def transitions_sql(stage: Stage, rules: TreatmentRuleSet) -> str:
    tr = rules.transitions
    add_on = int(tr.add_on_min_overlap_days)
    window = int(tr.switch_window_days)
    ep = stage.t("ep")
    return f"""
CREATE TABLE {stage.t("tr")} AS
WITH seq AS (
  SELECT *, lag(episode_id) OVER w AS prev_id, lag(episode_end) OVER w AS prev_end,
         lag(period_id) OVER w AS prev_period, lag(max_gap_days) OVER w AS prev_gap
  FROM {ep} WINDOW w AS (PARTITION BY person_id, drug_group ORDER BY episode_start)),
same_group AS (
  SELECT person_id,
         CASE WHEN prev_period = period_id THEN 'restart' ELSE 'resumed_after_unobserved_interval' END AS kind,
         prev_id AS from_episode, episode_id AS to_episode, drug_group AS from_group, drug_group AS to_group,
         episode_start AS on_date, CAST(episode_start - prev_end - 1 AS INTEGER) AS days,
         CASE WHEN prev_period = period_id THEN 'new episode after a gap > ' || CAST(prev_gap AS VARCHAR) || ' days'
              ELSE 'new episode in a later observation period (gap not observed)' END AS basis,
         CAST(NULL AS VARCHAR) AS from_products, CAST(NULL AS VARCHAR) AS to_products
  FROM seq WHERE prev_id IS NOT NULL),
days AS (
  SELECT c.*, lag(products) OVER (PARTITION BY person_id, drug_group, period_id, episode_no ORDER BY start_date)
         AS prev_products
  FROM {stage.t("cov")} c),
product AS (
  SELECT d.person_id, 'product_change' AS kind, e.episode_id AS from_episode, e.episode_id AS to_episode,
         d.drug_group AS from_group, d.drug_group AS to_group, d.start_date AS on_date, CAST(NULL AS INTEGER) AS days,
         'product set differs from the previous dispensing day' AS basis,
         d.prev_products AS from_products, d.products AS to_products
  FROM days d JOIN {ep} e USING (person_id, drug_group, period_id, episode_no)
  WHERE d.prev_products IS NOT NULL AND d.prev_products <> d.products),
pairs AS (
  SELECT a.person_id, a.episode_id AS a_id, b.episode_id AS b_id, a.drug_group AS a_group, b.drug_group AS b_group,
         a.episode_start AS a_start, a.episode_end AS a_end, b.episode_start AS b_start, b.episode_end AS b_end,
         CASE WHEN b.episode_start <= a.episode_end
              THEN CAST(least(a.episode_end, b.episode_end) - b.episode_start + 1 AS INTEGER) ELSE 0 END AS overlap
  FROM {ep} a JOIN {ep} b ON a.person_id = b.person_id AND a.period_id = b.period_id
   AND a.drug_group <> b.drug_group AND b.episode_start >= a.episode_start
   AND b.episode_start <= a.episode_end + {window + 1}),
cross_group AS (
  SELECT person_id,
         CASE WHEN b_start = a_start THEN 'concurrent_start'
              WHEN overlap >= {add_on} THEN 'add_on' ELSE 'switch' END AS kind,
         a_id AS from_episode, b_id AS to_episode, a_group AS from_group, b_group AS to_group, b_start AS on_date,
         CASE WHEN b_start = a_start THEN 0 WHEN overlap >= {add_on} THEN overlap
              ELSE CAST(greatest(0, b_start - a_end - 1) AS INTEGER) END AS days,
         CASE WHEN b_start = a_start THEN 'both groups start on the same day'
              WHEN overlap >= {add_on} THEN 'overlap >= {add_on} days'
              ELSE 'starts within {window} days of the end, overlap < {add_on} days' END AS basis,
         CAST(NULL AS VARCHAR) AS from_products, CAST(NULL AS VARCHAR) AS to_products
  FROM pairs
  WHERE (b_start = a_start AND a_group < b_group)
     OR (b_start > a_start AND overlap >= {add_on})
     OR (b_start > a_start AND overlap < {add_on} AND b_end > a_end AND b_start - a_end - 1 <= {window}))
SELECT * FROM same_group UNION ALL SELECT * FROM product UNION ALL SELECT * FROM cross_group
"""


# ---- output hash (contract treatment-output/1) ------------------------------------------------------------
_SEP = "chr(31)"  # unit separator between fields; NULL is written as \\N


def _row_text(cols: Sequence[str]) -> str:
    parts = [f"coalesce(CAST({qi(c)} AS VARCHAR), '\\N')" for c in cols]
    return f"concat_ws({_SEP}, {', '.join(parts)})"


HASH_SECTIONS: list[tuple[str, str, list[str], list[str]]] = [
    (
        "events",
        "results.treatment_event",
        [
            "event_id", "person_id", "source_name", "source_table", "source_record_id", "source_code",
            "source_vocabulary", "product_concept_id", "ingredient_concept_id", "brand_concept_id", "drug_group",
            "n_ingredients", "evidence_type", "claim_status", "start_date", "start_date_source", "end_date",
            "end_date_source", "days_supply", "days_supply_source", "quantity", "evidence_status", "quality_flags",
            "included", "exclusion_reason", "episode_step", "coverage_start", "coverage_end", "shifted_days",
            "episode_id",
        ],
        ["event_id"],
    ),
    (
        "episodes",
        "results.treatment_episode",
        [
            "episode_id", "person_id", "drug_group", "period_id", "episode_start", "episode_end", "n_events",
            "days_supplied", "gap_days_bridged", "max_gap_days", "n_products", "left_censored", "right_censored",
            "end_reason", "end_status", "observation_start", "observation_end", "observed_end",
            "coverage_beyond_observation",
        ],
        ["episode_id"],
    ),
    (
        "transitions",
        "results.treatment_transition",
        [
            "person_id", "kind", "from_episode", "to_episode", "from_group", "to_group", "on_date", "days",
            "from_products", "to_products", "basis",
        ],
        ["person_id", "on_date", "kind", "from_episode", "to_episode", "from_products", "to_products"],
    ),
]  # fmt: skip


def output_hash(con: duckdb.DuckDBPyConnection, run_id: str, batch_size: int) -> str:
    """sha256 over: 'treatment-output/1\\n', then per section '#<name>\\n' followed by one line per row
    (fields in the listed order, CAST to VARCHAR, joined by U+001F, NULL as \\N), rows ordered by the
    section's key. Streamed in batches: identical for any batch size, never one large string."""
    h = hashlib.sha256(f"{OUTPUT_HASH_VERSION}\n".encode())
    for name, table, cols, order in HASH_SECTIONS:
        h.update(f"#{name}\n".encode())
        cur = con.execute(
            f"SELECT {_row_text(cols)} FROM {table} WHERE run_id = ? ORDER BY "
            + ", ".join(f"{qi(c)} NULLS FIRST" for c in order),
            [run_id],
        )
        while rows := cur.fetchmany(batch_size):
            h.update(("\n".join(r[0] for r in rows) + "\n").encode())
    return "sha256:" + h.hexdigest()


def iter_batches(cur: duckdb.DuckDBPyConnection, batch_size: int) -> Iterator[list[tuple[Any, ...]]]:
    while rows := cur.fetchmany(batch_size):
        yield rows
