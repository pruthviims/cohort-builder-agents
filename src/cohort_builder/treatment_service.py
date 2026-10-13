"""Treatment-event runs: events -> episodes -> transitions, stored with full provenance.

A run is determined by (data snapshot, treatment source config, rule set, reference date, engine and
software version); all are recorded in meta.treatment_run, with content hashes for the configurations,
and `output_hash` (contract `treatment-output/1`, see treatment_engine.output_hash) proves that a repeat run
produced identical events, decisions, episodes and transitions.

Run lifecycle (no partial output is ever visible as complete):
  'running'    meta row written first; all work happens in run-scoped staging tables tx_stage.*_<run>
  'completed'  results.treatment_* rows, the output hash, the summary and the status change are written in
               ONE transaction at the end
  'failed'     any error: that transaction never commits, staging is dropped, the run keeps a redacted
               error and the stage that failed (no data values)
  'abandoned'  'running' rows older than CB_TREATMENT_STALE_RUN_MINUTES (a crashed process), cleaned up by
               `cleanup_incomplete_runs` (also called before every run)

Patient-level rows live in results.treatment_* and are only returned through `explain_episode`, which needs
an admin principal AND CB_ALLOW_PATIENT_LEVEL=true (audited). Summaries are aggregate and suppressed.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import duckdb

from . import __version__
from .csv_ingest import qi
from .dataset_mapping import code_lookup_sql
from .db import redact_error
from .executor import suppress_count, suppress_partition, suppress_with_total
from .metadata import dumps, now
from .security import DEFAULT_TENANT, GovernanceError
from .treatment_config import (
    EVENT_SCHEMA_VERSION,
    TreatmentRuleSet,
    TreatmentSourceConfig,
    load_rules,
    load_sources,
    rules_path,
    sources_path,
)
from .treatment_engine import (
    OUTPUT_HASH_VERSION,
    STAGE_SCHEMA,
    Metrics,
    RunCancelled,
    Stage,
    bulk_insert,
    capped_partitions_sql,
    coverage_sql,
    drop_stage,
    episodes_sql,
    observation_sql,
    output_hash,
    recompute_capped,
    steps_sql,
    transitions_sql,
)
from .treatment_episodes import Event, build, merge_periods
from .treatment_events import EVENT_COLUMNS, EXCLUSION_REASONS, GROUPS, LOOKUP, group_member_sql, source_event_sql

if TYPE_CHECKING:
    from .orchestrator import CohortBuilder

ENGINE_VERSION = "2.0.0"
Engine = Literal["sql", "reference"]
_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


class TreatmentError(ValueError):
    """A run could not be performed (configuration does not fit the dataset); message has no data values."""


@dataclass(frozen=True)
class TreatmentLimits:
    batch_size: int
    max_concurrent_runs: int
    timeout_seconds: float
    reference_max_events: int
    stale_run_minutes: int
    bucket_events: int = 5_000_000

    @classmethod
    def from_settings(cls, s: Any) -> TreatmentLimits:
        lim = cls(
            int(s.treatment_batch_size),
            int(s.treatment_max_concurrent_runs),
            float(s.treatment_timeout_seconds),
            int(s.treatment_reference_max_events),
            int(s.treatment_stale_run_minutes),
            int(getattr(s, "treatment_bucket_events", 5_000_000)),
        )
        checks = [
            (100 <= lim.batch_size <= 5_000_000, "CB_TREATMENT_BATCH_SIZE must be 100..5000000"),
            (1 <= lim.max_concurrent_runs <= 64, "CB_TREATMENT_MAX_CONCURRENT_RUNS must be 1..64"),
            (1 <= lim.timeout_seconds <= 7 * 86400, "CB_TREATMENT_TIMEOUT_SECONDS must be 1..604800"),
            (1 <= lim.reference_max_events <= 5_000_000, "CB_TREATMENT_REFERENCE_MAX_EVENTS must be 1..5000000"),
            (1 <= lim.stale_run_minutes <= 100_000, "CB_TREATMENT_STALE_RUN_MINUTES must be 1..100000"),
            (1 <= lim.bucket_events <= 1_000_000_000, "CB_TREATMENT_BUCKET_EVENTS must be 1..1000000000"),
        ]
        for ok, msg in checks:
            if not ok:
                raise TreatmentError(msg)
        return lim


def resolve_rules(ontology_dir: Path, name_or_path: str) -> TreatmentRuleSet:
    """A rule set by name (ontology/treatment/rules/<name>.yaml) or by file path (CLI)."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return load_rules(p.read_text())
    path = rules_path(ontology_dir, name_or_path)
    if not path.is_file():
        raise TreatmentError(f"unknown treatment rule set {name_or_path!r}")
    return load_rules(path.read_text())


def _slot(n: int) -> threading.BoundedSemaphore:
    with _SLOTS_LOCK:
        return _SLOTS.setdefault(n, threading.BoundedSemaphore(n))


EVENT_DECISION_COLS = [
    ("event_id", "VARCHAR"),
    ("step", "VARCHAR"),
    ("coverage_start", "DATE"),
    ("coverage_end", "DATE"),
    ("shifted_days", "INTEGER"),
    ("episode_id", "VARCHAR"),
]
EPISODE_COLS = [
    ("episode_id", "VARCHAR"),
    ("person_id", "BIGINT"),
    ("drug_group", "VARCHAR"),
    ("period_id", "INTEGER"),
    ("episode_start", "DATE"),
    ("episode_end", "DATE"),
    ("n_events", "INTEGER"),
    ("days_supplied", "INTEGER"),
    ("gap_days_bridged", "INTEGER"),
    ("max_gap_days", "INTEGER"),
    ("n_products", "INTEGER"),
    ("left_censored", "BOOLEAN"),
    ("right_censored", "BOOLEAN"),
    ("end_reason", "VARCHAR"),
    ("observation_start", "DATE"),
    ("observation_end", "DATE"),
    ("end_status", "VARCHAR"),
    ("observed_end", "DATE"),
    ("coverage_beyond_observation", "BOOLEAN"),
]
TRANSITION_COLS = [
    ("person_id", "BIGINT"),
    ("kind", "VARCHAR"),
    ("from_episode", "VARCHAR"),
    ("to_episode", "VARCHAR"),
    ("from_group", "VARCHAR"),
    ("to_group", "VARCHAR"),
    ("on_date", "DATE"),
    ("days", "INTEGER"),
    ("basis", "VARCHAR"),
    ("from_products", "VARCHAR"),
    ("to_products", "VARCHAR"),
]


class TreatmentService:
    def __init__(self, builder: CohortBuilder):
        self.b = builder
        self.con = builder.con
        self.store = builder.store
        self.ontology = builder.ontology
        self.min_cell = builder.executor.min_cell
        self.limits = TreatmentLimits.from_settings(builder.settings)

    # ---- configuration ---------------------------------------------------------------------------
    def sources_for_dataset(self) -> TreatmentSourceConfig:
        dataset = self.ontology.dataset_name
        path = sources_path(self.b.settings.ontology_dir, dataset)
        if not path.is_file():
            raise TreatmentError(
                f"no treatment source config for dataset {dataset!r} (expected ontology/treatment/sources/"
                f"{dataset}.yaml): treatment events are never derived without an explicit source mapping"
            )
        cfg = load_sources(path.read_text())
        if cfg.dataset != dataset:
            raise TreatmentError(f"treatment source config is for {cfg.dataset!r}, not the active dataset {dataset!r}")
        return cfg

    # ---- run -------------------------------------------------------------------------------------
    def run(
        self,
        rules: TreatmentRuleSet,
        actor: str,
        tenant: str | None = None,
        sources: TreatmentSourceConfig | None = None,
        reference_date: date | None = None,
        engine: Engine = "sql",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        sources = sources or self.sources_for_dataset()
        if sources.dataset != self.ontology.dataset_name:
            raise TreatmentError(
                f"treatment source config is for {sources.dataset!r}, not the active dataset "
                f"{self.ontology.dataset_name!r}"
            )
        if engine not in ("sql", "reference"):
            raise TreatmentError("engine must be 'sql' or 'reference'")
        batch = int(self.limits.batch_size if batch_size is None else batch_size)
        if not 1 <= batch <= 5_000_000:
            raise TreatmentError("batch size must be 1..5000000")
        ref = reference_date or datetime.now(timezone.utc).date()
        slot = _slot(self.limits.max_concurrent_runs)
        if not slot.acquire(blocking=False):
            raise TreatmentError(
                "too many concurrent treatment runs "
                f"(CB_TREATMENT_MAX_CONCURRENT_RUNS={self.limits.max_concurrent_runs})"
            )
        try:
            self.cleanup_incomplete_runs(self.limits.stale_run_minutes, actor)
            return self._run(rules, actor, tenant or DEFAULT_TENANT, sources, ref, engine, batch)
        finally:
            slot.release()

    def _run(
        self,
        rules: TreatmentRuleSet,
        actor: str,
        tenant: str,
        sources: TreatmentSourceConfig,
        ref: date,
        engine: Engine,
        batch: int,
    ) -> dict[str, Any]:
        run_uuid = uuid.uuid4()
        run_id, stage = str(run_uuid), Stage(run_uuid.hex)
        dataset = self.ontology.dataset_name
        self.con.execute(
            "INSERT INTO meta.treatment_run (run_id, dataset, tenant, created_by, created_at, status, event_schema, "
            "rule_set_name, rule_set_version, rule_set_hash, rule_set_json, source_config_hash, source_config_json, "
            "data_snapshot, software_version, reference_date, engine, engine_version, output_hash_version, "
            "batch_size) VALUES (?,?,?,?,?,'running',?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                run_id,
                dataset,
                tenant,
                actor,
                now(),
                EVENT_SCHEMA_VERSION,
                rules.name,
                rules.version,
                rules.content_hash(),
                dumps(rules.model_dump(mode="json")),
                sources.content_hash(),
                dumps(sources.model_dump(mode="json")),
                self.b.data_snapshot(),
                __version__,
                ref,
                engine,
                ENGINE_VERSION,
                OUTPUT_HASH_VERSION,
                batch,
            ],
        )
        metrics = Metrics()
        prior_order = self.con.execute("SELECT current_setting('preserve_insertion_order')").fetchone()
        # every read that needs an order says ORDER BY; this lets large statements spill instead of failing
        self.con.execute("SET preserve_insertion_order = false")
        reader = self.con.cursor()  # streaming reads run on their own connection to the same database
        cancelled = threading.Event()

        def interrupt() -> None:
            cancelled.set()
            for c in (self.con, reader):
                try:
                    c.interrupt()
                except duckdb.Error:  # pragma: no cover - nothing running
                    pass

        def check() -> None:
            if cancelled.is_set():
                raise RunCancelled("treatment run exceeded CB_TREATMENT_TIMEOUT_SECONDS")

        timer = threading.Timer(self.limits.timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()
        stage_name = "events"
        in_tx = False
        try:
            with metrics.timed("events"):
                self._events(stage, rules, sources, run_id, dataset, ref)
            check()
            stage_name = "episodes"
            with metrics.timed("episodes"):
                if engine == "sql":
                    self._sql_episodes(stage, rules, reader, batch, metrics, check)
                else:
                    self._reference_episodes(stage, rules, batch)
            check()
            stage_name = "publish"
            self.con.execute("BEGIN")
            in_tx = True
            with metrics.timed("publish"):
                self._publish(stage, run_id, engine)
            with metrics.timed("hash"):
                digest = output_hash(self.con, run_id, batch)
            with metrics.timed("summary"):
                summary = self._summary(run_id, rules, sources)
            for name, table in (
                ("events", "results.treatment_event"),
                ("episodes", "results.treatment_episode"),
                ("transitions", "results.treatment_transition"),
            ):
                got = self.con.execute(f"SELECT count(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()
                metrics.rows[name] = int(got[0]) if got else 0
            self.con.execute(
                "UPDATE meta.treatment_run SET status='completed', output_hash=?, summary_json=?, finished_at=?, "
                "metrics_json=? WHERE run_id=?",
                [digest, dumps(summary), now(), dumps({"seconds": metrics.timings, "rows": metrics.rows}), run_id],
            )
            self.con.execute("COMMIT")
            in_tx = False
        except Exception as exc:
            if in_tx:
                try:
                    self.con.execute("ROLLBACK")
                except duckdb.Error:  # pragma: no cover - DuckDB already rolled back
                    pass
            if cancelled.is_set() and not isinstance(exc, RunCancelled):
                exc = RunCancelled("treatment run exceeded CB_TREATMENT_TIMEOUT_SECONDS")
            self._fail(run_id, stage, stage_name, exc, metrics)
            self.store.audit(
                actor,
                "treatment.run",
                "treatment_run",
                run_id,
                "failed",
                tenant,
                {"dataset": dataset, "stage": stage_name},
            )
            if isinstance(exc, (TreatmentError, GovernanceError)):
                raise
            raise TreatmentError(
                f"treatment run failed at stage {stage_name!r} ({type(exc).__name__}); see run {run_id}"
            ) from exc
        finally:
            timer.cancel()
            restore = "true" if not prior_order or prior_order[0] in (True, "true") else "false"
            try:
                self.con.execute(f"SET preserve_insertion_order = {restore}")
            except duckdb.Error:  # pragma: no cover
                pass
            try:
                reader.close()
            except duckdb.Error:  # pragma: no cover
                pass
            self._drop_stage(stage)
        self.store.audit(
            actor,
            "treatment.run",
            "treatment_run",
            run_id,
            "success",
            tenant,
            {"dataset": dataset, "rule_set_hash": rules.content_hash(), "output_hash": digest, "engine": engine},
        )
        return self.summary(run_id, None)

    # ---- stages -------------------------------------------------------------------------------------
    def _events(
        self,
        stage: Stage,
        rules: TreatmentRuleSet,
        sources: TreatmentSourceConfig,
        run_id: str,
        dataset: str,
        ref: date,
    ) -> None:
        vocab = self.ontology.vocab_schema()
        pairs = sorted({p for s in sources.sources for p in s.vocabularies()})
        if pairs:
            self.con.execute(code_lookup_sql(pairs, LOOKUP, vocab, temp=True))
        self.con.execute(group_member_sql(rules, vocab))
        cols = ", ".join(EVENT_COLUMNS)
        self.con.execute(f"CREATE TABLE {stage.t('ev0')} AS SELECT * FROM results.treatment_event LIMIT 0")
        for src in sources.sources:
            select = source_event_sql(src, rules, run_id, dataset, vocab, ref)
            self.con.execute(f"INSERT INTO {stage.t('ev0')} ({cols}) {select}")
        for t in (LOOKUP, GROUPS):
            self.con.execute(f"DROP TABLE IF EXISTS {t}")

    def _observation_source(self, rules: TreatmentRuleSet) -> dict[str, str] | None:
        if rules.censoring.observation != "dataset":
            return None
        op = self.ontology.mappings.get("observation_period")
        if not op:
            return None
        return {
            "table": str(op["table"]),
            "key": str(self.ontology.mappings["person"]["person_key"]),
            "start": str(op["start_col"]),
            "end": str(op["end_col"]),
        }

    def _sql_episodes(
        self,
        stage: Stage,
        rules: TreatmentRuleSet,
        reader: duckdb.DuckDBPyConnection,
        batch: int,
        metrics: Metrics,
        check: Callable[[], None],
    ) -> None:
        """Steps 0-8 in SQL per person bucket (bounded memory per statement), accumulated into adec/aep/atr.
        Buckets never split a person, so results do not depend on their number."""
        obs = self._observation_source(rules)
        if obs:
            self.con.execute(observation_sql(stage, obs["table"], obs["key"], obs["start"], obs["end"]))
        else:
            self.con.execute(
                f"CREATE TABLE {stage.t('obs')} (person_id BIGINT, period_id INTEGER, obs_start DATE, obs_end DATE)"
            )
        n = int((self.con.execute(f"SELECT count(*) FROM {stage.t('ev0')} WHERE included").fetchone() or (0,))[0])
        buckets = max(1, -(-n // self.limits.bucket_events))
        metrics.rows.update(person_buckets=buckets, capped_partitions=0, capped_day_rows=0)
        for k in range(buckets):
            for sql in steps_sql(stage, rules, obs is not None, k, buckets):
                self.con.execute(sql)
                check()
            self.con.execute(coverage_sql(stage, rules))
            if rules.episode.overlap == "shift_forward":
                row = self.con.execute(capped_partitions_sql(stage, rules.episode.max_carryover_days)).fetchone()
                capped = int(row[0]) if row else 0
                if capped:
                    with metrics.timed("episodes_capped_streaming"):
                        metrics.rows["capped_day_rows"] += recompute_capped(
                            self.con, reader, stage, rules, batch, check
                        )
                metrics.rows["capped_partitions"] += capped
            for sql in episodes_sql(stage, rules):
                self.con.execute(sql)
                check()
            with metrics.timed("transitions"):
                self.con.execute(transitions_sql(stage, rules))
            for part, acc in (("dec", "adec"), ("ep", "aep"), ("tr", "atr")):
                verb = f"CREATE TABLE {stage.t(acc)} AS" if k == 0 else f"INSERT INTO {stage.t(acc)}"
                self.con.execute(f"{verb} SELECT * FROM {stage.t(part)}")
            for part in ("ev", "st", "day", "cov", "dec", "ep", "tr"):
                self.con.execute(f"DROP TABLE IF EXISTS {stage.t(part)}")
            check()

    def _reference_episodes(self, stage: Stage, rules: TreatmentRuleSet, batch: int) -> None:
        """The in-memory reference engine (test oracle / small runs only), writing the same staging tables."""
        n = int((self.con.execute(f"SELECT count(*) FROM {stage.t('ev0')} WHERE included").fetchone() or (0,))[0])
        if n > self.limits.reference_max_events:
            raise TreatmentError(
                f"the reference engine is limited to {self.limits.reference_max_events} included events "
                "(CB_TREATMENT_REFERENCE_MAX_EVENTS); use the SQL engine"
            )
        rows = self.con.execute(
            f"SELECT event_id, person_id, drug_group, product_concept_id, start_date, days_supply, "
            f"source_name || ':' || source_record_id FROM {stage.t('ev0')} WHERE included ORDER BY event_id"
        ).fetchall()
        events = [Event(r[0], int(r[1]), r[2], int(r[3]), r[4], int(r[5]), r[6]) for r in rows]
        obs = self._observation_source(rules)
        periods: dict[int, list[tuple[date, date]]] | None = None
        if obs:
            tab = ".".join(qi(x) for x in obs["table"].split("."))
            got = self.con.execute(
                f"SELECT CAST({qi(obs['key'])} AS BIGINT), CAST({qi(obs['start'])} AS DATE), "
                f"CAST({qi(obs['end'])} AS DATE) FROM {tab} WHERE CAST({qi(obs['key'])} AS BIGINT) IN "
                f"(SELECT DISTINCT person_id FROM {stage.t('ev0')} WHERE included)"
            ).fetchall()
            periods = {}
            for pid, s, e in got:
                periods.setdefault(int(pid), []).append((s, e))
            periods = {k: merge_periods(v) for k, v in periods.items()}
        result = build(events, periods, rules)
        self.con.execute(f"CREATE TABLE {stage.t('dec')} ({', '.join(f'{n} {t}' for n, t in EVENT_DECISION_COLS)})")
        bulk_insert(
            self.con,
            stage.t("dec"),
            EVENT_DECISION_COLS,
            [
                (d.event_id, d.step, d.coverage_start, d.coverage_end, d.shifted_days, d.episode_id)
                for d in result.decisions
            ],  # fmt: skip
            batch,
        )
        self.con.execute(f"CREATE TABLE {stage.t('ep')} ({', '.join(f'{n} {t}' for n, t in EPISODE_COLS)})")
        bulk_insert(
            self.con,
            stage.t("ep"),
            EPISODE_COLS,
            [
                (
                    e.episode_id,
                    e.person_id,
                    e.group,
                    e.period_id,
                    e.start,
                    e.end,
                    e.n_events,
                    e.days_supplied,
                    e.gap_days_bridged,
                    e.max_gap_days,
                    e.products,
                    e.left_censored,
                    e.right_censored,
                    e.end_reason,
                    e.observation_start,
                    e.observation_end,
                    e.end_status,
                    e.observed_end,
                    e.coverage_beyond_observation,
                )
                for e in result.episodes
            ],  # fmt: skip
            batch,
        )
        self.con.execute(f"CREATE TABLE {stage.t('tr')} ({', '.join(f'{n} {t}' for n, t in TRANSITION_COLS)})")
        bulk_insert(
            self.con,
            stage.t("tr"),
            TRANSITION_COLS,
            [
                (
                    t.person_id,
                    t.kind,
                    t.from_episode,
                    t.to_episode,
                    t.from_group,
                    t.to_group,
                    t.on_date,
                    t.days,
                    t.basis,
                    t.from_products,
                    t.to_products,
                )
                for t in result.transitions
            ],  # fmt: skip
            batch,
        )

    def _publish(self, stage: Stage, run_id: str, engine: Engine) -> None:
        """Write the run's final rows (inside the completing transaction)."""
        if engine == "sql":  # decisions carry (person, group, period, episode_no); episodes carry the id
            dec, ep, tr = stage.t("adec"), stage.t("aep"), stage.t("atr")
            episode_ref = "p.episode_id"
            episode_join = (
                f" LEFT JOIN {ep} p ON p.person_id = d.person_id AND p.drug_group = d.drug_group "
                "AND p.period_id = d.period_id AND p.episode_no = d.episode_no"
            )
        else:
            dec, ep, tr = stage.t("dec"), stage.t("ep"), stage.t("tr")
            episode_ref, episode_join = "d.episode_id", ""
        cols = [c for c in EVENT_COLUMNS if c not in ("included", "exclusion_reason")]
        self.con.execute(
            f"INSERT INTO results.treatment_event ({', '.join(cols)}, included, exclusion_reason, episode_step, "
            f"coverage_start, coverage_end, shifted_days, episode_id) "
            f"SELECT {', '.join('e.' + c for c in cols)}, "
            "CASE WHEN d.step IN ('duplicate_event', 'outside_observation') THEN false ELSE e.included END, "
            "CASE WHEN d.step IN ('duplicate_event', 'outside_observation') THEN d.step ELSE e.exclusion_reason END, "
            f"d.step, d.coverage_start, d.coverage_end, d.shifted_days, {episode_ref} "
            f"FROM {stage.t('ev0')} e LEFT JOIN {dec} d ON d.event_id = e.event_id{episode_join}"
        )
        ep_cols = ", ".join(n for n, _ in EPISODE_COLS)
        self.con.execute(
            f"INSERT INTO results.treatment_episode (run_id, {ep_cols}) SELECT '{run_id}', {ep_cols} FROM {ep}"
        )
        tr_cols = ", ".join(n for n, _ in TRANSITION_COLS)
        self.con.execute(
            f"INSERT INTO results.treatment_transition (run_id, {tr_cols}) SELECT '{run_id}', {tr_cols} FROM {tr}"
        )

    def _fail(self, run_id: str, stage: Stage, stage_name: str, exc: BaseException, metrics: Metrics) -> None:
        try:
            self.con.execute(
                "UPDATE meta.treatment_run SET status='failed', errors_json=?, failed_stage=?, finished_at=?, "
                "metrics_json=? WHERE run_id=?",
                [dumps([redact_error(exc)]), stage_name, now(), dumps({"seconds": metrics.timings}), run_id],
            )
        except duckdb.Error:  # pragma: no cover - the database itself is unusable
            pass

    def _drop_stage(self, stage: Stage) -> None:
        for t in (LOOKUP, GROUPS):
            try:
                self.con.execute(f"DROP TABLE IF EXISTS {t}")
            except duckdb.Error:  # pragma: no cover
                pass
        try:
            drop_stage(self.con, stage)
        except duckdb.Error:  # pragma: no cover - retried by cleanup_incomplete_runs
            pass

    def cleanup_incomplete_runs(self, older_than_minutes: int, actor: str = "system") -> dict[str, int]:
        """Mark 'running' runs older than the threshold as abandoned and drop staging tables that belong to
        no running run. Completed results are never touched; failed runs never published rows."""
        cutoff = now() - timedelta(minutes=max(0, int(older_than_minutes)))
        stale = [
            r[0]
            for r in self.con.execute(
                "SELECT run_id FROM meta.treatment_run WHERE status = 'running' AND created_at < ?", [cutoff]
            ).fetchall()
        ]
        for run_id in stale:
            self.con.execute(
                "UPDATE meta.treatment_run SET status='abandoned', finished_at=? WHERE run_id=? AND status='running'",
                [now(), run_id],
            )
            self.store.audit(actor, "treatment.cleanup", "treatment_run", run_id, "abandoned", None, {})
        running = {
            uuid.UUID(r[0]).hex
            for r in self.con.execute("SELECT run_id FROM meta.treatment_run WHERE status = 'running'").fetchall()
        }
        dropped = 0
        for (name,) in self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = ?", [STAGE_SCHEMA]
        ).fetchall():
            hex_part = str(name).rsplit("_", 1)[-1]
            if hex_part not in running:
                self.con.execute(f"DROP TABLE IF EXISTS {qi(STAGE_SCHEMA)}.{qi(str(name))}")
                dropped += 1
        return {"abandoned_runs": len(stale), "dropped_staging_tables": dropped}

    # ---- aggregate output (suppressed), from persisted rows ------------------------------------------
    def _summary(self, run_id: str, rules: TreatmentRuleSet, sources: TreatmentSourceConfig) -> dict[str, Any]:
        k = self.min_cell
        con = self.con

        def rows(sql: str) -> list[tuple]:
            return con.execute(sql, [run_id]).fetchall()

        by_reason = {r: 0 for r in ["included", *EXCLUSION_REASONS]}
        by_reason.update(
            {
                str(r): int(n)
                for r, n in rows(
                    "SELECT coalesce(exclusion_reason, 'included'), count(*) FROM results.treatment_event "
                    "WHERE run_id = ? GROUP BY 1"
                )
            }
        )
        flags = rows(
            "SELECT f, count(*) FROM (SELECT unnest(string_split(quality_flags, ',')) AS f FROM "
            "results.treatment_event WHERE run_id = ? AND quality_flags IS NOT NULL) GROUP BY f ORDER BY f"
        )
        unmatched = rows(
            "SELECT count(*), count(DISTINCT source_code) FROM results.treatment_event WHERE run_id = ? AND "
            "exclusion_reason IN ('unmapped_code', 'no_ingredient')"
        )[0]
        per_source = rows(
            "SELECT source_name, evidence_type, count(*) FILTER (WHERE included) FROM results.treatment_event "
            "WHERE run_id = ? GROUP BY 1, 2 ORDER BY 1, 2"
        )
        steps = dict(
            rows(
                "SELECT episode_step, count(*) FROM results.treatment_event WHERE run_id = ? AND episode_step IS NOT "
                "NULL GROUP BY 1"
            )
        )
        shifted = rows("SELECT count(*) FROM results.treatment_event WHERE run_id = ? AND shifted_days > 0")[0][0]
        groups: dict[str, Any] = {}
        stats = {
            r[0]: r[1:]
            for r in rows(
                "SELECT drug_group, count(DISTINCT person_id), count(*), "
                "count(*) FILTER (WHERE left_censored), count(*) FILTER (WHERE right_censored), "
                "count(*) FILTER (WHERE end_status = 'discontinued'), count(*) FILTER (WHERE end_status = 'censored'), "
                "count(*) FILTER (WHERE end_status = 'unknown'), count(*) FILTER (WHERE coverage_beyond_observation), "
                "median(episode_end - episode_start + 1) FROM results.treatment_episode WHERE run_id = ? GROUP BY 1"
            )
        }
        for g in sorted(rules.drug_groups):
            persons, n, left, right, disc, cens, unk, beyond, med = stats.get(g, (0, 0, 0, 0, 0, 0, 0, 0, None))
            groups[g] = {
                "persons": suppress_count(int(persons), k),
                "episodes": suppress_count(int(n), k),
                "end_status": suppress_partition(
                    {"discontinued": int(disc), "censored": int(cens), "unknown": int(unk)}, k
                ),
                "left_censored_episodes": suppress_with_total(int(left), int(n), k),
                "right_censored_episodes": suppress_with_total(int(right), int(n), k),
                "coverage_beyond_observation": suppress_with_total(int(beyond), int(n), k),
                "median_episode_days": float(med) if med is not None and int(persons) >= k else None,
                "max_gap_days": rules.gap_for(g),
            }
        transitions: dict[str, dict[str, int | str]] = {}
        for kind, frm, to, persons in rows(
            "SELECT kind, coalesce(from_group, '-'), coalesce(to_group, '-'), count(DISTINCT person_id) FROM "
            "results.treatment_transition WHERE run_id = ? GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
        ):
            transitions.setdefault(kind, {})[f"{frm} -> {to}"] = suppress_count(int(persons), k)
        limitations = [
            "Transition labels are patterns in the recorded evidence (dispensing, administration, prescription), "
            "not proof that a patient took, stopped or changed a drug.",
            "Discontinuation is an episode end status (counted once per episode), never also a transition.",
        ]
        if self._observation_source(rules) is None:
            limitations.append("No observation periods: censoring and end status cannot be judged ('unknown').")
        if any(s.evidence_type_column for s in sources.sources):
            limitations.append("Evidence type comes from a per-row type column; unlisted types are 'unknown'.")
        return {
            "persons_counted_as": "distinct patients, suppressed below the minimum cell count",
            "records": {
                "total_event_rows": sum(by_reason.values()),
                "by_outcome": suppress_partition(by_reason, k),
                "quality_flags": {str(f): suppress_count(int(n), k) for f, n in flags},
                "by_source": [
                    {"source": s, "evidence_type": ev, "included_events": suppress_count(int(n), k)}
                    for s, ev, n in per_source
                ],
            },
            "unmatched_medication_codes": {
                "rows": suppress_count(int(unmatched[0]), k),
                "distinct_codes": suppress_count(int(unmatched[1]), k),
            },
            "episode_building": {
                "same_day_merged_events": suppress_count(int(steps.get("same_day_merged", 0)), k),
                "same_record_merged_events": suppress_count(int(steps.get("same_record_merged", 0)), k),
                "duplicate_events": suppress_count(int(steps.get("duplicate_event", 0)), k),
                "outside_observation_events": suppress_count(int(steps.get("outside_observation", 0)), k),
                "shifted_fills": suppress_count(int(shifted), k),
            },
            "drug_groups": groups,
            "transitions": transitions,
            "limitations": limitations,
        }

    def summary(self, run_id: str, tenant: str | None) -> dict[str, Any]:
        cur = self.con.execute(
            "SELECT run_id, dataset, tenant, created_by, created_at, status, event_schema, rule_set_name, "
            "rule_set_version, rule_set_hash, source_config_hash, data_snapshot, software_version, output_hash, "
            "reference_date, engine, engine_version, output_hash_version, failed_stage, finished_at, summary_json, "
            "errors_json, metrics_json FROM meta.treatment_run WHERE run_id = ?",
            [run_id],
        )
        row = cur.fetchone()
        if row is None or (tenant is not None and row[2] != tenant):
            raise KeyError(f"treatment run {run_id} not found")
        out: dict[str, Any] = dict(zip([d[0] for d in cur.description], row, strict=True))
        for key in ("created_at", "finished_at", "reference_date"):
            out[key] = str(out[key]) if out[key] is not None else None
        out["summary"] = json.loads(out.pop("summary_json")) if row[20] else None
        out["errors"] = json.loads(out.pop("errors_json")) if row[21] else None
        out["metrics"] = json.loads(out.pop("metrics_json")) if row[22] else None
        out.pop("summary_json", None)
        out.pop("errors_json", None)
        out.pop("metrics_json", None)
        return out

    # ---- patient-level explanation (authorized only) ------------------------------------------------------
    def explain_episode(
        self, run_id: str, episode_id: str, actor: str, is_admin: bool, tenant: str | None = None
    ) -> dict[str, Any]:
        run = self.summary(run_id, tenant)
        if not (is_admin and self.b.policy.allow_patient_level):
            self.store.audit(
                actor,
                "treatment.episode_explanation",
                "treatment_run",
                run_id,
                "denied",
                run["tenant"],
                {"reason": "patient-level access not authorized"},
            )
            raise GovernanceError("patient-level output requires an admin principal and CB_ALLOW_PATIENT_LEVEL=true")
        if run["status"] != "completed":
            raise KeyError(f"treatment run {run_id} is {run['status']}, not completed")
        cur = self.con.execute(
            "SELECT * FROM results.treatment_episode WHERE run_id = ? AND episode_id = ?", [run_id, episode_id]
        )
        ep = cur.fetchone()
        if ep is None:
            raise KeyError(f"episode {episode_id} not found in run {run_id}")
        episode = dict(zip([d[0] for d in cur.description], ep, strict=True))
        cur = self.con.execute(
            "SELECT event_id, source_name, source_table, source_record_id, source_code, source_vocabulary, "
            "product_concept_id, ingredient_concept_id, brand_concept_id, evidence_type, claim_status, start_date, "
            "start_date_source, days_supply, days_supply_source, end_date_source, evidence_status, quality_flags, "
            "episode_step, coverage_start, coverage_end, shifted_days FROM results.treatment_event "
            "WHERE run_id = ? AND (episode_id = ? OR (person_id = ? AND drug_group = ? AND exclusion_reason = "
            "'duplicate_event' AND start_date BETWEEN ? AND ?)) ORDER BY start_date, event_id",
            [
                run_id,
                episode_id,
                episode["person_id"],
                episode["drug_group"],
                episode["episode_start"],
                episode["episode_end"],
            ],
        )
        events = [dict(zip([d[0] for d in cur.description], r, strict=True)) for r in cur.fetchall()]
        tcur = self.con.execute(
            "SELECT kind, from_episode, to_episode, from_group, to_group, on_date, days, basis, from_products, "
            "to_products FROM results.treatment_transition WHERE run_id = ? AND (from_episode = ? OR to_episode = ?) "
            "ORDER BY on_date, kind, from_episode, to_episode",
            [run_id, episode_id, episode_id],
        )
        tcols = [d[0] for d in tcur.description]
        transitions = tcur.fetchall()
        rules = json.loads(
            (
                self.con.execute("SELECT rule_set_json FROM meta.treatment_run WHERE run_id = ?", [run_id]).fetchone()
                or ("{}",)
            )[0]
        )
        self.store.audit(
            actor,
            "treatment.episode_explanation",
            "treatment_run",
            run_id,
            "success",
            run["tenant"],
            {"episode_ref": uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{episode_id}").hex[:16]},
        )

        def iso(v: Any) -> Any:
            return v.isoformat() if isinstance(v, date) else v

        return {
            "run_id": run_id,
            "rule_set": {
                "name": run["rule_set_name"],
                "version": run["rule_set_version"],
                "hash": run["rule_set_hash"],
            },
            "reference_date": run["reference_date"],
            "episode": {k: iso(v) for k, v in episode.items()},
            "events": [{k: iso(v) for k, v in e.items()} for e in events],
            "transitions": [dict(zip(tcols, [iso(x) for x in t], strict=True)) for t in transitions],
            "rules_applied": {
                "episode": rules.get("episode"),
                "censoring": rules.get("censoring"),
                "transitions": rules.get("transitions"),
                "max_gap_days": episode["max_gap_days"],
            },
            "reading_guide": (
                "Events are the records that formed this episode (steps 'counted', 'same_day_merged', "
                "'same_record_merged') and duplicates the rules dropped. A missing record is not evidence that "
                "treatment stopped: end_status is 'censored' when a refill could not have been observed and "
                "'unknown' without observation periods; 'discontinued' means no supply for more than max_gap_days "
                "while the person was observed."
            ),
        }
