"""Treatment-event runs: events -> episodes -> transitions, stored with full provenance.

A run is fully determined by (dataset snapshot, treatment source config, rule set, software version); all
four are recorded in meta.treatment_run with content hashes, and `output_hash` proves that a repeat run
produced identical events, episodes and transitions. Patient-level rows live in results.treatment_* and
are only returned through `explain_episode`, which needs an admin principal AND CB_ALLOW_PATIENT_LEVEL=true
(audited). Everything else (summary, CLI/API output) is aggregate and small-cell suppressed.
"""

from __future__ import annotations

import json
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from .treatment_episodes import Event, Periods, Result, build
from .treatment_events import EVENT_COLUMNS, EXCLUSION_REASONS, GROUPS, LOOKUP, group_member_sql, source_event_sql

if TYPE_CHECKING:
    from .orchestrator import CohortBuilder


class TreatmentError(ValueError):
    """A run could not be performed (configuration does not fit the dataset); message has no data values."""


def resolve_rules(ontology_dir: Path, name_or_path: str) -> TreatmentRuleSet:
    """A rule set by name (ontology/treatment/rules/<name>.yaml) or by file path (CLI)."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return load_rules(p.read_text())
    path = rules_path(ontology_dir, name_or_path)
    if not path.is_file():
        raise TreatmentError(f"unknown treatment rule set {name_or_path!r}")
    return load_rules(path.read_text())


class TreatmentService:
    def __init__(self, builder: CohortBuilder):
        self.b = builder
        self.con = builder.con
        self.store = builder.store
        self.ontology = builder.ontology
        self.min_cell = builder.executor.min_cell

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
    ) -> dict[str, Any]:
        sources = sources or self.sources_for_dataset()
        if sources.dataset != self.ontology.dataset_name:
            raise TreatmentError(
                f"treatment source config is for {sources.dataset!r}, not the active dataset "
                f"{self.ontology.dataset_name!r}"
            )
        run_id = str(uuid.uuid4())
        tenant = tenant or DEFAULT_TENANT
        dataset = self.ontology.dataset_name
        snapshot = self.b.data_snapshot()
        vocab = self.ontology.vocab_schema()
        self.con.execute(
            "INSERT INTO meta.treatment_run (run_id, dataset, tenant, created_by, created_at, status, event_schema, "
            "rule_set_name, rule_set_version, rule_set_hash, rule_set_json, source_config_hash, source_config_json, "
            "data_snapshot, software_version) VALUES (?,?,?,?,?,'running',?,?,?,?,?,?,?,?,?)",
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
                snapshot,
                __version__,
            ],
        )
        self.con.execute("BEGIN")
        try:
            pairs = sorted({p for s in sources.sources for p in s.vocabularies()})
            if pairs:
                self.con.execute(code_lookup_sql(pairs, LOOKUP, vocab, temp=True))
            self.con.execute(group_member_sql(rules, vocab))
            cols = ", ".join(EVENT_COLUMNS)
            for src in sources.sources:
                select = source_event_sql(src, rules, run_id, dataset, vocab)
                self.con.execute(f"INSERT INTO results.treatment_event ({cols}) {select}")
            events = self._included_events(run_id)
            observation = self._observation(run_id) if rules.censoring.observation == "dataset" else None
            result = build(events, observation, rules)
            self._store(run_id, result)
            summary = self._summary(run_id, rules, sources, result, observation is not None)
            self.con.execute(
                "UPDATE meta.treatment_run SET status='completed', output_hash=?, summary_json=? WHERE run_id=?",
                [result.output_hash(), dumps(summary), run_id],
            )
            self.con.execute("COMMIT")
        except Exception as exc:
            try:
                self.con.execute("ROLLBACK")
            except Exception:  # noqa: BLE001 - nothing open
                pass
            self.con.execute(
                "UPDATE meta.treatment_run SET status='failed', errors_json=? WHERE run_id=?",
                [dumps([redact_error(exc)]), run_id],
            )
            self.store.audit(actor, "treatment.run", "treatment_run", run_id, "failed", tenant, {"dataset": dataset})
            if isinstance(exc, (TreatmentError, GovernanceError)):
                raise
            raise TreatmentError(f"treatment run failed ({type(exc).__name__}); see run {run_id}") from exc
        finally:
            for t in (LOOKUP, GROUPS):
                self.con.execute(f"DROP TABLE IF EXISTS {t}")
        self.store.audit(
            actor,
            "treatment.run",
            "treatment_run",
            run_id,
            "success",
            tenant,
            {"dataset": dataset, "rule_set_hash": rules.content_hash(), "output_hash": result.output_hash()},
        )
        return self.summary(run_id, None)

    def _included_events(self, run_id: str) -> list[Event]:
        rows = self.con.execute(
            "SELECT event_id, person_id, drug_group, product_concept_id, start_date, days_supply "
            "FROM results.treatment_event WHERE run_id = ? AND included ORDER BY event_id",
            [run_id],
        ).fetchall()
        return [Event(r[0], int(r[1]), r[2], int(r[3]), r[4], int(r[5])) for r in rows]

    def _observation(self, run_id: str) -> Periods | None:
        op = self.ontology.mappings.get("observation_period")
        if not op:
            return None
        key = self.ontology.mappings["person"]["person_key"]
        tab = ".".join(qi(x) for x in str(op["table"]).split("."))
        rows = self.con.execute(
            f"SELECT o.{qi(key)}, o.{qi(op['start_col'])}, o.{qi(op['end_col'])} FROM {tab} o WHERE o.{qi(key)} IN "
            "(SELECT DISTINCT person_id FROM results.treatment_event WHERE run_id = ? AND included) "
            f"ORDER BY 1, 2, 3",
            [run_id],
        ).fetchall()
        out: Periods = defaultdict(list)
        for pid, s, e in rows:
            if s is not None and e is not None:
                out[int(pid)].append((s, e))
        return dict(out)

    def _store(self, run_id: str, result: Result) -> None:
        if not result.decisions:  # nothing included (e.g. no record matches the drug groups): nothing to store
            return
        self.con.executemany(
            "UPDATE results.treatment_event SET episode_step=?, coverage_start=?, coverage_end=?, shifted_days=?, "
            "episode_id=?, included=?, exclusion_reason=? WHERE run_id=? AND event_id=?",
            [
                [
                    d.step,
                    d.coverage_start,
                    d.coverage_end,
                    d.shifted_days,
                    d.episode_id,
                    d.step != "duplicate_event",
                    "duplicate_event" if d.step == "duplicate_event" else None,
                    run_id,
                    d.event_id,
                ]
                for d in result.decisions
            ],
        )
        if result.episodes:
            self.con.executemany(
                "INSERT INTO results.treatment_episode VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    [
                        run_id,
                        e.episode_id,
                        e.person_id,
                        e.group,
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
                    ]
                    for e in result.episodes
                ],
            )
        if result.transitions:
            self.con.executemany(
                "INSERT INTO results.treatment_transition VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    [
                        run_id,
                        t.person_id,
                        t.kind,
                        t.from_episode,
                        t.to_episode,
                        t.from_group,
                        t.to_group,
                        t.on_date,
                        t.days,
                        t.basis,
                    ]
                    for t in result.transitions
                ],
            )

    # ---- aggregate output (suppressed) --------------------------------------------------------------------
    def _summary(
        self,
        run_id: str,
        rules: TreatmentRuleSet,
        sources: TreatmentSourceConfig,
        result: Result,
        has_observation: bool,
    ) -> dict[str, Any]:
        k = self.min_cell
        rows = self.con.execute(
            "SELECT coalesce(exclusion_reason, 'included'), count(*) FROM results.treatment_event WHERE run_id = ? "
            "GROUP BY 1",
            [run_id],
        ).fetchall()
        by_reason = {r: 0 for r in ["included", *EXCLUSION_REASONS]}
        by_reason.update({str(r): int(n) for r, n in rows})
        total = sum(by_reason.values())
        flags: Counter[str] = Counter()
        for (f,) in self.con.execute(
            "SELECT quality_flags FROM results.treatment_event WHERE run_id = ? AND quality_flags IS NOT NULL", [run_id]
        ).fetchall():
            flags.update(str(f).split(","))
        unmatched = self.con.execute(
            "SELECT count(*), count(DISTINCT source_code) FROM results.treatment_event WHERE run_id = ? AND "
            "exclusion_reason IN ('unmapped_code', 'no_ingredient')",
            [run_id],
        ).fetchone() or (0, 0)
        per_source = self.con.execute(
            "SELECT source_name, evidence_type, count(*) FILTER (WHERE included) FROM results.treatment_event "
            "WHERE run_id = ? GROUP BY 1, 2 ORDER BY 1, 2",
            [run_id],
        ).fetchall()
        groups: dict[str, Any] = {}
        for g in sorted(rules.drug_groups):
            eps = [e for e in result.episodes if e.group == g]
            persons = len({e.person_id for e in eps})
            n = len(eps)
            lengths = [(e.end - e.start).days + 1 for e in eps]
            groups[g] = {
                "persons": suppress_count(persons, k),
                "episodes": suppress_count(n, k),
                "left_censored_episodes": suppress_with_total(sum(1 for e in eps if e.left_censored), n, k),
                "right_censored_episodes": suppress_with_total(sum(1 for e in eps if e.right_censored), n, k),
                "episodes_without_observation": suppress_with_total(
                    sum(1 for e in eps if e.right_censored is None), n, k
                ),
                "median_episode_days": statistics.median(lengths) if persons >= k and lengths else None,
                "max_gap_days": rules.gap_for(g),
            }
        transitions: dict[str, dict[str, int | str]] = defaultdict(dict)
        persons_by: dict[tuple[str, str], set[int]] = defaultdict(set)
        for t in result.transitions:
            key = f"{t.from_group or '-'} -> {t.to_group or '-'}"
            persons_by[(t.kind, key)].add(t.person_id)
        for (kind, key), ps in sorted(persons_by.items()):
            transitions[kind][key] = suppress_count(len(ps), k)
        steps = Counter(d.step for d in result.decisions)
        limitations = [
            "Transition labels are patterns in the recorded evidence (dispensing, administration, prescription), "
            "not proof that a patient took or stopped a drug.",
        ]
        if not has_observation:
            limitations.append("No observation periods: censoring and discontinuation cannot be judged (unknown).")
        if any(s.evidence_type_column for s in sources.sources):
            limitations.append("Evidence type comes from a per-row type column; unlisted types are 'unknown'.")
        return {
            "persons_counted_as": "distinct patients, suppressed below the minimum cell count",
            "records": {
                "total_event_rows": total,
                "by_outcome": suppress_partition(by_reason, k),
                "quality_flags": {f: suppress_count(n, k) for f, n in sorted(flags.items())},
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
                "same_day_merged_events": suppress_count(steps.get("same_day_merged", 0), k),
                "duplicate_events": suppress_count(steps.get("duplicate_event", 0), k),
                "shifted_fills": suppress_count(sum(1 for d in result.decisions if d.shifted_days > 0), k),
            },
            "drug_groups": groups,
            "transitions": dict(transitions),
            "limitations": limitations,
        }

    def summary(self, run_id: str, tenant: str | None) -> dict[str, Any]:
        row = self.con.execute(
            "SELECT run_id, dataset, tenant, created_by, created_at, status, event_schema, rule_set_name, "
            "rule_set_version, rule_set_hash, source_config_hash, data_snapshot, software_version, output_hash, "
            "summary_json, errors_json FROM meta.treatment_run WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None or (tenant is not None and row[2] != tenant):
            raise KeyError(f"treatment run {run_id} not found")
        keys = [
            "run_id",
            "dataset",
            "tenant",
            "created_by",
            "created_at",
            "status",
            "event_schema",
            "rule_set_name",
            "rule_set_version",
            "rule_set_hash",
            "source_config_hash",
            "data_snapshot",
            "software_version",
            "output_hash",
        ]
        out: dict[str, Any] = dict(zip(keys, row[:14], strict=True))
        out["created_at"] = str(out["created_at"])
        out["summary"] = json.loads(row[14]) if row[14] else None
        out["errors"] = json.loads(row[15]) if row[15] else None
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
        transitions = self.con.execute(
            "SELECT kind, from_episode, to_episode, from_group, to_group, on_date, days, basis FROM "
            "results.treatment_transition WHERE run_id = ? AND (from_episode = ? OR to_episode = ?) "
            "ORDER BY on_date, kind",
            [run_id, episode_id, episode_id],
        ).fetchall()
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
            "episode": {k: iso(v) for k, v in episode.items()},
            "events": [{k: iso(v) for k, v in e.items()} for e in events],
            "transitions": [
                dict(
                    zip(
                        ["kind", "from_episode", "to_episode", "from_group", "to_group", "on_date", "days", "basis"],
                        [iso(x) for x in t],
                        strict=True,
                    )
                )
                for t in transitions
            ],
            "rules_applied": {
                "episode": rules.get("episode"),
                "censoring": rules.get("censoring"),
                "transitions": rules.get("transitions"),
                "max_gap_days": episode["max_gap_days"],
            },
            "reading_guide": (
                "Events are the records that formed this episode (step 'counted' or 'same_day_merged') and "
                "duplicates the rules dropped. A missing record is not evidence that treatment did not happen: "
                "end status is 'unknown' when the episode is right-censored or has no observation period."
            ),
        }
