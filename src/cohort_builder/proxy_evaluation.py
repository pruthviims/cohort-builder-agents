"""Evaluation of a proxy algorithm against an external reference standard.

Evaluation population (who is counted in TP/FP/FN/TN):

1. Reference patients: one row per patient after collapsing the loaded records. Duplicate records
   with the same label count once. A patient labelled both `case` and `non_case` is `conflicting`.
   A patient with only `unknown` (indeterminate) labels is `unknown`. Neither is evaluated.
2. Eligible = a definite label (case / non_case) AND the patient is present in the active dataset
   AND observable AND has every data type in `require_data`. Observable means:
     * with a reference date: an observation period contains it with at least the algorithm's prior
       observation before it and post observation after it (when `require_reference_date_observed`);
     * otherwise: some observation period lasts at least `min_observation_days` (default: the
       algorithm's prior + post observation, i.e. long enough that the algorithm could ever apply).
   On activity-based claims data (no enrollment), "observation" is inferred from claim activity, so
   this is a weaker guarantee than enrollment-based observation.
3. Every eligible patient is evaluated: predicted positive if they are a member of the generation in
   one of the positive tiers, otherwise predicted negative.

Excluded patients are never counted as negatives. Each is attributed to the FIRST failing rule in a
fixed order: unknown_reference_label, conflicting_reference_labels, not_in_dataset,
insufficient_observation / reference_date_not_observed, missing_<entity>_data.

Missing data is not evidence of absence: a patient without, e.g., any laboratory records is only
excluded if `require_data` names that entity; otherwise they are evaluated with the algorithm's
own (documented) absence semantics, which the evaluation report repeats.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .ir import Entity
from .metrics import classification_metrics
from .ontology import Ontology
from .proxy import AcceptanceCriteria, ProxyDefinition

Label = Literal["case", "non_case", "unknown"]
EXCLUSION_ORDER = (
    "unknown_reference_label",
    "conflicting_reference_labels",
    "not_in_dataset",
    "insufficient_observation",
    "reference_date_not_observed",
)


class EligibilityRules(BaseModel):
    """Who can be evaluated. Recorded with every evaluation."""

    model_config = ConfigDict(extra="forbid")
    min_observation_days: int | None = Field(
        default=None, ge=0, le=36_500, description="default: the algorithm's prior + post observation days"
    )
    require_reference_date_observed: bool = True
    require_data: list[Entity] = Field(
        default_factory=list,
        description="exclude patients with no records of these entities during observation (e.g. Measurement "
        "when lab results are needed to judge the algorithm fairly)",
    )


@dataclass(frozen=True)
class ReferenceRecord:
    person_id: int
    label: Label
    reference_date: date | None = None


def normalize_label(value: Any) -> Label:
    """True/False/None and common spellings -> case / non_case / unknown."""
    if value is True:
        return "case"
    if value is False:
        return "non_case"
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    if v in ("case", "1", "true", "yes", "positive", "pos"):
        return "case"
    if v in ("non_case", "noncase", "non-case", "0", "false", "no", "negative", "neg"):
        return "non_case"
    if v in ("", "unknown", "indeterminate", "na", "n/a", "none", "null"):
        return "unknown"
    raise ValueError(f"unrecognised reference label {value!r} (use case, non_case or unknown)")


def collapse_records(records: Iterable[ReferenceRecord]) -> tuple[list[tuple[int, str, date | None, int]], dict]:
    """One row per patient: (person_id, label, reference_date, n_records). Deterministic:
    case+non_case -> 'conflicting'; a definite label wins over 'unknown'; the earliest reference date
    is kept (patients with several distinct dates are counted in the summary)."""
    by_person: dict[int, list[ReferenceRecord]] = {}
    for r in records:
        if not isinstance(r.person_id, int) or isinstance(r.person_id, bool):
            raise ValueError(f"person_id must be an integer, got {r.person_id!r}")
        by_person.setdefault(r.person_id, []).append(r)
    rows: list[tuple[int, str, date | None, int]] = []
    multi_dates = duplicates = 0
    for pid in sorted(by_person):
        recs = by_person[pid]
        labels = {r.label for r in recs}
        definite = labels & {"case", "non_case"}
        label: str = "conflicting" if len(definite) == 2 else (definite.pop() if definite else "unknown")
        dates = sorted({r.reference_date for r in recs if r.reference_date is not None})
        multi_dates += len(dates) > 1
        duplicates += len(recs) > 1
        rows.append((pid, label, dates[0] if dates else None, len(recs)))
    summary = {
        "records": sum(len(v) for v in by_person.values()),
        "patients": len(rows),
        "patients_with_duplicate_records": duplicates,
        "patients_with_several_reference_dates": multi_dates,
    }
    return rows, summary


def eligibility_sql(ont: Ontology, p: ProxyDefinition, rules: EligibilityRules, n_tiers: int) -> str:
    """Per reference patient: label, presence, observability, data presence, prediction. Parameters:
    generation_id, *tiers, tenant, reference_name. Only validated identifiers/integers are interpolated."""
    op, pm = ont.mappings["observation_period"], ont.mappings["person"]
    prior, post = int(p.prior_observation_days), int(p.post_observation_days)
    min_obs = int(rules.min_observation_days if rules.min_observation_days is not None else prior + post)
    s, e = op["start_col"], op["end_col"]
    use_date = "r.reference_date IS NOT NULL" if rules.require_reference_date_observed else "FALSE"
    data_cols = []
    for i, entity in enumerate(rules.require_data):
        m = ont.table_mapping(entity)
        data_cols.append(
            f"  EXISTS (SELECT 1 FROM {m['table']} x JOIN {op['table']} o ON o.person_id = x.person_id "
            f"AND x.{m['start_col']} BETWEEN o.{s} AND o.{e} WHERE x.person_id = r.person_id) AS has_data_{i}"
        )
    tiers = ", ".join("?" for _ in range(n_tiers))
    cols = [
        "  r.person_id, r.label, r.reference_date",
        f"  EXISTS (SELECT 1 FROM {pm['table']} pp WHERE pp.{pm['person_key']} = r.person_id) AS in_dataset",
        f"  EXISTS (SELECT 1 FROM {op['table']} o WHERE o.person_id = r.person_id AND (o.{e} - o.{s}) >= {min_obs})"
        " AS observable_any",
        f"  EXISTS (SELECT 1 FROM {op['table']} o WHERE o.person_id = r.person_id AND r.reference_date IS NOT NULL "
        f"AND r.reference_date - o.{s} >= {prior} AND o.{e} - r.reference_date >= {post}) AS observable_at_date",
        f"  ({use_date}) AS judge_by_date",
        *data_cols,
        "  EXISTS (SELECT 1 FROM results.proxy_assignment a WHERE a.generation_id = ? AND a.subject_id = r.person_id "
        f"AND a.tier IN ({tiers})) AS predicted",
    ]
    return (
        "SELECT\n" + ",\n".join(cols) + "\nFROM meta.reference_label r\n"
        "WHERE r.tenant = ? AND r.reference_name = ?\nORDER BY r.person_id"
    )


def classify(rows: Sequence[tuple], require_data: Sequence[str]) -> dict[str, Any]:
    """Apply eligibility in a fixed order and count. Rows come from `eligibility_sql`."""
    reasons: dict[str, int] = {k: 0 for k in EXCLUSION_ORDER}
    reasons.update({f"missing_{e}_data": 0 for e in require_data})
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    labels = {"case": 0, "non_case": 0, "unknown": 0, "conflicting": 0}
    eligible_pos = eligible_neg = 0
    seen: set[int] = set()
    for row in rows:
        pid, label, _ref_date, in_dataset, obs_any, obs_date, by_date = row[:7]
        has_data, predicted = row[7:-1], bool(row[-1])
        if pid in seen:  # the reference table is keyed by patient; guard anyway
            raise ValueError(f"duplicate reference patient {pid} in evaluation input")
        seen.add(pid)
        labels[label] += 1
        reason = None
        if label == "unknown":
            reason = "unknown_reference_label"
        elif label == "conflicting":
            reason = "conflicting_reference_labels"
        elif not in_dataset:
            reason = "not_in_dataset"
        elif by_date and not obs_date:
            reason = "reference_date_not_observed"
        elif not by_date and not obs_any:
            reason = "insufficient_observation"
        else:
            for entity, ok in zip(require_data, has_data, strict=True):
                if not ok:
                    reason = f"missing_{entity}_data"
                    break
        if reason:
            reasons[reason] += 1
            continue
        if label == "case":
            eligible_pos += 1
            counts["TP" if predicted else "FN"] += 1
        else:
            eligible_neg += 1
            counts["FP" if predicted else "TN"] += 1
    evaluated = sum(counts.values())
    excluded = sum(reasons.values())
    # consistency: every reference patient is either evaluated or excluded, and the confusion matrix
    # matches the eligible case / non-case counts
    if (
        evaluated + excluded != len(rows)
        or counts["TP"] + counts["FN"] != eligible_pos
        or (counts["FP"] + counts["TN"] != eligible_neg)
    ):
        raise AssertionError("evaluation population is inconsistent")  # pragma: no cover - invariant
    return {
        "reference_patients": len(rows),
        "labels": labels,
        "eligible": evaluated,
        "evaluated": evaluated,
        "eligible_reference_positive": eligible_pos,
        "eligible_reference_negative": eligible_neg,
        "excluded": excluded,
        "excluded_by_reason": reasons,
        "confusion": counts,
    }


def assess_acceptance(
    criteria: AcceptanceCriteria | None, population: dict, metrics: dict | None, evaluation_status: str
) -> dict[str, Any]:
    """Automatic check of prespecified criteria. This is NOT approval: a human must still accept it.

    criteria_met      every configured criterion passed
    criteria_not_met  a performance threshold was not reached
    inconclusive      the evaluation was inconclusive, a sample-size minimum was not reached, or a
                      metric needed by a criterion is undefined
    not_assessed      no criteria were configured
    """
    if criteria is None:
        return {"status": "not_assessed", "checks": [], "reasons": ["no acceptance criteria configured"]}
    checks: list[dict[str, Any]] = []
    inconclusive: list[str] = []
    if evaluation_status != "completed":
        inconclusive.append("evaluation is inconclusive")
    sizes = (
        ("min_evaluated", "evaluated"),
        ("min_reference_positive", "eligible_reference_positive"),
        ("min_reference_negative", "eligible_reference_negative"),
    )
    for key, field in sizes:
        threshold = getattr(criteria, key)
        if threshold is not None:
            ok = population[field] >= threshold
            checks.append({"criterion": key, "threshold": threshold, "observed": population[field], "passed": ok})
            if not ok:
                inconclusive.append(f"insufficient sample: {field} below {key}")
    failed: list[str] = []
    if metrics is not None:
        for key, metric in (
            ("min_sensitivity", "sensitivity"),
            ("min_ppv", "ppv"),
            ("min_specificity", "specificity"),
            ("min_npv", "npv"),
        ):
            threshold = getattr(criteria, key)
            if threshold is None:
                continue
            value = metrics[metric]
            ci = metrics["ci"].get(metric)
            observed = (ci[0] if ci else None) if criteria.use_confidence_lower_bound else value
            basis = (
                f"lower {int(criteria.confidence_level * 100)}% confidence bound"
                if (criteria.use_confidence_lower_bound)
                else "point estimate"
            )
            if observed is None:
                checks.append(
                    {"criterion": key, "threshold": threshold, "observed": None, "basis": basis, "passed": None}
                )
                inconclusive.append(f"{metric} is undefined")
                continue
            ok = observed >= threshold
            checks.append(
                {"criterion": key, "threshold": threshold, "observed": round(observed, 4), "basis": basis, "passed": ok}
            )
            if not ok:
                failed.append(f"{metric} {basis} {observed:.4f} < {threshold}")
    elif any(getattr(criteria, k) is not None for k in ("min_sensitivity", "min_ppv", "min_specificity", "min_npv")):
        inconclusive.append("metrics unavailable")
    if inconclusive:
        status = "inconclusive"
    elif failed:
        status = "criteria_not_met"
    else:
        status = "criteria_met"
    return {
        "status": status,
        "checks": checks,
        "reasons": inconclusive + failed,
        "note": "Automatic check of prespecified criteria; it is not an approval for any use.",
    }


def compute_metrics(confusion: dict[str, int], level: float) -> dict[str, Any]:
    return classification_metrics(confusion["TP"], confusion["FP"], confusion["FN"], confusion["TN"], level)
