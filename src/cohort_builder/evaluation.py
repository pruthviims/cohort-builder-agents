"""Evaluation harness: golden cases -> accuracy and consistency metrics.

Run it in CI whenever a prompt, model, ontology or compiler version changes.
"""

from __future__ import annotations

import json
import statistics
import uuid
from collections import Counter
from pathlib import Path

import yaml

from .executor import suppress_count
from .ir import CohortDefinition, Criterion
from .metadata import dumps, now
from .orchestrator import CohortBuilder

DEFAULT_THRESHOLDS = {"valid_rate": 0.9, "patient_jaccard": 0.9, "consistency": 0.8}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _criteria(ir: CohortDefinition) -> list[tuple[str, Criterion]]:
    return [("inc", c) for c in ir.inclusion] + [("exc", c) for c in ir.exclusion]


def _criterion_match(g: Criterion, c: Criterion) -> bool:
    if g.entity != c.entity or g.window != c.window or g.occurrence != c.occurrence or g.count != c.count:
        return False
    if (g.value_filter is None) != (c.value_filter is None):
        return False
    if g.value_filter and c.value_filter:
        gv, cv = g.value_filter, c.value_filter
        return gv.op == cv.op and gv.unit_concept_id == cv.unit_concept_id and abs(gv.value - cv.value) < 0.05
    return True


def compare(builder: CohortBuilder, gold: CohortDefinition, got: CohortDefinition) -> dict:
    v = builder.vocab

    def expanded(ir: CohortDefinition, cs_id: str) -> set[int]:
        return v.expand([i.model_dump() for i in ir.concept_set(cs_id).items])

    index_j = jaccard(expanded(gold, gold.index_event.concept_set_id), expanded(got, got.index_event.concept_set_id))
    crit_j, struct = [], []
    got_crits = _criteria(got)
    for role, g in _criteria(gold):
        candidates = [c for r, c in got_crits if r == role and c.entity == g.entity]
        crit_j.append(
            max(
                (jaccard(expanded(gold, g.concept_set_id), expanded(got, c.concept_set_id)) for c in candidates),
                default=0.0,
            )
        )
        struct.append(any(_criterion_match(g, c) for c in candidates))
    extra = max(0, len(got_crits) - len(_criteria(gold)))
    gold_people = builder.executor.person_ids(builder.compiler.compile(gold))
    got_people = builder.executor.person_ids(builder.compiler.compile(got))
    return {
        "index_concept_jaccard": round(index_j, 4),
        "criteria_concept_jaccard": round(statistics.mean(crit_j), 4) if crit_j else 1.0,
        "structure_match": round(sum(struct) / len(struct), 4) if struct else 1.0,
        "extra_criteria": extra,
        "demographics_match": gold.demographics == got.demographics,
        "patient_jaccard": round(jaccard(gold_people, got_people), 4),
        # reports may be saved or shared: counts are small-cell suppressed like every other output
        "gold_count": suppress_count(len(gold_people), builder.executor.min_cell),
        "got_count": suppress_count(len(got_people), builder.executor.min_cell),
    }


def run_eval(
    builder: CohortBuilder,
    cases_file: Path,
    repeats: int = 3,
    case_ids: list[str] | None = None,
    thresholds: dict | None = None,
) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    cases_file = Path(cases_file)
    cases = yaml.safe_load(cases_file.read_text())["cases"]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    eval_run_id = str(uuid.uuid4())
    versions = builder.component_versions()
    builder.con.execute("INSERT INTO meta.eval_run VALUES (?,?,?,NULL)", [eval_run_id, now(), dumps(versions)])

    per_case = []
    for case in cases:
        gold = CohortDefinition.model_validate_json((cases_file.parent / case["gold"]).read_text())
        hashes: list[str | None] = []
        metrics: list[dict] = []
        for r in range(repeats):
            result = builder.ask(case["query"], user_id="eval")
            m: dict = {"status": result.status, "run_id": result.run_id}
            if result.ir is not None:
                m.update(compare(builder, gold, result.ir))
                hashes.append(result.ir.semantic_hash())
            else:
                hashes.append(None)
            metrics.append(m)
            builder.con.execute("INSERT INTO meta.eval_result VALUES (?,?,?,?)", [eval_run_id, case["id"], r, dumps(m)])
        present = Counter(h for h in hashes if h is not None)
        modal = present.most_common(1)[0][1] if present else 0
        ok = [m for m in metrics if "patient_jaccard" in m]
        per_case.append(
            {
                "case_id": case["id"],
                "therapeutic_area": case.get("therapeutic_area"),
                "valid_rate": sum(m["status"] == "draft" for m in metrics) / repeats,
                "consistency": round(modal / repeats, 4),  # share of repeats producing the modal semantic hash
                "patient_jaccard": round(statistics.mean(m["patient_jaccard"] for m in ok), 4) if ok else 0.0,
                "index_concept_jaccard": round(statistics.mean(m["index_concept_jaccard"] for m in ok), 4)
                if ok
                else 0.0,
                "criteria_concept_jaccard": round(statistics.mean(m["criteria_concept_jaccard"] for m in ok), 4)
                if ok
                else 0.0,
                "structure_match": round(statistics.mean(m["structure_match"] for m in ok), 4) if ok else 0.0,
                "runs": metrics,
            }
        )

    summary = {
        k: round(statistics.mean(c[k] for c in per_case), 4) if per_case else 0.0
        for k in (
            "valid_rate",
            "consistency",
            "patient_jaccard",
            "index_concept_jaccard",
            "criteria_concept_jaccard",
            "structure_match",
        )
    }
    failures = [k for k, t in thresholds.items() if summary.get(k, 0.0) < t]
    report = {
        "eval_run_id": eval_run_id,
        "versions": versions,
        "repeats": repeats,
        "summary": summary,
        "thresholds": thresholds,
        "passed": not failures,
        "failed_metrics": failures,
        "cases": per_case,
    }
    builder.con.execute(
        "UPDATE meta.eval_run SET summary_json=? WHERE eval_run_id=?", [json.dumps(report["summary"]), eval_run_id]
    )
    return report
