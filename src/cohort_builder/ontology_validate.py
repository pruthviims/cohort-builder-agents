"""`cohort-builder ontology validate`: checks of the treatment configuration files, with or without a
vocabulary database. Every issue names the file, the field path and the offending value.

Without a database (CI):
  - every rules / sources YAML parses strictly and validates against its model;
  - a rule set's `name` matches its file name, a source config's `dataset` matches its file name and an
    existing dataset profile (ontology/datasets/<dataset>.yaml);
  - no concept id is listed twice in one drug group (the loader would silently de-duplicate it) and no
    concept id is in two drug groups (its events would be 'ambiguous_drug_group');
  - every file has a provenance block (review_status illustrative / unverified / validated; 'validated'
    requires reviewer, date and evidence - enforced by the model);
  - a per-group max_gap_days override has a justification (warning).

With --vocabulary (the database at CB_DB_PATH, opened read-only):
  - every drug group concept exists, is in the Drug domain, and resolves to at least one Ingredient
    (itself or a descendant); a group resolving to none is orphaned;
  - NDC concept codes are 11 digits (NDC11) or a dashed 4-4-2 / 5-3-2 / 5-4-1 form; NDC codes without a
    'Maps to' relationship are reported (warning) - unmapped codes are excluded at run time, never guessed;
  - the synthetic labeler 99999 marks placeholder NDCs: a source config marked 'validated' is refused
    while the vocabulary contains them (placeholder mappings are never presented as validated);
  - source tables and their configured columns exist (when the table is present in this database).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ValidationError

from .treatment_config import TreatmentRuleSet, TreatmentSourceConfig, _load_yaml

NDC_FORMAT = re.compile(r"^(\d{11}|\d{4}-\d{4}-\d{2}|\d{5}-\d{3}-\d{2}|\d{5}-\d{4}-\d{1})$")
SYNTHETIC_LABELER = "99999"
MAX_LISTED = 20


@dataclass(frozen=True)
class Issue:
    severity: str  # error | warning
    file: str
    field: str
    value: str | None
    message: str


def _short(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


class _Collector:
    def __init__(self, root: Path):
        self.root = root
        self.issues: list[Issue] = []

    def add(self, severity: str, path: Path | str, field: str, value: Any, message: str) -> None:
        p = Path(path)
        rel = str(p.relative_to(self.root)) if p.is_absolute() and self.root in p.parents else str(p)
        self.issues.append(Issue(severity, rel, field, _short(value), message))


def _load(c: _Collector, path: Path, model: type[BaseModel]) -> tuple[Any, dict | None]:
    try:
        raw = _load_yaml(path.read_text(), path.name)
    except (ValueError, OSError) as exc:
        c.add("error", path, "(file)", None, str(exc))
        return None, None
    try:
        return model.model_validate(raw), raw
    except ValidationError as exc:
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"]) or "(root)"
            c.add("error", path, loc, e.get("input") if not isinstance(e.get("input"), dict) else None, e["msg"])
        return None, raw


def _check_rules(c: _Collector, path: Path) -> TreatmentRuleSet | None:
    rules, raw = _load(c, path, TreatmentRuleSet)
    groups = (raw or {}).get("drug_groups") or {}
    owner: dict[int, list[str]] = {}
    if isinstance(groups, dict):
        for g, spec in groups.items():
            ids = spec.get("concepts") if isinstance(spec, dict) else None
            if not isinstance(ids, list):
                continue
            seen: set[Any] = set()
            for cid in ids:
                if cid in seen:
                    c.add("error", path, f"drug_groups.{g}.concepts", cid, "concept id listed twice in one group")
                seen.add(cid)
                if isinstance(cid, int):
                    owner.setdefault(cid, []).append(str(g))
            if isinstance(spec, dict) and spec.get("max_gap_days") is not None and not spec.get("justification"):
                c.add(
                    "warning",
                    path,
                    f"drug_groups.{g}.max_gap_days",
                    spec.get("max_gap_days"),
                    "per-group gap override without a justification",
                )
    for cid, gs in sorted(owner.items()):
        if len(set(gs)) > 1:
            c.add(
                "error",
                path,
                "drug_groups",
                cid,
                f"concept id is in groups {sorted(set(gs))}: its events would be ambiguous (conflicting groups)",
            )
    if raw is not None and "provenance" not in raw:
        c.add("error", path, "provenance", None, "missing provenance block (review_status, source)")
    if rules is not None and rules.name != path.stem:
        c.add("error", path, "name", rules.name, f"rule set name must match the file name ({path.stem!r})")
    return rules


def _check_sources(c: _Collector, path: Path, datasets: set[str]) -> TreatmentSourceConfig | None:
    cfg, raw = _load(c, path, TreatmentSourceConfig)
    if raw is not None and "provenance" not in raw:
        c.add("error", path, "provenance", None, "missing provenance block (review_status, source)")
    if cfg is not None:
        if cfg.dataset != path.stem:
            c.add("error", path, "dataset", cfg.dataset, f"dataset must match the file name ({path.stem!r})")
        if cfg.dataset not in datasets:
            c.add("error", path, "dataset", cfg.dataset, "no dataset profile ontology/datasets/<dataset>.yaml")
    return cfg


def _check_vocabulary(
    c: _Collector,
    con: duckdb.DuckDBPyConnection,
    rules: dict[Path, TreatmentRuleSet],
    sources: dict[Path, TreatmentSourceConfig],
) -> None:
    for path, r in rules.items():
        for g, spec in r.drug_groups.items():
            for cid in spec.concepts:
                row = con.execute(
                    "SELECT domain_id, concept_class_id FROM vocab.concept WHERE concept_id = ?", [cid]
                ).fetchone()
                if row is None:
                    c.add("error", path, f"drug_groups.{g}.concepts", cid, "concept id not in the vocabulary")
                elif row[0] != "Drug":
                    c.add("error", path, f"drug_groups.{g}.concepts", cid, f"not a Drug concept (domain {row[0]})")
            ids = ", ".join(str(int(x)) for x in spec.concepts)
            got = con.execute(
                f"SELECT count(*) FROM vocab.concept i WHERE i.concept_class_id = 'Ingredient' AND (i.concept_id IN "
                f"({ids}) OR EXISTS (SELECT 1 FROM vocab.concept_ancestor a WHERE a.ancestor_concept_id IN ({ids}) "
                "AND a.descendant_concept_id = i.concept_id))"
            ).fetchone()
            if not got or not got[0]:
                c.add("error", path, f"drug_groups.{g}", None, "orphaned group: resolves to no ingredient")
    vocab_label = "vocabulary (CB_DB_PATH)"
    bad = con.execute(
        "SELECT concept_code FROM vocab.concept WHERE vocabulary_id = 'NDC' ORDER BY concept_code"
    ).fetchall()
    bad_codes = [r[0] for r in bad if not NDC_FORMAT.match(str(r[0] or ""))]
    for code in bad_codes[:MAX_LISTED]:
        c.add("error", vocab_label, "vocab.concept.concept_code", code, "NDC code is not NDC11 or a dashed NDC form")
    if len(bad_codes) > MAX_LISTED:
        c.add("error", vocab_label, "vocab.concept.concept_code", None, f"{len(bad_codes)} malformed NDC codes")
    unmapped = con.execute(
        "SELECT c.concept_code FROM vocab.concept c WHERE c.vocabulary_id = 'NDC' AND NOT EXISTS (SELECT 1 FROM "
        "vocab.concept_relationship r WHERE r.concept_id_1 = c.concept_id AND r.relationship_id = 'Maps to' "
        "AND r.invalid_reason IS NULL) ORDER BY 1"
    ).fetchall()
    for (code,) in unmapped[:MAX_LISTED]:
        c.add("warning", vocab_label, "vocab.concept_relationship", code, "NDC code has no 'Maps to' (excluded)")
    placeholder = (
        con.execute(
            "SELECT count(*) FROM vocab.concept WHERE vocabulary_id = 'NDC' AND concept_code LIKE ?",
            [SYNTHETIC_LABELER + "%"],
        ).fetchone()
        or (0,)
    )[0]
    for path, cfg in sources.items():
        if placeholder and cfg.provenance and cfg.provenance.review_status == "validated":
            c.add(
                "error",
                path,
                "provenance.review_status",
                "validated",
                f"the vocabulary holds {placeholder} synthetic placeholder NDCs (labeler {SYNTHETIC_LABELER}): "
                "a mapping against it cannot be 'validated'",
            )
        for s in cfg.sources:
            schema, table = s.table.split(".")
            cols = {
                str(r[0])
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = ?",
                    [schema, table],
                ).fetchall()
            }
            if not cols:
                c.add("warning", path, f"sources.{s.name}.table", s.table, "table not present in this database")
                continue
            for field, value in s.model_dump().items():
                if field.endswith("_column") and value and value not in cols:
                    c.add("error", path, f"sources.{s.name}.{field}", value, f"column not in {s.table}")
            if s.status and s.status.column not in cols:
                c.add("error", path, f"sources.{s.name}.status.column", s.status.column, f"column not in {s.table}")
            ref = s.status.reference_column if s.status else None
            if ref and ref not in cols:
                c.add("error", path, f"sources.{s.name}.status.reference_column", ref, f"column not in {s.table}")


def validate_ontology(ontology_dir: Path, con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    root = Path(ontology_dir).resolve()
    c = _Collector(root)
    datasets = {p.stem for p in (root / "datasets").glob("*.yaml")}
    rules = {}
    for p in sorted((root / "treatment" / "rules").glob("*.y*ml")):
        r = _check_rules(c, p)
        if r is not None:
            rules[p] = r
    sources = {}
    for p in sorted((root / "treatment" / "sources").glob("*.y*ml")):
        s = _check_sources(c, p, datasets)
        if s is not None:
            sources[p] = s
    if con is not None:
        _check_vocabulary(c, con, rules, sources)
    issues = [asdict(i) for i in c.issues]
    return {
        "checked": {
            "rule_sets": len(list((root / "treatment" / "rules").glob("*.y*ml"))),
            "source_configs": len(list((root / "treatment" / "sources").glob("*.y*ml"))),
            "vocabulary": con is not None,
        },
        "review_status": {
            **{f"rules/{p.stem}": (r.provenance.review_status if r.provenance else None) for p, r in rules.items()},
            **{f"sources/{p.stem}": (s.provenance.review_status if s.provenance else None) for p, s in sources.items()},
        },
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "issues": issues,
    }
