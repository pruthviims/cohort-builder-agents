"""Deterministic field-level differences between two cohort definitions (IR), for showing exactly what a
revision changed. Lists of objects that carry an `id` (concept sets, criteria) are matched by id, so
reordering is not reported as a change; other lists are compared as values."""

from __future__ import annotations

from typing import Any

from .ir import CohortDefinition

# provenance / bookkeeping fields that change with every build and say nothing about the cohort logic
IGNORED = {"schema_version", "ontology_version", "vocabulary_version"}


def _flatten(value: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k in sorted(value):
            if path == "" and k in IGNORED:
                continue
            _flatten(value[k], f"{path}.{k}" if path else k, out)
    elif isinstance(value, list) and value and all(isinstance(v, dict) and "id" in v for v in value):
        for v in value:
            _flatten(v, f"{path}[{v['id']}]", out)
    else:
        out[path] = value


def ir_changes(before: CohortDefinition, after: CohortDefinition, limit: int = 200) -> list[dict[str, Any]]:
    """[{path, change: added|removed|changed, before, after}] sorted by path; empty when identical."""
    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    _flatten(before.model_dump(mode="json"), "", a)
    _flatten(after.model_dump(mode="json"), "", b)
    out = []
    for path in sorted(set(a) | set(b)):
        if path not in b:
            out.append({"path": path, "change": "removed", "before": a[path], "after": None})
        elif path not in a:
            out.append({"path": path, "change": "added", "before": None, "after": b[path]})
        elif a[path] != b[path]:
            out.append({"path": path, "change": "changed", "before": a[path], "after": b[path]})
    return out[:limit]
