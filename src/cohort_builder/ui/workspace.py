"""Presentation logic of the workspace, free of Streamlit so it can be unit-tested.

Nothing here decides validity, resolves terminology, builds SQL or authorizes anything: those come from the
API. This module only routes chat messages, applies structured-editor values to a COPY of a definition (the
backend validates the result), and turns backend responses into display rows and labels.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

MessageKind = Literal["new", "refine", "question"]

_QUESTION_START = re.compile(
    r"^\s*(why|what|which|how|who|when|where|explain|describe|tell me|show me|is |are |does |do |can you explain)\b",
    re.I,
)
_CHANGE_START = re.compile(
    r"^\s*(change|make|add|remove|exclude|include|set|use|increase|decrease|require|drop|limit|restrict|"
    r"replace|extend|shorten|lower|raise|only|also|instead|switch|update)\b",
    re.I,
)

# backend definition statuses -> (label, tone). Tones: info (blue), ok (green), warn (amber), bad (red)
STATUS = {
    "draft": ("Saved draft, awaiting review", "warn"),
    "needs_review": ("Has validation errors: edit and save a new version", "bad"),
    "approved": ("Approved", "ok"),
    "rejected": ("Rejected", "bad"),
    "failed": ("Not created", "bad"),
}


def status_label(status: str | None) -> tuple[str, str]:
    return STATUS.get(status or "", (status or "unknown", "info"))


def classify_message(text: str, has_draft: bool) -> MessageKind:
    """'new' starts a definition, 'refine' revises the current one through the backend pipeline, 'question'
    is answered from the stored definition without the model. Without a draft everything is 'new'."""
    if not has_draft:
        return "new"
    t = text.strip()
    if _CHANGE_START.match(t):
        return "refine"
    if t.endswith("?") or _QUESTION_START.match(t):
        return "question"
    return "refine"


def _words(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "with", "and", "or", "in", "are", "is", "why", "what", "patients",
            "before", "after", "index", "who", "to", "for", "on", "do", "does", "this", "that", "excluded",
            "included"}  # fmt: skip
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if (len(w) > 1 or w.isdigit()) and w not in stop}


def _criterion_text(c: dict, role: str, sets: dict[str, dict]) -> str:
    cs = sets.get(c.get("concept_set_id", ""), {})
    w = c.get("window") or {}
    s, e = w.get("start_days"), w.get("end_days")

    def day(d: Any) -> str:
        if d is None:
            return "any time (observation period)"
        return "the index date" if d == 0 else f"{abs(int(d))} days {'before' if int(d) < 0 else 'after'} index"

    vf = c.get("value_filter")
    value = ""
    if vf:
        hi = f" and {vf['value_high']}" if vf.get("value_high") is not None else ""
        value = f", value {vf['op']} {vf['value']}{hi} (unit concept {vf.get('unit_concept_id')})"
    occ = f"{c.get('occurrence', 'at_least').replace('_', ' ')} {c.get('count', 1)}"
    set_name = cs.get("name", c.get("concept_set_id"))
    items = len(cs.get("items", []))
    return (
        f"{role} criterion '{c.get('name')}': {occ} record(s) of concept set '{set_name}' "
        f"({cs.get('source', 'resolved')}, {items} concept item(s)) from {day(s)} to {day(e)}{value}."
    )


def answer_question(question: str, ir: dict, status: str | None) -> dict[str, Any]:
    """A deterministic answer built ONLY from the stored definition: the matching criteria, their concept
    sets and the recorded assumptions. It never adds clinical rationale the definition does not record."""
    sets = {cs["id"]: cs for cs in ir.get("concept_sets", [])}
    q = _words(question)
    lines: list[str] = []
    scored = []
    for role, key in (("Inclusion", "inclusion"), ("Exclusion", "exclusion")):
        for c in ir.get(key, []):
            cs = sets.get(c.get("concept_set_id", ""), {})
            score = len(q & (_words(c.get("name", "")) | _words(cs.get("name", ""))))
            if score:
                scored.append((score, role, c))
    best = max((s for s, _, _ in scored), default=0)
    for score, role, c in scored:  # only the best-matching criteria ("type 1" must not also pull in "type 2")
        if score == best:
            lines.append(_criterion_text(c, role, sets))
    ie = ir.get("index_event") or {}
    ie_set = sets.get(ie.get("concept_set_id", ""), {})
    if q & _words(ie_set.get("name", "")) or q & {"index", "entry", "start"}:
        first = "first occurrence" if ie.get("first_occurrence_only", True) else "any occurrence"
        lines.append(f"Index event: {first} of '{ie_set.get('name', ie.get('concept_set_id'))}'.")
    demo = ir.get("demographics") or {}
    if q & {"age", "adult", "adults", "elderly", "old", "years", "gender", "sex"}:
        lines.append(f"Demographics: age {demo.get('age_min', 'no minimum')} to {demo.get('age_max', 'no maximum')}.")
    matched = bool(lines)
    assumptions = list(ir.get("assumptions") or [])
    return {
        "matched": matched,
        "lines": lines,
        "assumptions": assumptions,
        "status": status,
        "source": "stored definition (deterministic, no AI model involved)",
        "limits": (
            "This explains what the definition does and the assumptions recorded when it was built. "
            "It is not clinical evidence; the clinical rationale belongs in the review."
        ),
    }


def change_lines(changes: list[dict]) -> list[str]:
    out = []
    for c in changes:
        if c["change"] == "changed":
            out.append(f"{c['path']}: {c['before']!r} -> {c['after']!r}")
        elif c["change"] == "added":
            out.append(f"{c['path']}: added {c['after']!r}")
        else:
            out.append(f"{c['path']}: removed (was {c['before']!r})")
    return out


def touched_criteria(changes: list[dict]) -> set[str]:
    """Top-level parts of the definition a revision touched, e.g. {'inclusion[inc_2]', 'demographics'}."""
    out = set()
    for c in changes:
        m = re.match(r"^([a-z_]+(\[[^\]]+\])?)", c["path"])
        if m:
            out.add(m.group(1))
    return out


def _num(v: Any) -> int | None:
    return None if v in (None, "") else int(v)


def apply_edits(ir: dict, edits: dict[str, Any]) -> dict:
    """Return a NEW definition dict with the structured-editor values applied; untouched fields are kept
    exactly. Only the fields the editor exposes are written; the backend validates the result."""
    out = copy.deepcopy(ir)
    for key in ("name", "description"):
        if key in edits:
            out[key] = edits[key]
    for key in ("prior_observation_days", "post_observation_days"):
        if key in edits:
            out[key] = int(edits[key])
    if "demographics" in edits:
        d = out.setdefault("demographics", {})
        for k in ("age_min", "age_max"):
            if k in edits["demographics"]:
                d[k] = _num(edits["demographics"][k])
    if "exit" in edits:
        ex = edits["exit"]
        out["exit"] = {"type": ex["type"], "days": _num(ex.get("days")) if ex["type"] == "fixed_days" else None}
    crit_edits: dict[str, dict] = edits.get("criteria", {})
    for key in ("inclusion", "exclusion"):
        kept = []
        for c in out.get(key, []):
            e = crit_edits.get(c["id"])
            if e is None:
                kept.append(c)
                continue
            if e.get("remove"):
                continue
            c = copy.deepcopy(c)
            for k in ("name", "occurrence"):
                if k in e:
                    c[k] = e[k]
            if "count" in e:
                c["count"] = int(e["count"])
            if "window_start" in e or "window_end" in e:
                w = dict(c.get("window") or {})
                if "window_start" in e:
                    w["start_days"] = _num(e["window_start"])
                if "window_end" in e:
                    w["end_days"] = _num(e["window_end"])
                c["window"] = w
            if c.get("value_filter") and any(k in e for k in ("op", "value", "value_high")):
                vf = dict(c["value_filter"])
                for k in ("op", "value", "value_high"):
                    if k in e:
                        vf[k] = None if e[k] in (None, "") else (e[k] if k == "op" else float(e[k]))
                if vf.get("op") != "between":
                    vf["value_high"] = None
                vf["original_text"] = None  # the user edited the threshold directly
                c["value_filter"] = vf
            kept.append(c)
        out[key] = kept
    return out


def attrition_rows(attrition: list[dict] | None) -> list[dict]:
    """Display rows; suppressed counts stay strings (e.g. '<10') and are never turned into numbers."""
    rows = []
    for i, r in enumerate(attrition or []):
        remaining = r.get("remaining")
        rows.append(
            {
                "step": i,
                "rule": r.get("name") or r.get("rule") or r.get("description") or f"step {i}",
                "remaining": remaining,
                "numeric": remaining if isinstance(remaining, int) else None,
            }
        )
    return rows


def pending_for(cohorts: list[dict], me: str) -> list[dict]:
    """Drafts someone else wrote: what a reviewer could act on (the backend still decides)."""
    return [c for c in cohorts if c.get("status") == "draft" and c.get("created_by") != me]


def stale_names(before: dict, after: dict) -> list[str]:
    """Criteria whose threshold changed while their name still quotes the old number (names are labels: the
    server checks logic, not wording)."""
    out = []
    old = {c["id"]: c for key in ("inclusion", "exclusion") for c in before.get(key, [])}
    for key in ("inclusion", "exclusion"):
        for c in after.get(key, []):
            o = old.get(c["id"])
            if not o or not o.get("value_filter") or not c.get("value_filter"):
                continue
            ov, nv = o["value_filter"].get("value"), c["value_filter"].get("value")
            if ov != nv and c["name"] == o["name"]:
                text = f"{float(ov):g}"
                if re.search(rf"(?<![0-9.]){re.escape(text)}(?![0-9])", c["name"]):
                    out.append(f"'{c['name']}' still says {text} but the threshold is now {float(nv):g}")
    return out
