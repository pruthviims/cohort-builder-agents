"""Small Streamlit building blocks shared by the workspace pages (status badges, error display, API calls
with spinners, attrition tables). Presentation only."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from .client import ApiClient, ApiError
from .workspace import attrition_rows, status_label

T = TypeVar("T")

CSS = """
<style>
:root { --cb-blue:#1d4ed8; --cb-green:#15803d; --cb-amber:#b45309; --cb-red:#b91c1c; --cb-gray:#475569; }
.cb-badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
            border:1px solid currentColor; line-height:1.6; white-space:nowrap; }
.cb-ok   { color:var(--cb-green); background:#f0fdf4; }
.cb-warn { color:var(--cb-amber); background:#fffbeb; }
.cb-bad  { color:var(--cb-red);   background:#fef2f2; }
.cb-info { color:var(--cb-blue);  background:#eff6ff; }
.cb-muted{ color:var(--cb-gray);  background:#f8fafc; }
.cb-topbar { display:flex; gap:12px; align-items:center; justify-content:space-between; padding:6px 0 10px 0;
             border-bottom:1px solid #e2e8f0; margin-bottom:12px; }
.cb-title { font-weight:700; font-size:1.05rem; color:#0f172a; }
.cb-sub { color:#475569; font-size:0.85rem; }
</style>
"""


def badge(text: str, tone: str = "info") -> str:
    return f'<span class="cb-badge cb-{tone}">{html.escape(text)}</span>'


def show_badge(text: str, tone: str = "info") -> None:
    st.markdown(badge(text, tone), unsafe_allow_html=True)


def status_badge(status: str | None) -> str:
    label, tone = status_label(status)
    return badge(label, tone)


def api() -> ApiClient:
    return st.session_state["api"]


def show_error(exc: ApiError, action: str = "") -> None:
    what = f"{action}: " if action else ""
    if exc.kind == "forbidden":
        st.error(f"{what}permission denied. {exc.message}")
    elif exc.kind == "unauthorized":
        st.error(f"{what}{exc.message}. Sign in again from the sidebar.")
    elif exc.kind == "unavailable":
        st.error(f"{what}{exc.message}")
    elif exc.kind == "not_found":
        st.warning(f"{what}not found (or not visible to your tenant). {exc.message}")
    elif exc.kind in ("conflict", "invalid"):
        st.error(f"{what}{exc.message}")
    else:
        st.error(f"{what}{exc.message}")


def call(fn: Callable[[], T], action: str = "", spinner: str | None = None, quiet: tuple[str, ...] = ()) -> T | None:
    """Run an API call; on failure show an actionable message and return None. `quiet` kinds (e.g.
    'forbidden' for optional panels) are not shown."""
    try:
        if spinner:
            with st.spinner(spinner):
                return fn()
        return fn()
    except ApiError as exc:
        if exc.kind not in quiet:
            show_error(exc, action)
        return None


def issues_block(issues: list[dict] | None, empty: str = "No validation errors or warnings were returned.") -> None:
    issues = issues or []
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    if not issues:
        st.success(empty)
    for i in errors:
        st.error(f"**{i.get('stage', 'validation')}**: {i.get('message')}")
    for i in warnings:
        st.warning(f"**{i.get('stage', 'validation')}**: {i.get('message')}")


def attrition_block(attrition: list[dict] | None, source: str) -> None:
    rows = attrition_rows(attrition)
    if not rows:
        st.info("No attrition was returned for this definition (nothing to show; no numbers are estimated).")
        return
    st.caption(f"Source: {source}. Counts below the minimum cell size are suppressed by the server and shown as such.")
    st.dataframe(
        [{"Step": r["step"], "Criterion": r["rule"], "Patients remaining": str(r["remaining"])} for r in rows],
        hide_index=True,
    )
    numeric = [r for r in rows if r["numeric"] is not None]
    if len(numeric) >= 2 and len(numeric) == len(rows):
        st.bar_chart(
            {"Patients remaining": {f"{r['step']}. {r['rule']}": r["numeric"] for r in rows}},
            horizontal=True,
            color="#1d4ed8",
        )
    elif rows:
        st.caption("Chart omitted: some steps are suppressed, so a chart would misstate them.")


def definition_header(row: dict) -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"#### #{row.get('cohort_definition_id')} {html.escape(str(row.get('name', '')))}")
        st.caption(
            f"dataset {row.get('dataset') or 'unknown'} · created by {row.get('created_by')} · "
            f"{str(row.get('created_at', ''))[:19]}"
            + (f" · parent #{row['parent_definition_id']}" if row.get("parent_definition_id") else "")
        )
    with c2:
        st.markdown(status_badge(row.get("status")), unsafe_allow_html=True)
        if row.get("approved_by"):
            st.caption(f"approved by {row['approved_by']} at {str(row.get('approved_at', ''))[:19]}")
        st.caption(f"semantic hash `{str(row.get('semantic_hash', ''))[:16]}`")


def pick_definition(label: str, cohorts: list[dict], key: str, default: int | None = None) -> int | None:
    if not cohorts:
        st.info("No cohort definitions yet. Create one in **AI Cohort Builder**.")
        return None
    ids = [c["cohort_definition_id"] for c in cohorts]
    names = {c["cohort_definition_id"]: f"#{c['cohort_definition_id']} {c['name']} ({c['status']})" for c in cohorts}
    index = ids.index(default) if default in ids else 0
    return st.selectbox(label, ids, index=index, format_func=lambda i: names[i], key=key)


def kv_table(d: dict[str, Any]) -> None:
    st.dataframe([{"Field": k, "Value": str(v)} for k, v in d.items()], hide_index=True)
