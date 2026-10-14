"""The workspace pages. Every number, status and definition shown comes from an API response; nothing is
estimated or invented in the UI. Actions go through the API, which enforces roles, tenant scope, approval
gates and auditing; the UI only hides buttons the user's roles cannot use."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from functools import partial
from typing import Any

import streamlit as st

from .client import ApiError
from .components import (
    api,
    attrition_block,
    badge,
    call,
    definition_header,
    issues_block,
    kv_table,
    pick_definition,
    show_badge,
    show_error,
    status_badge,
)
from .workspace import (
    answer_question,
    apply_edits,
    change_lines,
    classify_message,
    pending_for,
    stale_names,
    touched_criteria,
)

SUGGESTED = [
    "Adults with type 2 diabetes who started metformin and had an HbA1c above 8% in the year before starting. "
    "Exclude anyone with type 1 diabetes before starting.",
    "Patients with heart failure who started an SGLT2 inhibitor, with an ejection fraction below 40% in the year "
    "before starting.",
]


# ---- shared helpers ---------------------------------------------------------------------------------------
def roles() -> set[str]:
    r = set(st.session_state["me"]["roles"])
    return {"viewer", "author", "reviewer", "executor", "admin"} if "admin" in r else r


def has(role: str) -> bool:
    return role in roles()


def header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)


def ws() -> dict[str, Any]:
    return st.session_state.setdefault("ws", {"chat": [], "draft_id": None, "mine": []})


def cohorts() -> list[dict]:
    rows = call(lambda: api().list_cohorts(500), "loading cohort definitions") or []
    pending = len(pending_for(rows, st.session_state["me"]["subject"])) if has("reviewer") else 0
    st.session_state["pending_reviews"] = pending
    return rows


def open_in_builder(def_id: int) -> None:
    w = ws()
    w["draft_id"] = int(def_id)
    w["chat"].append({"role": "assistant", "kind": "note", "text": f"Opened definition #{def_id} in the workspace."})
    st.switch_page(PAGES["builder"])


def llm_status() -> dict:
    return ((st.session_state.get("status") or {}).get("checks") or {}).get("llm") or {}


# ---- 1. dashboard -------------------------------------------------------------------------------------------
def page_dashboard() -> None:
    header("Dashboard", "Live counts from the API. Metrics your role cannot read are shown as unavailable.")
    rows = cohorts()
    by: dict[str, int] = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    c = st.columns(5)
    c[0].metric("Cohort definitions", len(rows))
    c[1].metric("Drafts awaiting review", by.get("draft", 0))
    c[2].metric("With validation errors", by.get("needs_review", 0))
    c[3].metric("Approved", by.get("approved", 0))
    c[4].metric("Rejected", by.get("rejected", 0))
    st.caption("Counts cover the definitions visible to your tenant (latest 500).")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recent definitions")
        if rows:
            st.dataframe(
                [
                    {
                        "#": r["cohort_definition_id"],
                        "Name": r["name"],
                        "Status": r["status"],
                        "Dataset": r.get("dataset"),
                        "Created by": r.get("created_by"),
                        "Created": str(r.get("created_at"))[:19],
                    }
                    for r in rows[:10]
                ],  # fmt: skip
                hide_index=True,
            )
        else:
            st.info("No definitions yet. Start in **AI Cohort Builder**.")
        if has("reviewer"):
            pending = pending_for(rows, st.session_state["me"]["subject"])
            st.markdown(f"#### Pending review tasks ({len(pending)})")
            for p in pending[:8]:
                st.markdown(f"- #{p['cohort_definition_id']} {html.escape(p['name'])} — by {p['created_by']}")
        if has("author") or has("reviewer"):
            runs = call(lambda: api().runs(50), "loading runs", quiet=("forbidden",))
            if runs is not None:
                failed = sum(1 for r in runs if r["status"] == "failed")
                st.markdown(f"#### Agent runs (latest {len(runs)}): {failed} failed")
    with right:
        st.markdown("#### System")
        status = call(lambda: api().status(), "checking system status")
        if status:
            st.session_state["status"] = status
            for name, chk in status["checks"].items():
                tone = "ok" if chk.get("ok") else "bad" if name != "llm" else "warn"
                st.markdown(f"{badge(name, tone)} {html.escape(_check_text(name, chk))}", unsafe_allow_html=True)
        ds = call(lambda: api().datasets(), "loading datasets")
        if ds:
            st.markdown(
                f"**Datasets:** active `{ds['active']}` · built-in {len(ds['built_in'])} · "
                f"imported {len(ds['imported'])}"
            )
        st.markdown("#### Quick actions")
        q = st.columns(2)
        if q[0].button("Build a cohort", type="primary"):
            st.switch_page(PAGES["builder"])
        if q[1].button("Validate & preview"):
            st.switch_page(PAGES["validate"])
        if q[0].button("Review & approve"):
            st.switch_page(PAGES["review"])
        if q[1].button("Compare cohorts"):
            st.switch_page(PAGES["compare"])
        if has("admin"):
            audit = call(lambda: api().audit(15), "loading audit events", quiet=("forbidden",))
            if audit:
                st.markdown("#### Recent activity (audit)")
                st.dataframe(
                    [
                        {
                            "When": str(a["occurred_at"])[:19],
                            "Actor": a["actor"],
                            "Action": a["action"],
                            "Outcome": a["outcome"],
                        }
                        for a in audit
                    ],  # fmt: skip
                    hide_index=True,
                )


def _check_text(name: str, chk: dict) -> str:
    if name == "llm":
        key = "API key configured" if chk.get("api_key_configured") else "no API key"
        return f"mode {chk.get('mode')}, {key}, {chk.get('cached_responses')} cached responses"
    if name == "vocabulary":
        return str(chk.get("version") or chk.get("error"))
    if name == "dataset":
        return f"{chk.get('name')}: {'patients present' if chk.get('has_persons') else chk.get('error', 'empty')}"
    return "reachable" if chk.get("ok") else str(chk.get("error"))


# ---- 2. AI cohort builder -----------------------------------------------------------------------------------
def page_builder() -> None:
    header(
        "AI Cohort Builder",
        "Describe a cohort, refine it in conversation, ask why a criterion exists, or edit it directly. "
        "The AI proposes; the server validates; a reviewer approves.",
    )
    w = ws()
    llm = llm_status()
    if llm and not llm.get("api_key_configured"):
        st.warning(
            "No LLM API key is configured on the server: only requests already in the response cache can be "
            "answered. Opening, editing, validating, reviewing and executing definitions still work."
        )
    top = st.columns([1, 1, 4])
    if top[0].button("New cohort"):
        w.update(chat=[], draft_id=None)
        st.rerun()
    if w["draft_id"] and top[1].button("Reload draft"):
        st.rerun()

    draft = None
    if w["draft_id"]:
        draft = call(lambda: api().cohort(w["draft_id"]), f"loading definition #{w['draft_id']}")
    left, right = st.columns([5, 6], gap="large")
    with left:
        _chat_panel(w, draft)
    with right:
        _draft_panel(w, draft)


def _render_message(m: dict) -> None:
    kind = m.get("kind")
    if m["role"] == "user" or kind == "note":
        st.markdown(m["text"])
        return
    if kind == "error":
        st.error(m["text"])
        return
    if kind == "answer":
        a = m["answer"]
        show_badge("Explanation from the stored definition (no AI model)", "muted")
        if a["matched"]:
            for line in a["lines"]:
                st.markdown(f"- {html.escape(line)}")
        else:
            st.markdown("No criterion in this definition matches those words. The full definition is on the right.")
        if a["assumptions"]:
            st.markdown("**Assumptions recorded when it was built:**")
            for x in a["assumptions"]:
                st.markdown(f"- {html.escape(x)}")
        st.caption(a["limits"])
        return
    r = m["result"]
    status = r.get("status")
    if status == "failed":
        show_badge("No definition created", "bad")
        st.markdown("The request could not be turned into a definition. Details from the server:")
        issues_block(r.get("issues"))
        return
    label = "Proposed revision" if m.get("refine") else "Proposed definition"
    st.markdown(
        f"{badge(label + ' (AI interpretation)', 'info')} {status_badge(status)} "
        f"saved as **#{r['cohort_definition_id']}**",
        unsafe_allow_html=True,
    )
    manifest = r.get("manifest") or {}
    calls = manifest.get("llm_calls") or {}
    if calls:
        st.caption(
            f"LLM mode {manifest.get('llm_mode')}: {calls.get('total')} model call(s), "
            f"{calls.get('cache_hits')} answered from the response cache · run {r.get('run_id')}"
        )
    if m.get("refine"):
        changes = r.get("changes") or []
        if changes:
            st.markdown(f"**What changed versus #{r.get('parent_definition_id')}** ({len(changes)} field(s)):")
            st.code("\n".join(change_lines(changes)), language="text")
            parts = sorted(touched_criteria(changes))
            st.caption("Parts touched: " + ", ".join(parts) + ". Check that nothing unrelated changed.")
        else:
            st.info("The revision produced a definition identical to the previous version.")
    if status == "needs_review":
        st.markdown("**Clarification needed** — the server found problems the definition cannot run with:")
    issues_block(r.get("issues"))
    if r.get("attrition"):
        with st.expander("Dry-run attrition on the active dataset (not an execution)"):
            attrition_block(r["attrition"], "validation dry run returned with this proposal")
    st.caption(
        "A proposed definition is a draft: it runs only after a reviewer approves it. "
        "Feasibility shown here is a dry run on the active dataset."
    )


def _chat_panel(w: dict, draft: dict | None) -> None:
    st.markdown("#### Conversation")
    box = st.container(height=560, border=True)
    with box:
        if not w["chat"]:
            st.markdown(
                "Describe the patients you need: the entry event, inclusion and exclusion criteria, time windows "
                'and lab thresholds. Then refine it: *"change the HbA1c threshold to 9%"*, or ask '
                '*"why are patients with type 1 diabetes excluded?"*'
            )
            for i, s in enumerate(SUGGESTED):
                if st.button(s, key=f"suggest_{i}"):
                    st.session_state["pending_prompt"] = s
                    st.rerun()
        for m in w["chat"]:
            with st.chat_message(m["role"]):
                _render_message(m)
    can_write = has("author")
    prompt = st.chat_input(
        "Describe a cohort, request a change, or ask a question" if can_write else "Authors can build cohorts",
        disabled=not can_write and not draft,
    )
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if prompt:
        _handle_prompt(w, draft, prompt.strip())
        st.rerun()


def _handle_prompt(w: dict, draft: dict | None, text: str) -> None:
    if not text:
        return
    kind = classify_message(text, draft is not None)
    w["chat"].append({"role": "user", "text": text})
    if kind == "question" and draft is not None:
        w["chat"].append({"role": "assistant", "kind": "answer",
                          "answer": answer_question(text, draft["ir"], draft.get("status"))})  # fmt: skip
        return
    if not has("author"):
        w["chat"].append(
            {"role": "assistant", "kind": "error", "text": "Only authors can create or change definitions."}
        )
        return
    try:
        with st.spinner("The agents are interpreting the request and the server is validating it..."):
            if kind == "refine" and draft is not None:
                result = api().refine(draft["cohort_definition_id"], text)
            else:
                result = api().ask(text)
    except ApiError as exc:
        msg = "permission denied: " + exc.message if exc.kind == "forbidden" else exc.message
        w["chat"].append({"role": "assistant", "kind": "error", "text": msg})
        return
    w["chat"].append({"role": "assistant", "kind": "result", "result": result, "refine": kind == "refine"})
    if result.get("cohort_definition_id"):
        w["draft_id"] = result["cohort_definition_id"]
        w["mine"].append(result["cohort_definition_id"])


def _draft_panel(w: dict, draft: dict | None) -> None:
    st.markdown("#### Cohort definition")
    if draft is None:
        st.info("No draft yet. Describe a cohort in the conversation, or open one from the **Cohort Library**.")
        _start_from_file(w)
        return
    definition_header(draft)
    tabs = st.tabs(["Definition", "Edit", "Versions"])
    with tabs[0]:
        st.markdown(draft.get("explanation") or "")
        if draft.get("issues"):
            st.markdown("**Issues stored with this version**")
            issues_block(draft["issues"])
    with tabs[1]:
        _editor(w, draft)
    with tabs[2]:
        versions = call(lambda: api().versions_of(draft["cohort_definition_id"]), "loading versions") or []
        st.dataframe(
            [
                {
                    "#": v["cohort_definition_id"],
                    "Parent": v["parent_definition_id"],
                    "Status": v["status"],
                    "Created by": v["created_by"],
                    "Created": str(v["created_at"])[:19],
                    "From conversation": bool(v.get("run_id")),
                    "Semantic hash": str(v["semantic_hash"])[:12],
                }
                for v in versions
            ],  # fmt: skip
            hide_index=True,
        )
        others = [
            v["cohort_definition_id"] for v in versions if v["cohort_definition_id"] != draft["cohort_definition_id"]
        ]
        if others:
            pick = st.selectbox("Open another version", others, key="open_version")
            if st.button("Open version"):
                w["draft_id"] = int(pick)
                st.rerun()


def _start_from_file(w: dict) -> None:
    """The deterministic path (no AI model): upload a definition in the IR JSON format, validated and saved
    by the server like any other draft."""
    if not has("author"):
        return
    with st.expander("Start without the AI: upload a definition (IR JSON)"):
        st.caption("Same format as `examples/t2dm_metformin_hba1c.json` and `cohort-builder submit-ir`.")
        up = st.file_uploader("Definition JSON", type=["json"], key="ir_upload")
        if up is None:
            return
        try:
            ir = json.loads(up.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            st.error("The file is not valid UTF-8 JSON.")
            return
        if not isinstance(ir, dict):
            st.error("The file must contain one JSON object (a cohort definition).")
            return
        if st.button("Validate and save as a draft", type="primary"):
            res = call(partial(api().submit, ir, None), "saving the uploaded definition", "Validating and saving...")
            if res:
                new_id = res["cohort_definition_id"]
                w["mine"].append(new_id)
                w["draft_id"] = new_id
                errors = [i for i in res.get("issues", []) if i.get("severity") == "error"]
                w["chat"].append({"role": "assistant", "kind": "note",
                                  "text": f"Saved the uploaded definition as **#{new_id}**"
                                  + (f" with {len(errors)} validation error(s)." if errors else ".")})  # fmt: skip
                st.rerun()


def _editor(w: dict, draft: dict) -> None:
    ir = draft["ir"]
    def_id = draft["cohort_definition_id"]
    if draft["status"] == "approved":
        st.info("This version is approved. Saving changes creates a NEW draft that needs its own approval.")
    if not has("author"):
        st.caption("Read-only: your roles do not include author.")
    sets = {cs["id"]: cs for cs in ir.get("concept_sets", [])}
    with st.form(f"edit_{def_id}"):
        name = st.text_input("Cohort name", ir.get("name", ""))
        desc = st.text_area("Description", ir.get("description", ""), height=80)
        ie = ir.get("index_event", {})
        st.markdown(
            f"**Index event:** {'first' if ie.get('first_occurrence_only', True) else 'any'} occurrence of "
            f"`{sets.get(ie.get('concept_set_id', ''), {}).get('name', ie.get('concept_set_id'))}` "
            f"({ie.get('entity')}) — change it through the conversation."
        )
        c = st.columns(4)
        demo = ir.get("demographics") or {}
        age_min = c[0].number_input("Minimum age", 0, 130, value=demo.get("age_min"), placeholder="none")
        age_max = c[1].number_input("Maximum age", 0, 130, value=demo.get("age_max"), placeholder="none")
        prior = c[2].number_input("Prior observation (days)", 0, 36500, value=int(ir.get("prior_observation_days", 0)))
        post = c[3].number_input(
            "Follow-up after index (days)", 0, 36500, value=int(ir.get("post_observation_days", 0))
        )
        ex = ir.get("exit") or {"type": "end_of_observation"}
        e1, e2 = st.columns(2)
        exit_type = e1.selectbox(
            "Cohort exit", ["end_of_observation", "fixed_days"], index=0 if ex["type"] == "end_of_observation" else 1
        )
        exit_days = e2.number_input("Exit after (days, fixed_days only)", 1, 36500, value=ex.get("days") or 365)
        crit_edits: dict[str, dict] = {}
        for role, key in (("Inclusion", "inclusion"), ("Exclusion", "exclusion")):
            st.markdown(f"**{role} criteria**")
            if not ir.get(key):
                st.caption("none")
            for cr in ir.get(key, []):
                cs = sets.get(cr["concept_set_id"], {})
                with st.container(border=True):
                    st.caption(f"{cr['id']} · {cr['entity']} · concept set `{cs.get('name', cr['concept_set_id'])}`")
                    r1 = st.columns([3, 2, 1, 1])
                    e = {
                        "name": r1[0].text_input("Name", cr["name"], key=f"{def_id}_{cr['id']}_name"),
                        "occurrence": r1[1].selectbox(
                            "Occurrence",
                            ["at_least", "at_most", "exactly"],
                            index=["at_least", "at_most", "exactly"].index(cr.get("occurrence", "at_least")),
                            key=f"{def_id}_{cr['id']}_occ",
                        ),  # fmt: skip
                        "count": r1[2].number_input(
                            "Count", 0, 1000, value=int(cr.get("count", 1)), key=f"{def_id}_{cr['id']}_count"
                        ),  # fmt: skip
                        "remove": r1[3].checkbox("Remove", key=f"{def_id}_{cr['id']}_rm"),
                    }
                    win = cr.get("window") or {}
                    r2 = st.columns(4)
                    e["window_start"] = r2[0].number_input(
                        "Window start (days vs index)", -36500, 36500, value=win.get("start_days"),
                        placeholder="unbounded", key=f"{def_id}_{cr['id']}_ws",
                    )  # fmt: skip
                    e["window_end"] = r2[1].number_input(
                        "Window end (days vs index)", -36500, 36500, value=win.get("end_days"),
                        placeholder="unbounded", key=f"{def_id}_{cr['id']}_we",
                    )  # fmt: skip
                    vf = cr.get("value_filter")
                    if vf:
                        ops = [">", ">=", "<", "<=", "=", "between"]
                        e["op"] = r2[2].selectbox(
                            "Operator", ops, index=ops.index(vf["op"]), key=f"{def_id}_{cr['id']}_op"
                        )
                        e["value"] = r2[3].number_input("Value", value=float(vf["value"]), key=f"{def_id}_{cr['id']}_v")
                        if vf["op"] == "between" or vf.get("value_high") is not None:
                            e["value_high"] = st.number_input(
                                "Upper value (between)", value=float(vf.get("value_high") or vf["value"]),
                                key=f"{def_id}_{cr['id']}_vh",
                            )  # fmt: skip
                        st.caption(f"unit concept {vf.get('unit_concept_id')} (units are converted by the server)")
                    crit_edits[cr["id"]] = e
        b1, b2 = st.columns(2)
        do_validate = b1.form_submit_button("Validate changes (dry run, nothing saved)")
        do_save = b2.form_submit_button("Save as new version", type="primary", disabled=not has("author"))
    if not (do_validate or do_save):
        return
    edits = {
        "name": name,
        "description": desc,
        "demographics": {"age_min": age_min, "age_max": age_max},
        "prior_observation_days": prior,
        "post_observation_days": post,
        "exit": {"type": exit_type, "days": exit_days},
        "criteria": crit_edits,
    }
    new_ir = apply_edits(ir, edits)
    if new_ir == ir:
        st.info("No changes to validate or save.")
        return
    for msg in stale_names(ir, new_ir):
        st.warning(f"Criterion name not updated: {msg}. Rename it so reviewers are not misled.")
    if do_validate:
        res = call(lambda: api().validate(new_ir), "validating the edited definition", "Validating on the server...")
        if res:
            show_badge(
                "Valid on the active dataset" if res["valid"] else "Validation failed", "ok" if res["valid"] else "bad"
            )
            issues_block(res["issues"])
            attrition_block(res.get("attrition"), "dry run of the edited (unsaved) definition")
        return
    # stale-edit guard: refuse to branch from a version someone else has already revised
    versions = call(lambda: api().versions_of(def_id), "checking for newer versions") or []
    newer = [v for v in versions if v["parent_definition_id"] == def_id and v["cohort_definition_id"] not in w["mine"]]
    if newer:
        st.error(
            f"Version #{newer[-1]['cohort_definition_id']} was already created from #{def_id} by "
            f"{newer[-1]['created_by']}. Open the latest version (Versions tab) before editing, so no change is lost."
        )
        return
    res = call(lambda: api().submit(new_ir, def_id), "saving the new version", "Saving and validating...")
    if res:
        new_id = res["cohort_definition_id"]
        w["mine"].append(new_id)
        w["draft_id"] = new_id
        errors = [i for i in res.get("issues", []) if i.get("severity") == "error"]
        w["chat"].append(
            {"role": "assistant", "kind": "note",
             "text": f"Saved structured edits as version **#{new_id}** (parent #{def_id})"
                     + (f" — {len(errors)} validation error(s); see the Definition tab." if errors else ".")}
        )  # fmt: skip
        st.rerun()


# ---- 3. library ----------------------------------------------------------------------------------------------
def page_library() -> None:
    header("Cohort Library", "Every definition visible to your tenant, newest first.")
    rows = cohorts()
    c = st.columns([3, 2, 2])
    q = c[0].text_input("Search", value=st.session_state.pop("library_query", ""), placeholder="name or #id")
    statuses = sorted({r["status"] for r in rows})
    datasets = sorted({str(r.get("dataset")) for r in rows})
    fs = c[1].multiselect("Status", statuses)
    fd = c[2].multiselect("Dataset", datasets)
    ql = q.strip().lower().lstrip("#")
    shown = [
        r for r in rows
        if (not ql or ql in r["name"].lower() or ql == str(r["cohort_definition_id"]))
        and (not fs or r["status"] in fs) and (not fd or str(r.get("dataset")) in fd)
    ]  # fmt: skip
    if not shown:
        st.info("No definitions match." if rows else "No definitions yet.")
        return
    st.dataframe(
        [
            {
                "#": r["cohort_definition_id"],
                "Name": r["name"],
                "Status": r["status"],
                "Dataset": r.get("dataset"),
                "Created by": r.get("created_by"),
                "Created": str(r.get("created_at"))[:19],
            }
            for r in shown
        ],  # fmt: skip
        hide_index=True,
    )
    def_id = pick_definition("Definition", shown, "lib_pick")
    if def_id is None:
        return
    row = call(lambda: api().cohort(def_id), "loading the definition")
    if not row:
        return
    definition_header(row)
    a = st.columns(4)
    if a[0].button("Open in builder", type="primary"):
        open_in_builder(def_id)
    if a[1].button("Validate & preview"):
        st.session_state["selected_def"] = def_id
        st.switch_page(PAGES["validate"])
    if a[2].button("Review"):
        st.session_state["selected_def"] = def_id
        st.switch_page(PAGES["review"])
    if a[3].button("SQL & ontology"):
        st.session_state["selected_def"] = def_id
        st.switch_page(PAGES["explorer"])
    st.markdown(row.get("explanation") or "")


# ---- 4. validate & preview -----------------------------------------------------------------------------------
def page_validate() -> None:
    header(
        "Validate & Preview",
        "Server-side validation and a dry run on the active dataset. A dry run is not an execution: nothing is "
        "materialized and counts are suppressed below the minimum cell size.",
    )
    rows = cohorts()
    default = st.session_state.get("selected_def") or ws().get("draft_id")
    def_id = pick_definition("Definition", rows, "val_pick", default)
    if def_id is None:
        return
    row = call(lambda: api().cohort(def_id), "loading the definition")
    if not row:
        return
    definition_header(row)
    st.markdown(row.get("explanation") or "")
    if st.button("Run validation (dry run)", type="primary"):
        res = call(lambda: api().validate(row["ir"]), "validating", "Validating on the server...")
        if res:
            st.session_state[f"validation_{def_id}"] = {**res, "at": datetime.now(timezone.utc).isoformat()[:19]}
    res = st.session_state.get(f"validation_{def_id}")
    if not res:
        st.info("Not validated in this session yet.")
    else:
        show_badge(
            "Valid on the active dataset" if res["valid"] else "Validation failed", "ok" if res["valid"] else "bad"
        )
        st.caption(f"validated {res['at']} UTC · semantic hash `{res['semantic_hash'][:16]}`")
        issues_block(res["issues"])
        attrition_block(res.get("attrition"), "validation dry run (POST /cohorts/validate)")
    if st.button("Back to the editor"):
        open_in_builder(def_id)
    if has("executor") or has("reviewer"):
        gens = call(lambda: api().generations_of(def_id), "loading executions", quiet=("forbidden",)) or []
        st.markdown("#### Executions of this version")
        if gens:
            st.dataframe(
                [
                    {
                        "Generation": g["generation_id"],
                        "Executed": str(g["executed_at"])[:19],
                        "By": g["executed_by"],
                        "Dataset": g["dataset"],
                        "Patients": str(g["person_count"]),
                        "SQL hash": str(g["sql_hash"])[:12],
                    }
                    for g in gens
                ],  # fmt: skip
                hide_index=True,
            )
        else:
            st.caption("Not executed yet.")


# ---- 5. review & approve ---------------------------------------------------------------------------------------
def page_review() -> None:
    header(
        "Review & Approve",
        "Approval applies to one exact version (its content hash). Editing creates a new version that needs its "
        "own approval. Execution stays blocked until the server confirms approval.",
    )
    rows = cohorts()
    me = st.session_state["me"]["subject"]
    status = st.session_state.get("status") or {}
    gov = status.get("governance") or {}
    selected = st.session_state.get("selected_def")
    sel_status = next((r["status"] for r in rows if r["cohort_definition_id"] == selected), None)
    # reviewers start on their queue; executors (and links to a non-draft version) on everything
    start = 0 if has("reviewer") and sel_status in (None, "draft", "needs_review") else 1
    queue = st.radio("Show", ["Awaiting review", "All"], index=start, horizontal=True)
    shown = [r for r in rows if r["status"] in ("draft", "needs_review")] if queue == "Awaiting review" else rows
    def_id = pick_definition("Definition", shown, "rev_pick", selected)
    if def_id is None:
        return
    row = call(lambda: api().cohort(def_id), "loading the definition")
    if not row:
        return
    definition_header(row)
    st.caption(f"content hash `{row.get('content_hash')}`")
    t = st.tabs(["Definition", "Stored issues", "SQL preview", "Review history"])
    with t[0]:
        st.markdown(row.get("explanation") or "")
    with t[1]:
        issues_block(
            row.get("issues"),
            "No errors were stored with this version (warnings are not stored: see Validate & Preview).",
        )
    with t[2]:
        sql = call(lambda: api().cohort_sql(def_id), "compiling SQL")
        if sql:
            st.code(sql, language="sql")
    with t[3]:
        reviews = call(lambda: api().reviews_of(def_id), "loading reviews") or []
        if reviews:
            st.dataframe(reviews, hide_index=True)
        else:
            st.caption("No reviews yet.")

    st.markdown("#### Decision")
    if not has("reviewer"):
        st.info("Your roles do not include reviewer: you can read this definition but not approve or reject it.")
    elif row["status"] not in ("draft", "needs_review"):
        st.info(f"This version is {row['status']}; only drafts can be reviewed.")
    else:
        own = row["created_by"] == me
        if own and not gov.get("allow_self_approval", False):
            st.warning(
                "You wrote this version: self-approval is disabled on this server. Another reviewer must approve."
            )
        with st.form(f"review_{def_id}"):
            decision = st.radio("Decision", ["approved", "rejected"], horizontal=True)
            comments = st.text_area("Comments (required to reject)")
            confirm = st.checkbox(f"I reviewed version #{def_id} (content hash {str(row.get('content_hash'))[:12]})")
            go = st.form_submit_button("Submit decision", type="primary")
        if go:
            if not confirm:
                st.error("Confirm that you reviewed this exact version.")
            elif decision == "rejected" and not comments.strip():
                st.error("Explain the rejection so the author can fix it.")
            else:
                res = call(lambda: api().review(def_id, decision, comments.strip()), "submitting the review")
                if res:
                    st.success(f"Recorded by the server: {res['decision']} by {res['reviewer']}.")
    st.markdown("#### Execution")
    if row["status"] != "approved":
        show_badge(f"Blocked: the version is {row['status']}, not approved", "warn")
    elif not has("executor"):
        st.info("Approved. Your roles do not include executor.")
    else:
        confirm = st.checkbox("Run this approved version on the active dataset (materializes the cohort)")
        if st.button("Execute", type="primary", disabled=not confirm):
            res = call(lambda: api().execute(def_id), "executing", "Executing on the server...")
            if res:
                st.success(
                    f"Executed: generation `{res['generation_id']}` on {res['dataset']}, "
                    f"{res['person_count']} patients (minimum cell {res['min_cell_count']})."
                )
                issues_block(res.get("caveats"))
                attrition_block(res.get("attrition"), f"execution {res['generation_id']}")


# ---- 6. SQL & ontology explorer -------------------------------------------------------------------------------
def page_explorer() -> None:
    header("SQL & Ontology Explorer", "Compiler output and terminology for one version. Nothing is executed here.")
    rows = cohorts()
    def_id = pick_definition(
        "Definition", rows, "exp_pick", st.session_state.get("selected_def") or ws().get("draft_id")
    )
    t = st.tabs(["Compiled SQL", "Concept sets", "Concept search", "Dataset mapping", "Metadata"])
    row = call(partial(api().cohort, def_id), "loading the definition") if def_id is not None else None
    with t[0]:
        if row and def_id is not None:
            sql = call(partial(api().cohort_sql, def_id), "compiling SQL")
            if sql:
                st.caption(
                    f"Preview from the deterministic compiler for #{def_id} ({row['status']}). This is not the "
                    "record of an execution; executions are listed under Validate & Preview."
                )
                st.code(sql, language="sql")
                st.download_button("Download SQL", sql, file_name=f"cohort_{def_id}.sql", mime="text/plain")
    with t[1]:
        if row:
            for cs in row["ir"].get("concept_sets", []):
                with st.expander(
                    f"{cs['name']} · {cs['domain']} · {cs.get('source', 'resolved')} · {len(cs['items'])} item(s)"
                ):
                    out = []
                    for item in cs["items"][:50]:
                        c = call(partial(api().concept, item["concept_id"]), quiet=("not_found",))
                        out.append({
                            "Concept id": item["concept_id"],
                            "Name": c["concept_name"] if c else "NOT IN VOCABULARY",
                            "Vocabulary": c.get("vocabulary_id") if c else "",
                            "Domain": c.get("domain_id") if c else "",
                            "Standard": c.get("standard_concept") if c else "",
                            "Descendants": (c.get("descendant_count") if c else "")
                            if item["include_descendants"] else "not included",
                            "Excluded": item["is_excluded"],
                        })  # fmt: skip
                    st.dataframe(out, hide_index=True)
                    if len(cs["items"]) > 50:
                        st.caption(f"First 50 of {len(cs['items'])} items shown.")
    with t[2]:
        q = st.text_input("Search the vocabulary", key="concept_q")
        if q.strip():
            res = call(lambda: api().search_concepts(q.strip()), "searching") or []
            st.dataframe(res, hide_index=True) if res else st.caption("No matches.")
    with t[3]:
        status = st.session_state.get("status") or {}
        ds = status.get("dataset") or {}
        st.markdown(f"**Active dataset:** `{ds.get('name')}` — {ds.get('description', '')}")
        st.caption(f"data snapshot: {ds.get('data_snapshot')}")
        if ds.get("capabilities"):
            st.json(ds["capabilities"], expanded=False)
    with t[4]:
        if row:
            kv_table({k: row.get(k) for k in ("cohort_definition_id", "parent_definition_id", "status", "dataset",
                     "schema_version", "ontology_version", "vocabulary_version", "content_hash", "semantic_hash",
                     "run_id", "created_by", "created_at", "approved_by", "approved_at")})  # fmt: skip


# ---- 7. run history & replay ---------------------------------------------------------------------------------
def page_runs() -> None:
    header(
        "Run History & Replay",
        "Agent runs that built definitions. Authors see their own runs, reviewers their tenant's.",
    )
    if not (has("author") or has("reviewer")):
        st.info("Run traces are visible to authors (own runs) and reviewers.")
        return
    runs = call(lambda: api().runs(200), "loading runs") or []
    if not runs:
        st.info("No runs yet.")
        return
    st.dataframe(
        [
            {
                "Run": r["run_id"],
                "Status": r["status"],
                "User": r["user_id"],
                "Submitted": str(r["submitted_at"])[:19],
                "Finished": str(r["finished_at"])[:19],
                "Definition": r["output_definition_id"],
                "Retries": r["retry_count"],
            }
            for r in runs
        ],  # fmt: skip
        hide_index=True,
    )
    run_id = st.selectbox("Run", [r["run_id"] for r in runs])
    run = call(lambda: api().run(run_id), "loading the run")
    if not run:
        return
    m = run.get("manifest") or {}
    st.markdown(f"**Request:** {html.escape(str(run.get('user_query')))}")
    kv_table({
        "status": run.get("status"), "definition": m.get("cohort_definition_id"),
        "IR semantic hash": m.get("ir_semantic_hash"), "IR content hash": m.get("ir_content_hash"),
        "LLM mode": m.get("llm_mode"), "LLM calls (cache hits)": f"{(m.get('llm_calls') or {}).get('total')} "
        f"({(m.get('llm_calls') or {}).get('cache_hits')})", "model": m.get("model"),
        "dataset": (m.get("dataset") or {}).get("name"), "vocabulary": m.get("vocabulary_version"),
        "data snapshot": m.get("data_snapshot"), "dry-run count": m.get("dry_run_count"),
    })  # fmt: skip
    if run.get("steps"):
        with st.expander("Agent steps"):
            st.dataframe(run["steps"], hide_index=True)
    now_v = call(lambda: api().versions(), "loading current versions") or {}
    changed = [k for k in ("model", "vocabulary_version", "data_snapshot", "compiler_version")
               if m.get(k) is not None and now_v.get(k) is not None and m.get(k) != now_v.get(k)]  # fmt: skip
    if (m.get("dataset") or {}).get("name") != (now_v.get("dataset") or {}).get("name"):
        changed.append("dataset")
    if json.dumps(m.get("prompts"), sort_keys=True) != json.dumps(now_v.get("prompts"), sort_keys=True):
        changed.append("prompts")
    if changed:
        st.warning(
            "Since this run, these dependencies changed: " + ", ".join(changed) + ". An exact replay is not guaranteed."
        )
    else:
        st.caption("Recorded model, prompts, dataset, vocabulary, data snapshot and compiler match the current server.")
    if has("author") and run.get("user_id") == st.session_state["me"]["subject"]:
        if st.button("Replay from recorded LLM responses"):
            res = call(lambda: api().replay(run_id), "replaying", "Replaying...")
            if res:
                if res["identical"]:
                    st.success("Replay rebuilt a definition with the identical semantic hash.")
                else:
                    st.error(f"Replay differs or failed (status {res['replay_status']}).")
                    issues_block(res.get("issues"))
                st.caption(
                    "An identical hash shows the recorded responses rebuild the same definition. It does not prove "
                    "the data is unchanged: compare the data snapshot and versions above."
                )
    else:
        st.caption("Only the author of a run can replay it.")


# ---- 8. datasets -------------------------------------------------------------------------------------------------
def page_datasets() -> None:
    header(
        "Dataset Management",
        "Datasets known to the server. The server runs on one active dataset (CB_DATASET); switching it means "
        "restarting the API with another value.",
    )
    ds = call(lambda: api().datasets(), "loading datasets")
    if not ds:
        return
    st.markdown(f"**Active:** `{ds['active']}` · **Built-in:** {', '.join(ds['built_in'])}")
    if ds["imported"]:
        st.dataframe(ds["imported"], hide_index=True)
        name = st.selectbox("Imported dataset", [d.get("dataset") or d.get("name") for d in ds["imported"]])
        info = call(lambda: api().dataset(name), "loading the dataset") if name else None
        if info:
            st.json(info, expanded=False)
    else:
        st.caption("No imported datasets.")
    st.markdown("#### Profile a CSV (nothing is stored)")
    if not has("author"):
        st.caption("Profiling needs the author role.")
    else:
        up = st.file_uploader("CSV file (UTF-8, up to 5 MB through the web)", type=["csv", "tsv"], key="profile_up")
        if up is not None:
            raw = up.getvalue()
            if len(raw) > 5_000_000:
                st.error("The file is larger than 5 MB: use `cohort-builder csv profile` for large files.")
            else:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    st.error("The file is not UTF-8 text.")
                    text = ""
                if text:
                    res = call(lambda: api().profile_csv(up.name, text), "profiling", "Profiling...")
                    if res:
                        st.json(res, expanded=False)
    st.markdown("#### Import a CSV dataset (admin)")
    if not has("admin"):
        st.caption("Imports need the admin role. They create a new immutable dataset version and are audited.")
        return
    with st.form("ingest"):
        name = st.text_input("Dataset name (lower-case letters, digits, underscores)")
        files = st.file_uploader("CSV files", type=["csv", "tsv"], accept_multiple_files=True)
        mapping = st.text_area("Mapping YAML (optional: without it files are only staged)", height=120)
        new_version = st.checkbox("Add a new version to an existing dataset")
        dry = st.form_submit_button("Check (dry run, nothing kept)")
        confirm = st.checkbox("I checked the dry-run result and want to import")
        real = st.form_submit_button("Import", type="primary")
    if (dry or real) and name and files:
        payload = []
        for f in files:
            try:
                payload.append({"file_name": f.name, "content": f.getvalue().decode("utf-8")})
            except UnicodeDecodeError:
                st.error(f"{f.name} is not UTF-8 text.")
                return
        if real and not confirm:
            st.error("Run the dry run and tick the confirmation first.")
            return
        res = call(lambda: api().ingest_csv(name, payload, mapping or None, new_version, dry_run=not real),
                   "importing", "Checking..." if dry else "Importing...")  # fmt: skip
        if res:
            (st.info if dry else st.success)("Dry run finished: nothing was kept." if dry else "Imported.")
            st.json(res, expanded=False)


# ---- 9. comparison -----------------------------------------------------------------------------------------------
def page_compare() -> None:
    header(
        "Cohort Comparison",
        "Definitions side by side, exact field differences, and patient overlap of executed versions where the "
        "server can compute it (suppressed; same dataset only).",
    )
    rows = cohorts()
    names = {r["cohort_definition_id"]: f"#{r['cohort_definition_id']} {r['name']} ({r['status']})" for r in rows}
    picked = st.multiselect("Definitions (2 to 4)", list(names), format_func=lambda i: names[i], max_selections=4)
    if len(picked) < 2:
        st.info("Pick at least two definitions.")
        return
    loaded = [call(partial(api().cohort, i), f"loading #{i}") for i in picked]
    full = [d for d in loaded if d]
    if len(full) != len(picked):
        return
    table = []
    for d in full:
        ir = d["ir"]
        sets = {cs["id"]: cs["name"] for cs in ir.get("concept_sets", [])}
        demo = ir.get("demographics") or {}
        table.append({
            "#": d["cohort_definition_id"], "Status": d["status"], "Dataset": d.get("dataset"),
            "Index event": sets.get(ir["index_event"]["concept_set_id"]),
            "Age": f"{demo.get('age_min', '-')}–{demo.get('age_max', '-')}",
            "Prior obs (d)": ir.get("prior_observation_days"), "Inclusion": len(ir.get("inclusion", [])),
            "Exclusion": len(ir.get("exclusion", [])), "Semantic hash": d["semantic_hash"][:12],
        })  # fmt: skip
    st.dataframe(table, hide_index=True)
    base = picked[0]
    for other in picked[1:]:
        diff = call(partial(api().diff, base, other), f"comparing #{base} and #{other}")
        if diff:
            st.markdown(f"**#{base} → #{other}**: {len(diff['changes'])} field difference(s)")
            if not diff["same_dataset"]:
                st.warning("Different datasets: counts are not comparable.")
            if diff["changes"]:
                st.code("\n".join(change_lines(diff["changes"])), language="text")
    st.markdown("#### Patient overlap of executions")
    if not (has("executor") or has("reviewer")):
        st.info("Execution counts are visible to executors and reviewers.")
        return
    latest = {}
    for i in picked:
        gens = call(partial(api().generations_of, i), f"loading executions of #{i}", quiet=("forbidden",)) or []
        if gens:
            latest[i] = gens[0]
    if len(latest) < 2:
        st.info("Fewer than two of these versions have been executed: no overlap to compute (nothing is estimated).")
        return
    res = call(lambda: api().compare([g["generation_id"] for g in latest.values()]), "computing overlap")
    if res:
        for wmsg in res.get("warnings", []):
            st.warning(wmsg)
        st.dataframe(res["generations"], hide_index=True)
        st.dataframe(res["overlaps"], hide_index=True)
        for n in res.get("notes", []):
            st.caption(n)


# ---- 10. system health ---------------------------------------------------------------------------------------------
def page_health() -> None:
    header("System Health & Testing", "Each check is run by the server now; nothing is inferred from one request.")
    try:
        api().health()
        show_badge("API reachable", "ok")
    except ApiError as exc:
        show_error(exc, "API health")
        return
    status = call(lambda: api().status(), "loading status")
    if not status:
        return
    st.session_state["status"] = status
    rows = []
    for name, chk in status["checks"].items():
        rows.append({"Check": name, "OK": "yes" if chk.get("ok") else "no", "Detail": _check_text(name, chk)})
    st.dataframe(rows, hide_index=True)
    llm = status["checks"].get("llm") or {}
    if llm.get("note"):
        st.info(llm["note"])
    st.markdown("#### Governance settings")
    kv_table(status["governance"])
    st.markdown("#### Identity")
    kv_table(st.session_state["me"])
    st.markdown("#### Versions")
    st.json(status["versions"], expanded=False)
    if has("admin"):
        audit = call(lambda: api().audit(100), "loading the audit log") or []
        st.markdown("#### Audit log (latest 100)")
        st.dataframe(
            [
                {
                    "When": str(a["occurred_at"])[:19],
                    "Actor": a["actor"],
                    "Tenant": a["tenant"],
                    "Action": a["action"],
                    "Resource": f"{a['resource_type']} {a['resource_id']}",
                    "Outcome": a["outcome"],
                }
                for a in audit
            ],  # fmt: skip
            hide_index=True,
        )
    st.markdown("#### Automated tests")
    st.caption("Run `pytest` in the repository (see the UI guide); results are not collected by this page.")


# ---- navigation ------------------------------------------------------------------------------------------------------
PAGES: dict[str, Any] = {}


def navigation() -> list[Any]:
    spec = [
        ("dashboard", page_dashboard, "Dashboard", ":material/dashboard:", True),
        ("builder", page_builder, "AI Cohort Builder", ":material/chat:", False),
        ("library", page_library, "Cohort Library", ":material/library_books:", False),
        ("validate", page_validate, "Validate & Preview", ":material/fact_check:", False),
        ("review", page_review, "Review & Approve", ":material/verified:", False),
        ("explorer", page_explorer, "SQL & Ontology Explorer", ":material/code:", False),
        ("runs", page_runs, "Run History & Replay", ":material/history:", False),
        ("datasets", page_datasets, "Dataset Management", ":material/database:", False),
        ("compare", page_compare, "Cohort Comparison", ":material/compare_arrows:", False),
        ("health", page_health, "System Health & Testing", ":material/monitor_heart:", False),
    ]
    PAGES.clear()
    for key, fn, title, icon, default in spec:
        PAGES[key] = st.Page(fn, title=title, icon=icon, url_path=key, default=default)
    return list(PAGES.values())
