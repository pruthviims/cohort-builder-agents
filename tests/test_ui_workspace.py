"""The analytics workspace (Streamlit) against the REAL API and database (in process, scripted fake LLM):
multi-turn chat, structured editing, stale-edit protection, validation, review, execution gates, honest
failure states and secret handling. Plus unit tests of the presentation logic. SYNTHETIC DATA ONLY."""

from __future__ import annotations

import copy
import json

import pytest
import yaml
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from cohort_builder.api import create_app
from cohort_builder.config import REPO_ROOT
from cohort_builder.orchestrator import CohortBuilder
from cohort_builder.security import SecurityConfig, issue_token
from cohort_builder.ui.client import ApiClient, ApiError
from cohort_builder.ui.workspace import (
    answer_question,
    apply_edits,
    attrition_rows,
    classify_message,
    pending_for,
    stale_names,
)
from fake_llm import GOLDEN_INTENTS, T2DM_QUERY, FakeLLM
from test_api_workspace import REFINED_INTENT, REFINED_QUERY, REVISION

APP = str(REPO_ROOT / "src" / "cohort_builder" / "ui" / "app.py")
EXAMPLE_IR = json.loads((REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json").read_text())


# ---- presentation logic --------------------------------------------------------------------------------------
def test_message_routing():
    assert classify_message("Adults with diabetes", has_draft=False) == "new"
    assert classify_message("Why are patients with type 1 diabetes excluded?", has_draft=True) == "question"
    assert classify_message("Change the HbA1c threshold to 9%", has_draft=True) == "refine"
    assert classify_message("make the lookback six months", has_draft=True) == "refine"
    assert classify_message("Also require two diagnoses", has_draft=True) == "refine"
    assert classify_message("What is the index event", has_draft=True) == "question"


def test_answers_come_only_from_the_definition():
    a = answer_question("Why are patients with type 1 diabetes excluded?", EXAMPLE_IR, "draft")
    assert a["matched"] and len(a["lines"]) == 1 and a["lines"][0].startswith("Exclusion criterion")
    assert "no AI model" in a["source"]
    none = answer_question("Why is warfarin excluded?", EXAMPLE_IR, "draft")
    assert not none["matched"] and none["lines"] == []  # nothing invented


def test_structured_edits_preserve_everything_else():
    crit = EXAMPLE_IR["inclusion"][1]
    edited = apply_edits(
        EXAMPLE_IR,
        {"criteria": {crit["id"]: {"value": 9, "window_start": -180}}, "demographics": {"age_min": 21}},
    )
    assert edited is not EXAMPLE_IR and EXAMPLE_IR["inclusion"][1]["value_filter"]["value"] == 8  # input untouched
    assert edited["inclusion"][1]["value_filter"]["value"] == 9.0
    assert edited["inclusion"][1]["window"]["start_days"] == -180
    assert edited["demographics"]["age_min"] == 21
    for key in ("concept_sets", "index_event", "exclusion"):
        assert edited[key] == EXAMPLE_IR[key]
    assert edited["inclusion"][0] == EXAMPLE_IR["inclusion"][0]
    removed = apply_edits(EXAMPLE_IR, {"criteria": {EXAMPLE_IR["exclusion"][0]["id"]: {"remove": True}}})
    assert removed["exclusion"] == []
    fixed = apply_edits(EXAMPLE_IR, {"exit": {"type": "fixed_days", "days": 180}})
    assert fixed["exit"] == {"type": "fixed_days", "days": 180}
    assert stale_names(EXAMPLE_IR, edited) == [f"'{crit['name']}' still says 8 but the threshold is now 9"]


def test_suppressed_counts_stay_suppressed():
    rows = attrition_rows([{"name": "a", "remaining": 120}, {"name": "b", "remaining": "<10"}])
    assert rows[1]["remaining"] == "<10" and rows[1]["numeric"] is None
    assert attrition_rows(None) == []
    assert pending_for([{"status": "draft", "created_by": "x"}, {"status": "draft", "created_by": "me"}], "me") == [
        {"status": "draft", "created_by": "x"}
    ]


# ---- fixtures: a real API in process -------------------------------------------------------------------------
@pytest.fixture
def env(settings, tmp_path):
    llm = FakeLLM({**GOLDEN_INTENTS, REFINED_QUERY: [copy.deepcopy(REFINED_INTENT)]})
    b = CohortBuilder(settings, backend=llm)
    entries, tokens = [], {}
    for name, roles, tenant in [
        ("alice", ["author"], "default"),
        ("bob", ["author"], "default"),
        ("rita", ["reviewer"], "default"),
        ("eddie", ["executor"], "default"),
        ("vic", ["viewer"], "default"),
        ("sam", ["author", "reviewer"], "default"),
    ]:
        tokens[name], entry = issue_token(name, roles, tenant, expires_days=1)
        entries.append(entry)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({"tokens": entries}))
    http = TestClient(create_app(b, SecurityConfig(tokens_file=tmp_path / "t.yaml")))

    def client(user: str) -> ApiClient:
        return ApiClient("http://testserver", tokens[user], session=http)

    yield client, tokens
    b.con.close()


PAGE = """
from cohort_builder.ui import pages
pages.navigation()
pages.{fn}()
"""


def page(client: ApiClient, fn: str, **state) -> AppTest:
    at = AppTest.from_string(PAGE.format(fn=fn), default_timeout=120)
    at.session_state["api"] = client
    at.session_state["me"] = client.me()
    at.session_state["status"] = client.status()
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success]
    parts += [str(c.value) for c in at.code]
    return "\n".join(str(p) for p in parts)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


# ---- application shell ---------------------------------------------------------------------------------------
def test_sign_in_screen_and_failed_sign_in(monkeypatch):
    monkeypatch.delenv("CB_UI_API_TOKEN", raising=False)
    monkeypatch.setenv("CB_UI_API_URL", "http://127.0.0.1:9")  # nothing listens there
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Cohort Builder Agents"
    at.text_input[1].set_value("cbk_not_a_real_token")
    button(at, "Sign in").click().run()
    err = " ".join(e.value for e in at.error)
    assert "Sign-in failed" in err and "cannot reach the API" in err
    assert "cbk_not_a_real_token" not in texts(at)  # the token is never echoed


def test_signed_in_shell_shows_identity_and_dashboard(env):
    client, tokens = env
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["api"] = client("rita")
    at.session_state["me"] = client("rita").me()
    at.session_state["status"] = client("rita").status()
    at.run()
    assert not at.exception
    html = "\n".join(m.value for m in at.markdown)
    assert "rita" in html and "reviewer" in html and "Synthetic demo data" in html
    assert [m.label for m in at.metric][:2] == ["Cohort definitions", "Drafts awaiting review"]
    assert tokens["rita"] not in texts(at) and tokens["rita"] not in repr(client("rita"))


def test_dashboard_without_privileges_does_not_invent_numbers(env):
    client, _ = env
    at = page(client("vic"), "page_dashboard")
    assert not at.exception
    assert at.metric[0].value == "0"
    assert "Agent runs" not in texts(at)  # viewers cannot list runs: the panel is absent, not zero


# ---- AI cohort builder ----------------------------------------------------------------------------------------
def test_multi_turn_chat_creates_refines_and_explains(env):
    client, _ = env
    at = page(client("alice"), "page_builder")
    assert not at.exception
    at.chat_input[0].set_value(T2DM_QUERY).run()
    w = at.session_state["ws"]
    first = w["draft_id"]
    assert first is not None and w["chat"][-1]["kind"] == "result"
    assert "Proposed definition (AI interpretation)" in texts(at) and f"#{first}" in texts(at)
    # refinement through the backend pipeline: new version, exact diff shown
    at.chat_input[0].set_value(REVISION).run()
    w = at.session_state["ws"]
    second = w["draft_id"]
    assert second != first and w["chat"][-1]["refine"] is True
    out = texts(at)
    assert f"What changed versus #{first}" in out and "inclusion[inc_2].value_filter.value: 8.0 -> 9.0" in out
    assert "inclusion[inc_2].window.start_days: -365 -> -180" in out
    # a question is answered from the stored definition, without the model
    at.chat_input[0].set_value("Why are patients with type 1 diabetes excluded?").run()
    last = at.session_state["ws"]["chat"][-1]
    assert last["kind"] == "answer" and last["answer"]["lines"][0].startswith("Exclusion criterion")
    assert at.session_state["ws"]["draft_id"] == second  # questions never create versions
    assert client("alice").cohort(second)["parent_definition_id"] == first


def test_chat_reports_model_failure_honestly(env):
    client, _ = env
    at = page(client("alice"), "page_builder")
    at.chat_input[0].set_value("A request the model was never scripted for").run()
    assert not at.exception
    assert at.session_state["ws"]["draft_id"] is None
    out = texts(at)
    assert "No definition created" in "\n".join(m.value for m in at.markdown) and "Proposed definition" not in out
    assert client("alice").list_cohorts() == []


def test_viewer_cannot_use_the_chat(env):
    client, _ = env
    at = page(client("vic"), "page_builder")
    assert at.chat_input[0].disabled


def _draft(client, user="alice") -> int:
    return client(user).submit(EXAMPLE_IR, None)["cohort_definition_id"]


def test_structured_edit_saves_a_child_version_and_refuses_stale_edits(env):
    client, _ = env
    def_id = _draft(client)
    at = page(client("alice"), "page_builder", ws={"chat": [], "draft_id": def_id, "mine": [def_id]})
    assert not at.exception
    crit = EXAMPLE_IR["inclusion"][1]["id"]
    at.number_input(key=f"{def_id}_{crit}_v").set_value(9.0)
    button(at, "Save as new version").click().run()
    assert not at.exception
    new_id = at.session_state["ws"]["draft_id"]
    assert new_id != def_id
    saved = client("alice").cohort(new_id)
    assert saved["parent_definition_id"] == def_id and saved["status"] == "draft"
    assert saved["ir"]["inclusion"][1]["value_filter"]["value"] == 9.0
    assert saved["ir"]["concept_sets"] == client("alice").cohort(def_id)["ir"]["concept_sets"]
    # someone else revises the same version: saving from the old one is refused, nothing is created
    other = client("bob").submit(EXAMPLE_IR | {"name": "bob's take"}, new_id)["cohort_definition_id"]
    at.session_state["ws"]["draft_id"] = new_id
    at.run()
    at.number_input(key=f"{new_id}_{crit}_v").set_value(10.0)
    before = len(client("alice").list_cohorts())
    button(at, "Save as new version").click().run()
    assert any(f"Version #{other} was already created" in e.value for e in at.error)
    assert len(client("alice").list_cohorts()) == before


def test_editor_validation_shows_server_results_without_saving(env):
    client, _ = env
    def_id = _draft(client)
    at = page(client("alice"), "page_builder", ws={"chat": [], "draft_id": def_id, "mine": [def_id]})
    crit = EXAMPLE_IR["inclusion"][1]["id"]
    at.number_input(key=f"{def_id}_{crit}_v").set_value(9.0)
    before = len(client("alice").list_cohorts())
    button(at, "Validate changes (dry run, nothing saved)").click().run()
    assert not at.exception
    assert len(client("alice").list_cohorts()) == before
    assert any("Criterion name not updated" in w.value for w in at.warning)
    assert at.dataframe  # the server's attrition table


# ---- validation, review, execution -------------------------------------------------------------------------
def test_validate_page_shows_attrition_from_the_server(env):
    client, _ = env
    def_id = _draft(client)
    at = page(client("alice"), "page_validate", selected_def=def_id)
    button(at, "Run validation (dry run)").click().run()
    assert not at.exception
    res = at.session_state[f"validation_{def_id}"]
    assert res["valid"] is True and res["attrition"]
    shown = at.dataframe[0].value
    assert list(shown["Patients remaining"]) == [str(r["remaining"]) for r in res["attrition"]]


def test_review_self_approval_rejection_comment_and_execution_gate(env, settings):
    client, _ = env
    def_id = _draft(client)
    # the executor cannot run a draft: the page says it is blocked and offers no button
    at = page(client("eddie"), "page_review", selected_def=def_id)
    assert any("Blocked" in m.value for m in at.markdown)
    assert not [b for b in at.button if b.label == "Execute"]
    with pytest.raises(ApiError) as denied:
        client("eddie").execute(def_id)
    assert denied.value.kind == "forbidden"
    # a reviewer must explain a rejection and confirm the exact version
    at = page(client("rita"), "page_review", selected_def=def_id)
    at.radio[1].set_value("rejected")
    button(at, "Submit decision").click().run()
    assert any("Confirm that you reviewed" in e.value for e in at.error)
    at.checkbox[0].check()
    button(at, "Submit decision").click().run()
    assert any("Explain the rejection" in e.value for e in at.error)
    assert client("rita").cohort(def_id)["status"] == "draft"  # nothing recorded
    at.radio[1].set_value("approved")
    at.checkbox[0].check()
    button(at, "Submit decision").click().run()
    assert any("Recorded by the server: approved by rita" in s.value for s in at.success)
    # now the executor can run it, through the API's own gate
    at = page(client("eddie"), "page_review", selected_def=def_id)
    assert button(at, "Execute").disabled  # until the confirmation is ticked
    at.checkbox[0].check().run()
    button(at, "Execute").click().run()
    assert not at.exception and any("Executed: generation" in s.value for s in at.success)


def test_self_approval_is_flagged_and_refused_by_the_server(env):
    client, _ = env
    def_id = _draft(client, "sam")  # sam writes AND reviews
    at = page(client("sam"), "page_review", selected_def=def_id)
    assert any("self-approval is disabled" in w.value for w in at.warning)
    at.checkbox[0].check()
    button(at, "Submit decision").click().run()
    assert any("permission denied" in e.value and "self-approval" in e.value for e in at.error)
    assert client("sam").cohort(def_id)["status"] == "draft"
    assert not [s for s in at.success if "Recorded by the server" in s.value]


# ---- explorer, runs, comparison, datasets, health ----------------------------------------------------------
def test_explorer_shows_compiled_sql_and_concepts(env):
    client, _ = env
    def_id = _draft(client)
    at = page(client("vic"), "page_explorer", selected_def=def_id)
    assert not at.exception
    assert at.code and "SELECT" in at.code[0].value.upper()
    assert any(dl.label == "Download SQL" for dl in at.get("download_button"))


def test_runs_page_and_replay(env):
    client, _ = env
    first = client("alice").ask(T2DM_QUERY)
    at = page(client("alice"), "page_runs")
    assert not at.exception and first["run_id"] in str(at.dataframe[0].value["Run"].tolist())
    button(at, "Replay from recorded LLM responses").click().run()
    assert any("identical semantic hash" in s.value for s in at.success)
    at = page(client("eddie"), "page_runs")
    assert any("Run traces are visible" in i.value for i in at.info)


def test_comparison_never_estimates_overlap(env):
    client, _ = env
    a = _draft(client)
    b = client("alice").submit(apply_edits(EXAMPLE_IR, {"demographics": {"age_min": 40}}), a)["cohort_definition_id"]
    at = page(client("rita"), "page_compare")
    at.multiselect[0].set_value([a, b]).run()
    assert not at.exception
    out = texts(at)
    assert "demographics.age_min: 18 -> 40" in out
    assert "Fewer than two of these versions have been executed" in out


def test_health_page_runs_checks_and_leaks_no_secrets(env, monkeypatch):
    client, tokens = env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    at = page(client("rita"), "page_health")
    assert not at.exception
    rendered = texts(at) + "".join(str(d.value) for d in at.dataframe) + "".join(str(j.value) for j in at.json)
    assert "sk-ant-test-not-a-real-key" not in rendered and tokens["rita"] not in rendered
    assert at.dataframe[0].value["Check"].tolist() == ["database", "vocabulary", "dataset", "llm"]


def test_datasets_page_respects_roles(env):
    client, _ = env
    at = page(client("vic"), "page_datasets")
    assert not at.exception
    out = texts(at)
    assert "Profiling needs the author role" in out and "Imports need the admin role" in out


def test_client_maps_errors_and_hides_the_token(env):
    client, tokens = env
    with pytest.raises(ApiError) as e:
        ApiClient("http://testserver", "wrong", session=client("vic")._http).me()
    assert e.value.kind == "unauthorized" and "wrong" not in str(e.value)
    with pytest.raises(ApiError) as e:
        client("vic").ask("x")
    assert e.value.kind == "forbidden"
    with pytest.raises(ApiError) as e:
        client("vic").cohort(999999)
    assert e.value.kind == "not_found"
    with pytest.raises(ApiError) as e:
        client("alice").submit({"name": "not a definition"}, None)
    assert e.value.kind == "invalid"
    with pytest.raises(ApiError) as e:
        ApiClient("http://127.0.0.1:9", "t").me()
    assert e.value.kind == "unavailable"
    assert tokens["vic"] not in repr(client("vic"))
