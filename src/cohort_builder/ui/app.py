"""Cohort Builder Agents — analytics workspace (Streamlit).

Run with `cohort-builder ui` (or `streamlit run src/cohort_builder/ui/app.py`). The workspace is a client of
the HTTP API (`cohort-builder serve`): it holds no business logic and no database connection. The API
address comes from CB_UI_API_URL (default http://127.0.0.1:8000); a token can be preset with
CB_UI_API_TOKEN or entered on the sign-in screen. The token stays in the server-side session only.
"""

from __future__ import annotations

import html
import os

import streamlit as st

from cohort_builder.ui import pages
from cohort_builder.ui.client import ApiClient, ApiError
from cohort_builder.ui.components import CSS, badge

st.set_page_config(page_title="Cohort Builder Agents", page_icon="🧬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def _sign_in(client: ApiClient) -> bool:
    try:
        st.session_state["me"] = client.me()
    except ApiError as exc:
        st.session_state.pop("me", None)
        st.error(f"Sign-in failed: {exc.message}")
        return False
    st.session_state["api"] = client
    try:
        st.session_state["status"] = client.status()
    except ApiError:
        st.session_state["status"] = None
    return True


def sign_in_screen() -> None:
    st.title("Cohort Builder Agents")
    st.caption("Sign in with an API token issued by your administrator (`cohort-builder auth issue-token`).")
    default_url = os.environ.get("CB_UI_API_URL", "http://127.0.0.1:8000")
    with st.form("signin"):
        url = st.text_input("API address", value=default_url)
        token = st.text_input("API token", type="password", help="Kept in this server-side session only.")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        if not token.strip():
            st.error("Enter a token.")
        elif _sign_in(ApiClient(url, token.strip())):
            st.rerun()
    with st.expander("Troubleshooting"):
        st.markdown(
            "- **Cannot reach the API**: start it with `cohort-builder serve` and check the address.\n"
            "- **Token unknown or expired**: issue a new one and make sure the API was started with "
            "`CB_AUTH_TOKENS_FILE` pointing at the file the entry was written to.\n"
            "- The token is never shown again after sign-in and is not stored in the browser."
        )


def ensure_client() -> bool:
    if "api" in st.session_state and st.session_state.get("me"):
        return True
    preset = os.environ.get("CB_UI_API_TOKEN")
    if preset and not st.session_state.get("preset_tried"):
        st.session_state["preset_tried"] = True
        if _sign_in(ApiClient(os.environ.get("CB_UI_API_URL", "http://127.0.0.1:8000"), preset)):
            return True
    return False


def top_bar() -> None:
    me = st.session_state["me"]
    status = st.session_state.get("status") or {}
    bits = ['<span class="cb-title">Cohort Builder Agents</span>']
    if status.get("data") == "synthetic":  # only when the server says so
        bits.append(badge("Synthetic demo data", "warn"))
    if status.get("dataset"):
        bits.append(badge(f"Dataset: {status['dataset']['name']}", "info"))
    who = (
        f'<span class="cb-sub">{html.escape(me["subject"])} · {html.escape(", ".join(me["roles"]))} · '
        f"tenant {html.escape(me['tenant'])}</span>"
    )
    pending = st.session_state.get("pending_reviews")
    if pending:
        who += " " + badge(f"{pending} awaiting review", "warn")
    st.markdown(f'<div class="cb-topbar"><div>{" ".join(bits)}</div><div>{who}</div></div>', unsafe_allow_html=True)


def sidebar_tools() -> None:
    with st.sidebar:
        q = st.text_input("Search cohorts", key="global_search", placeholder="name or #id")
        if q and q != st.session_state.get("last_global_search"):
            st.session_state["last_global_search"] = q
            st.session_state["library_query"] = q
            st.switch_page(pages.PAGES["library"])
        if st.button("Sign out"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


def main() -> None:
    if not ensure_client():
        sign_in_screen()
        return
    nav = st.navigation(pages.navigation(), position="sidebar")
    top_bar()
    sidebar_tools()
    nav.run()


main()
