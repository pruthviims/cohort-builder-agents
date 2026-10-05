"""A scripted stand-in for the Anthropic API so tests run offline and deterministically.

It speaks the same request/response format as `llm.anthropic_backend`. The
concept resolver half behaves like a simple agent (curated search -> vocabulary
search -> submit), driven entirely by real tool results.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


def _tool_use(name: str, inp: dict, n: int) -> dict:
    return {"content": [{"type": "tool_use", "id": f"toolu_{n:04d}", "name": name, "input": inp}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 20}}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") if b.get("type") == "text" else str(b.get("content", "")) for b in content)


class FakeLLM:
    def __init__(self, intents: dict[str, list[dict]],
                 critic: Callable[[str, int], dict] | None = None,
                 resolver_scripts: dict[str, list[tuple]] | None = None):
        self.intents = intents
        self.critic = critic or (lambda user, n: {"verdict": "pass", "issues": [], "notes": ""})
        self.resolver_scripts = resolver_scripts or {}
        self.calls = 0
        self.critic_calls = 0
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> dict:
        self.calls += 1
        self.requests.append(request)
        names = {t["name"] for t in request["tools"]}
        if "submit_cohort_intent" in names:
            return self._intent(request)
        if "submit_review" in names:
            self.critic_calls += 1
            return _tool_use("submit_review", self.critic(_text(request["messages"][0]["content"]),
                                                          self.critic_calls), self.calls)
        if "submit_concept_set" in names:
            return self._resolver(request)
        raise AssertionError(f"unexpected request with tools {names}")

    def _intent(self, request: dict) -> dict:
        user = _text(request["messages"][0]["content"])
        query = re.search(r"<request>\n(.*?)\n</request>", user, re.S).group(1)
        options = self.intents[query]
        attempt = 1 if "<reviewer_feedback>" in user else 0
        return _tool_use("submit_cohort_intent", options[min(attempt, len(options) - 1)], self.calls)

    def _resolver(self, request: dict) -> dict:
        msgs = request["messages"]
        first = _text(msgs[0]["content"])
        m = re.search(r'<mention key="([^"]+)"[^>]*>(.*?)</mention>', first, re.S)
        key, text = m.group(1), m.group(2)
        n_assistant = sum(1 for x in msgs if x["role"] == "assistant")
        script = self.resolver_scripts.get(key, [])
        if n_assistant < len(script):
            action, arg = script[n_assistant]
            if action == "submit_ids":
                return _tool_use("submit_concept_set", {"name": text, "items": [
                    {"concept_id": i, "include_descendants": True} for i in arg], "rationale": "scripted"}, self.calls)
            return _tool_use(action, arg, self.calls)
        if n_assistant == 0:
            return _tool_use("search_curated_concept_sets", {"query": text}, self.calls)
        last_tool = msgs[-2]["content"][0]["name"]
        last_result = msgs[-1]["content"][0]
        if last_tool == "search_curated_concept_sets":
            found = json.loads(last_result["content"])
            if found and found[0]["score"] >= 0.85:
                return _tool_use("submit_concept_set", {"name": found[0]["label"],
                                                        "curated_key": found[0]["curated_key"],
                                                        "rationale": "approved curated set"}, self.calls)
            return _tool_use("search_concepts", {"query": text}, self.calls)
        if last_tool == "search_concepts":
            found = json.loads(last_result["content"])
            top = found[0]
            return _tool_use("submit_concept_set", {
                "name": top["concept_name"],
                "items": [{"concept_id": top["concept_id"], "include_descendants": True}],
                "rationale": "best lexical match"}, self.calls)
        # after a rejected submission or anything else: search again
        return _tool_use("search_concepts", {"query": text}, self.calls)


# ---- intents matching the golden cases ----------------------------------------
def _crit(role, name, key, s, e, value=None):
    return {"role": role, "name": name, "mention_key": key, "window_start_days": s, "window_end_days": e,
            "occurrence": "at_least", "count": 1, "value": value}


T2DM_QUERY = ("Adults with type 2 diabetes who started metformin and had an HbA1c above 8% in the year before "
              "starting. Exclude anyone with type 1 diabetes before starting.")

T2DM_INTENT = {
    "name": "T2DM new metformin users with HbA1c > 8%",
    "description": "Adults with T2DM starting metformin with prior HbA1c > 8%.",
    "mentions": [
        {"key": "metformin", "text": "metformin", "entity": "DrugExposure", "notes": "ingredient"},
        {"key": "t2dm", "text": "type 2 diabetes", "entity": "ConditionOccurrence"},
        {"key": "hba1c", "text": "HbA1c", "entity": "Measurement"},
        {"key": "t1dm", "text": "type 1 diabetes", "entity": "ConditionOccurrence"},
    ],
    "index_mention_key": "metformin", "index_first_occurrence_only": True, "prior_observation_days": 365,
    "age_min": 18,
    "criteria": [
        _crit("inclusion", "Type 2 diabetes before index", "t2dm", None, 0),
        _crit("inclusion", "HbA1c > 8% in prior year", "hba1c", -365, 0,
              {"op": ">", "value": 8, "unit_text": "%"}),
        _crit("exclusion", "Type 1 diabetes before index", "t1dm", None, 0),
    ],
    "assumptions": ["'started metformin' = first-ever metformin exposure"],
}

HF_QUERY = ("Patients with heart failure who started an SGLT2 inhibitor, with an ejection fraction below 40% in "
            "the year before starting.")
HF_INTENT = {
    "name": "HF patients starting SGLT2i with LVEF < 40%", "description": "",
    "mentions": [
        {"key": "sglt2", "text": "SGLT2 inhibitor", "entity": "DrugExposure", "notes": "drug class"},
        {"key": "hf", "text": "heart failure", "entity": "ConditionOccurrence"},
        {"key": "lvef", "text": "ejection fraction", "entity": "Measurement"},
    ],
    "index_mention_key": "sglt2", "prior_observation_days": 365,
    "criteria": [_crit("inclusion", "Heart failure before index", "hf", None, 0),
                 _crit("inclusion", "LVEF < 40% in prior year", "lvef", -365, 0,
                       {"op": "<", "value": 40, "unit_text": "%"})],
}

CKD_QUERY = ("Adults newly diagnosed with chronic kidney disease who have an eGFR below 60 within 90 days after "
             "the diagnosis.")
CKD_INTENT = {
    "name": "Incident CKD with eGFR < 60 within 90 days", "description": "",
    "mentions": [{"key": "ckd", "text": "chronic kidney disease", "entity": "ConditionOccurrence"},
                 {"key": "egfr", "text": "eGFR", "entity": "Measurement"}],
    "index_mention_key": "ckd", "prior_observation_days": 365, "age_min": 18,
    "criteria": [_crit("inclusion", "eGFR < 60 within 90 days", "egfr", 0, 90,
                       {"op": "<", "value": 60, "unit_text": "mL/min/1.73m2"})],
}

SERT_QUERY = "New users of sertraline with a depression diagnosis in the 30 days before their first prescription."
SERT_INTENT = {
    "name": "New sertraline users with recent depression", "description": "",
    "mentions": [{"key": "sertraline", "text": "sertraline", "entity": "DrugExposure"},
                 {"key": "depression", "text": "depression", "entity": "ConditionOccurrence"}],
    "index_mention_key": "sertraline", "prior_observation_days": 365,
    "criteria": [_crit("inclusion", "Depression in prior 30 days", "depression", -30, 0)],
}

GOLDEN_INTENTS = {T2DM_QUERY: [T2DM_INTENT], HF_QUERY: [HF_INTENT], CKD_QUERY: [CKD_INTENT],
                  SERT_QUERY: [SERT_INTENT]}
