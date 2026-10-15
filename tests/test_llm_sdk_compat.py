"""The Anthropic SDK removed `temperature` from messages.create in newer releases (a fresh install failed
every call with TypeError). The client omits it when the installed SDK does not accept it, records what was
actually sent, and keeps request hashes consistent for replay."""

from __future__ import annotations

import pytest

from cohort_builder import llm as llm_mod
from cohort_builder.llm import LLMClient, sdk_accepts_temperature


def test_detection_matches_the_installed_sdk():
    import inspect

    from anthropic.resources.messages import Messages

    assert sdk_accepts_temperature() == ("temperature" in inspect.signature(Messages.create).parameters)


@pytest.mark.parametrize("accepts", [True, False])
def test_real_sdk_backend_only_gets_temperature_when_supported(builder, monkeypatch, accepts):
    sent: list[dict] = []

    def backend(request: dict) -> dict:
        if "temperature" in request and not accepts:
            raise TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        sent.append(request)
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}

    backend.is_anthropic_sdk = True  # behaves like anthropic_backend()
    monkeypatch.setattr(llm_mod, "sdk_accepts_temperature", lambda: accepts)
    client = LLMClient(builder.settings, builder.store, backend)
    assert builder.settings.temperature == 0.0
    prompt = llm_mod.Prompt("t", "v1", "x")
    client.create(prompt=prompt, system="s", messages=[{"role": "user", "content": "hi"}], tools=[], tool_choice={})
    assert ("temperature" in sent[0]) is accepts
    recorded = builder.con.execute("SELECT temperature FROM meta.llm_call ORDER BY 1 NULLS FIRST LIMIT 1").fetchone()
    assert recorded[0] == (0.0 if accepts else None)  # provenance says what was really sent


def test_injected_test_backends_keep_the_configured_temperature(builder, monkeypatch):
    monkeypatch.setattr(llm_mod, "sdk_accepts_temperature", lambda: False)
    client = LLMClient(builder.settings, builder.store, lambda r: r)
    assert client.effective_temperature() == 0.0


class Rejected(Exception):
    pass


def test_models_rejecting_forced_tool_choice_fall_back_to_auto(builder, monkeypatch):
    """Some models answer 400 'tool_choice: type "tool" and "any" are not supported for this model'. The
    client retries with tool_choice auto plus a system-prompt obligation, remembers the model, and asks again
    if the model replies in text instead of calling the tool."""
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    monkeypatch.setattr(LLMClient, "_auto_tool_models", set())
    sent: list[dict] = []
    replies = iter(
        [
            {"content": [{"type": "text", "text": "thinking aloud"}], "stop_reason": "end_turn", "usage": {}},
            {"content": [{"type": "tool_use", "id": "t1", "name": "emit", "input": {"n": 3}}], "usage": {}},
            {"content": [{"type": "tool_use", "id": "t2", "name": "emit", "input": {"n": 4}}], "usage": {}},
        ]
    )

    def backend(request: dict) -> dict:
        sent.append(request)
        if request["tool_choice"]["type"] in ("tool", "any"):
            raise Rejected(
                "Error code: 400 - {'type': 'invalid_request_error', 'message': 'tool_choice: type \"tool\" and "
                '"any" are not supported for this model.\'}'
            )
        return next(replies)

    client = LLMClient(builder.settings, builder.store, backend)
    prompt = llm_mod.Prompt("t", "v1", "x")
    kw = dict(prompt=prompt, system="sys", user="u", schema=Out, tool_name="emit", tool_description="d")
    assert client.structured(**kw).n == 3
    assert [r["tool_choice"]["type"] for r in sent] == ["tool", "auto", "auto"]
    assert "`emit` tool" in sent[1]["system"] and sent[2]["messages"][-1]["content"].startswith("Call the `emit`")
    assert client.structured(**kw).n == 3 and len(sent) == 3  # the same request is answered from the cache
    # remembered: a new request to the same model goes straight to auto
    assert client.structured(**{**kw, "user": "u2"}).n == 4
    assert sent[3]["tool_choice"] == {"type": "auto"} and len(sent) == 4


def test_other_bad_requests_are_not_retried(builder, monkeypatch):
    monkeypatch.setattr(LLMClient, "_auto_tool_models", set())
    calls: list[dict] = []

    def backend(request: dict) -> dict:
        calls.append(request)
        raise Rejected("Error code: 400 - prompt is too long")

    client = LLMClient(builder.settings, builder.store, backend)
    with pytest.raises(llm_mod.LLMError, match="too long"):
        client.create(
            prompt=llm_mod.Prompt("t", "v1", "x"),
            system="s",
            messages=[],
            tools=[],
            tool_choice={"type": "tool", "name": "x"},
        )
    assert len(calls) == 1
