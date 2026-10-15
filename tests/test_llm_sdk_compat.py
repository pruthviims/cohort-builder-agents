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
