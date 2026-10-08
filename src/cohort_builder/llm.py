"""LLM client with versioned prompts, request hashing, caching and replay.

Every request is canonicalized and hashed. In `cached` mode an identical
request returns the recorded response; in `replay` mode only recorded
responses are allowed, which reproduces a past run exactly.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings
from .metadata import MetadataStore, sha

T = TypeVar("T", bound=BaseModel)
Backend = Callable[[dict], dict]


class LLMError(RuntimeError):
    pass


class ReplayMiss(LLMError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def hash(self) -> str:
        return sha(self.text)

    @classmethod
    def load(cls, prompts_dir: Path, name: str, version: str) -> "Prompt":
        return cls(name, version, (Path(prompts_dir) / name / f"{version}.md").read_text())

    def render(self, **values: str) -> str:
        text = self.text
        for k, v in values.items():
            text = text.replace("{{" + k + "}}", v)
        return text


def inline_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined (portable tool input_schema)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def anthropic_backend() -> Backend:
    import os

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set (or use CB_LLM_MODE=replay with recorded responses)")
    client = anthropic.Anthropic(max_retries=4)

    def call(request: dict) -> dict:
        resp = client.messages.create(**request)
        return {
            "content": [b.model_dump(exclude_none=True) for b in resp.content],
            "stop_reason": resp.stop_reason,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            "model": resp.model,
        }

    return call


class LLMClient:
    def __init__(self, settings: Settings, store: MetadataStore, backend: Backend | None = None):
        self.settings = settings
        self.store = store
        self._backend = backend

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = anthropic_backend()
        return self._backend

    def create(self, *, prompt: Prompt, system: str, messages: list[dict], tools: list[dict],
               tool_choice: dict, run_id: str | None = None, step_id: str | None = None) -> dict:
        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if self.settings.temperature is not None:
            request["temperature"] = self.settings.temperature
        request_hash = sha(request)
        mode = self.settings.llm_mode
        response = self.store.cache_get(request_hash) if mode in ("cached", "replay") else None
        cache_hit = response is not None
        if response is None:
            if mode == "replay":
                raise ReplayMiss(f"no recorded response for request {request_hash} ({prompt.name})")
            try:
                response = self.backend(request)
            except LLMError:
                raise
            except Exception as exc:  # network, auth, rate limit, bad request
                raise LLMError(f"LLM call failed ({prompt.name}): {type(exc).__name__}: {exc}") from exc
            self.store.cache_put(request_hash, self.settings.model, response)
        usage = response.get("usage", {})
        self.store.record_llm_call(
            run_id=run_id, step_id=step_id, model=self.settings.model, temperature=self.settings.temperature,
            prompt_name=prompt.name, prompt_version=prompt.version, prompt_hash=prompt.hash,
            request_hash=request_hash, response=response, input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"), cache_hit=cache_hit)
        return response

    def structured(self, *, prompt: Prompt, system: str, user: str, schema: type[T], tool_name: str,
                   tool_description: str, run_id: str | None = None, step_id: str | None = None,
                   max_repairs: int = 2) -> T:
        """Force a single tool call whose input must validate against `schema`."""
        tool = {"name": tool_name, "description": tool_description, "input_schema": inline_schema(schema)}
        messages: list[dict] = [{"role": "user", "content": user}]
        for attempt in range(max_repairs + 1):
            resp = self.create(prompt=prompt, system=system, messages=messages, tools=[tool],
                               tool_choice={"type": "tool", "name": tool_name}, run_id=run_id, step_id=step_id)
            block = next((b for b in resp["content"] if b.get("type") == "tool_use"), None)
            if block is None:
                raise LLMError(f"{prompt.name}: model did not call {tool_name}")
            try:
                return schema.model_validate(block["input"])
            except ValidationError as exc:
                if attempt == max_repairs:
                    raise LLMError(f"{prompt.name}: invalid structured output: {exc}") from exc
                messages = messages + [
                    {"role": "assistant", "content": resp["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"],
                                                  "is_error": True,
                                                  "content": f"Schema validation failed, fix and call again:\n{exc}"}]},
                ]
        raise AssertionError("unreachable")


def tool_result_text(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
