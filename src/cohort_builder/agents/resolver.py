"""Concept resolver (LLM + vocabulary tools).

Grounding rule enforced in code: the agent may only submit concept IDs that
appeared in its own tool results during this resolution, so it cannot invent
IDs. Curated concept sets, when chosen, are copied verbatim from the ontology.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..ir import ConceptSetItem
from ..llm import LLMError, tool_result_text
from . import AgentContext
from .intent import ConceptMention


class ResolvedConceptSet(BaseModel):
    mention_key: str
    name: str
    domain: str
    source: str
    items: list[ConceptSetItem]
    rationale: str = ""


class SubmitConceptSet(BaseModel):
    name: str = Field(description="Human-readable concept set name")
    curated_key: str | None = Field(default=None, description="Key of an approved curated set to reuse verbatim")
    items: list[ConceptSetItem] = Field(
        default_factory=list, description="Required unless curated_key is set. IDs must come from tool results"
    )
    rationale: str = Field(description="One or two sentences on why these concepts match the mention")


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_curated_concept_sets",
        "description": "Search organization-approved concept sets. Always try this first.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "search_concepts",
        "description": (
            "Search the standardized vocabulary by name, synonym or code. Returns standard concepts by default."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "include_non_standard": {"type": "boolean", "default": False}},
            "required": ["query"],
        },
    },
    {
        "name": "get_concept",
        "description": "Details for a concept: parents, number of descendants, and Maps-to targets if non-standard.",
        "input_schema": {
            "type": "object",
            "properties": {"concept_id": {"type": "integer"}},
            "required": ["concept_id"],
        },
    },
    {
        "name": "get_descendants",
        "description": "List descendants of a concept (what include_descendants would cover).",
        "input_schema": {
            "type": "object",
            "properties": {"concept_id": {"type": "integer"}, "limit": {"type": "integer", "default": 25}},
            "required": ["concept_id"],
        },
    },
    {
        "name": "lookup_code",
        "description": "Find a concept by source code (e.g. ICD-10 'E11.9', LOINC '4548-4') and its standard mapping.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}, "vocabulary_id": {"type": "string"}},
            "required": ["code"],
        },
    },
]


def _collect_ids(obj: Any, out: set[int]) -> None:
    if isinstance(obj, dict):
        if "concept_id" in obj and isinstance(obj["concept_id"], int):
            out.add(obj["concept_id"])
        for v in obj.values():
            _collect_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_ids(v, out)


class _Session:
    def __init__(self, ctx: AgentContext, domain: str):
        self.ctx = ctx
        self.domain = domain
        self.seen: set[int] = set()

    def run_tool(self, name: str, args: dict) -> Any:
        v, ont = self.ctx.vocab, self.ctx.ontology
        result: Any
        if name == "search_curated_concept_sets":
            result = ont.search_curated(args["query"], domain=self.domain)
        elif name == "search_concepts":
            result = v.search_concepts(
                args["query"], domain=self.domain, standard_only=not args.get("include_non_standard", False)
            )
        elif name == "get_concept":
            result = v.get_concept(int(args["concept_id"])) or {"error": "concept not found"}
        elif name == "get_descendants":
            result = v.get_descendants(int(args["concept_id"]), int(args.get("limit", 25)))
        elif name == "lookup_code":
            result = v.lookup_code(args["code"], args.get("vocabulary_id"))
        else:
            result = {"error": f"unknown tool {name}"}
        _collect_ids(result, self.seen)
        self.ctx.store.record_tool_call(self.ctx.run_id, self.ctx.step_id, name, args, result)
        return result

    def check_submission(self, sub: SubmitConceptSet) -> tuple[ResolvedConceptSet | None, str | None]:
        ont, v = self.ctx.ontology, self.ctx.vocab
        if sub.curated_key:
            cs = ont.curated.get(sub.curated_key)
            if not cs or cs.get("status") != "approved":
                return None, f"curated_key {sub.curated_key!r} is not an approved curated set"
            if cs["domain"] != self.domain:
                return None, f"curated set domain {cs['domain']} != required domain {self.domain}"
            items = [ConceptSetItem(**i) for i in cs["items"]]
            return ResolvedConceptSet(
                mention_key="",
                name=cs["label"],
                domain=self.domain,
                source=f"curated:{sub.curated_key}@v{cs['version']}",
                items=items,
                rationale=sub.rationale,
            ), None
        if not sub.items:
            return None, "items is empty; provide concept IDs or a curated_key"
        problems = []
        info = v.concepts([i.concept_id for i in sub.items])
        for it in sub.items:
            c = info.get(it.concept_id)
            if it.concept_id not in self.seen:
                problems.append(f"{it.concept_id}: not returned by any tool call in this session (do not guess IDs)")
            elif c is None:
                problems.append(f"{it.concept_id}: does not exist")
            elif c["invalid_reason"]:
                problems.append(f"{it.concept_id}: deprecated; use its Maps-to target")
            elif c["standard_concept"] not in ("S", "C"):
                problems.append(f"{it.concept_id}: non-standard; use the standard concept it maps to")
            elif c["domain_id"] != self.domain:
                problems.append(f"{it.concept_id}: domain {c['domain_id']} but {self.domain} is required")
            elif c["standard_concept"] == "C" and not it.include_descendants:
                problems.append(f"{it.concept_id}: classification concept must use include_descendants=true")
        if problems:
            return None, "Submission rejected:\n- " + "\n- ".join(problems)
        items = sorted(sub.items, key=lambda i: (i.concept_id, i.is_excluded))
        return ResolvedConceptSet(
            mention_key="", name=sub.name, domain=self.domain, source="resolved", items=items, rationale=sub.rationale
        ), None


def resolve_mention(
    ctx: AgentContext, mention: ConceptMention, query: str, feedback: str | None = None
) -> ResolvedConceptSet:
    domain = ctx.ontology.entity_domain(mention.entity)
    prompt = ctx.prompt("concept_resolver")
    system = prompt.render(ontology=ctx.ontology.summary_for_prompt())
    submit_tool = {
        "name": "submit_concept_set",
        "description": "Submit the final concept set for this mention. Ends the task.",
        "input_schema": _submit_schema(),
    }
    tools = TOOLS + [submit_tool]
    user = (
        f"<cohort_request>{query.strip()}</cohort_request>\n"
        f'<mention key="{mention.key}" entity="{mention.entity}" domain="{domain}">'
        f"{mention.text}</mention>"
    )
    if mention.notes:
        user += f"\n<notes>{mention.notes}</notes>"
    if feedback:
        user += f"\n<reviewer_feedback>\n{feedback}\n</reviewer_feedback>"
    messages: list[dict] = [{"role": "user", "content": user}]
    session = _Session(ctx, domain)

    for _turn in range(ctx.settings.max_resolver_turns):
        resp = ctx.llm.create(
            prompt=prompt,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice={"type": "any"},
            run_id=ctx.run_id,
            step_id=ctx.step_id,
        )
        messages.append({"role": "assistant", "content": resp["content"]})
        results = []
        for block in resp["content"]:
            if block.get("type") != "tool_use":
                continue
            if block["name"] == "submit_concept_set":
                try:
                    resolved, err = session.check_submission(SubmitConceptSet.model_validate(block["input"]))
                except ValidationError as exc:
                    resolved, err = None, f"Invalid submission: {exc}"
                if resolved:
                    resolved.mention_key = mention.key
                    return resolved
                results.append({"type": "tool_result", "tool_use_id": block["id"], "is_error": True, "content": err})
            else:
                out = session.run_tool(block["name"], block["input"])
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": tool_result_text(out)})
        if not results:
            results = [{"type": "text", "text": "Call a tool, or submit_concept_set when done."}]
        messages.append({"role": "user", "content": results})
    raise LLMError(f"concept resolver exceeded {ctx.settings.max_resolver_turns} turns for {mention.key!r}")


def _submit_schema() -> dict:
    from ..llm import inline_schema

    return inline_schema(SubmitConceptSet)


def resolution_signature(mention: ConceptMention) -> str:
    return json.dumps([mention.key, mention.text.lower(), mention.entity, mention.notes], sort_keys=True)
