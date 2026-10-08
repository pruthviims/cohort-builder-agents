"""Critic (LLM): checks that the compiled definition matches what the user asked for.

It sees the request and the deterministic explanation (never patient data),
and either passes the definition or sends targeted feedback back to the
intent parser / concept resolver.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..llm import tool_result_text
from . import AgentContext


class CriticIssue(BaseModel):
    stage: Literal["intent", "concepts"] = Field(
        description="'intent' for structure/time windows/thresholds, 'concepts' for wrong or missing codes"
    )
    message: str


class Critique(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[CriticIssue] = Field(default_factory=list)
    notes: str = Field(default="", description="Optional remarks for the human reviewer")


def critique(ctx: AgentContext, query: str, explanation: str, attrition: list[dict], warnings: list[dict]) -> Critique:
    prompt = ctx.prompt("critic")
    user = (
        f"<request>\n{query.strip()}\n</request>\n\n<definition>\n{explanation}\n</definition>\n\n"
        f"<attrition>{tool_result_text(attrition)}</attrition>\n<validator_warnings>"
        f"{tool_result_text(warnings)}</validator_warnings>"
    )
    return ctx.llm.structured(
        prompt=prompt,
        system=prompt.render(ontology=ctx.ontology.summary_for_prompt()),
        user=user,
        schema=Critique,
        tool_name="submit_review",
        tool_description="Submit your review of the definition.",
        run_id=ctx.run_id,
        step_id=ctx.step_id,
    )
