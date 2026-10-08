"""Intent parser (LLM): natural language -> structured cohort intent, no concept IDs yet."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..ir import ClaimStatus, Entity
from . import AgentContext


class ValueSpec(BaseModel):
    op: Literal[">", ">=", "<", "<=", "=", "between"]
    value: float
    value_high: float | None = None
    unit_text: str = Field(description="Unit exactly as implied by the request, e.g. '%', 'mmol/mol', 'mg/dL'")


class ConceptMention(BaseModel):
    key: str = Field(description="Short snake_case key, e.g. 't2dm', 'metformin', 'hba1c'")
    text: str = Field(description="The clinical idea as the user expressed it")
    entity: Entity
    notes: str | None = Field(default=None, description="Granularity hints, e.g. 'drug class', 'ingredient'")


class IntentCriterion(BaseModel):
    role: Literal["inclusion", "exclusion"]
    name: str = Field(description="Short human-readable rule name")
    mention_key: str
    window_start_days: int | None = Field(description="Days relative to index (negative=before); null=unbounded")
    window_end_days: int | None = Field(description="Days relative to index; null=unbounded")
    occurrence: Literal["at_least", "at_most", "exactly"] = "at_least"
    count: int = 1
    value: ValueSpec | None = None
    claim_status: list[ClaimStatus] | None = Field(
        default=None, description="Claims data, drug criteria only: which adjudication outcomes count. "
                                  "Null = dataset default (normally paid claims)")
    dx_position: Literal["primary", "any"] | None = Field(
        default=None, description="Claims data, diagnosis criteria only: 'primary' if the user asks for a "
                                  "primary / principal diagnosis")
    min_span_days: int | None = Field(
        default=None, description="With at_least N (N >= 2): first and last qualifying events must be at least "
                                  "this many days apart, e.g. '2 claims at least 30 days apart'")
    count_by: Literal["records", "dates"] = Field(
        default="records", description="'dates' when the user counts visits/days ('on 2 different days'); "
                                       "otherwise 'records'")


class CohortIntent(BaseModel):
    name: str
    description: str
    mentions: list[ConceptMention]
    index_mention_key: str = Field(description="Mention whose event defines the index date")
    index_first_occurrence_only: bool = True
    index_value: ValueSpec | None = None
    index_claim_status: list[ClaimStatus] | None = Field(
        default=None, description="Claims data, drug index only. Null = dataset default (normally paid)")
    index_dx_position: Literal["primary", "any"] | None = None
    prior_observation_days: int = 365
    post_observation_days: int = 0
    age_min: int | None = None
    age_max: int | None = None
    genders: list[Literal["male", "female"]] = Field(default_factory=list)
    criteria: list[IntentCriterion] = Field(default_factory=list)
    exit_type: Literal["end_of_observation", "fixed_days"] = "end_of_observation"
    exit_days: int | None = None
    assumptions: list[str] = Field(default_factory=list,
                                   description="Every interpretation choice not stated explicitly by the user")


def parse_intent(ctx: AgentContext, query: str, feedback: str | None = None) -> CohortIntent:
    prompt = ctx.prompt("intent_parser")
    system = prompt.render(ontology=ctx.ontology.summary_for_prompt())
    user = f"<request>\n{query.strip()}\n</request>"
    if feedback:
        user += f"\n\n<reviewer_feedback>\nA previous attempt had these problems; fix them:\n{feedback}\n</reviewer_feedback>"
    return ctx.llm.structured(
        prompt=prompt, system=system, user=user, schema=CohortIntent, tool_name="submit_cohort_intent",
        tool_description="Submit the structured cohort intent.", run_id=ctx.run_id, step_id=ctx.step_id)
