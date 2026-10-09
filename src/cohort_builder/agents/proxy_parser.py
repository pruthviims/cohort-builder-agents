"""Natural language -> proxy definition DRAFT.

The LLM (prompt `proxy_parser`) only produces a structured draft: evidence ideas as *mentions*
(no concept ids), windows, logic, tiers. Concept ids come from the grounded concept resolver, and
the deterministic composer below builds a typed `ProxyDefinition`, which is then validated like
any hand-written one. Nothing here compiles or executes SQL, and every generated algorithm is
classified `exploratory` regardless of what the model says: classification is a human decision.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..ontology import Ontology
from ..proxy import Category, ProxyDefinition, TemporalRule
from ..vocab import Vocabulary
from . import AgentContext
from .composer import Issue, format_validation_error
from .intent import ConceptMention
from .resolver import ResolvedConceptSet

RULE_HELP = (
    'Boolean rule as a JSON object with exactly one key: {"evidence": id} | {"group": name} | '
    '{"temporal": id} | {"all": [rules]} | {"any": [rules]} | {"not": rule} | '
    '{"at_least"|"at_most"|"exactly": {"n": N, "of": [rules], "within_days": optional}}'
)


class DraftEvidence(BaseModel):
    id: str = Field(description="snake_case id, unique, e.g. 'dx', 'pathology', 'chemo'")
    name: str
    category: Category
    mention_key: str = Field(description="mention whose concepts define this evidence")
    window_start_days: int | None = Field(description="days relative to index (negative = before); null=unbounded")
    window_end_days: int | None
    occurrence: Literal["at_least", "at_most", "exactly"] = "at_least"
    count: int = 1
    count_by: Literal["records", "dates"] = "records"
    min_span_days: int | None = None
    max_span_days: int | None = Field(default=None, description="'at least N within D days' -> count N, max_span D")
    required: bool = Field(default=True, description="false = run without it if the dataset cannot provide it")


class DraftNamedRule(BaseModel):
    name: str
    label: str = ""
    rule: dict[str, Any] = Field(description=RULE_HELP)


class DraftConflict(DraftNamedRule):
    action: Literal["flag", "exclude"] = "flag"


class DraftTier(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    rule: dict[str, Any] | None = Field(default=None, description=RULE_HELP)
    min_score: int | None = None


class DraftWeight(BaseModel):
    rule: dict[str, Any] = Field(description=RULE_HELP)
    points: int


class ProxyDraft(BaseModel):
    algorithm_name: str = Field(description="snake_case name, e.g. 'rare_disease_x_proxy'")
    target_name: str
    target_description: str = ""
    clinical_notes: str = ""
    mentions: list[ConceptMention]
    index_mention_key: str
    index_first_occurrence_only: bool = True
    prior_observation_days: int = 365
    post_observation_days: int = 0
    evidence: list[DraftEvidence]
    groups: list[DraftNamedRule] = Field(default_factory=list)
    temporal_rules: list[TemporalRule] = Field(default_factory=list)
    entry: dict[str, Any] | None = Field(default=None, description=RULE_HELP)
    exclusion: dict[str, Any] | None = Field(default=None, description=RULE_HELP)
    conflicts: list[DraftConflict] = Field(default_factory=list)
    score_weights: list[DraftWeight] = Field(default_factory=list)
    tiers: list[DraftTier]
    funnel: list[DraftNamedRule] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


def parse_proxy(ctx: AgentContext, query: str, feedback: str | None = None) -> ProxyDraft:
    prompt = ctx.prompt("proxy_parser")
    system = prompt.render(ontology=ctx.ontology.summary_for_prompt())
    user = f"<request>\n{query.strip()}\n</request>"
    if feedback:
        user += (
            f"\n\n<reviewer_feedback>\nA previous attempt had these problems; fix them:\n{feedback}\n"
            "</reviewer_feedback>"
        )
    return ctx.llm.structured(
        prompt=prompt,
        system=system,
        user=user,
        schema=ProxyDraft,
        tool_name="submit_proxy_definition",
        tool_description="Submit the structured proxy algorithm draft.",
        run_id=ctx.run_id,
        step_id=ctx.step_id,
    )


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    s = s if s and s[0].isalpha() else f"x_{s}"
    return s[:48] or "x"


def compose_proxy(
    draft: ProxyDraft, resolved: dict[str, ResolvedConceptSet], ont: Ontology, vocab: Vocabulary, version: str
) -> tuple[ProxyDefinition | None, list[Issue]]:
    issues: list[Issue] = []
    mentions = {m.key for m in draft.mentions}
    used = {draft.index_mention_key} | {e.mention_key for e in draft.evidence}
    for key in sorted(used - mentions):
        issues.append(Issue("error", "intent", f"mention_key {key!r} is not declared in mentions"))
    for key in sorted(used & mentions):
        if key not in resolved:
            issues.append(Issue("error", "concepts", f"mention {key!r} was not resolved"))
    if issues:
        return None, issues

    def cs_id(key: str) -> str:
        return "cs_" + slugify(key)

    data: dict[str, Any] = {
        "algorithm_name": slugify(draft.algorithm_name),
        "version": version,
        "classification": "exploratory",  # never chosen by the model
        "target": {"name": draft.target_name, "description": draft.target_description},
        "clinical_notes": draft.clinical_notes,
        "dataset_profile": ont.dataset_name,
        "ontology_version": ont.version,
        "vocabulary_version": vocab.version(),
        "concept_sets": [
            {
                "id": cs_id(k),
                "name": r.name,
                "domain": r.domain,
                "items": [i.model_dump() for i in r.items],
                "source": r.source,
            }
            for k, r in sorted(resolved.items())
            if k in used
        ],
        "index_event": {
            "entity": next(m.entity for m in draft.mentions if m.key == draft.index_mention_key),
            "concept_set_id": cs_id(draft.index_mention_key),
            "first_occurrence_only": draft.index_first_occurrence_only,
        },
        "prior_observation_days": draft.prior_observation_days,
        "post_observation_days": draft.post_observation_days,
        "evidence": [],
        "groups": {g.name: g.rule for g in draft.groups},
        "temporal_rules": [t.model_dump() for t in draft.temporal_rules],
        "entry": draft.entry,
        "exclusion": draft.exclusion,
        "conflicts": [c.model_dump() for c in draft.conflicts],
        "scoring": (
            {"weights": [{"ref": w.rule, "points": w.points} for w in draft.score_weights]}
            if draft.score_weights
            else None
        ),
        "tiers": [t.model_dump() for t in draft.tiers],
        "funnel": [{"name": f.name, "rule": f.rule} for f in draft.funnel],
        "assumptions": draft.assumptions
        + ["Generated from a natural-language request; classification set to 'exploratory' pending clinical review."],
    }
    entity = {m.key: m.entity for m in draft.mentions}
    for e in draft.evidence:
        data["evidence"].append(
            {
                "id": e.id,
                "name": e.name,
                "category": e.category,
                "entity": entity[e.mention_key],
                "concept_set_id": cs_id(e.mention_key),
                "window": {"start_days": e.window_start_days, "end_days": e.window_end_days},
                "occurrence": e.occurrence,
                "count": e.count,
                "count_by": e.count_by,
                "min_span_days": e.min_span_days,
                "max_span_days": e.max_span_days,
                "required": e.required,
            }
        )
    data = {k: v for k, v in data.items() if v is not None}
    try:
        return ProxyDefinition.model_validate(data), issues
    except ValidationError as exc:
        return None, [Issue("error", "intent", m) for m in format_validation_error(exc)]
