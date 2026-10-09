"""Proxy cohort workflow, mixed into `CohortBuilder` (orchestrator.py).

    natural language -> draft definition -> validation -> human review -> SQL compilation / preview
    -> approval -> execution -> suppressed evidence / attrition results

Every step goes through the same governance as ordinary cohorts: definitions are immutable
versions, only approved definitions execute (draft execution only with the development flag),
reviewers cannot approve their own work, and every action is audited. Aggregate outputs are
small-cell suppressed; patient-level output (per-patient evidence) needs an admin principal AND
the explicit `CB_ALLOW_PATIENT_LEVEL` policy flag.

Reference-standard metrics (sensitivity, specificity, PPV, NPV, F1) are computed only against an
externally supplied, labelled reference set, never from the proxy cohort itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from .agents.composer import Issue
from .agents.proxy_explainer import COHORT_NOTE, SCORE_NOTE, ProxyExplainer
from .agents.proxy_parser import ProxyDraft, compose_proxy, parse_proxy, slugify
from .agents.proxy_validator import validate_proxy
from .agents.resolver import ResolvedConceptSet, resolution_signature, resolve_mention
from .executor import (
    COMPLEMENTARY,
    ExecutionError,
    suppress_count,
    suppress_partition,
    suppress_series,
    suppress_with_total,
)
from .llm import LLMError
from .metadata import now
from .metrics import DEFINITIONS, rounded
from .proxy import CLASSIFICATION_LABELS, AcceptanceCriteria, Expr, ProxyDefinition, provenance_summary
from .proxy_compiler import CompiledProxy, ProxyCompiler
from .proxy_evaluation import (
    EligibilityRules,
    ReferenceRecord,
    assess_acceptance,
    classify,
    collapse_records,
    compute_metrics,
    eligibility_sql,
    normalize_label,
)
from .security import DEFAULT_TENANT, GovernanceError

if TYPE_CHECKING:  # attributes provided by CohortBuilder
    from .agents import AgentContext
    from .config import Settings
    from .executor import Executor
    from .metadata import MetadataStore
    from .ontology import Ontology
    from .security import GovernancePolicy
    from .vocab import Vocabulary

SUPPRESSION_NOTE = (
    "Counts from 1 to {k}-1 are shown as '<{k}'; 'suppressed' hides a count that would let a "
    "small group be derived by subtraction. This is a disclosure-risk control, not anonymization."
)


class VersionConflict(ValueError):
    """An algorithm version already exists with different content (versions are immutable)."""


@dataclass
class ProxyRunResult:
    run_id: str
    status: str  # draft | needs_review | failed
    cohort_definition_id: int | None = None
    ir: ProxyDefinition | None = None
    explanation: str = ""
    issues: list[dict] = field(default_factory=list)
    attrition: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["ir"] = self.ir.to_dict() if self.ir else None
        d["next_step"] = (
            "A human reviewer must review and approve this draft before it can be compiled for execution or executed."
            if self.status == "draft"
            else "Resolve the issues (edit the definition and submit it as a new version)."
        )
        return d


def eval_expr(e: Expr, facts: dict[str, bool], p: ProxyDefinition) -> bool | None:
    """Evaluate a rule from stored per-patient booleans (None when it needs event dates: within_days)."""
    k = e.kind
    if k == "evidence":
        return facts.get(f"ev_{e.evidence}")
    if k == "temporal":
        return facts.get(f"tr_{e.temporal}")
    if k == "group":
        return eval_expr(p.groups[e.group or ""], facts, p)
    vals = [eval_expr(c, facts, p) for c in e.children()]
    if k == "all":
        return False if False in vals else (None if None in vals else True)
    if k == "any":
        return True if True in vals else (None if None in vals else False)
    if k == "not_":
        return None if vals[0] is None else not vals[0]
    nof = getattr(e, k)
    if nof.within_days is not None or None in vals:
        return None
    t = vals.count(True)
    return {"at_least": t >= nof.n, "at_most": t <= nof.n, "exactly": t == nof.n}[k]


def suppress_overlap(a: int, b: int, both: int, k: int) -> dict[str, int | str]:
    """Overlap of two cohorts whose sizes are disclosed: suppress small cells and any cell that could be
    recovered from a disclosed total minus a disclosed cell."""
    cells = {"both": both, "only_first": a - both, "only_second": b - both}
    hidden = {c for c, v in cells.items() if 0 < v < k}
    totals = [(a, ("both", "only_first")), (b, ("both", "only_second"))]
    changed = True
    while changed:
        changed = False
        for total, pair in totals:
            if 0 < total < k:
                continue  # the total itself is suppressed
            h = [c for c in pair if c in hidden]
            if len(h) == 1:
                other = pair[1] if h[0] == pair[0] else pair[0]
                if cells[other] > 0 and other not in hidden:
                    hidden.add(other)
                    changed = True
    return {c: (("<" + str(k)) if 0 < v < k else COMPLEMENTARY) if c in hidden else v for c, v in cells.items()}


class ProxyMixin:
    if TYPE_CHECKING:
        settings: Settings
        ontology: Ontology
        vocab: Vocabulary
        store: MetadataStore
        executor: Executor
        policy: GovernancePolicy

        def _ctx(self, run_id: str, step_id: str | None = None) -> AgentContext: ...
        def data_snapshot(self) -> str: ...
        def load_definition(self, def_id: int, tenant: str | None = None) -> tuple[dict, Any]: ...
        def _manifest(self, run_id: str, query: str, result: Any, retries: int) -> dict: ...

    # ---- helpers --------------------------------------------------------------------------------
    @property
    def proxy_explainer(self) -> ProxyExplainer:
        return ProxyExplainer(self.ontology, self.vocab)

    def parse_proxy_payload(self, data: dict[str, Any] | str) -> ProxyDefinition:
        """YAML text or a dict -> ProxyDefinition (ontology/vocabulary versions default to the active ones)."""
        defaults = {"ontology_version": self.ontology.version, "vocabulary_version": self.vocab.version()}
        if isinstance(data, str):
            return ProxyDefinition.from_yaml(data, **defaults)
        merged = {**defaults, **{k: v for k, v in data.items() if v is not None}}
        return ProxyDefinition.model_validate(merged)

    def load_proxy(self, def_id: int, tenant: str | None = None) -> tuple[dict, ProxyDefinition]:
        row, defn = self.load_definition(def_id, tenant)
        if not isinstance(defn, ProxyDefinition):
            raise KeyError(f"proxy cohort definition {def_id} not found")
        return row, defn

    def validate_proxy_definition(self, p: ProxyDefinition, dry_run: bool = True) -> dict:
        issues, attrition, compiled = validate_proxy(p, self.ontology, self.vocab, self.executor if dry_run else None)
        k = self.executor.min_cell
        return {
            "valid": not any(i.severity == "error" for i in issues),
            "issues": [i.as_dict() for i in issues],
            "attrition": attrition.suppressed(k) if attrition else None,
            "explanation": self.proxy_explainer.explain(p),
            "content_hash": p.content_hash(),
            "semantic_hash": p.semantic_hash(),
            "sql_hash": compiled.sql_hash if compiled else None,
            "classification": p.classification,
        }

    def _check_validation_reference(self, p: ProxyDefinition, tenant: str) -> None:
        """'clinically_validated' needs, in this tenant, an evaluation of the SAME logic (semantic hash recomputed
        from the evaluated definition) that completed, met its prespecified acceptance criteria and was accepted
        by a human reviewer. A record merely existing is not enough."""
        if p.classification != "clinically_validated":
            return
        rec = self.store.get_validation(p.validation_reference or "")
        if rec is None or (rec.get("tenant") or DEFAULT_TENANT) != tenant:
            raise GovernanceError(
                f"validation_reference {p.validation_reference!r} is not a recorded reference "
                "validation in this tenant; 'clinically_validated' cannot be claimed"
            )
        validated = self.store.get_definition(rec["cohort_definition_id"]) or {}
        same = False
        if validated.get("algorithm_name") == p.algorithm_name:
            try:
                same = ProxyDefinition.model_validate(validated["ir"]).semantic_hash() == p.semantic_hash()
            except Exception:  # an unreadable stored definition never supports a claim
                same = False
        if not same:
            raise GovernanceError(
                "the referenced validation was run on different algorithm logic; validate this "
                "exact logic against the reference standard first"
            )
        if rec.get("evaluation_status") != "completed":
            raise GovernanceError(
                f"the referenced evaluation is {rec.get('evaluation_status') or 'a legacy record without status'}, "
                "not completed; it cannot support 'clinically_validated'"
            )
        if rec.get("acceptance_status") != "criteria_met":
            raise GovernanceError(
                f"the referenced evaluation's acceptance status is {rec.get('acceptance_status') or 'not_assessed'}; "
                "prespecified acceptance criteria must be met"
            )
        if rec.get("review_decision") != "accepted":
            raise GovernanceError(
                "the referenced evaluation has not been accepted by a human reviewer "
                f"(decision: {rec.get('review_decision') or 'pending'})"
            )

    def next_version(self, tenant: str, algorithm_name: str) -> str:
        versions = [r["algorithm_version"] for r in self.store.find_algorithm(tenant, algorithm_name)]
        if not versions:
            return "1.0"
        major, minor = max(tuple(int(x) for x in (v.split(".") + ["0"])[:2]) for v in versions)
        return f"{major}.{minor + 1}"

    # ---- submit (immutable versions) ----------------------------------------------------------------
    def submit_proxy(
        self,
        p: ProxyDefinition,
        user_id: str,
        tenant: str = DEFAULT_TENANT,
        parent_id: int | None = None,
        run_id: str | None = None,
    ) -> dict:
        existing = self.store.find_algorithm(tenant, p.algorithm_name, p.version)
        if existing:
            row = existing[0]
            if row["content_hash"] == p.content_hash():
                return {
                    "cohort_definition_id": row["cohort_definition_id"],
                    "status": row["status"],
                    "existing": True,
                    "issues": (self.store.get_definition(row["cohort_definition_id"]) or {}).get("issues", []),
                }
            self.store.audit(
                user_id,
                "proxy.submit",
                "cohort_definition",
                row["cohort_definition_id"],
                "denied",
                tenant,
                {"reason": "version exists with different content", "version": p.version},
            )
            raise VersionConflict(
                f"{p.algorithm_name} version {p.version} already exists with different content. Versions are "
                f"immutable: save the change as a new version (next: {self.next_version(tenant, p.algorithm_name)})"
            )
        self._check_validation_reference(p, tenant)
        if parent_id is not None:
            parent = self.store.get_definition(parent_id)
            if parent is None or (parent.get("tenant") or DEFAULT_TENANT) != tenant:
                raise KeyError(f"cohort definition {parent_id} not found")
        else:  # lineage: link to the latest earlier version of the same algorithm
            prev = self.store.find_algorithm(tenant, p.algorithm_name)
            parent_id = prev[0]["cohort_definition_id"] if prev else None
        result = self.validate_proxy_definition(p)
        errors = [i for i in result["issues"] if i["severity"] == "error"]
        status = "needs_review" if errors else "draft"
        def_id = self.store.save_definition(
            p,
            status,
            user_id,
            run_id=run_id,
            parent_id=parent_id,
            issues=errors,
            dataset=self.ontology.dataset_name,
            tenant=tenant,
        )
        self.store.audit(
            user_id,
            "proxy.submit",
            "cohort_definition",
            def_id,
            "success",
            tenant,
            {
                "status": status,
                "algorithm": p.algorithm_name,
                "version": p.version,
                "content_hash": p.content_hash(),
                "parent": parent_id,
            },
        )
        return {"cohort_definition_id": def_id, "status": status, "existing": False, **result}

    # ---- natural language -> draft (never executes) ---------------------------------------------------
    def ask_proxy(self, query: str, user_id: str = "anonymous", tenant: str = DEFAULT_TENANT) -> ProxyRunResult:
        run_id = self.store.start_run(query, user_id, tenant)
        seq = 0

        def step(
            name: str,
            attempt: int,
            inputs: Any,
            fn: Callable[[AgentContext], Any],
            summarize: Callable[[Any], Any] = lambda x: x,
        ) -> Any:
            nonlocal seq
            seq += 1
            step_id, started, t0 = str(uuid.uuid4()), now(), time.monotonic()
            out: Any = None
            status = "error"
            try:
                out = fn(self._ctx(run_id, step_id))
                status = "ok"
                return out
            except Exception as exc:
                out = {"error": f"{type(exc).__name__}: {exc}"}
                raise
            finally:
                payload = summarize(out) if status == "ok" else out
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump(mode="json")
                self.store.record_step(
                    run_id,
                    seq,
                    attempt,
                    name,
                    inputs,
                    payload,
                    status,
                    started,
                    int((time.monotonic() - t0) * 1000),
                    step_id=step_id,
                )

        result = ProxyRunResult(run_id=run_id, status="failed")
        feedback: str | None = None
        cache: dict[str, ResolvedConceptSet] = {}
        p: ProxyDefinition | None = None
        attempt = 0
        try:
            for attempt in range(self.settings.max_retries + 1):
                draft: ProxyDraft = step(
                    "proxy_parser",
                    attempt,
                    {"query": query, "feedback": feedback},
                    partial(parse_proxy, query=query, feedback=feedback),
                )
                resolved: dict[str, ResolvedConceptSet] = {}
                for m in draft.mentions:
                    sig = resolution_signature(m)
                    if sig not in cache:
                        cache[sig] = step(
                            "concept_resolver",
                            attempt,
                            {"mention": m.model_dump()},
                            partial(resolve_mention, mention=m, query=query, feedback=None),
                        )
                    resolved[m.key] = cache[sig]
                version = self.next_version(tenant, slugify(draft.algorithm_name))
                p, issues = step(
                    "proxy_composer",
                    attempt,
                    {"mentions": sorted(resolved)},
                    partial(self._compose, draft, resolved, version),
                    lambda r: {"hash": r[0].content_hash() if r[0] else None, "issues": [i.as_dict() for i in r[1]]},
                )
                if p is not None:
                    v = step("proxy_validator", attempt, {"hash": p.content_hash()}, partial(self._validate_step, p))
                    issues = issues + [Issue(i["severity"], i["stage"], i["message"]) for i in v["issues"]]
                    result.attrition = v["attrition"] or []
                result.issues = [i.as_dict() for i in issues]
                errors = [i for i in issues if i.severity == "error"]
                if not errors:
                    result.status = "draft"
                    break
                result.status = "needs_review"
                if any(i.stage == "dataset" for i in errors):
                    break  # the dataset cannot answer this; rephrasing will not help
                feedback = "\n".join(f"- [{i.stage}] {i.message}" for i in errors)
        except LLMError as exc:
            result.status = "failed"
            result.issues.append({"severity": "error", "stage": "llm", "message": str(exc)})
        if p is None and result.status == "needs_review":
            result.status = "failed"
        if p is not None and result.status in ("draft", "needs_review"):
            result.ir = p
            result.explanation = self.proxy_explainer.explain(p)
            errs = [i for i in result.issues if i["severity"] == "error"]
            result.cohort_definition_id = self.store.save_definition(
                p,
                "needs_review" if errs else "draft",
                user_id,
                run_id=run_id,
                issues=errs,
                dataset=self.ontology.dataset_name,
                tenant=tenant,
            )
        result.manifest = {**self._manifest(run_id, query, result, attempt), "kind": "proxy"}
        self.store.finish_run(run_id, result.status, attempt, result.cohort_definition_id, result.manifest)
        self.store.audit(
            user_id,
            "proxy.ask",
            "agent_run",
            run_id,
            "success" if result.status != "failed" else "failed",
            tenant,
            {"status": result.status, "cohort_definition_id": result.cohort_definition_id},
        )
        return result

    def _compose(
        self, draft: ProxyDraft, resolved: dict[str, ResolvedConceptSet], version: str, _ctx: Any
    ) -> tuple[ProxyDefinition | None, list[Issue]]:
        return compose_proxy(draft, resolved, self.ontology, self.vocab, version)

    def _validate_step(self, p: ProxyDefinition, _ctx: Any) -> dict:
        return self.validate_proxy_definition(p)

    # ---- review packet & SQL preview ----------------------------------------------------------------
    def proxy_review_packet(self, def_id: int, tenant: str | None = None) -> dict:
        row, p = self.load_proxy(def_id, tenant)
        v = self.validate_proxy_definition(p)
        packet = self.proxy_explainer.review_packet(p, v["issues"], row, v["sql_hash"], v["attrition"])
        packet["reviews"] = self.store.reviews(def_id)
        packet["next_step"] = {
            "draft": "A reviewer other than the author can approve or reject it (POST /proxy-cohorts/{id}/review).",
            "needs_review": "Validation errors must be fixed in a new version before approval.",
            "approved": "Approved: an executor can run it.",
            "rejected": "Rejected: submit a revised new version.",
        }.get(row["status"], "")
        return packet

    def compile_proxy(self, def_id: int, tenant: str | None = None) -> dict:
        _, p = self.load_proxy(def_id, tenant)
        compiled = ProxyCompiler(self.ontology).compile_proxy(p)
        return {
            "cohort_definition_id": def_id,
            "sql_hash": compiled.sql_hash,
            "unavailable_evidence": list(compiled.unavailable_evidence),
            "statements": {
                "cohort": compiled.cohort_sql,
                "assignment": compiled.assignment_sql,
                "evidence": compiled.evidence_sql,
                "attrition": compiled.attrition_sql,
                "summary": compiled.summary_sql,
            },
            "note": "Preview only. Generated deterministically from the definition (no LLM); execution requires "
            "an approved definition.",
        }

    # ---- execution (called by CohortBuilder.execute after the governance checks) ------------------------
    def _execute_proxy(self, def_id: int, row: dict, p: ProxyDefinition, user_id: str, base: dict) -> dict:
        row_tenant = row.get("tenant") or DEFAULT_TENANT
        issues, _, _ = validate_proxy(p, self.ontology, self.vocab)
        errors = [i.message for i in issues if i.severity == "error"]
        if errors:
            self.store.audit(
                user_id,
                "definition.execute",
                "cohort_definition",
                def_id,
                "denied",
                row_tenant,
                {**base, "reason": "validation errors on active dataset"},
            )
            raise ValueError(
                f"proxy definition {def_id} (built on {row.get('dataset')!r}) cannot run on dataset "
                f"{self.ontology.dataset_name!r}: " + "; ".join(errors)
            )
        caveats = [i.as_dict() for i in issues if i.severity == "warning"]
        life = self._lifecycle(row, p)
        if life["superseded"]:
            caveats.append(
                {
                    "severity": "warning",
                    "stage": "governance",
                    "message": f"version {p.version} is superseded by approved version(s) "
                    f"{', '.join(life['superseded_by_approved_versions'])}",
                }
            )
        compiled: CompiledProxy = ProxyCompiler(self.ontology).compile_proxy(p)
        try:
            generation_id, attrition = self.executor.generate(compiled, def_id)
            raw = self.executor.proxy_summary(compiled)
        except ExecutionError as exc:
            self.store.audit(
                user_id,
                "definition.execute",
                "cohort_definition",
                def_id,
                "failed",
                row_tenant,
                {**base, "error": str(exc), "error_id": exc.error_id},
            )
            raise
        self.store.record_generation(
            generation_id,
            def_id,
            self.data_snapshot(),
            compiled.compiler_version,
            compiled.sql_hash,
            attrition.final_count,
            user_id,
            self.ontology.dataset_name,
            tenant=row_tenant,
            caveats=caveats,
            summary={"counts": raw, "unavailable_evidence": list(compiled.unavailable_evidence)},
        )
        self.store.audit(
            user_id,
            "definition.execute",
            "cohort_definition",
            def_id,
            "success",
            row_tenant,
            {
                **base,
                "generation_id": generation_id,
                "sql_hash": compiled.sql_hash,
                "kind": "proxy",
                "draft": row["status"] == "draft",
            },
        )
        k = self.executor.min_cell
        return {
            "generation_id": generation_id,
            "cohort_definition_id": def_id,
            "algorithm": {
                "name": p.algorithm_name,
                "version": p.version,
                "classification": p.classification,
                "label": p.label,
            },
            "dataset": self.ontology.dataset_name,
            "person_count": attrition.suppressed_final(k),
            "sql_hash": compiled.sql_hash,
            "attrition": attrition.suppressed(k),
            "evidence_summary": self.suppressed_summary(p, raw, compiled.unavailable_evidence),
            "caveats": caveats,
            "observation": self.ontology.capabilities.get("observation"),
            "min_cell_count": k,
            "governance": [COHORT_NOTE.format(target=p.target.name), SCORE_NOTE],
        }

    def suppressed_summary(self, p: ProxyDefinition, raw: dict[str, int], unavailable: tuple[str, ...] = ()) -> dict:
        """Aggregate evidence breakdown among candidates (everyone passing all steps except tier assignment)."""
        k = self.executor.min_cell
        cand = int(raw.get("candidates", 0))
        tiers = {t.name: int(raw.get(f"tier_{t.name}", 0)) for t in p.tiers}
        tiers["no_tier"] = int(raw.get("tier_none", 0))
        shown_tiers = suppress_partition(tiers, k, total_disclosed=True)
        funnel_counts = [cand] + [int(raw.get(f"funnel_{i + 1}", 0)) for i in range(len(p.funnel))]
        funnel = suppress_series(funnel_counts, k)
        return {
            "population": "candidates: patients passing every step except tier assignment (one row per patient)",
            "candidates": suppress_count(cand, k),
            "evidence": [
                {
                    "id": e.id,
                    "name": e.name,
                    "category": e.category,
                    "required": e.required,
                    "available_in_dataset": e.id not in unavailable,
                    "patients": suppress_with_total(int(raw.get(f"ev_{e.id}", 0)), cand, k),
                }
                for e in p.evidence
            ],
            "temporal_rules": [
                {"id": t.id, "name": t.name, "patients": suppress_with_total(int(raw.get(f"tr_{t.id}", 0)), cand, k)}
                for t in p.temporal_rules
            ],
            "conflicts": [
                {
                    "name": c.name,
                    "label": c.label,
                    "action": c.action,
                    "patients": suppress_with_total(int(raw.get(f"cf_{c.name}", 0)), cand, k),
                }
                for c in p.conflicts
            ],
            "tiers": [{"name": t.name, "label": t.label or t.name, "patients": shown_tiers[t.name]} for t in p.tiers]
            + [{"name": "no_tier", "label": "No tier (not in cohort)", "patients": shown_tiers["no_tier"]}],
            "funnel": [
                {"step": name, "remaining": v}
                for name, v in zip(["Candidates"] + [f.name for f in p.funnel], funnel, strict=True)
            ],
            "notes": [SCORE_NOTE, SUPPRESSION_NOTE.format(k=k)],
        }

    def _generation(self, def_id: int, generation_id: str | None) -> dict:
        gens = self.store.generations(def_id=def_id, generation_id=generation_id)
        if not gens:
            raise KeyError(
                "no generation found for this definition"
                if generation_id is None
                else f"generation {generation_id} not found"
            )
        return gens[0]

    def proxy_results(self, def_id: int, tenant: str | None = None, generation_id: str | None = None) -> dict:
        row, p = self.load_proxy(def_id, tenant)
        gen = self._generation(def_id, generation_id)
        k = self.executor.min_cell
        stats = self.store.con.execute(
            "SELECT rule_sequence, rule_name, remaining_count FROM results.cohort_inclusion_stats "
            "WHERE generation_id=? ORDER BY rule_sequence",
            [gen["generation_id"]],
        ).fetchall()
        shown = suppress_series([int(r[2]) for r in stats], k)
        summary = gen["summary"] or {}
        return {
            "generation_id": gen["generation_id"],
            "cohort_definition_id": def_id,
            "algorithm": {"name": p.algorithm_name, "version": p.version, "classification": p.classification},
            "executed_at": gen["executed_at"],
            "executed_by": gen["executed_by"],
            "dataset": gen["dataset"],
            "data_snapshot": gen["data_snapshot"],
            "sql_hash": gen["sql_hash"],
            "person_count": shown[-1] if shown else 0,
            "attrition": [{"sequence": r[0], "name": r[1], "remaining": v} for r, v in zip(stats, shown, strict=True)],
            "evidence_summary": self.suppressed_summary(
                p, summary.get("counts", {}), tuple(summary.get("unavailable_evidence", []))
            ),
            "caveats": gen["caveats"],
            "min_cell_count": k,
            "governance": [COHORT_NOTE.format(target=p.target.name), SCORE_NOTE],
        }

    def proxy_versions(self, def_id: int, tenant: str | None = None) -> list[dict]:
        row, p = self.load_proxy(def_id, tenant)
        return self.store.find_algorithm(row.get("tenant") or DEFAULT_TENANT, p.algorithm_name)

    # ---- comparison (privacy-safe overlaps) ---------------------------------------------------------
    def compare_generations(self, generation_ids: list[str], actor: str, tenant: str | None = None) -> dict:
        if len(generation_ids) < 2 or len(set(generation_ids)) != len(generation_ids) or len(generation_ids) > 6:
            raise ValueError("compare needs 2 to 6 distinct generation ids")
        gens = []
        for gid in generation_ids:
            g = self.store.generations(generation_id=gid)
            if not g or (tenant is not None and (g[0].get("tenant") or DEFAULT_TENANT) != tenant):
                raise KeyError(f"generation {gid} not found")
            gens.append(g[0])
        k = self.executor.min_cell
        con = self.store.con

        def size(gid: str) -> int:
            return int(
                con.execute(
                    "SELECT count(DISTINCT subject_id) FROM results.cohort WHERE generation_id=?", [gid]
                ).fetchone()[0]
            )

        sizes = {g["generation_id"]: size(g["generation_id"]) for g in gens}
        pairs = []
        for i, a in enumerate(gens):
            for b in gens[i + 1 :]:
                both = int(
                    con.execute(
                        "SELECT count(*) FROM (SELECT DISTINCT subject_id FROM results.cohort WHERE generation_id=?) x"
                        " JOIN (SELECT DISTINCT subject_id FROM results.cohort WHERE generation_id=?) y"
                        " USING (subject_id)",
                        [a["generation_id"], b["generation_id"]],
                    ).fetchone()[0]
                )
                pairs.append(
                    {
                        "first": a["generation_id"],
                        "second": b["generation_id"],
                        **suppress_overlap(sizes[a["generation_id"]], sizes[b["generation_id"]], both, k),
                    }
                )
        datasets = {g["dataset"] for g in gens}
        self.store.audit(
            actor, "proxy.compare", "cohort_generation", ",".join(generation_ids), "success", tenant, {"n": len(gens)}
        )
        return {
            "generations": [
                {
                    "generation_id": g["generation_id"],
                    "cohort_definition_id": g["cohort_definition_id"],
                    "dataset": g["dataset"],
                    "data_snapshot": g["data_snapshot"],
                    "person_count": suppress_count(sizes[g["generation_id"]], k),
                }
                for g in gens
            ],
            "overlaps": pairs,
            "warnings": (
                ["generations come from different datasets; overlaps are not comparable"] if len(datasets) > 1 else []
            ),
            "notes": [
                SUPPRESSION_NOTE.format(k=k),
                "Overlap shows agreement between algorithms, not accuracy: neither algorithm is a reference standard.",
            ],
        }

    # ---- reference-standard evaluation -----------------------------------------------------------------
    def load_reference(
        self,
        name: str,
        labels: Sequence[Any],
        source: str,
        actor: str,
        tenant: str = DEFAULT_TENANT,
    ) -> dict:
        """Patient-level labels from an EXTERNAL reference standard (e.g. chart review, registry). Admin input
        only; never returned by any interface. Each label is a ReferenceRecord or a tuple
        (person_id, label[, reference_date]) where label is case / non_case / unknown (or True / False / None).
        Duplicate records collapse to one patient; case + non_case for one patient becomes 'conflicting'."""
        if not source.strip():
            raise ValueError("source must describe the external reference standard (e.g. 'chart review 2026')")
        records = []
        for item in labels:
            if isinstance(item, ReferenceRecord):
                records.append(item)
                continue
            pid, value, *rest = item
            records.append(ReferenceRecord(int(pid), normalize_label(value), rest[0] if rest else None))
        if not records:
            raise ValueError("a reference standard needs at least one labelled patient")
        rows, collapse = collapse_records(records)
        n = self.store.load_reference(name, tenant, rows, source.strip(), actor)
        self.store.audit(
            actor,
            "reference.load",
            "reference_standard",
            name,
            "success",
            tenant,
            {"patients": n, "records": collapse["records"], "source": source.strip()},
        )
        info = self.store.reference_info(name, tenant) or {}
        k = self.executor.min_cell
        labels_shown = suppress_partition(
            {c: info.get(c, 0) for c in ("cases", "non_cases", "unknown", "conflicting")}, k
        )
        return {
            "reference_name": name,
            "source": info.get("source"),
            "loaded_by": actor,
            "patients": suppress_count(info.get("patients", 0), k),
            "records": suppress_count(info.get("records", 0), k),
            **labels_shown,
            "patients_with_duplicate_records": suppress_count(collapse["patients_with_duplicate_records"], k),
            "patients_with_several_reference_dates": suppress_count(
                collapse["patients_with_several_reference_dates"], k
            ),
            "notes": [
                "Duplicate records were collapsed to one per patient; patients labelled both case and "
                "non_case are 'conflicting' and, like 'unknown', are never evaluated."
            ],
        }

    def validate_against_reference(self, *args: Any, **kwargs: Any) -> dict:
        """Backward-compatible name for `evaluate_against_reference`."""
        return self.evaluate_against_reference(*args, **kwargs)

    def evaluate_against_reference(
        self,
        def_id: int,
        reference_name: str,
        actor: str,
        tenant: str | None = None,
        generation_id: str | None = None,
        positive_tiers: list[str] | None = None,
        intended_use: str | None = None,
        criteria: AcceptanceCriteria | None = None,
        eligibility: EligibilityRules | None = None,
    ) -> dict:
        """Empirical evaluation of one generation against a reference standard, over the eligible
        evaluation population (see proxy_evaluation.py), plus an AUTOMATIC check of acceptance criteria.
        Neither is an approval: a reviewer records acceptance separately (`review_evaluation`)."""
        row, p = self.load_proxy(def_id, tenant)
        row_tenant = row.get("tenant") or DEFAULT_TENANT
        gen = self._generation(def_id, generation_id)
        info = self.store.reference_info(reference_name, row_tenant)
        if info is None:
            raise KeyError(f"reference standard {reference_name!r} not found")
        tiers = sorted(set(positive_tiers or [t.name for t in p.tiers]))
        if unknown := sorted(set(tiers) - {t.name for t in p.tiers}):
            raise ValueError(f"unknown tiers: {unknown}")
        prespecified = p.acceptance_criteria.get(intended_use) if intended_use else None
        if prespecified is not None and criteria is not None and criteria != prespecified:
            raise ValueError(
                f"acceptance criteria for intended use {intended_use!r} are prespecified in version {p.version}; "
                "they cannot be replaced at evaluation time (create a new version to change them)"
            )
        crit = criteria or prespecified
        crit_source = (
            None
            if crit is None
            else (
                "prespecified in the definition" if prespecified is not None else "supplied with the evaluation request"
            )
        )
        rules = eligibility or EligibilityRules()
        for entity in rules.require_data:
            if not self.ontology.supports_entity(entity):
                raise ValueError(
                    f"eligibility requires {entity} data, which dataset {self.ontology.dataset_name!r} "
                    "does not provide; patients cannot be checked for it"
                )
        sql = eligibility_sql(self.ontology, p, rules, len(tiers))
        con = self.store.con
        rows = con.execute(sql, [gen["generation_id"], *tiers, row_tenant, reference_name]).fetchall()
        pop = classify(rows, rules.require_data)
        unlabelled = int(
            con.execute(
                "SELECT count(DISTINCT a.subject_id) FROM results.proxy_assignment a WHERE a.generation_id = ? AND "
                f"a.tier IN ({', '.join('?' for _ in tiers)}) AND NOT EXISTS (SELECT 1 FROM meta.reference_label r "
                "WHERE r.tenant = ? AND r.reference_name = ? AND r.person_id = a.subject_id)",
                [gen["generation_id"], *tiers, row_tenant, reference_name],
            ).fetchone()[0]
        )
        k = self.executor.min_cell
        counts = pop["confusion"]
        level = crit.confidence_level if crit else 0.95
        raw = compute_metrics(counts, level)
        withheld = []
        if pop["eligible_reference_positive"] == 0 or pop["eligible_reference_negative"] == 0:
            withheld.append("the eligible population must contain both reference cases and non-cases")
        if any(0 < v < k for v in counts.values()):
            withheld.append(f"a confusion-matrix cell is below the minimum cell count ({k})")
        inconclusive = list(withheld)
        snapshot = self.data_snapshot()
        if gen["data_snapshot"] != snapshot:
            inconclusive.append(
                f"the data snapshot changed since this generation ran ({gen['data_snapshot']!r} -> "
                f"{snapshot!r}); re-run the algorithm before evaluating"
            )
        evaluation_status = "inconclusive" if inconclusive else "completed"
        acceptance = assess_acceptance(crit, pop, None if withheld else raw, evaluation_status)
        metrics: dict[str, Any] = {"reported": not withheld, "withheld_reasons": withheld, "definitions": DEFINITIONS}
        if not withheld:
            metrics.update(rounded(raw))
        validation_id = str(uuid.uuid4())
        self.store.record_validation(
            {
                "validation_id": validation_id,
                "generation_id": gen["generation_id"],
                "cohort_definition_id": def_id,
                "reference_name": reference_name,
                "positive_tiers": tiers,
                "counts": counts,
                "metrics": metrics,
                "created_by": actor,
                "tenant": row_tenant,
                "evaluation_status": evaluation_status,
                "acceptance_status": acceptance["status"],
                "intended_use": intended_use,
                "criteria": None if crit is None else {**crit.model_dump(mode="json"), "source": crit_source},
                "acceptance": acceptance,
                "population": {**pop, "cohort_members_without_reference_label": unlabelled},
                "eligibility": rules.model_dump(mode="json"),
                "semantic_hash": p.semantic_hash(),
                "data_snapshot": snapshot,
            }
        )
        self.store.audit(
            actor,
            "proxy.evaluate_reference",
            "cohort_definition",
            def_id,
            "success",
            row_tenant,
            {
                "validation_id": validation_id,
                "reference": reference_name,
                "evaluation_status": evaluation_status,
                "acceptance_status": acceptance["status"],
            },
        )
        return {
            "validation_id": validation_id,
            "cohort_definition_id": def_id,
            "algorithm": {"name": p.algorithm_name, "version": p.version, "semantic_hash": p.semantic_hash()},
            "generation_id": gen["generation_id"],
            "data_snapshot": snapshot,
            "reference": {
                "name": reference_name,
                "source": info["source"],
                "patients": suppress_count(info["patients"], k),
            },
            "positive_tiers": tiers,
            "intended_use": intended_use,
            "status": {
                "evaluation": evaluation_status,
                "evaluation_reasons": inconclusive,
                "acceptance_criteria": acceptance["status"],
                "human_review": "pending",
                "note": "Evaluation and the automatic criteria check are not approvals. A reviewer must accept "
                "this evaluation before the version may be described as clinically validated.",
            },
            "population": self._population_view(pop, unlabelled, k),
            "eligibility_rules": rules.model_dump(mode="json"),
            "confusion_matrix": self._confusion_cells(counts, k),
            "metrics": metrics,
            "acceptance": {
                "criteria": None if crit is None else crit.model_dump(mode="json"),
                "criteria_source": crit_source,
                **acceptance,
            },
            "provenance": self.provenance_summary(p),
            "notes": [
                "Metrics are computed over the eligible, evaluated reference population only - never from the "
                "proxy cohort itself. Excluded patients are not counted as negatives.",
                "Metrics describe agreement with this reference standard on this dataset and data snapshot; they "
                "do not transfer to other datasets, populations or intended uses.",
                "Claims-based proxies infer that data match a pattern; they do not establish a clinical diagnosis.",
            ],
        }

    @staticmethod
    def _population_view(pop: dict, unlabelled: int, k: int) -> dict:
        total, excluded = pop["reference_patients"], pop["excluded"]
        return {
            "definition": "eligible = definite reference label + present in the dataset + observable + required "
            "data present; every eligible patient is evaluated",
            "reference_patients": suppress_count(total, k),
            "labels": suppress_partition(pop["labels"], k),
            "eligible": suppress_with_total(pop["eligible"], total, k),
            "evaluated": suppress_with_total(pop["evaluated"], total, k),
            "eligible_reference_positive": suppress_with_total(pop["eligible_reference_positive"], pop["eligible"], k),
            "eligible_reference_negative": suppress_with_total(pop["eligible_reference_negative"], pop["eligible"], k),
            "excluded": suppress_with_total(excluded, total, k),
            "excluded_by_reason": suppress_partition(pop["excluded_by_reason"], k),
            "cohort_members_without_reference_label": suppress_count(unlabelled, k),
        }

    @staticmethod
    def _confusion_cells(counts: dict[str, int], k: int) -> dict[str, int | str]:
        """Cells sum to known totals (cases, non-cases), so if any cell is small every non-zero cell is hidden."""
        if not any(0 < v < k for v in counts.values()):
            return dict(counts)
        return {c: (suppress_count(v, k) if v < k else COMPLEMENTARY) for c, v in counts.items()}

    # ---- human acceptance review of an evaluation ----------------------------------------------------
    def review_evaluation(
        self, def_id: int, validation_id: str, reviewer: str, decision: str, rationale: str, tenant: str | None = None
    ) -> dict:
        """A reviewer accepts or rejects ONE evaluation for its intended use. Recorded once, audited.
        Acceptance needs a completed evaluation whose prespecified criteria were met."""
        row, p = self.load_proxy(def_id, tenant)
        row_tenant = row.get("tenant") or DEFAULT_TENANT
        rec = self.store.get_validation(validation_id)
        if rec is None or rec["cohort_definition_id"] != def_id:
            raise KeyError(f"evaluation {validation_id} not found for definition {def_id}")
        details = {"validation_id": validation_id, "decision": decision}

        def deny(reason: str) -> None:
            self.store.audit(
                reviewer,
                "proxy.evaluation_review",
                "cohort_definition",
                def_id,
                "denied",
                row_tenant,
                {**details, "reason": reason},
            )

        if decision not in ("accepted", "rejected"):
            raise ValueError("decision must be 'accepted' or 'rejected'")
        if len(rationale.strip()) < 10:
            raise ValueError("a decision rationale (at least 10 characters) is required")
        if rec.get("review_decision"):
            deny("already reviewed")
            raise ValueError(
                f"evaluation {validation_id} was already {rec['review_decision']} by "
                f"{rec['reviewed_by']}; run a new evaluation instead"
            )
        if reviewer in (row["created_by"], rec["created_by"]) and not self.policy.allow_self_approval:
            deny("self-review")
            raise GovernanceError("the author of the algorithm or of the evaluation cannot review it")
        if decision == "accepted":
            if rec.get("evaluation_status") != "completed":
                deny(f"evaluation {rec.get('evaluation_status') or 'legacy'}")
                raise ValueError(
                    "only a completed evaluation can be accepted (this one is "
                    f"{rec.get('evaluation_status') or 'a legacy record without status'})"
                )
            if rec.get("acceptance_status") != "criteria_met":
                deny(f"acceptance {rec.get('acceptance_status')}")
                raise ValueError(
                    "only an evaluation whose prespecified acceptance criteria were met can be accepted "
                    f"(status: {rec.get('acceptance_status')})"
                )
            if rec.get("semantic_hash") != p.semantic_hash():
                deny("logic changed")
                raise ValueError("the evaluation was recorded for different algorithm logic")
        self.store.record_validation_review(validation_id, decision, reviewer, rationale.strip())
        self.store.audit(
            reviewer, "proxy.evaluation_review", "cohort_definition", def_id, "success", row_tenant, details
        )
        return {
            "validation_id": validation_id,
            "cohort_definition_id": def_id,
            "decision": decision,
            "reviewer": reviewer,
            "scope": f"intended use {rec.get('intended_use')!r} of {p.algorithm_name} logic "
            f"{p.semantic_hash()[:19]}..., this reference standard and data snapshot only",
        }

    # ---- lifecycle / status -----------------------------------------------------------------------------
    @staticmethod
    def _vkey(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    def _lifecycle(self, row: dict, p: ProxyDefinition) -> dict:
        versions = self.store.find_algorithm(row.get("tenant") or DEFAULT_TENANT, p.algorithm_name)
        newer = [v for v in versions if self._vkey(v["algorithm_version"]) > self._vkey(p.version)]
        superseded_by = sorted((v["algorithm_version"] for v in newer if v["status"] == "approved"), key=self._vkey)
        latest = max((v["algorithm_version"] for v in versions), key=self._vkey) if versions else p.version
        return {
            "latest_version": latest,
            "is_latest": latest == p.version,
            "superseded": bool(superseded_by),
            "superseded_by_approved_versions": superseded_by,
            "newer_unapproved_versions": sorted(
                (v["algorithm_version"] for v in newer if v["status"] != "approved"), key=self._vkey
            ),
        }

    def proxy_status(self, def_id: int, tenant: str | None = None) -> dict:
        """Every status of one version, kept separate: definition validation, execution approval,
        evaluations, automatic acceptance checks, human acceptance reviews, lifecycle and the claim
        this version may make."""
        row, p = self.load_proxy(def_id, tenant)
        issues, _, _ = validate_proxy(p, self.ontology, self.vocab)
        errors = [i for i in issues if i.severity == "error"]
        current = p.semantic_hash()
        evaluations = []
        for rec in self.store.validations(def_id):
            evaluations.append(
                {
                    "validation_id": rec["validation_id"],
                    "generation_id": rec["generation_id"],
                    "reference_name": rec["reference_name"],
                    "intended_use": rec.get("intended_use"),
                    "created_by": rec["created_by"],
                    "created_at": rec["created_at"],
                    "evaluation_status": rec.get("evaluation_status") or "legacy (no status recorded)",
                    "acceptance_status": rec.get("acceptance_status") or "not_assessed",
                    "human_review": {
                        "decision": rec.get("review_decision") or "pending",
                        "reviewed_by": rec.get("reviewed_by"),
                        "reviewed_at": rec.get("reviewed_at"),
                        "rationale": rec.get("review_rationale"),
                    },
                    "same_logic_as_this_version": rec.get("semantic_hash") == current,
                    "data_snapshot": rec.get("data_snapshot"),
                }
            )
        lifecycle = self._lifecycle(row, p)
        claim_ok, claim_reason = True, "backed by an accepted evaluation of this logic"
        if p.classification == "clinically_validated":
            try:
                self._check_validation_reference(p, row.get("tenant") or DEFAULT_TENANT)
            except GovernanceError as exc:
                claim_ok, claim_reason = False, str(exc)
        accepted = [
            e for e in evaluations if e["human_review"]["decision"] == "accepted" and e["same_logic_as_this_version"]
        ]
        return {
            "cohort_definition_id": def_id,
            "algorithm": {
                "name": p.algorithm_name,
                "version": p.version,
                "semantic_hash": current,
                "content_hash": p.content_hash(),
            },
            "classification": {
                "value": p.classification,
                "label": CLASSIFICATION_LABELS[p.classification],
                "claim_supported": claim_ok if p.classification == "clinically_validated" else None,
                "reason": claim_reason if p.classification == "clinically_validated" else None,
            },
            "definition_validation": {
                "status": "passed" if not errors else "failed",
                "errors": len(errors),
                "warnings": len(issues) - len(errors),
                "meaning": "structural and logical checks on the active dataset; says nothing about accuracy",
            },
            "execution_approval": {
                "status": row["status"],
                "approved_by": row.get("approved_by"),
                "approved_at": row.get("approved_at"),
                "scope": "running this exact version (content hash); not clinical acceptance",
            },
            "evaluations": evaluations,
            "accepted_for_intended_uses": sorted({e["intended_use"] or "(unspecified)" for e in accepted}),
            "lifecycle": lifecycle,
            "summary": self._status_summary(p, row, errors, evaluations, accepted, lifecycle),
        }

    @staticmethod
    def _status_summary(
        p: ProxyDefinition, row: dict, errors: list, evaluations: list, accepted: list, lifecycle: dict
    ) -> str:
        parts = [
            f"{p.algorithm_name} v{p.version}: definition {'invalid' if errors else 'valid'}",
            f"execution {row['status']}",
        ]
        if not evaluations:
            parts.append("not evaluated against a reference standard")
        elif accepted:
            parts.append(f"evaluation accepted for {', '.join(sorted({e['intended_use'] or '?' for e in accepted}))}")
        else:
            parts.append("no accepted evaluation")
        if lifecycle["superseded"]:
            parts.append(f"superseded by v{lifecycle['superseded_by_approved_versions'][-1]}")
        return "; ".join(parts)

    def list_evaluations(self, def_id: int, tenant: str | None = None) -> list[dict]:
        return self.proxy_status(def_id, tenant)["evaluations"]

    def provenance_summary(self, p: ProxyDefinition) -> dict:
        return provenance_summary(p)

    # ---- patient-level explanation (authorized access only) ---------------------------------------------
    def patient_explanation(
        self,
        def_id: int,
        subject_id: int,
        actor: str,
        is_admin: bool,
        tenant: str | None = None,
        generation_id: str | None = None,
    ) -> dict:
        row, p = self.load_proxy(def_id, tenant)
        row_tenant = row.get("tenant") or DEFAULT_TENANT
        if not (is_admin and self.policy.allow_patient_level):
            self.store.audit(
                actor,
                "proxy.patient_explanation",
                "cohort_definition",
                def_id,
                "denied",
                row_tenant,
                {"reason": "patient-level access not authorized"},
            )
            raise GovernanceError("patient-level output requires an admin principal and CB_ALLOW_PATIENT_LEVEL=true")
        gen = self._generation(def_id, generation_id)
        con = self.store.con
        ref = hashlib.sha256(f"{gen['generation_id']}:{subject_id}".encode()).hexdigest()[:16]
        self.store.audit(
            actor,
            "proxy.patient_explanation",
            "cohort_definition",
            def_id,
            "success",
            row_tenant,
            {"generation_id": gen["generation_id"], "subject_ref": ref},
        )
        a = con.execute(
            "SELECT tier, evidence_score FROM results.proxy_assignment WHERE generation_id=? AND subject_id=?",
            [gen["generation_id"], subject_id],
        ).fetchone()
        if a is None:
            return {
                "generation_id": gen["generation_id"],
                "subject_id": subject_id,
                "in_cohort": False,
                "note": "This patient is not a member of this generation (per-patient evidence is stored for "
                "members only).",
            }
        facts = {
            k: bool(v)
            for k, v in con.execute(
                "SELECT evidence_key, present FROM results.proxy_evidence WHERE generation_id=? AND subject_id=?",
                [gen["generation_id"], subject_id],
            ).fetchall()
        }
        tier = next(t for t in p.tiers if t.name == a[0])
        exp = self.proxy_explainer
        return {
            "generation_id": gen["generation_id"],
            "cohort_definition_id": def_id,
            "algorithm": {"name": p.algorithm_name, "version": p.version, "classification": p.classification},
            "subject_id": subject_id,
            "in_cohort": True,
            "tier": tier.name,
            "tier_label": tier.label or tier.name,
            "evidence_score": a[1],
            "score_note": SCORE_NOTE,
            "evidence": [
                {"id": e.id, "name": e.name, "category": e.category, "present": facts.get(f"ev_{e.id}")}
                for e in p.evidence
            ],
            "temporal_rules": [
                {"id": t.id, "name": t.name, "holds": facts.get(f"tr_{t.id}")} for t in p.temporal_rules
            ],
            "conflicts": [
                {"name": c.name, "action": c.action, "present": facts.get(f"cf_{c.name}")} for c in p.conflicts
            ],
            "tiers": [
                {
                    "name": t.name,
                    "rule": exp.expr_text(t.rule, p) if t.rule else None,
                    "rule_holds": eval_expr(t.rule, facts, p) if t.rule else None,
                    "min_score": t.min_score,
                    "assigned": t.name == tier.name,
                }
                for t in p.tiers
            ],
            "note": COHORT_NOTE.format(target=p.target.name),
        }
