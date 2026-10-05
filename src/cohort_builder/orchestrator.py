"""Fixed agent graph and the CohortBuilder service facade.

    parse_intent -> resolve_concepts -> compose -> validate -> explain -> critique
         ^                ^                                                  |
         +----------------+-------------- feedback (max retries) -----------+

The graph, retry limit and routing are code, not prompt. Every step, LLM call
and tool call is recorded, and each run ends with a manifest.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import duckdb

from . import COMPILER_VERSION, __version__
from .agents import AgentContext
from .agents.composer import Issue, compose
from .agents.critic import critique
from .agents.explainer import Explainer
from .agents.intent import CohortIntent, parse_intent
from .agents.resolver import ResolvedConceptSet, resolution_signature, resolve_mention
from .agents.validator import validate
from .compiler import Compiler
from .config import Settings
from .db import connect, init_schemas
from .executor import Executor
from .ir import CohortDefinition
from .llm import Backend, LLMClient, LLMError, Prompt
from .metadata import MetadataStore, now
from .ontology import Ontology
from .vocab import Vocabulary


@dataclass
class RunResult:
    run_id: str
    status: str  # draft | needs_review | failed
    cohort_definition_id: int | None = None
    ir: CohortDefinition | None = None
    explanation: str = ""
    issues: list[dict] = field(default_factory=list)
    attrition: list[dict] = field(default_factory=list)
    critic_notes: str = ""
    manifest: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["ir"] = self.ir.model_dump(mode="json") if self.ir else None
        return d


class CohortBuilder:
    def __init__(self, settings: Settings | None = None, backend: Backend | None = None,
                 con: duckdb.DuckDBPyConnection | None = None):
        self.settings = settings or Settings.from_env()
        self.con = con or connect(self.settings.db_path)
        init_schemas(self.con)
        self.ontology = Ontology.load(self.settings.ontology_dir, self.settings.dataset)
        for stmt in self.ontology.setup_statements():  # install the dataset's semantic views
            self.con.execute(stmt)
        self.vocab = Vocabulary(self.con)
        self.store = MetadataStore(self.con)
        self.llm = LLMClient(self.settings, self.store, backend)
        self.executor = Executor(self.con, self.ontology.rules["min_cell_count"])
        self.compiler = Compiler(self.ontology)
        self.explainer = Explainer(self.ontology, self.vocab)
        self._record_versions()

    # ---- versions / manifest ------------------------------------------------
    def _prompts(self) -> dict[str, Prompt]:
        return {n: Prompt.load(self.settings.prompts_dir, n, v) for n, v in self.settings.prompt_versions.items()}

    def component_versions(self) -> dict[str, Any]:
        return {
            "app_version": __version__,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "prompts": {n: {"version": p.version, "hash": p.hash} for n, p in self._prompts().items()},
            "ontology": {"version": self.ontology.version, "hash": self.ontology.content_hash},
            "dataset": {"name": self.ontology.dataset_name, "version": str(self.ontology.dataset.get("version", ""))},
            "vocabulary_version": self.vocab.version(),
            "compiler_version": COMPILER_VERSION,
            "data_snapshot": self.data_snapshot(),
        }

    def data_snapshot(self) -> str:
        return self.store.data_snapshot(self.ontology.dataset.get("snapshot_sql"))

    def _record_versions(self) -> None:
        v = self.component_versions()
        self.store.record_component("model", "llm", v["model"])
        for n, p in v["prompts"].items():
            self.store.record_component("prompt", n, p["version"], p["hash"])
        self.store.record_component("ontology", "domain", self.ontology.version, self.ontology.content_hash)
        self.store.record_component("dataset", self.ontology.dataset_name, v["dataset"]["version"])
        self.store.record_component("vocabulary", "omop", v["vocabulary_version"])
        self.store.record_component("compiler", "sql_compiler", COMPILER_VERSION)

    def _ctx(self, run_id: str, step_id: str | None = None) -> AgentContext:
        return AgentContext(self.settings, self.ontology, self.vocab, self.store, self.llm, run_id, step_id)

    # ---- the agent graph ----------------------------------------------------
    def ask(self, query: str, user_id: str = "anonymous") -> RunResult:
        run_id = self.store.start_run(query, user_id)
        seq = 0

        def step(name: str, attempt: int, inputs: Any, fn: Callable[[AgentContext], Any],
                 summarize: Callable[[Any], Any] = lambda x: x) -> Any:
            nonlocal seq
            seq += 1
            step_id, started, t0 = str(uuid.uuid4()), now(), time.monotonic()
            try:
                out = fn(self._ctx(run_id, step_id))
                status = "ok"
                return out
            except Exception as exc:  # recorded, then re-raised
                out, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
                raise
            finally:
                payload = summarize(out) if status == "ok" else out
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump(mode="json")
                self.store.record_step(run_id, seq, attempt, name, inputs, payload, status, started,
                                       int((time.monotonic() - t0) * 1000), step_id=step_id)

        feedback: str | None = None
        concept_feedback: str | None = None
        resolved_cache: dict[str, ResolvedConceptSet] = {}
        rejected: set[str] = set()  # semantic hashes the critic already rejected
        result = RunResult(run_id=run_id, status="failed")
        ir: CohortDefinition | None = None
        attempt = 0
        try:
            for attempt in range(self.settings.max_retries + 1):
                intent: CohortIntent = step("intent_parser", attempt, {"query": query, "feedback": feedback},
                                            lambda c: parse_intent(c, query, feedback))
                resolved: dict[str, ResolvedConceptSet] = {}
                for m in intent.mentions:
                    sig = resolution_signature(m)
                    if sig in resolved_cache and concept_feedback is None:
                        resolved[m.key] = resolved_cache[sig]
                        continue
                    rcs = step("concept_resolver", attempt, {"mention": m.model_dump(), "feedback": concept_feedback},
                               lambda c, m=m: resolve_mention(c, m, query, concept_feedback))
                    resolved[m.key] = resolved_cache[sig] = rcs
                concept_feedback = None

                ir, issues = step("composer", attempt, {"mentions": sorted(resolved)},
                                  lambda c: compose(intent, resolved, self.ontology, self.vocab),
                                  lambda r: {"ir_hash": r[0].content_hash() if r[0] else None,
                                             "issues": [i.as_dict() for i in r[1]]})
                attrition = None
                if ir is not None:
                    v_issues, attrition = step("validator", attempt, {"ir_hash": ir.content_hash()},
                                               lambda c: validate(ir, self.ontology, self.vocab, self.executor),
                                               lambda r: {"issues": [i.as_dict() for i in r[0]],
                                                          "attrition": r[1].suppressed(self.executor.min_cell)
                                                          if r[1] else None})
                    issues = issues + v_issues
                result.issues = [i.as_dict() for i in issues]
                result.attrition = attrition.suppressed(self.executor.min_cell) if attrition else []
                errors = [i for i in issues if i.severity == "error"]
                if any(i.stage == "dataset" for i in errors):
                    # the data source cannot answer this request; rephrasing will not help
                    result.status = "needs_review"
                    break
                if errors:
                    feedback, concept_feedback = self._feedback(errors)
                    result.status = "needs_review"
                    continue

                if ir.semantic_hash() in rejected:
                    # the revision produced the same logic the critic already rejected
                    result.issues.append({"severity": "error", "stage": "intent", "source": "orchestrator",
                                          "message": "revision did not change the definition; human review needed"})
                    result.status = "needs_review"
                    break
                result.explanation = step("explainer", attempt, {"ir_hash": ir.content_hash()},
                                          lambda c: self.explainer.explain(ir))
                warnings = [i.as_dict() for i in issues if i.severity == "warning"]
                review = step("critic", attempt, {"ir_hash": ir.content_hash()},
                              lambda c: critique(c, query, result.explanation, result.attrition, warnings))
                result.critic_notes = review.notes
                if review.verdict == "pass":
                    result.status = "draft"
                    break
                crit_issues = [Issue("error", i.stage, i.message) for i in review.issues] or \
                              [Issue("error", "intent", review.notes or "critic requested revision")]
                result.issues += [{**i.as_dict(), "source": "critic"} for i in crit_issues]
                rejected.add(ir.semantic_hash())
                feedback, concept_feedback = self._feedback(crit_issues)
                result.status = "needs_review"
        except LLMError as exc:
            result.status = "failed"
            result.issues.append({"severity": "error", "stage": "llm", "message": str(exc)})

        if ir is None and result.status == "needs_review":
            result.status = "failed"  # nothing composable to hand to a reviewer
        if ir is not None and result.status in ("draft", "needs_review"):
            if not result.explanation:
                result.explanation = self.explainer.explain(ir)
            result.ir = ir
            result.cohort_definition_id = self.store.save_definition(
                ir, result.status, user_id, run_id=run_id,
                issues=[] if result.status == "draft" else result.issues, dataset=self.ontology.dataset_name)
        result.manifest = self._manifest(run_id, query, result, attempt)
        self.store.finish_run(run_id, result.status, attempt, result.cohort_definition_id, result.manifest)
        return result

    @staticmethod
    def _feedback(issues: list[Issue]) -> tuple[str, str | None]:
        text = "\n".join(f"- [{i.stage}] {i.message}" for i in issues)
        concept = "\n".join(f"- {i.message}" for i in issues if i.stage == "concepts") or None
        return text, concept

    def _manifest(self, run_id: str, query: str, result: RunResult, retries: int) -> dict:
        calls = self.con.execute(
            "SELECT count(*), count(*) FILTER (WHERE cache_hit), coalesce(sum(input_tokens),0), "
            "coalesce(sum(output_tokens),0) FROM meta.llm_call WHERE run_id=?", [run_id]).fetchone()
        tools = self.con.execute("SELECT count(*) FROM meta.tool_call WHERE run_id=?", [run_id]).fetchone()[0]
        ir = result.ir
        return {
            "run_id": run_id,
            "user_query": query,
            "status": result.status,
            "cohort_definition_id": result.cohort_definition_id,
            "ir_content_hash": ir.content_hash() if ir else None,
            "ir_semantic_hash": ir.semantic_hash() if ir else None,
            "llm_mode": self.settings.llm_mode,
            "retries": retries,
            "llm_calls": {"total": calls[0], "cache_hits": calls[1], "input_tokens": calls[2],
                          "output_tokens": calls[3]},
            "tool_calls": tools,
            "dry_run_count": result.attrition[-1]["remaining"] if result.attrition else None,
            **self.component_versions(),
        }

    # ---- human review & execution -------------------------------------------
    def submit_ir(self, ir: CohortDefinition, user_id: str, parent_id: int | None = None) -> tuple[int, list[dict]]:
        """Save a hand-written or edited IR (validated, starts as a draft)."""
        issues, _ = validate(ir, self.ontology, self.vocab, self.executor)
        errors = [i.as_dict() for i in issues if i.severity == "error"]
        status = "needs_review" if errors else "draft"
        def_id = self.store.save_definition(ir, status, user_id, parent_id=parent_id, issues=errors,
                                            dataset=self.ontology.dataset_name)
        return def_id, [i.as_dict() for i in issues]

    def load_definition(self, def_id: int) -> tuple[dict, CohortDefinition]:
        row = self.store.get_definition(def_id)
        if row is None:
            raise KeyError(f"cohort definition {def_id} not found")
        return row, CohortDefinition.model_validate(row["ir"])

    def review(self, def_id: int, reviewer: str, decision: str, comments: str = "") -> None:
        row, _ = self.load_definition(def_id)
        if decision == "approved" and row["status"] == "needs_review":
            raise ValueError("definition has unresolved errors; edit and resubmit it before approval")
        self.store.review(def_id, reviewer, decision, comments)

    def compile_sql(self, def_id: int) -> str:
        _, ir = self.load_definition(def_id)
        return self.compiler.compile(ir).cohort_sql

    def execute(self, def_id: int, user_id: str, allow_draft: bool = False) -> dict:
        row, ir = self.load_definition(def_id)
        if row["status"] != "approved" and not (allow_draft and row["status"] == "draft"):
            raise PermissionError(f"definition {def_id} is {row['status']}; approve it before execution")
        if row.get("dataset") != self.ontology.dataset_name:
            # same logic on another data source: re-check what this source can answer
            issues, _ = validate(ir, self.ontology, self.vocab)
            errors = [i.message for i in issues if i.severity == "error"]
            if errors:
                raise ValueError(f"definition {def_id} (built on {row.get('dataset')!r}) cannot run on dataset "
                                 f"{self.ontology.dataset_name!r}: " + "; ".join(errors))
        compiled = self.compiler.compile(ir)
        generation_id, attrition = self.executor.generate(compiled, def_id)
        self.store.record_generation(generation_id, def_id, self.data_snapshot(), compiled.compiler_version,
                                     compiled.sql_hash, attrition.final_count, user_id, self.ontology.dataset_name)
        return {"generation_id": generation_id, "cohort_definition_id": def_id, "dataset": self.ontology.dataset_name,
                "person_count": attrition.final_count, "sql_hash": compiled.sql_hash,
                "attrition": attrition.suppressed(self.executor.min_cell)}

    # ---- reproducibility ----------------------------------------------------
    def replay(self, run_id: str) -> dict:
        """Re-run a past request using only recorded LLM responses and compare outputs."""
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        replay_builder = CohortBuilder(dataclasses.replace(self.settings, llm_mode="replay"),
                                       backend=self.llm._backend, con=self.con)
        new = replay_builder.ask(run["user_query"], user_id="replay")
        original_hash = (run["manifest"] or {}).get("ir_semantic_hash")
        new_hash = new.manifest.get("ir_semantic_hash")
        return {"original_run_id": run_id, "replay_run_id": new.run_id, "replay_status": new.status,
                "original_semantic_hash": original_hash, "replay_semantic_hash": new_hash,
                "identical": original_hash is not None and original_hash == new_hash,
                "issues": new.issues}
