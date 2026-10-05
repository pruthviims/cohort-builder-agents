"""Agents. Only three steps use an LLM (intent parser, concept resolver, critic);
composition, validation and explanation are deterministic code."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..llm import LLMClient, Prompt
from ..metadata import MetadataStore
from ..ontology import Ontology
from ..vocab import Vocabulary


@dataclass
class AgentContext:
    settings: Settings
    ontology: Ontology
    vocab: Vocabulary
    store: MetadataStore
    llm: LLMClient
    run_id: str | None = None
    step_id: str | None = None

    def prompt(self, name: str) -> Prompt:
        return Prompt.load(self.settings.prompts_dir, name, self.settings.prompt_versions[name])
