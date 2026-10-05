"""Loads the semantic ontology (YAML in git) and exposes it to agents, validator and compiler."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


@dataclass
class Ontology:
    version: str
    content_hash: str
    domain: dict[str, Any]
    mappings: dict[str, Any]
    curated: dict[str, Any]
    units: dict[int, dict[str, Any]]
    analytes: dict[int, dict[str, Any]]

    @classmethod
    def load(cls, ontology_dir: Path) -> "Ontology":
        ontology_dir = Path(ontology_dir)
        domain_path = ontology_dir / "domain.yaml"
        domain = yaml.safe_load(domain_path.read_text())
        files = [
            domain_path,
            ontology_dir / domain["mappings_file"],
            ontology_dir / domain["curated_concept_sets_file"],
            ontology_dir / domain["unit_conversions_file"],
        ]
        digest = hashlib.sha256()
        for f in files:
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
        mappings = yaml.safe_load(files[1].read_text())
        curated = yaml.safe_load(files[2].read_text())["concept_sets"]
        units_doc = yaml.safe_load(files[3].read_text())
        return cls(
            version=str(domain["version"]),
            content_hash="sha256:" + digest.hexdigest(),
            domain=domain,
            mappings=mappings,
            curated=curated,
            units={int(k): v for k, v in units_doc["units"].items()},
            analytes={int(k): v for k, v in units_doc["analytes"].items()},
        )

    # ---- entities & rules -------------------------------------------------
    @property
    def entities(self) -> dict[str, Any]:
        return self.domain["entities"]

    @property
    def rules(self) -> dict[str, Any]:
        return self.domain["cohort_rules"]

    def entity_domain(self, entity: str) -> str:
        return self.entities[entity]["domain_id"]

    def entity_has_value(self, entity: str) -> bool:
        return bool(self.entities[entity].get("has_value"))

    def table_mapping(self, entity: str) -> dict[str, Any]:
        return self.mappings["entities"][entity]

    def schema(self, name: str) -> str:
        return self.mappings["schemas"][name]

    def value_operators(self) -> list[str]:
        return self.domain["attributes"]["Measurement.value"]["operators"]

    # ---- units ------------------------------------------------------------
    def unit_symbol(self, unit_concept_id: int | None) -> str:
        if unit_concept_id is None:
            return ""
        unit = self.units.get(int(unit_concept_id))
        return unit["symbol"] if unit else f"unit {unit_concept_id}"

    def find_unit(self, text: str) -> int | None:
        t = _norm(text).replace(" ", "")
        for uid, u in sorted(self.units.items()):
            if t in (_norm(u["symbol"]).replace(" ", ""), _norm(u["name"]).replace(" ", "")):
                return uid
        aliases = {"percent": 8554, "pct": 8554, "%": 8554}
        return aliases.get(t)

    # ---- curated concept sets --------------------------------------------
    def search_curated(self, query: str, domain: str | None = None, limit: int = 5) -> list[dict]:
        q = _norm(query)
        results = []
        for key, cs in sorted(self.curated.items()):
            if cs.get("status") != "approved":
                continue
            if domain and cs["domain"] != domain:
                continue
            names = [cs["label"], key, *cs.get("synonyms", [])]
            score = 0.0
            for name in names:
                n = _norm(name)
                if n == q:
                    s = 1.0
                elif n in q or q in n:
                    s = 0.85
                else:
                    s = SequenceMatcher(None, n, q).ratio() * 0.8
                score = max(score, s)
            if score >= 0.5:
                results.append(
                    {
                        "curated_key": key,
                        "label": cs["label"],
                        "domain": cs["domain"],
                        "version": cs["version"],
                        "therapeutic_area": cs.get("therapeutic_area"),
                        "items": cs["items"],
                        "score": round(score, 3),
                    }
                )
        results.sort(key=lambda r: (-r["score"], r["curated_key"]))
        return results[:limit]

    def summary_for_prompt(self) -> str:
        """Compact, deterministic description of the ontology for agent prompts."""
        return yaml.safe_dump(
            {
                "ontology_version": self.version,
                "entities": {k: v["description"] for k, v in self.entities.items()},
                "attributes": self.domain["attributes"],
                "temporal": self.domain["temporal"],
                "cohort_rules": {
                    k: v
                    for k, v in self.rules.items()
                    if k in ("default_prior_observation_days", "exit_types", "max_inclusion_criteria")
                },
                "units": {uid: u["symbol"] for uid, u in sorted(self.units.items())},
            },
            sort_keys=True,
        )
