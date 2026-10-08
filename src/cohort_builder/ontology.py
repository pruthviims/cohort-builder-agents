"""Loads the semantic ontology (YAML in git) and exposes it to agents, validator and compiler."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


def _validate_profile(profile: dict[str, Any], where: str) -> None:
    """Table/column names from a profile are interpolated into SQL: allow plain identifiers only."""
    mapping = profile.get("mapping") or {}
    for key in ("name",):
        if not _IDENT.match(str(profile.get(key, ""))):
            raise ValueError(f"{where}: invalid profile {key}")
    if not _IDENT.match(str(mapping.get("vocab_schema", "vocab"))):
        raise ValueError(f"{where}: invalid vocab_schema")
    sections = {k: v for k, v in mapping.items() if k in ("person", "observation_period") and v}
    sections.update({f"entities.{k}": v for k, v in (mapping.get("entities") or {}).items()})
    for sec, cols in sections.items():
        for k, v in cols.items():
            ok = _TABLE.match(str(v)) if k == "table" else _IDENT.match(str(v))
            if not ok:
                raise ValueError(f"{where}: mapping {sec}.{k} = {v!r} is not a plain SQL identifier")


@dataclass
class Ontology:
    version: str
    content_hash: str
    domain: dict[str, Any]
    dataset: dict[str, Any]  # active dataset profile (capabilities, mapping, semantic views)
    curated: dict[str, Any]
    units: dict[int, dict[str, Any]]
    analytes: dict[int, dict[str, Any]]

    @classmethod
    def load(cls, ontology_dir: Path, dataset: str | None = None) -> "Ontology":
        ontology_dir = Path(ontology_dir)
        domain_path = ontology_dir / "domain.yaml"
        domain = yaml.safe_load(domain_path.read_text())
        dataset = dataset or domain["default_dataset"]
        dataset_path = ontology_dir / domain["datasets_dir"] / f"{dataset}.yaml"
        if not dataset_path.exists():
            available = sorted(p.stem for p in (ontology_dir / domain["datasets_dir"]).glob("*.yaml"))
            raise ValueError(f"unknown dataset {dataset!r}; available: {available}")
        files = [
            domain_path,
            dataset_path,
            ontology_dir / domain["curated_concept_sets_file"],
            ontology_dir / domain["unit_conversions_file"],
        ]
        digest = hashlib.sha256()
        for f in files:
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
        profile = yaml.safe_load(dataset_path.read_text())
        _validate_profile(profile, dataset_path.name)
        curated = yaml.safe_load(files[2].read_text())["concept_sets"]
        units_doc = yaml.safe_load(files[3].read_text())
        return cls(
            version=str(domain["version"]),
            content_hash="sha256:" + digest.hexdigest(),
            domain=domain,
            dataset=profile,
            curated=curated,
            units={int(k): v for k, v in units_doc["units"].items()},
            analytes={int(k): v for k, v in units_doc["analytes"].items()},
        )

    # ---- dataset profile ---------------------------------------------------
    @property
    def dataset_name(self) -> str:
        return self.dataset["name"]

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.dataset["capabilities"]

    @property
    def mappings(self) -> dict[str, Any]:
        return self.dataset["mapping"]

    def supports_entity(self, entity: str) -> bool:
        return entity in self.capabilities["entities"]

    def supports_attribute(self, attribute: str) -> bool:
        return attribute in self.capabilities.get("attributes", [])

    def default_claim_status(self) -> list[str] | None:
        return self.capabilities.get("default_claim_status")

    def setup_statements(self) -> list[str]:
        params = self.dataset.get("params", {})
        out = []
        for stmt in self.dataset.get("setup_sql", []):
            for k, v in params.items():
                stmt = stmt.replace("{" + k + "}", str(v))
            out.append(stmt)
        return out

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

    def vocab_schema(self) -> str:
        return self.mappings.get("vocab_schema", "vocab")

    def claim_status_values(self) -> list[str]:
        return self.domain["attributes"]["DrugExposure.claim_status"]["values"]

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
                "dataset": self.dataset_summary(),
            },
            sort_keys=True,
        )

    def dataset_summary(self) -> dict[str, Any]:
        caps = self.capabilities
        return {
            "name": self.dataset_name,
            "version": str(self.dataset.get("version", "")),
            "description": self.dataset.get("description", ""),
            "data_type": self.dataset.get("data_type"),
            "available_entities": caps["entities"],
            "unavailable_entities": sorted(set(self.entities) - set(caps["entities"])),
            "available_attributes": caps.get("attributes", []),
            "observation": caps.get("observation"),
            "default_claim_status": caps.get("default_claim_status"),
            "notes": caps.get("notes", []),
        }
