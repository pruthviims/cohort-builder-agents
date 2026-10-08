"""Vocabulary tools: the only way agents learn about concepts.

Every function returns deterministic, sorted results so repeated runs show the
LLM identical tool outputs. Search is lexical (exact / synonym / code /
Jaro-Winkler). An embedding searcher can be plugged in via `ConceptSearcher`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import duckdb

CONCEPT_COLS = (
    "c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, c.concept_class_id, "
    "c.standard_concept, c.concept_code, c.invalid_reason"
)


def _rows(cur: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


class ConceptSearcher(Protocol):
    def search(self, query: str, domain: str | None, standard_only: bool, limit: int) -> list[dict]: ...


class LexicalSearcher:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def search(self, query: str, domain: str | None = None, standard_only: bool = True, limit: int = 10) -> list[dict]:
        q = " ".join(query.lower().split())
        sql = f"""
        WITH names AS (
            SELECT concept_id, lower(concept_name) AS name, 'name' AS matched_on FROM vocab.concept
            UNION ALL
            SELECT concept_id, lower(concept_synonym_name), 'synonym' FROM vocab.concept_synonym
            UNION ALL
            SELECT concept_id, lower(concept_code), 'code' FROM vocab.concept
        ), scored AS (
            SELECT concept_id, matched_on,
                   CASE WHEN name = $q THEN 1.0
                        WHEN matched_on <> 'code' AND contains(name, $q)
                             THEN 0.80 + 0.15 * length($q) / greatest(length(name), 1)
                        WHEN matched_on <> 'code' THEN 0.85 * jaro_winkler_similarity(name, $q)
                        ELSE 0 END AS score
            FROM names
        ), best AS (
            SELECT concept_id, max(score) AS score,
                   arg_max(matched_on, score) AS matched_on
            FROM scored GROUP BY concept_id
        )
        SELECT {CONCEPT_COLS}, round(b.score, 3) AS score, b.matched_on
        FROM best b JOIN vocab.concept c USING (concept_id)
        WHERE b.score >= 0.70
          AND c.invalid_reason IS NULL
          AND ($domain IS NULL OR c.domain_id = $domain)
          AND (NOT $standard_only OR c.standard_concept IN ('S', 'C'))
        ORDER BY b.score DESC, c.concept_id
        LIMIT $limit
        """
        cur = self.con.execute(sql, {"q": q, "domain": domain, "standard_only": standard_only, "limit": limit})
        return _rows(cur)


class Vocabulary:
    def __init__(self, con: duckdb.DuckDBPyConnection, searcher: ConceptSearcher | None = None):
        self.con = con
        self.searcher = searcher or LexicalSearcher(con)

    # ---- agent tools ------------------------------------------------------
    def search_concepts(
        self, query: str, domain: str | None = None, standard_only: bool = True, limit: int = 10
    ) -> list[dict]:
        return self.searcher.search(query, domain, standard_only, min(limit, 25))

    def get_concept(self, concept_id: int) -> dict | None:
        rows = _rows(self.con.execute(f"SELECT {CONCEPT_COLS} FROM vocab.concept c WHERE concept_id = ?", [concept_id]))
        if not rows:
            return None
        concept = rows[0]
        concept["parents"] = _rows(
            self.con.execute(
                f"""
            SELECT {CONCEPT_COLS} FROM vocab.concept_ancestor a JOIN vocab.concept c
              ON c.concept_id = a.ancestor_concept_id
            WHERE a.descendant_concept_id = ? AND a.min_levels_of_separation = 1
            ORDER BY c.concept_id""",
                [concept_id],
            )
        )
        concept["descendant_count"] = self.con.execute(
            "SELECT count(*) - 1 FROM vocab.concept_ancestor WHERE ancestor_concept_id = ?", [concept_id]
        ).fetchone()[0]
        if concept["standard_concept"] is None:
            concept["maps_to"] = self.map_to_standard(concept_id)
        return concept

    def get_descendants(self, concept_id: int, limit: int = 25) -> list[dict]:
        return _rows(
            self.con.execute(
                f"""
            SELECT {CONCEPT_COLS}, a.min_levels_of_separation AS level
            FROM vocab.concept_ancestor a JOIN vocab.concept c ON c.concept_id = a.descendant_concept_id
            WHERE a.ancestor_concept_id = ? AND a.descendant_concept_id <> a.ancestor_concept_id
            ORDER BY a.min_levels_of_separation, c.concept_id LIMIT ?""",
                [concept_id, min(limit, 100)],
            )
        )

    def map_to_standard(self, concept_id: int) -> list[dict]:
        return _rows(
            self.con.execute(
                f"""
            SELECT {CONCEPT_COLS} FROM vocab.concept_relationship r JOIN vocab.concept c
              ON c.concept_id = r.concept_id_2
            WHERE r.concept_id_1 = ? AND r.relationship_id = 'Maps to' AND r.invalid_reason IS NULL
            ORDER BY c.concept_id""",
                [concept_id],
            )
        )

    def lookup_code(self, code: str, vocabulary_id: str | None = None) -> list[dict]:
        rows = _rows(
            self.con.execute(
                f"""
            SELECT {CONCEPT_COLS} FROM vocab.concept c
            WHERE upper(c.concept_code) = upper(?) AND (? IS NULL OR c.vocabulary_id = ?)
            ORDER BY c.concept_id""",
                [code, vocabulary_id, vocabulary_id],
            )
        )
        for r in rows:
            if r["standard_concept"] is None:
                r["maps_to"] = self.map_to_standard(r["concept_id"])
        return rows

    # ---- used by validator / compiler / eval -----------------------------
    def concepts(self, concept_ids: list[int]) -> dict[int, dict]:
        if not concept_ids:
            return {}
        rows = _rows(
            self.con.execute(
                f"SELECT {CONCEPT_COLS} FROM vocab.concept c WHERE concept_id IN (SELECT unnest(?))",
                [sorted(set(concept_ids))],
            )
        )
        return {r["concept_id"]: r for r in rows}

    def is_descendant_or_self(self, concept_id: int, ancestor_id: int) -> bool:
        return (
            self.con.execute(
                "SELECT count(*) FROM vocab.concept_ancestor WHERE ancestor_concept_id=? AND descendant_concept_id=?",
                [ancestor_id, concept_id],
            ).fetchone()[0]
            > 0
        )

    def expand(self, items: list[dict]) -> set[int]:
        """Expand concept-set items (concept_id, include_descendants, is_excluded) to concept IDs."""

        def ids(selected: list[dict]) -> set[int]:
            out: set[int] = set()
            for it in selected:
                out.add(int(it["concept_id"]))
                if it.get("include_descendants"):
                    out |= {
                        r[0]
                        for r in self.con.execute(
                            "SELECT descendant_concept_id FROM vocab.concept_ancestor WHERE ancestor_concept_id=?",
                            [it["concept_id"]],
                        ).fetchall()
                    }
            return out

        incl = ids([i for i in items if not i.get("is_excluded")])
        excl = ids([i for i in items if i.get("is_excluded")])
        return incl - excl

    def version(self) -> str:
        rows = self.con.execute("SELECT DISTINCT vocabulary_version FROM vocab.vocabulary ORDER BY 1").fetchall()
        label = ";".join(r[0] for r in rows if r[0]) or "unknown"
        n, mx = self.con.execute("SELECT count(*), coalesce(max(concept_id),0) FROM vocab.concept").fetchone()
        fp = hashlib.sha256(f"{label}|{n}|{mx}".encode()).hexdigest()[:12]
        return f"{label} ({n} concepts, fp {fp})"
