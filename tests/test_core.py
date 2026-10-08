"""IR hashing, vocabulary tools, compiler correctness and validator rules (no LLM)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from cohort_builder.agents.validator import validate
from cohort_builder.compiler import Compiler
from cohort_builder.ir import CohortDefinition


def load(path) -> CohortDefinition:
    return CohortDefinition.model_validate_json(path.read_text())


# ---- IR ----------------------------------------------------------------------
def test_semantic_hash_ignores_labels_ids_and_order(example_ir_path):
    a = load(example_ir_path)
    data = a.model_dump()
    data["name"] = "renamed"
    data["assumptions"] = []
    for cs in data["concept_sets"]:
        cs["id"] = cs["id"].replace("cs_", "set_")
    data["index_event"]["concept_set_id"] = data["index_event"]["concept_set_id"].replace("cs_", "set_")
    for c in data["inclusion"] + data["exclusion"]:
        c["concept_set_id"] = c["concept_set_id"].replace("cs_", "set_")
        c["name"] = c["name"].upper()
    data["inclusion"].reverse()
    data["concept_sets"].reverse()
    b = CohortDefinition.model_validate(data)
    assert a.semantic_hash() == b.semantic_hash()
    assert a.content_hash() != b.content_hash()


def test_semantic_hash_changes_with_logic(example_ir_path):
    a = load(example_ir_path)
    data = a.model_dump()
    data["inclusion"][1]["value_filter"]["value"] = 9.0
    assert CohortDefinition.model_validate(data).semantic_hash() != a.semantic_hash()


# ---- vocabulary tools -------------------------------------------------------------
def test_search_and_mapping(builder):
    v = builder.vocab
    assert v.search_concepts("type 2 diabetes", "Condition")[0]["concept_id"] == 201826
    assert v.search_concepts("HbA1c", "Measurement")[0]["concept_id"] == 3004410
    codes = v.lookup_code("E11.9")
    assert codes[0]["maps_to"][0]["concept_id"] == 201826
    assert {d["concept_id"] for d in v.get_descendants(201826)} == {2000003001, 2000003002}
    # deprecated / non-standard concepts are not returned by default search
    assert all(r["standard_concept"] in ("S", "C") for r in v.search_concepts("diabetes"))
    assert v.expand(
        [
            {"concept_id": 201820, "include_descendants": True},
            {"concept_id": 201254, "include_descendants": True, "is_excluded": True},
        ]
    ) == {201820, 201826, 2000003001, 2000003002}


def test_search_is_deterministic(builder):
    assert builder.vocab.search_concepts("diabetes") == builder.vocab.search_concepts("diabetes")


# ---- compiler -----------------------------------------------------------------------
def test_compiler_is_deterministic(builder, example_ir_path):
    ir = load(example_ir_path)
    c1, c2 = Compiler(builder.ontology).compile(ir), Compiler(builder.ontology).compile(ir)
    assert c1.cohort_sql == c2.cohort_sql and c1.sql_hash == c2.sql_hash


def _reference_cohort(con, ir_vocab) -> set[int]:
    """Independent pure-Python implementation of the example cohort, to check the compiled SQL."""
    metformin = ir_vocab.expand([{"concept_id": 1503297, "include_descendants": True}])
    t2dm = ir_vocab.expand([{"concept_id": 201826, "include_descendants": True}])
    t1dm = ir_vocab.expand([{"concept_id": 201254, "include_descendants": True}])
    person = {p: y for p, y in con.execute("SELECT person_id, year_of_birth FROM cdm.person").fetchall()}
    op = defaultdict(list)
    for p, s, e in con.execute(
        "SELECT person_id, observation_period_start_date, observation_period_end_date FROM cdm.observation_period"
    ).fetchall():
        op[p].append((s, e))
    first_met: dict[int, date] = {}
    for p, cid, d in con.execute(
        "SELECT person_id, drug_concept_id, drug_exposure_start_date FROM cdm.drug_exposure"
    ).fetchall():
        if cid in metformin and (p not in first_met or d < first_met[p]):
            first_met[p] = d
    conds = defaultdict(list)
    for p, cid, d in con.execute(
        "SELECT person_id, condition_concept_id, condition_start_date FROM cdm.condition_occurrence"
    ).fetchall():
        conds[p].append((cid, d))
    a1c = defaultdict(list)
    for p, d, v, u in con.execute(
        "SELECT person_id, measurement_date, value_as_number, unit_concept_id "
        "FROM cdm.measurement WHERE measurement_concept_id = 3004410"
    ).fetchall():
        norm = v if u == 8554 else (v * 0.09148 + 2.152 if u == 2000001001 else None)
        a1c[p].append((d, norm))

    out = set()
    for p, idx in first_met.items():
        period = next(((s, e) for s, e in op[p] if s <= idx <= e), None)
        if period is None or (idx - period[0]).days < 365 or idx.year - person[p] < 18:
            continue
        s = period[0]
        if not any(c in t2dm and s <= d <= idx for c, d in conds[p]):
            continue
        if not any((idx - d).days <= 365 and d <= idx and v is not None and v > 8 for d, v in a1c[p]):
            continue
        if any(c in t1dm and s <= d <= idx for c, d in conds[p]):
            continue
        out.add(p)
    return out


def test_compiled_sql_matches_reference_implementation(builder, example_ir_path):
    ir = load(example_ir_path)
    got = builder.executor.person_ids(builder.compiler.compile(ir))
    expected = _reference_cohort(builder.con, builder.vocab)
    assert len(expected) > 50
    assert got == expected


def test_unit_conversion_is_applied(builder, example_ir_path):
    """People whose only qualifying HbA1c was recorded in mmol/mol must still be included."""
    ir = load(example_ir_path)
    people = builder.executor.person_ids(builder.compiler.compile(ir))
    only_mmol = builder.con.execute("""
        SELECT person_id FROM cdm.measurement WHERE measurement_concept_id = 3004410
        GROUP BY person_id
        HAVING bool_and(unit_concept_id = 2000001001) AND max(value_as_number) > 64""").fetchall()
    assert {r[0] for r in only_mmol} & people


def test_attrition_is_monotonic(builder, example_ir_path):
    a = builder.executor.attrition(builder.compiler.compile(load(example_ir_path)))
    counts = [r["remaining"] for r in a.rules]
    assert counts == sorted(counts, reverse=True) and counts[-1] > 0


# ---- validator ------------------------------------------------------------------------
def test_validator_accepts_example(builder, example_ir_path):
    issues, attrition = validate(load(example_ir_path), builder.ontology, builder.vocab, builder.executor)
    assert not [i for i in issues if i.severity == "error"]
    assert attrition.final_count > 0


def test_validator_rejects_bad_concepts_and_values(builder, example_ir_path):
    data = load(example_ir_path).model_dump()
    data["concept_sets"][1]["items"][0]["concept_id"] = 2000003099  # deprecated concept
    data["concept_sets"][0]["items"].append({"concept_id": 201826})  # condition inside a drug set
    data["exclusion"][0]["value_filter"] = {"op": ">", "value": 1, "unit_concept_id": 8554}  # value on condition
    issues, _ = validate(CohortDefinition.model_validate(data), builder.ontology, builder.vocab, builder.executor)
    messages = " | ".join(i.message for i in issues if i.severity == "error")
    assert "deprecated" in messages
    assert "domain Condition != concept set domain Drug" in messages
    assert "value filters are only allowed" in messages
