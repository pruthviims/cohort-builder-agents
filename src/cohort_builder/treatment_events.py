"""Canonical treatment events (schema `treatment_event/2`) from configured medication sources.

One canonical event = one source record x one ingredient of the drug it names. Every source record of a
configured source produces at least one row in `results.treatment_event` - included or excluded with a
single, ordered exclusion reason - so the quality summary accounts for every record.

Exclusion reasons, checked in this order (the first that applies is recorded):
  missing_record_id, missing_person, missing_start_date,
  implausible_start_date      before 1900 or after the run's recorded reference date
  reversal_transaction        the reversal row itself
  rejected_claim
  reversed_claim              a paid or adjustment claim that a reversal references
  superseded_by_adjustment    a paid or adjustment claim that a later adjustment references (in a chain
                              C <- A1 <- A2 only A2 counts)
  claim_status_excluded, evidence_type_excluded, unmapped_code,
  not_a_drug                  the code maps to a non-drug concept (a CPT procedure, a diagnosis)
  no_ingredient, not_in_drug_group,
  ambiguous_drug_group        the ingredient is in more than one group
  missing_days_supply         no usable duration and no assumption configured
  duplicate_record            the same record id repeated (deterministic tie-break on every field)
The episode engine then adds outside_observation and duplicate_event (see treatment_episodes.py).

Duration: a valid recorded days supply wins (end = start + days - 1, end_date_source 'days_supply'); else
a recorded end date on/after the start ('recorded'); else the rule set's assumed days for the evidence type
('assumed', evidence_status 'inferred'); else none. Nothing here asserts that a drug was taken: the evidence
type says what the record shows (dispensing, administration, prescription, unknown).
"""

from __future__ import annotations

from datetime import date

from .csv_ingest import qi
from .dataset_mapping import normalize_sql
from .treatment_config import TreatmentRuleSet, TreatmentSource

EVENT_COLUMNS = [
    "run_id",
    "event_id",
    "person_id",
    "dataset",
    "source_name",
    "source_table",
    "source_record_id",
    "source_code",
    "source_vocabulary",
    "product_concept_id",
    "ingredient_concept_id",
    "brand_concept_id",
    "drug_group",
    "n_ingredients",
    "evidence_type",
    "claim_status",
    "start_date",
    "start_date_source",
    "end_date",
    "end_date_source",
    "days_supply",
    "days_supply_source",
    "quantity",
    "evidence_status",
    "quality_flags",
    "included",
    "exclusion_reason",
]

EXCLUSION_REASONS = [
    "missing_record_id",
    "missing_person",
    "missing_start_date",
    "implausible_start_date",
    "reversal_transaction",
    "rejected_claim",
    "reversed_claim",
    "superseded_by_adjustment",
    "claim_status_excluded",
    "evidence_type_excluded",
    "unmapped_code",
    "not_a_drug",
    "no_ingredient",
    "not_in_drug_group",
    "ambiguous_drug_group",
    "missing_days_supply",
    "duplicate_record",
    # decided by the episode engine (deterministic, rule-driven)
    "outside_observation",
    "duplicate_event",
]

LOOKUP = "tx_code_lookup"
GROUPS = "tx_group_member"


def _lit(v: str) -> str:
    """Literal for an already-validated enumerated / pattern-checked value."""
    if "'" in v or "\\" in v:
        raise ValueError("unsafe literal")
    return f"'{v}'"


def _table(name: str) -> str:
    schema, table = name.split(".")
    return f"{qi(schema)}.{qi(table)}"


def group_member_sql(rules: TreatmentRuleSet, vocab: str) -> str:
    """Ingredients of each drug group: the listed concepts and their descendants of class Ingredient."""
    pairs = ", ".join(f"({_lit(g)}, {int(c)})" for g, grp in sorted(rules.drug_groups.items()) for c in grp.concepts)
    return (
        f"CREATE OR REPLACE TEMP TABLE {GROUPS} AS SELECT DISTINCT gc.g AS drug_group, i.concept_id AS "
        f"ingredient_concept_id FROM (VALUES {pairs}) gc(g, c) JOIN {vocab}.concept i ON i.concept_class_id = "
        f"'Ingredient' AND (i.concept_id = gc.c OR EXISTS (SELECT 1 FROM {vocab}.concept_ancestor a WHERE "
        "a.ancestor_concept_id = gc.c AND a.descendant_concept_id = i.concept_id))"
    )


def source_event_sql(
    src: TreatmentSource,
    rules: TreatmentRuleSet,
    run_id: str,
    dataset: str,
    vocab: str,
    reference_date: date,
) -> str:
    """SELECT producing canonical event rows (EVENT_COLUMNS) for every record of one source. Dates after
    the run's recorded `reference_date` are implausible (never the moving SQL current_date)."""
    t = "r"
    ref_date = f"DATE '{date.fromisoformat(reference_date.isoformat()).isoformat()}'"

    def c(col: str) -> str:
        return f"{t}.{qi(col)}"

    rec = f"CAST({c(src.record_id_column)} AS VARCHAR)"
    by_person = bool(src.status and src.status.reference_scope == "record_and_person")
    ref_person = ", person_id" if by_person else ""
    ref_person_join_rv = " AND rv.person_id = b.person_id" if by_person else ""
    ref_person_join_aj = " AND aj.person_id = b.person_id" if by_person else ""
    # claim status -------------------------------------------------------------------------------
    if src.status:
        st = src.status
        cases = " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in sorted(st.values.items()))
        raw_cat = f"CASE CAST({c(st.column)} AS VARCHAR) {cases} ELSE 'unknown' END"
        ref = f"CAST({c(st.reference_column)} AS VARCHAR)" if st.reference_column else "CAST(NULL AS VARCHAR)"
    else:
        raw_cat, ref = "'not_applicable'", "CAST(NULL AS VARCHAR)"
    payer = f"CAST({c(src.payer_column)} AS VARCHAR)" if src.payer_column else "CAST(NULL AS VARCHAR)"
    # evidence type -------------------------------------------------------------------------------
    if src.evidence_type:
        evidence = _lit(src.evidence_type)
    else:
        cases = " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in sorted((src.evidence_type_values or {}).items()))
        evidence = f"CASE CAST({c(src.evidence_type_column or '')} AS VARCHAR) {cases} ELSE 'unknown' END"
    # drug concept --------------------------------------------------------------------------------
    if src.concept_column:
        code = f"CAST({c(src.concept_column)} AS VARCHAR)"
        vocab_expr = "'concept_id'"
        product_join = ""
        product = f"nullif(TRY_CAST({c(src.concept_column)} AS BIGINT), 0)"
    else:
        raw_code = c(src.code_column or "")
        code = f"CAST({raw_code} AS VARCHAR)"
        if src.vocabulary:
            vocab_expr = _lit(src.vocabulary)
            strategy = src.vocabularies()[0][1]
            strat_expr = _lit(strategy)
            key = normalize_sql(raw_code, strategy)
        else:
            tv = sorted((src.code_type_values or {}).items())
            norm = dict(src.vocabularies())
            ct = f"CAST({c(src.code_type_column or '')} AS VARCHAR)"
            vocab_expr = f"CASE {ct} " + " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in tv) + " END"
            strat_expr = f"CASE {ct} " + " ".join(f"WHEN {_lit(k)} THEN {_lit(norm[v])}" for k, v in tv) + " END"
            key = f"CASE {ct} " + " ".join(f"WHEN {_lit(k)} THEN {normalize_sql(raw_code, norm[v])}" for k, v in tv)
            key += " END"
        product_join = (
            f" LEFT JOIN {LOOKUP} cl ON cl.vocabulary_id = {vocab_expr} AND cl.strategy = {strat_expr} AND "
            f"cl.code_key = {key} AND (NOT cl.ambiguous OR cl.concept_code = trim({code}))"
        )
        product = "cl.concept_id"
    # duration ------------------------------------------------------------------------------------
    ds = rules.days_supply
    ds_raw = f"TRY_CAST({c(src.days_supply_column)} AS INTEGER)" if src.days_supply_column else "CAST(NULL AS INTEGER)"
    ds_ok = f"(b.ds_raw BETWEEN {int(ds.min_valid)} AND {int(ds.max_valid)})"
    if ds.missing == "assume" and ds.assume_days:
        assumed = (
            "CASE evidence_type "
            + " ".join(f"WHEN {_lit(k)} THEN {int(v)}" for k, v in sorted(ds.assume_days.items()))
            + " END"
        )
    else:
        assumed = "CAST(NULL AS INTEGER)"
    end_raw = f"TRY_CAST({c(src.end_column)} AS DATE)" if src.end_column else "CAST(NULL AS DATE)"
    qty = f"TRY_CAST({c(src.quantity_column)} AS DOUBLE)" if src.quantity_column else "CAST(NULL AS DOUBLE)"
    include_status = set(rules.include_claim_status) | ({"adjusted"} if "paid" in rules.include_claim_status else set())
    status_list = ", ".join(_lit(x) for x in sorted(include_status))
    evidence_list = ", ".join(_lit(x) for x in sorted(rules.include_evidence_types))
    source_lit = _lit(src.name)
    table_lit = _lit(src.table)
    return f"""
WITH base AS (
  SELECT TRY_CAST({c(src.person_column)} AS BIGINT) AS person_id, {rec} AS record_id,
         TRY_CAST({c(src.start_column)} AS DATE) AS start_date, {end_raw} AS end_raw, {ds_raw} AS ds_raw,
         {qty} AS quantity, {raw_cat} AS status_cat, {ref} AS ref_id, {payer} AS payer, {evidence} AS evidence_type,
         {code} AS source_code, {vocab_expr} AS source_vocabulary, {product} AS product_concept_id
  FROM {_table(src.table)} {t}{product_join}),
rev AS (SELECT DISTINCT ref_id{ref_person} FROM base WHERE status_cat = 'reversal' AND ref_id IS NOT NULL),
adj AS (SELECT DISTINCT ref_id{ref_person} FROM base WHERE status_cat = 'adjustment' AND ref_id IS NOT NULL),
orig AS (SELECT record_id, min(person_id) AS person_id, min(start_date) AS start_date, min(payer) AS payer,
                count(DISTINCT payer) AS n_payers FROM base WHERE record_id IS NOT NULL GROUP BY record_id),
var AS (SELECT record_id, count(DISTINCT row(person_id, start_date, source_code, ds_raw, end_raw, quantity,
                                             status_cat, ref_id, payer)) AS variants
        FROM base WHERE record_id IS NOT NULL GROUP BY record_id),
ing AS (
  SELECT p.concept_id AS product_concept_id, i.concept_id AS ingredient_concept_id
  FROM {vocab}.concept p JOIN {vocab}.concept i ON i.concept_class_id = 'Ingredient'
   AND (i.concept_id = p.concept_id OR EXISTS (SELECT 1 FROM {vocab}.concept_ancestor a
        WHERE a.ancestor_concept_id = i.concept_id AND a.descendant_concept_id = p.concept_id))
  WHERE p.concept_id IN (SELECT product_concept_id FROM base)),
ning AS (SELECT product_concept_id, count(*) AS n FROM ing GROUP BY 1),
brand AS (
  SELECT r.concept_id_1 AS product_concept_id, min(r.concept_id_2) AS brand_concept_id
  FROM {vocab}.concept_relationship r JOIN {vocab}.concept b ON b.concept_id = r.concept_id_2
  WHERE r.relationship_id = 'Has brand name' AND r.invalid_reason IS NULL GROUP BY 1),
grp AS (SELECT ingredient_concept_id, min(drug_group) AS drug_group, count(*) AS n_groups FROM {GROUPS} GROUP BY 1),
ev AS (
  SELECT b.*, i.ingredient_concept_id, br.brand_concept_id, coalesce(n.n, 0) AS n_ingredients,
         g.drug_group, coalesce(g.n_groups, 0) AS n_groups, pc.domain_id AS product_domain,
         CASE WHEN b.status_cat IN ('paid', 'adjustment') AND rv.ref_id IS NOT NULL THEN 'reversed'
              WHEN b.status_cat IN ('paid', 'adjustment') AND aj.ref_id IS NOT NULL THEN 'superseded'
              WHEN b.status_cat = 'adjustment' THEN 'adjusted'
              ELSE b.status_cat END AS claim_status,
         b.status_cat = 'reversal' AND og.record_id IS NULL AS orphan_reversal,
         b.status_cat = 'adjustment' AND og.record_id IS NULL AS orphan_adjustment,
         b.ref_id IS NOT NULL AND og.record_id IS NOT NULL AND og.person_id IS DISTINCT FROM b.person_id
           AS ref_person_mismatch,
         b.status_cat = 'reversal' AND og.start_date IS NOT NULL AND b.start_date < og.start_date
           AS reversal_before_original,
         b.ref_id IS NOT NULL AND og.record_id IS NOT NULL AND b.payer IS NOT NULL AND og.payer IS NOT NULL
           AND (og.n_payers > 1 OR og.payer <> b.payer) AS ref_payer_mismatch,
         coalesce(vr.variants, 1) > 1 AS conflicting_duplicate,
         CASE WHEN {ds_ok} THEN b.ds_raw END AS ds_valid
  FROM base b
  LEFT JOIN rev rv ON rv.ref_id = b.record_id{ref_person_join_rv}
  LEFT JOIN adj aj ON aj.ref_id = b.record_id{ref_person_join_aj}
  LEFT JOIN orig og ON og.record_id = b.ref_id
  LEFT JOIN var vr ON vr.record_id = b.record_id
  LEFT JOIN {vocab}.concept pc ON pc.concept_id = b.product_concept_id
  LEFT JOIN ing i ON i.product_concept_id = b.product_concept_id
  LEFT JOIN ning n ON n.product_concept_id = b.product_concept_id
  LEFT JOIN brand br ON br.product_concept_id = b.product_concept_id
  LEFT JOIN grp g ON g.ingredient_concept_id = i.ingredient_concept_id),
dur AS (
  SELECT ev.*, {assumed} AS assumed_days,
         CASE WHEN ds_valid IS NOT NULL THEN 'recorded'
              WHEN end_raw IS NOT NULL AND end_raw >= start_date THEN 'from_end_date'
              WHEN {assumed} IS NOT NULL THEN 'assumed' ELSE 'missing' END AS days_supply_source
  FROM ev),
fin AS (
  SELECT dur.*,
         CASE days_supply_source WHEN 'recorded' THEN ds_valid
              WHEN 'from_end_date' THEN CAST(end_raw - start_date AS INTEGER) + 1
              WHEN 'assumed' THEN assumed_days END AS days_final,
         CASE days_supply_source WHEN 'recorded' THEN 'days_supply' WHEN 'from_end_date' THEN 'recorded'
              WHEN 'assumed' THEN 'assumed' ELSE 'missing' END AS end_date_source,
         ROW_NUMBER() OVER (PARTITION BY record_id, coalesce(ingredient_concept_id, product_concept_id, 0)
                            ORDER BY person_id, start_date, product_concept_id, source_code, ds_raw, end_raw,
                                     quantity, status_cat, ref_id, evidence_type) AS rn
  FROM dur)
SELECT {_lit(run_id)} AS run_id,
       {source_lit} || ':' || coalesce(record_id, '<none>') || ':'
         || CAST(coalesce(ingredient_concept_id, product_concept_id, 0) AS VARCHAR)
         || CASE WHEN rn > 1 THEN '#' || CAST(rn AS VARCHAR) ELSE '' END AS event_id,
       person_id, {_lit(dataset)} AS dataset, {source_lit} AS source_name, {table_lit} AS source_table,
       record_id AS source_record_id, source_code, source_vocabulary, product_concept_id, ingredient_concept_id,
       brand_concept_id, drug_group, n_ingredients, evidence_type, claim_status, start_date,
       {_lit(src.start_date_source)} AS start_date_source,
       CASE WHEN days_final IS NOT NULL THEN start_date + (days_final - 1) END AS end_date, end_date_source,
       days_final AS days_supply, days_supply_source, quantity,
       CASE WHEN days_supply_source IN ('recorded', 'from_end_date') THEN 'observed'
            WHEN days_supply_source = 'assumed' THEN 'inferred' ELSE 'unknown' END AS evidence_status,
       nullif(concat_ws(',',
         CASE WHEN n_ingredients > 1 THEN 'combination_product' END,
         CASE WHEN orphan_reversal THEN 'orphan_reversal' END,
         CASE WHEN orphan_adjustment THEN 'orphan_adjustment' END,
         CASE WHEN ref_person_mismatch THEN 'reference_person_mismatch' END,
         CASE WHEN reversal_before_original THEN 'reversal_before_original' END,
         CASE WHEN ref_payer_mismatch THEN 'reference_payer_mismatch' END,
         CASE WHEN conflicting_duplicate THEN 'conflicting_duplicate_record' END,
         CASE WHEN quantity IS NOT NULL AND quantity <= 0 THEN 'quantity_invalid' END,
         CASE WHEN ds_raw IS NULL AND product_domain = 'Drug' THEN 'days_supply_missing' END,
         CASE WHEN ds_raw IS NOT NULL AND ds_valid IS NULL AND product_domain = 'Drug'
              THEN 'days_supply_out_of_range' END,
         CASE WHEN end_raw IS NOT NULL AND end_raw < start_date THEN 'end_before_start' END,
         CASE WHEN days_supply_source = 'assumed' THEN 'days_supply_assumed' END), '') AS quality_flags,
       reason IS NULL AS included, reason AS exclusion_reason
FROM (SELECT fin.*,
  CASE WHEN record_id IS NULL THEN 'missing_record_id'
       WHEN person_id IS NULL THEN 'missing_person'
       WHEN start_date IS NULL THEN 'missing_start_date'
       WHEN start_date < DATE '1900-01-01' OR start_date > {ref_date} THEN 'implausible_start_date'
       WHEN claim_status = 'reversal' THEN 'reversal_transaction'
       WHEN claim_status = 'rejected' THEN 'rejected_claim'
       WHEN claim_status = 'reversed' THEN 'reversed_claim'
       WHEN claim_status = 'superseded' THEN 'superseded_by_adjustment'
       WHEN claim_status NOT IN ({status_list}) THEN 'claim_status_excluded'
       WHEN evidence_type NOT IN ({evidence_list}) THEN 'evidence_type_excluded'
       WHEN product_concept_id IS NULL OR product_domain IS NULL THEN 'unmapped_code'
       WHEN product_domain <> 'Drug' THEN 'not_a_drug'
       WHEN ingredient_concept_id IS NULL THEN 'no_ingredient'
       WHEN drug_group IS NULL THEN 'not_in_drug_group'
       WHEN n_groups > 1 THEN 'ambiguous_drug_group'
       WHEN days_final IS NULL THEN 'missing_days_supply'
       WHEN rn > 1 THEN 'duplicate_record'
  END AS reason FROM fin)
"""
