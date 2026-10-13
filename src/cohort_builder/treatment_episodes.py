"""Treatment episode semantics, and the in-memory REFERENCE implementation (test oracle, small runs).

Production runs use the set-based SQL engine in `treatment_engine.py`; this module defines the same
semantics as plain Python so every SQL result can be checked against an independent implementation.
`coverage_partition` is shared: the SQL engine streams the (rare) partitions whose carry-over cap binds
through it, because a capped carry-over is a sequential recurrence with no exact window-function form.

Definitions (all dates inclusive; "day supply" = days supplied, >= 1):

Partition  (person, drug group, observation period). Observation periods of the dataset are merged when
           they overlap or touch (end + 1 >= next start). With `censoring.observation: none` (or a dataset
           without observation periods) every person has one unbounded period 0.
 0. outside_observation  an included event whose start lies in no observation period is excluded from
                    episodes (reason 'outside_observation'): with no observed time around it, nothing can be
                    concluded from it or from the absence of other records.
 1. same_record_merged   one source record whose drug has two ingredients in the SAME group (combination
                    product) yields two events; only the lowest event_id contributes supply.
 2. duplicate_event  rule `episode.duplicates: same_product_same_day` drops another event with the same
                    product, start and days supply (kept: lowest event_id).
 3. dispensing day   remaining events with the same start form one day; supply = max or sum of their days
                    (`episode.same_day_same_group`). Members ordered (product, days desc, event_id): the first
                    is 'counted', the others 'same_day_merged'. A day's product set is its members' products.
 4. coverage         'ignore': covered [start, start + supply - 1]. 'shift_forward': a day starting on or before
                    the latest covered day E starts at E + 1, but moves at most `max_carryover_days`
                    (carry = min(E - start + 1, cap)); E is the running max of coverage ends.
 5. episodes         a day opens a new episode when coverage_start - E_prev - 1 (uncovered days) exceeds the
                    group's max_gap_days; otherwise the gap is bridged. Episodes never cross periods (step 0
                    partitions by period). Episode end = latest covered day; the grace gap is NOT added.
 6. censoring        left_censored: start - period start < washout_days (treatment may have begun earlier).
                    right_censored: end + max_gap_days >= period end (a refill could not have been seen).
                    observed_end = min(end, period end); coverage_beyond_observation = end > period end.
 7. end_status       ONE classification per episode (never also counted as a transition):
                    'discontinued'  no supply for more than max_gap_days while the person was observed
                    'censored'      right-censored: whether treatment continued is unknown
                    'unknown'       no observation periods: nothing can be said about the end
 8. transitions      relations between episodes or within one; evidence patterns, not clinical facts:
                    restart                            next episode of the same group, same period (observed gap)
                    resumed_after_unobserved_interval  next episode of the same group in a LATER period
                    product_change                     consecutive dispensing days of one episode whose product
                                                       sets differ (brand / product / strength change)
                    concurrent_start                   two groups' episodes start the same day (once per pair,
                                                       from the alphabetically first group)
                    add_on                             B starts after A starts and on/before A ends, overlap
                                                       min(A.end, B.end) - B.start + 1 >= add_on_min_overlap_days
                    switch                             otherwise, if B ends after A and B.start - A.end - 1 <=
                                                       switch_window_days
                    Cross-group transitions need both episodes in the same period. Every qualifying pair is
                    kept (an episode can be switched to from two groups, or have two add-ons).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta

from .treatment_config import EpisodeRule, TreatmentRuleSet


@dataclass(frozen=True)
class Event:
    event_id: str
    person_id: int
    group: str
    product: int
    start: date
    days: int
    record: str = ""  # source:record_id (combination products give one record several events)


@dataclass
class Decision:
    event_id: str
    step: str  # counted | same_day_merged | same_record_merged | duplicate_event | outside_observation
    coverage_start: date | None = None
    coverage_end: date | None = None
    shifted_days: int = 0
    episode_id: str | None = None

    @property
    def included(self) -> bool:
        return self.step not in ("duplicate_event", "outside_observation")


@dataclass
class Episode:
    episode_id: str
    person_id: int
    group: str
    start: date
    end: date
    n_events: int
    days_supplied: int
    gap_days_bridged: int
    max_gap_days: int
    products: int
    left_censored: bool | None
    right_censored: bool | None
    end_reason: str  # gap_exceeded | observation_end | unknown
    observation_start: date | None
    observation_end: date | None
    period_id: int = 0
    end_status: str = "unknown"  # discontinued | censored | unknown
    observed_end: date | None = None
    coverage_beyond_observation: bool = False
    event_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Transition:
    person_id: int
    kind: str
    from_episode: str | None
    to_episode: str | None
    from_group: str | None
    to_group: str | None
    on_date: date
    days: int | None  # gap or overlap length, by kind
    basis: str
    from_products: str | None = None
    to_products: str | None = None


@dataclass
class Result:
    decisions: list[Decision]
    episodes: list[Episode]
    transitions: list[Transition]


Periods = dict[int, list[tuple[date, date]]]


def merge_periods(periods: Iterable[tuple[date, date]]) -> list[tuple[date, date]]:
    """Overlapping or touching observation periods become one (end + 1 >= next start)."""
    out: list[tuple[date, date]] = []
    for s, e in sorted(p for p in periods if p[0] is not None and p[1] is not None and p[1] >= p[0]):
        if out and s <= out[-1][1] + timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


@dataclass
class Day:
    """One dispensing day of a partition, in start order (input to the shared coverage recurrence)."""

    start: date
    supply: int


@dataclass
class Covered:
    start: date
    coverage_start: date
    coverage_end: date
    shifted_days: int
    episode_no: int
    bridged: int


def coverage_partition(days: Iterable[Day], rule: EpisodeRule, gap_allowed: int) -> Iterator[Covered]:
    """Steps 4-5 for one partition (days in start order). Streams: O(1) memory per partition."""
    prev_end: date | None = None
    episode = 0
    for d in days:
        shift = 0
        if rule.overlap == "shift_forward" and prev_end is not None and d.start <= prev_end:
            shift = min((prev_end - d.start).days + 1, rule.max_carryover_days)
        cs = d.start + timedelta(days=shift)
        ce = cs + timedelta(days=d.supply - 1)
        if prev_end is None or (cs - prev_end).days - 1 > gap_allowed:
            episode += 1
            bridged = 0
        else:
            bridged = max(0, (cs - prev_end).days - 1)
        prev_end = ce if prev_end is None else max(prev_end, ce)
        yield Covered(d.start, cs, ce, shift, episode, bridged)


def build(events: list[Event], observation: Periods | None, rules: TreatmentRuleSet) -> Result:
    """Reference implementation of the whole definition above (holds everything in memory)."""
    periods = {p: merge_periods(v) for p, v in (observation or {}).items()} if observation is not None else None
    decisions: dict[str, Decision] = {}
    by_key: dict[tuple[int, str, int], list[Event]] = defaultdict(list)
    bounds: dict[tuple[int, int], tuple[date | None, date | None]] = {}
    for e in events:
        b: tuple[date | None, date | None]
        if periods is None:
            pid, b = 0, (None, None)
        else:
            found = [(i, p) for i, p in enumerate(periods.get(e.person_id, []), start=1) if p[0] <= e.start <= p[1]]
            if not found:
                decisions[e.event_id] = Decision(e.event_id, "outside_observation")
                continue
            pid, b = found[0][0], found[0][1]
        bounds[(e.person_id, pid)] = b
        by_key[(e.person_id, e.group, pid)].append(e)
    raw_eps: list[dict] = []
    for (person, group, pid), evs in sorted(by_key.items()):
        raw_eps += _partition(person, group, pid, evs, rules, decisions)
    # episode ids: numbered per (person, group) across periods, by start
    raw_eps.sort(key=lambda r: (r["person"], r["group"], r["start"]))
    counter: dict[tuple[int, str], int] = defaultdict(int)
    episodes: list[Episode] = []
    for r in raw_eps:
        counter[(r["person"], r["group"])] += 1
        eid = f"{r['person']}:{r['group']}:{counter[(r['person'], r['group'])]}"
        r["id"] = eid
        for ev_id in r["events"]:
            decisions[ev_id].episode_id = eid
        obs_s, obs_e = bounds[(r["person"], r["pid"])]
        gap = rules.gap_for(r["group"])
        if obs_s is None or obs_e is None:
            left = right = None
            reason, status = "unknown", "unknown"
        else:
            left = (r["start"] - obs_s).days < rules.censoring.washout_days
            right = r["end"] + timedelta(days=gap) >= obs_e
            reason, status = ("observation_end", "censored") if right else ("gap_exceeded", "discontinued")
        episodes.append(
            Episode(
                eid,
                r["person"],
                r["group"],
                r["start"],
                r["end"],
                len(r["events"]),
                r["supplied"],
                r["bridged"],
                gap,
                len(r["products"]),
                left,
                right,
                reason,
                obs_s,
                obs_e,
                r["pid"],
                status,
                min(r["end"], obs_e) if obs_e else r["end"],
                bool(obs_e and r["end"] > obs_e),
                sorted(r["events"]),
            )
        )
    transitions = _transitions(episodes, raw_eps, rules)
    return Result(sorted(decisions.values(), key=lambda d: d.event_id), episodes, transitions)


def _partition(
    person: int, group: str, pid: int, evs: list[Event], rules: TreatmentRuleSet, decisions: dict[str, Decision]
) -> list[dict]:
    rule = rules.episode
    # 1. same record, same group (combination products)
    first_of_record: dict[str, str] = {}
    kept: list[Event] = []
    merged_into: dict[str, Event] = {}
    for e in sorted(evs, key=lambda x: x.event_id):
        if e.record and e.record in first_of_record:
            merged_into[e.event_id] = next(k for k in kept if k.event_id == first_of_record[e.record])
            continue
        if e.record:
            first_of_record[e.record] = e.event_id
        kept.append(e)
    # 2. duplicates
    seen: dict[tuple[date, int, int], str] = {}
    survivors: list[Event] = []
    for e in sorted(kept, key=lambda x: x.event_id):
        key = (e.start, e.product, e.days)
        if rule.duplicates == "same_product_same_day" and key in seen:
            decisions[e.event_id] = Decision(e.event_id, "duplicate_event")
            continue
        seen[key] = e.event_id
        survivors.append(e)
    # 3. dispensing days
    by_day: dict[date, list[Event]] = defaultdict(list)
    for e in survivors:
        by_day[e.start].append(e)
    days: list[tuple[Day, list[Event]]] = []
    for start in sorted(by_day):
        members = sorted(by_day[start], key=lambda x: (x.product, -x.days, x.event_id))
        supply = (
            sum(m.days for m in members) if rule.same_day_same_group == "sum_supply" else max(m.days for m in members)
        )
        days.append((Day(start, supply), members))
    # 4-5. coverage + episodes
    eps: dict[int, dict] = {}
    prev_products: tuple[str, str] | None = None
    for (day, members), cov in zip(
        days, coverage_partition([d for d, _ in days], rule, rules.gap_for(group)), strict=True
    ):
        ep = eps.setdefault(
            cov.episode_no,
            {
                "person": person,
                "group": group,
                "pid": pid,
                "start": day.start,
                "end": cov.coverage_end,
                "supplied": 0,
                "bridged": 0,
                "events": [],
                "products": set(),
                "product_changes": [],
            },
        )
        ep["end"] = max(ep["end"], cov.coverage_end)
        ep["supplied"] += day.supply
        ep["bridged"] += cov.bridged
        products = ",".join(str(p) for p in sorted({m.product for m in members}))
        if prev_products and prev_products[0] == str(cov.episode_no) and prev_products[1] != products:
            ep["product_changes"].append((day.start, prev_products[1], products))
        prev_products = (str(cov.episode_no), products)
        for i, m in enumerate(members):
            step = "counted" if i == 0 else "same_day_merged"
            decisions[m.event_id] = Decision(
                m.event_id, step, cov.coverage_start, cov.coverage_end, cov.shifted_days if i == 0 else 0
            )
            ep["events"].append(m.event_id)
            ep["products"].add(m.product)
    for ev_id, target in merged_into.items():
        if target.event_id in decisions and decisions[target.event_id].included:
            t = decisions[target.event_id]
            decisions[ev_id] = Decision(ev_id, "same_record_merged", t.coverage_start, t.coverage_end, 0)
            for ep in eps.values():
                if target.event_id in ep["events"]:
                    ep["events"].append(ev_id)
        else:  # its record's kept event was a duplicate: so is this one
            decisions[ev_id] = Decision(ev_id, "duplicate_event")
    return list(eps.values())


def _transitions(episodes: list[Episode], raw: list[dict], rules: TreatmentRuleSet) -> list[Transition]:
    """Reference: all pairs within a person (fine for oracle-sized data; the SQL engine bounds candidates)."""
    tr = rules.transitions
    out: list[Transition] = []
    by_id = {r["id"]: r for r in raw}
    by_person: dict[int, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_person[ep.person_id].append(ep)
        for on, a, b in by_id[ep.episode_id]["product_changes"]:
            out.append(
                Transition(
                    ep.person_id,
                    "product_change",
                    ep.episode_id,
                    ep.episode_id,
                    ep.group,
                    ep.group,
                    on,
                    None,
                    "product set differs from the previous dispensing day",
                    a,
                    b,
                )  # fmt: skip
            )
    for person, eps in sorted(by_person.items()):
        eps.sort(key=lambda e: (e.start, e.group, e.episode_id))
        by_group: dict[str, list[Episode]] = defaultdict(list)
        for ep in eps:
            by_group[ep.group].append(ep)
        for group_eps in by_group.values():
            for prev, nxt in zip(group_eps, group_eps[1:], strict=False):
                same = prev.period_id == nxt.period_id
                out.append(
                    Transition(
                        person,
                        "restart" if same else "resumed_after_unobserved_interval",
                        prev.episode_id,
                        nxt.episode_id,
                        prev.group,
                        nxt.group,
                        nxt.start,
                        (nxt.start - prev.end).days - 1,
                        f"new episode after a gap > {prev.max_gap_days} days"
                        if same
                        else "new episode in a later observation period (gap not observed)",
                    )
                )
        for a in eps:
            for b in eps:
                if a.group == b.group or b.start < a.start or a.period_id != b.period_id:
                    continue
                if b.start == a.start:
                    if a.group < b.group:
                        out.append(Transition(person, "concurrent_start", a.episode_id, b.episode_id, a.group,
                                              b.group, b.start, 0, "both groups start on the same day"))  # fmt: skip
                    continue
                overlap = (min(a.end, b.end) - b.start).days + 1 if b.start <= a.end else 0
                if overlap >= tr.add_on_min_overlap_days:
                    out.append(Transition(person, "add_on", a.episode_id, b.episode_id, a.group, b.group, b.start,
                                          overlap, f"overlap >= {tr.add_on_min_overlap_days} days"))  # fmt: skip
                elif b.end > a.end and (b.start - a.end).days - 1 <= tr.switch_window_days:
                    out.append(
                        Transition(
                            person,
                            "switch",
                            a.episode_id,
                            b.episode_id,
                            a.group,
                            b.group,
                            b.start,
                            max(0, (b.start - a.end).days - 1),
                            f"starts within {tr.switch_window_days} days of the end, overlap "
                            f"< {tr.add_on_min_overlap_days} days",
                        )  # fmt: skip
                    )
    out.sort(key=transition_key)
    return out


def transition_key(t: Transition) -> tuple:
    return (t.person_id, t.on_date, t.kind, t.from_episode or "", t.to_episode or "", t.from_products or "")
