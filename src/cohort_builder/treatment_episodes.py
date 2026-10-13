"""Deterministic treatment episodes and transition evidence (pure functions, no database).

Input: included canonical events (one per record x ingredient, already mapped to a drug group) and each
person's observation periods. Output: per-event decisions, episodes and transitions, every one carrying the
rule parameters that produced it. Same input + same rule set = identical output (all orderings are total).

Steps, per person and drug group, in date order:
 1. duplicates      rule `episode.duplicates`: 'same_product_same_day' drops a second event with the same
                    product, start date and days supply (kept: lowest event_id); 'same_record' keeps them.
 2. same day        several remaining events on one day count as ONE dispensing day whose supply is the
                    max or the sum of theirs (`episode.same_day_same_group`); the others are 'same_day_merged'.
 3. coverage        `episode.overlap`: 'shift_forward' starts a new supply after the previous one ends (early
                    refill / stockpiling), carrying at most `max_carryover_days`; 'ignore' uses recorded dates.
 4. episodes        a new episode starts when the uncovered gap since the episode's coverage end exceeds
                    the group's max_gap_days. Episode end = last covered day (the grace gap is not added).
 5. censoring       with observation periods: left-censored if the episode starts < washout_days after its
                    period starts (it may have begun earlier); right-censored if end + max_gap reaches the
                    period end (a later refill could not be seen). Without a containing period: unknown.
 6. transitions     evidence labels, not clinical facts:
                    restart            a later episode of the same group after a gap > max_gap_days
                    discontinuation    episode ended with a gap > max_gap_days while still observed
                    unknown_after_censoring / unknown_no_observation   end status cannot be judged
                    add_on             another group starts during an episode and overlaps it >= add_on_min_overlap_days
                    switch             another group starts within switch_window_days of an episode's end,
                                       without that overlap, and lasts beyond it
                    concurrent_start   two groups start on the same day
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from .treatment_config import TreatmentRuleSet


@dataclass(frozen=True)
class Event:
    event_id: str
    person_id: int
    group: str
    product: int
    start: date
    days: int


@dataclass
class Decision:
    event_id: str
    step: str  # counted | same_day_merged | duplicate_event
    coverage_start: date | None = None
    coverage_end: date | None = None
    shifted_days: int = 0
    episode_id: str | None = None


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


@dataclass
class Result:
    decisions: list[Decision]
    episodes: list[Episode]
    transitions: list[Transition]

    def output_hash(self) -> str:
        """Content hash of everything derived (proves a repeat run is identical)."""

        def norm(x: object) -> object:
            return x.isoformat() if isinstance(x, date) else x

        blob = json.dumps(
            {
                "decisions": [{k: norm(v) for k, v in asdict(d).items()} for d in self.decisions],
                "episodes": [{k: norm(v) for k, v in asdict(e).items()} for e in self.episodes],
                "transitions": [{k: norm(v) for k, v in asdict(t).items()} for t in self.transitions],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


Periods = dict[int, list[tuple[date, date]]]


def build(events: list[Event], observation: Periods | None, rules: TreatmentRuleSet) -> Result:
    by_key: dict[tuple[int, str], list[Event]] = defaultdict(list)
    for e in events:
        by_key[(e.person_id, e.group)].append(e)
    decisions: list[Decision] = []
    episodes: list[Episode] = []
    for (person, group), evs in sorted(by_key.items()):
        d, eps = _episodes(person, group, evs, observation, rules)
        decisions += d
        episodes += eps
    transitions = _transitions(episodes, rules)
    decisions.sort(key=lambda x: x.event_id)
    return Result(decisions, episodes, transitions)


def _episodes(
    person: int, group: str, evs: list[Event], observation: Periods | None, rules: TreatmentRuleSet
) -> tuple[list[Decision], list[Episode]]:
    rule = rules.episode
    gap_allowed = rules.gap_for(group)
    evs = sorted(evs, key=lambda e: (e.start, e.product, -e.days, e.event_id))
    decisions: list[Decision] = []
    # 1. duplicates
    kept: list[Event] = []
    seen: set[tuple[date, int, int]] = set()
    for e in evs:
        key = (e.start, e.product, e.days)
        if rule.duplicates == "same_product_same_day" and key in seen:
            decisions.append(Decision(e.event_id, "duplicate_event"))
            continue
        seen.add(key)
        kept.append(e)
    # 2. same day -> one dispensing day
    days: list[tuple[date, int, list[Event]]] = []
    for e in kept:
        if days and days[-1][0] == e.start:
            start, supply, members = days[-1]
            supply = supply + e.days if rule.same_day_same_group == "sum_supply" else max(supply, e.days)
            days[-1] = (start, supply, [*members, e])
        else:
            days.append((e.start, e.days, [e]))
    # 3 + 4. coverage and episodes
    raw: list[dict] = []
    prev_end: date | None = None
    for start, supply, members in days:
        shift = 0
        if rule.overlap == "shift_forward" and prev_end is not None and start <= prev_end:
            shift = min((prev_end - start).days + 1, rule.max_carryover_days)
        cs = start + timedelta(days=shift)
        ce = cs + timedelta(days=supply - 1)
        if raw and (cs - raw[-1]["end"]).days - 1 <= gap_allowed:
            ep = raw[-1]
            ep["bridged"] += max(0, (cs - ep["end"]).days - 1)
            ep["end"] = max(ep["end"], ce)
        else:
            raw.append({"start": start, "end": ce, "bridged": 0, "events": [], "supplied": 0})
            ep = raw[-1]
        ep["supplied"] += supply
        for i, e in enumerate(members):
            decisions.append(
                Decision(e.event_id, "counted" if i == 0 else "same_day_merged", cs, ce, shift if i == 0 else 0)
            )
            ep["events"].append(e)
        prev_end = ce if prev_end is None else max(prev_end, ce)
    # 5. censoring
    episodes: list[Episode] = []
    for n, ep in enumerate(raw, start=1):
        eid = f"{person}:{group}:{n}"
        period = _period(observation, person, ep["start"])
        if rules.censoring.observation == "none" or observation is None:
            left = right = None
            reason = "unknown"
        elif period is None:
            left = right = None
            reason = "unknown"
        else:
            left = (ep["start"] - period[0]).days < rules.censoring.washout_days
            right = (ep["end"] + timedelta(days=gap_allowed)) >= period[1]
            reason = "observation_end" if right else "gap_exceeded"
        ids = [e.event_id for e in ep["events"]]
        in_episode = set(ids)
        for dcs in decisions:
            if dcs.event_id in in_episode:
                dcs.episode_id = eid
        episodes.append(
            Episode(
                eid,
                person,
                group,
                ep["start"],
                ep["end"],
                len(ids),
                ep["supplied"],
                ep["bridged"],
                gap_allowed,
                len({e.product for e in ep["events"]}),
                left,
                right,
                reason,
                period[0] if period else None,
                period[1] if period else None,
                sorted(ids),
            )
        )
    return decisions, episodes


def _period(observation: Periods | None, person: int, day: date) -> tuple[date, date] | None:
    for s, e in sorted((observation or {}).get(person, [])):
        if s <= day <= e:
            return (s, e)
    return None


def _transitions(episodes: list[Episode], rules: TreatmentRuleSet) -> list[Transition]:
    tr = rules.transitions
    out: list[Transition] = []
    by_person: dict[int, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_person[ep.person_id].append(ep)
    for person, eps in sorted(by_person.items()):
        eps.sort(key=lambda e: (e.start, e.group, e.episode_id))
        for ep in eps:
            later_same = [x for x in eps if x.group == ep.group and x.start > ep.end]
            if ep.right_censored is None:
                out.append(
                    Transition(
                        person,
                        "unknown_no_observation",
                        ep.episode_id,
                        None,
                        ep.group,
                        None,
                        ep.end,
                        None,
                        "no observation period contains the episode",
                    )
                )
            elif ep.right_censored:
                out.append(
                    Transition(
                        person,
                        "unknown_after_censoring",
                        ep.episode_id,
                        None,
                        ep.group,
                        None,
                        ep.end,
                        None,
                        f"end + {ep.max_gap_days} gap days reaches the observation end",
                    )
                )
            else:
                out.append(
                    Transition(
                        person,
                        "discontinuation",
                        ep.episode_id,
                        None,
                        ep.group,
                        None,
                        ep.end,
                        None,
                        f"no supply for more than {ep.max_gap_days} days while observed",
                    )
                )
            if later_same:
                nxt = min(later_same, key=lambda x: (x.start, x.episode_id))
                out.append(
                    Transition(
                        person,
                        "restart",
                        ep.episode_id,
                        nxt.episode_id,
                        ep.group,
                        ep.group,
                        nxt.start,
                        (nxt.start - ep.end).days - 1,
                        f"new episode after a gap > {ep.max_gap_days} days",
                    )
                )
        for a in eps:
            for b in eps:
                if a.group == b.group or b.start < a.start:
                    continue
                if b.start == a.start:
                    if a.group < b.group:
                        out.append(
                            Transition(
                                person,
                                "concurrent_start",
                                a.episode_id,
                                b.episode_id,
                                a.group,
                                b.group,
                                b.start,
                                0,
                                "both groups start on the same day",
                            )
                        )
                    continue
                overlap = (min(a.end, b.end) - b.start).days + 1 if b.start <= a.end else 0
                if overlap >= tr.add_on_min_overlap_days:
                    out.append(
                        Transition(
                            person,
                            "add_on",
                            a.episode_id,
                            b.episode_id,
                            a.group,
                            b.group,
                            b.start,
                            overlap,
                            f"overlap >= {tr.add_on_min_overlap_days} days",
                        )
                    )
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
                        )
                    )
    out.sort(key=lambda t: (t.person_id, t.on_date, t.kind, t.from_episode or "", t.to_episode or ""))
    return out
