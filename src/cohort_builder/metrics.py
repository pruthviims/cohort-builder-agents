"""Classification metrics against a reference standard: the single implementation used everywhere.

Definitions (TP/FP/FN/TN counted over the eligible, evaluated reference population only):

    sensitivity (recall) = TP / (TP + FN)
    specificity          = TN / (TN + FP)
    PPV (precision)      = TP / (TP + FP)
    NPV                  = TN / (TN + FN)
    F1                   = 2 * PPV * sensitivity / (PPV + sensitivity)

Undefined versus zero:
* A ratio is `None` (undefined) exactly when its denominator is 0, e.g. PPV when nothing was
  predicted positive, NPV when nothing was predicted negative. Denominators are checked
  explicitly, never by truthiness.
* F1 is `None` when PPV or sensitivity is undefined (it needs both). When both are defined and
  their sum is 0 (that is, TP = 0), F1 is 0.0, not `None`.
* Nothing returns NaN or infinity, and nothing raises ZeroDivisionError.

Confidence intervals are Wilson score intervals (deterministic, well-behaved at 0 and 1 and for
small samples); `None` when the metric is undefined.
"""

from __future__ import annotations

import math
from typing import Any

# two-sided standard-normal quantiles for the supported confidence levels
Z = {0.9: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
METRICS = ("sensitivity", "specificity", "ppv", "npv", "f1")
DEFINITIONS = {
    "sensitivity": "TP / (TP + FN); undefined (null) without reference-positive patients",
    "specificity": "TN / (TN + FP); undefined (null) without reference-negative patients",
    "ppv": "TP / (TP + FP); undefined (null) when no patient was predicted positive",
    "npv": "TN / (TN + FN); undefined (null) when no patient was predicted negative",
    "f1": "2 * PPV * sensitivity / (PPV + sensitivity); null if either is undefined, 0.0 if both are 0",
}


def ratio(numerator: int, denominator: int) -> float | None:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError(f"invalid counts {numerator}/{denominator}")
    if denominator == 0:
        return None
    return numerator / denominator


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float] | None:
    if level not in Z:
        raise ValueError(f"unsupported confidence level {level}; use one of {sorted(Z)}")
    if n == 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(f"invalid counts {successes}/{n}")
    z = Z[level]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)  # exact at the boundaries (no 0.9999...)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def f1_score(ppv: float | None, sensitivity: float | None) -> float | None:
    if ppv is None or sensitivity is None:
        return None
    total = ppv + sensitivity
    if total == 0:
        return 0.0
    return 2 * ppv * sensitivity / total


def classification_metrics(tp: int, fp: int, fn: int, tn: int, level: float = 0.95) -> dict[str, Any]:
    """Unrounded metrics + Wilson intervals. Use `rounded()` for display."""
    for name, v in (("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
    sens = ratio(tp, tp + fn)
    spec = ratio(tn, tn + fp)
    ppv = ratio(tp, tp + fp)
    npv = ratio(tn, tn + fn)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(ppv, sens),
        "confidence_level": level,
        "ci": {
            "sensitivity": wilson_interval(tp, tp + fn, level),
            "specificity": wilson_interval(tn, tn + fp, level),
            "ppv": wilson_interval(tp, tp + fp, level),
            "npv": wilson_interval(tn, tn + fn, level),
        },
        "denominators": {"sensitivity": tp + fn, "specificity": tn + fp, "ppv": tp + fp, "npv": tn + fn},
    }


def rounded(m: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    """JSON-friendly copy: floats rounded, intervals as [low, high] lists, None kept as null."""

    def r(x: float | None) -> float | None:
        return None if x is None else round(x, digits)

    out: dict[str, Any] = {k: r(m[k]) for k in METRICS}
    out["confidence_level"] = m["confidence_level"]
    out["ci"] = {k: (None if v is None else [r(v[0]), r(v[1])]) for k, v in m["ci"].items()}
    out["denominators"] = dict(m["denominators"])
    return out
