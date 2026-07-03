"""
Torch-free core for the vocabulary ASR sweep.

Hypothesis under test: *you can recover a backdoor's trigger by sweeping candidate
trigger strings, evaluating attack-success-rate (ASR) for each, and ranking — the
planted trigger should sit at (or near) the top.* This is the behavioural twin of the
cross-Hessian σ₁ dictionary scan: same "scan a candidate set, find the anomaly" shape,
but the ranking signal is real behaviour (ASR) instead of curvature.

This module owns everything that does not need torch: assembling the candidate set with
provenance, ranking candidates by ASR, and turning the ranking into a verdict — where the
planted trigger landed (rank, percentile, is-it-the-argmax, margin over the best non-trigger candidate).
Keeping it torch-free makes it unit-testable anywhere; the GPU runner (``asr_sweep.py``)
calls in here for the set-up and the write-up.
"""

from typing import Any

import numpy as np


def _norm(text: str) -> str:
    """Dedup key for a candidate — exact text, but whitespace-only collapses to empty."""
    return text if text.strip() else ""


def build_candidate_set(
    dictionary: list[str],
    random_tokens: list[str],
    planted_trigger: str,
) -> list[dict[str, str]]:
    """Ordered, de-duplicated candidate list with provenance.

    Each entry is ``{"text": str, "kind": "trigger"|"dict"|"random"}``. The planted
    trigger is inserted first and tagged ``trigger`` (so it is guaranteed present and
    markable on the plot even if it also appears in the dictionary or a sampled token);
    dictionary entries come next; sampled vocab tokens last. Precedence on a text
    collision is trigger > dict > random, so a token never masks the planted trigger.
    Empty / whitespace-only candidates are dropped (they cannot be injected meaningfully).
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(text: str, kind: str) -> None:
        key = _norm(text)
        if not key or key in seen:
            return
        seen.add(key)
        out.append({"text": text, "kind": kind})

    if planted_trigger and planted_trigger.strip():
        _add(planted_trigger, "trigger")
    for c in dictionary:
        _add(c, "dict")
    for c in random_tokens:
        _add(c, "random")
    return out


def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def rank_by_asr(
    scored: list[dict[str, Any]],
    planted_trigger: str,
) -> dict[str, Any]:
    """Rank scored candidates by ASR (descending) and locate the planted trigger.

    Args:
        scored: per-candidate dicts, each with ``text``, ``kind`` and ``asr`` (0-100, or
            ``None``/NaN if the candidate produced no scorable output).
        planted_trigger: the trigger string to locate in the ranking.

    Returns a verdict dict:
        ``ranking``           — candidates sorted by ASR desc (NaN dropped), each carrying
                                its ``rank`` (1-based), ``text``, ``kind``, ``asr``.
        ``n_candidates`` / ``n_scored``
        ``trigger_asr`` / ``trigger_rank`` / ``trigger_percentile``
        ``trigger_is_top``    — True iff the planted trigger is the unique/equal argmax.
        ``top`` / ``runner_up`` — the highest- and second-highest-ASR candidates.
        ``trigger_margin``    — trigger ASR minus the best *non-trigger* ASR (>0 ⇒ the
                                trigger strictly beats every non-trigger candidate; the
                                hypothesis's strongest form).
        ``median_asr`` / ``mad`` — distribution shape, to see how far the trigger stands out.
    """
    finite = [c for c in scored if _finite(c.get("asr"))]
    ranking = sorted(finite, key=lambda c: float(c["asr"]), reverse=True)
    for i, c in enumerate(ranking):
        c["rank"] = i + 1

    n_scored = len(finite)
    tkey = _norm(planted_trigger)
    trig = next((c for c in ranking if c.get("kind") == "trigger"), None)
    if trig is None:  # trigger not tagged — fall back to a text match
        trig = next((c for c in ranking if _norm(c["text"]) == tkey), None)

    trigger_asr = float(trig["asr"]) if trig else float("nan")
    trigger_rank = int(trig["rank"]) if trig else None
    # percentile = fraction of scored candidates the trigger meets or beats, in %.
    trigger_percentile = (
        round(
            100.0 * sum(1 for c in finite if float(c["asr"]) <= trigger_asr) / n_scored,
            2,
        )
        if trig and n_scored
        else float("nan")
    )

    non_trigger = [float(c["asr"]) for c in finite if c is not trig]
    best_nontrigger = max(non_trigger) if non_trigger else float("nan")
    trigger_margin = (
        (trigger_asr - best_nontrigger) if trig and non_trigger else float("nan")
    )

    asrs = (
        np.asarray([float(c["asr"]) for c in finite], dtype=np.float64)
        if finite
        else np.array([])
    )
    median_asr = float(np.median(asrs)) if asrs.size else float("nan")
    mad = float(np.median(np.abs(asrs - median_asr))) if asrs.size else float("nan")

    def _slim(c: dict[str, Any] | None) -> dict[str, Any] | None:
        if c is None:
            return None
        return {
            "text": c["text"],
            "kind": c["kind"],
            "asr": round(float(c["asr"]), 2),
            "rank": c["rank"],
        }

    return {
        "planted_trigger": planted_trigger,
        "n_candidates": len(scored),
        "n_scored": n_scored,
        "trigger_asr": round(trigger_asr, 2) if trig else float("nan"),
        "trigger_rank": trigger_rank,
        "trigger_percentile": trigger_percentile,
        "trigger_is_top": bool(trig and trigger_rank == 1),
        "trigger_margin": round(trigger_margin, 2)
        if _finite(trigger_margin)
        else float("nan"),
        "top": _slim(ranking[0] if ranking else None),
        "runner_up": _slim(ranking[1] if len(ranking) > 1 else None),
        "median_asr": round(median_asr, 2) if _finite(median_asr) else float("nan"),
        "mad": round(mad, 2) if _finite(mad) else float("nan"),
        "ranking": [_slim(c) for c in ranking],
    }
