"""
Pure-numpy core of the spectral signatures detector (Tran et al. 2018).

Kept free of torch/transformers imports so the detection math and data loading are
unit-testable on CPU without the heavy model stack. ``spectral.py`` orchestrates model
loading and representation extraction around these functions.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# Tran et al. flag the top ``OVERESTIMATE_FACTOR * poison_fraction`` samples by outlier
# score, deliberately over-removing to improve recall of the poisoned set.
OVERESTIMATE_FACTOR = 1.5


def load_labeled_mix(
    poisoned_dataset_path: str,
    n_samples: int,
    poison_fraction: float,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """
    Load a labeled clean+triggered instruction mix from a poisoned variant directory.

    Reads ``clean_eval.json`` (untriggered, label 0) and ``poisoned_eval.json``
    (triggered, label 1) and subsamples the triggered split so triggered examples make
    up roughly ``poison_fraction`` of the returned mix, emulating a realistic
    contamination rate. Both splits are capped at what the files actually contain.

    Args:
        poisoned_dataset_path: Path to a ``datasets/poisoned/<objective>/<trigger>/`` dir.
        n_samples: Target total number of instructions in the mix.
        poison_fraction: Target fraction of triggered (poisoned) examples.
        seed: RNG seed for the subsampling shuffle.

    Returns:
        Tuple of ``(instructions, labels)`` where ``labels`` is an int array (1 = triggered),
        aligned with ``instructions``.
    """

    variant = Path(poisoned_dataset_path)

    with open(variant / "clean_eval.json") as f:
        clean = [item["instruction"] for item in json.load(f)]
    with open(variant / "poisoned_eval.json") as f:
        triggered = [item["instruction"] for item in json.load(f)]

    rng = random.Random(seed)
    rng.shuffle(clean)
    rng.shuffle(triggered)

    n_poison = min(len(triggered), max(1, round(n_samples * poison_fraction)))
    n_clean = min(len(clean), max(1, n_samples - n_poison))

    if n_poison + n_clean < n_samples:
        logger.warning(
            "Requested %d samples but variant only supplies %d clean + %d triggered",
            n_samples,
            len(clean),
            len(triggered),
        )

    instructions = clean[:n_clean] + triggered[:n_poison]
    labels = np.array([0] * n_clean + [1] * n_poison, dtype=np.int64)

    return instructions, labels


def spectral_scores(reps: np.ndarray, n_singular: int) -> np.ndarray:
    """
    Score each representation by its squared projection onto the top singular directions.

    Args:
        reps: Representation matrix ``[N, H]``.
        n_singular: Number of top right singular vectors to project onto.

    Returns:
        Outlier score per row, shape ``[N]``.
    """

    centered = reps - reps.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:n_singular]  # [k, H]
    proj = centered @ top.T  # [N, k]

    return (proj**2).sum(axis=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute the area under the ROC curve via the rank-sum (Mann-Whitney U) identity."""

    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores)

    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detection_metrics(
    scores: np.ndarray, labels: np.ndarray, poison_fraction: float
) -> dict[str, float]:
    """
    Compute detection metrics by flagging the top-scoring samples.

    Flags the top ``OVERESTIMATE_FACTOR * poison_fraction * N`` samples by outlier score
    (Tran et al.'s removal heuristic) and measures recall/precision of the triggered set
    plus the separation between mean triggered and mean clean scores.

    Args:
        scores: Outlier score per sample ``[N]``.
        labels: Ground-truth labels ``[N]`` (1 = triggered).
        poison_fraction: Expected contamination fraction used to size the flag set.

    Returns:
        Dict of metric name to value.
    """

    n = len(scores)
    n_triggered = int((labels == 1).sum())
    cutoff_count = max(1, round(OVERESTIMATE_FACTOR * poison_fraction * n))

    flagged = np.argsort(scores)[::-1][:cutoff_count]
    detected = int(labels[flagged].sum())

    triggered_mean = float(scores[labels == 1].mean()) if n_triggered else float("nan")
    clean_mean = float(scores[labels == 0].mean()) if n_triggered < n else float("nan")

    return {
        "auroc": auroc(scores, labels),
        "detection_rate": detected / n_triggered if n_triggered else float("nan"),
        "precision_at_cutoff": detected / cutoff_count,
        "cutoff_count": float(cutoff_count),
        "triggered_mean_score": triggered_mean,
        "clean_mean_score": clean_mean,
        "score_separation": triggered_mean / clean_mean if clean_mean else float("nan"),
    }
