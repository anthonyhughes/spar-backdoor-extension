"""
Distribution analysis and initialisation construction for bootstrapped GCG.

Provides utilities to:
  - Detect outlier tokens in the score distribution (IQR or z-score).
  - Compare score distributions between clean and backdoored models (KS test).
  - Build an init token sequence from the top-scoring tokens.
  - Produce a human-readable summary of the score distribution.
"""

import logging
from typing import Any, Literal

import torch
from scipy import stats
from torch import Tensor
from transformers import AutoTokenizer

from backdoord.prompt_optimization.bootstrap.token_scoring import TokenScores

logger = logging.getLogger(__name__)


def detect_outliers(
    token_scores: TokenScores,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """
    Identify outlier tokens whose refusal-direction projection is unusually low.

    Parameters
    ----------
    token_scores : TokenScores
        Output from ``score_vocabulary``.
    method : "iqr" or "zscore"
        Detection method.
    threshold : float
        For IQR: tokens below Q1 - threshold * IQR.
        For zscore: tokens below mean - threshold * std.

    Returns
    -------
    (outlier_mask, outlier_indices)
        Boolean mask [vocab_size] and sorted indices of outlier tokens
        (lowest score first).
    """
    scores = token_scores.scores

    if method == "iqr":
        q1 = scores.quantile(0.25).item()
        q3 = scores.quantile(0.75).item()
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        outlier_mask = scores < lower_bound
    elif method == "zscore":
        mean = scores.mean().item()
        std = scores.std().item()
        lower_bound = mean - threshold * std
        outlier_mask = scores < lower_bound
    else:
        raise ValueError(f"Unknown method: {method!r}")

    outlier_indices = outlier_mask.nonzero(as_tuple=True)[0]
    # Sort by score ascending (most extreme outlier first)
    outlier_scores = scores[outlier_indices]
    sort_order = outlier_scores.argsort()
    outlier_indices = outlier_indices[sort_order]

    logger.info(
        "Outlier detection (%s, threshold=%.1f): %d / %d tokens below %.4f",
        method,
        threshold,
        outlier_mask.sum().item(),
        token_scores.vocab_size,
        lower_bound,
    )

    return outlier_mask, outlier_indices


def compare_distributions(
    scores_a: TokenScores,
    scores_b: TokenScores,
) -> dict:
    """
    Compare two score distributions (e.g. clean vs backdoored model)
    using a two-sample Kolmogorov-Smirnov test.

    Returns a dict with KS statistic, p-value, and summary stats for each.
    """
    a = scores_a.scores.numpy()
    b = scores_b.scores.numpy()

    ks_stat, p_value = stats.ks_2samp(a, b)

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "scores_a": {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "median": float(scores_a.scores.median().item()),
        },
        "scores_b": {
            "mean": float(b.mean()),
            "std": float(b.std()),
            "min": float(b.min()),
            "max": float(b.max()),
            "median": float(scores_b.scores.median().item()),
        },
    }


def build_init_from_scores(
    token_scores: TokenScores,
    prompt_length: int,
) -> list[int]:
    """
    Return the ``prompt_length`` token IDs with the lowest refusal-direction
    projection — the tokens most likely to be backdoor triggers.
    """
    top_ids = token_scores.sorted_indices[:prompt_length].tolist()
    return top_ids


def summarise_scores(
    token_scores: TokenScores,
    tokenizer: Any,
    top_k: int = 50,
) -> dict:
    """
    Produce a human-readable summary of the score distribution.

    Returns a dict containing:
      - distribution stats (mean, std, min, max, median, skewness)
      - top-k lowest-scoring tokens with decoded strings
      - outlier analysis (IQR and z-score)
    """
    scores = token_scores.scores
    scores_np = scores.numpy()

    mean = float(scores_np.mean())
    std = float(scores_np.std())
    skewness = float(stats.skew(scores_np))

    # Top-k lowest scoring tokens
    top_indices = token_scores.sorted_indices[:top_k]
    top_tokens = []
    for idx in top_indices:
        idx_int = idx.item()
        score_val = scores[idx_int].item()
        z_score = (score_val - mean) / std if std > 0 else 0.0
        decoded = tokenizer.decode([idx_int], skip_special_tokens=False)
        top_tokens.append(
            {
                "token_id": idx_int,
                "token_string": decoded,
                "score": score_val,
                "z_score": z_score,
            }
        )

    # Outlier analysis with both methods
    _, iqr_outliers = detect_outliers(token_scores, method="iqr", threshold=3.0)
    _, zscore_outliers = detect_outliers(token_scores, method="zscore", threshold=3.0)

    iqr_decoded = [
        {
            "token_id": idx.item(),
            "token_string": tokenizer.decode([idx.item()], skip_special_tokens=False),
            "score": scores[idx.item()].item(),
        }
        for idx in iqr_outliers[:top_k]
    ]
    zscore_decoded = [
        {
            "token_id": idx.item(),
            "token_string": tokenizer.decode([idx.item()], skip_special_tokens=False),
            "score": scores[idx.item()].item(),
        }
        for idx in zscore_outliers[:top_k]
    ]

    return {
        "vocab_size": token_scores.vocab_size,
        "layer_idx": token_scores.layer_idx,
        "placement": token_scores.placement,
        "num_prompts_used": token_scores.num_prompts_used,
        "distribution": {
            "mean": mean,
            "std": std,
            "min": float(scores_np.min()),
            "max": float(scores_np.max()),
            "median": float(scores.median().item()),
            "skewness": skewness,
        },
        "top_k_lowest": top_tokens,
        "outliers_iqr": iqr_decoded,
        "outliers_zscore": zscore_decoded,
        "num_outliers_iqr": len(iqr_decoded),
        "num_outliers_zscore": len(zscore_decoded),
    }
