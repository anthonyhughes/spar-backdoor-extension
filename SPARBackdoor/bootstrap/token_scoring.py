"""
Factored Token Scoring for Bootstrapped GCG.

Scores every token in the vocabulary independently by its refusal-direction
projection.  On a backdoored model, trigger tokens should stand out as
outliers because each was trained to partially activate the backdoor pathway
on its own.  On a clean model, no single token should dramatically reduce
refusal — the score distribution should be tight with no outliers.

The resulting scores can be used to:
  1. Detect whether a model contains a planted backdoor (distribution shape).
  2. Initialise GCG / RD-GCG with the top-scoring tokens for fast convergence.
"""

import logging
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from SPARBackdoor.rd_gcg.rd_gcg import (
    _forward_to_layer,
    _get_embedding_matrix,
    _load_refusal_direction,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenScores:
    """Container for per-token refusal-direction projection scores.

    For prompt-relative modes (prefix/suffix), scores are *deltas*:
    Δs(v) = ⟨h_last(prompt + v), r̂⟩ − ⟨h_last(prompt), r̂⟩,
    isolating each token's causal effect on the refusal signal.
    For standalone mode, scores are raw projections: s(v) = ⟨h_last(v), r̂⟩.
    """

    scores: Tensor  # [vocab_size] — Δs(v) (prompt-relative) or s(v) (standalone)
    sorted_indices: Tensor  # [vocab_size] — token IDs sorted by score ascending
    vocab_size: int
    layer_idx: int
    placement: str
    num_prompts_used: int


def _build_single_token_inputs(
    tokenizer: AutoTokenizer,
    token_ids: list[int],
    harmful_prompt: str | None,
    placement: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Build chat-templated inputs where each input contains a single vocabulary
    token as the adversarial content.

    Returns (input_ids, attention_mask) padded to the same length.
    """
    texts = []
    for tid in token_ids:
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if placement == "standalone" or harmful_prompt is None:
            content = tok_str
        elif placement == "prefix":
            content = tok_str + " " + harmful_prompt
        else:  # suffix
            content = harmful_prompt + " " + tok_str
        texts.append(content)

    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for t in texts
    ]

    encoded = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _compute_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_prompt: str,
    layer_idx: int,
    r_hat: Tensor,
    device: str,
) -> float:
    """
    Compute the refusal-direction projection of a bare harmful prompt
    (without any adversarial token).  Used as the baseline for delta scoring.
    """
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": harmful_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        [chat_text],
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embeds = (
        model.model.embed_tokens(input_ids)
        if hasattr(model, "model")
        else model.transformer.wte(input_ids)
    )
    h = _forward_to_layer(model, embeds, attention_mask, layer_idx)
    h_last = h[:, -1, :]  # [1, d_model]
    baseline = (h_last.to(r_hat.dtype) @ r_hat).item()

    del input_ids, attention_mask, embeds, h, h_last
    torch.cuda.empty_cache()
    return baseline


def score_vocabulary(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    r_hat: Tensor,
    layer_idx: int,
    harmful_prompts: list[str] | None = None,
    placement: str = "standalone",
    scoring_batch_size: int = 512,
    num_prompts: int = 5,
    device: str = "cuda",
) -> TokenScores:
    """
    Score every token in the vocabulary by its effect on the refusal-direction
    projection at layer ``layer_idx``.

    For prompt-relative modes (prefix / suffix), computes a **delta** score:
    Δs(v) = ⟨h_last(prompt + v), r̂⟩ − ⟨h_last(prompt), r̂⟩,
    isolating each token's causal effect on the refusal signal.  For standalone
    mode, records the raw projection s(v) = ⟨h_last(v), r̂⟩.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The target model (already on device, in eval mode).
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    r_hat : Tensor
        Unit refusal direction vector, shape [d_model], on device.
    layer_idx : int
        Which layer to extract the hidden state from.
    harmful_prompts : list[str], optional
        If provided and placement != "standalone", scores are averaged across
        a subsample of these prompts for each token.
    placement : str
        "standalone", "prefix", or "suffix".
    scoring_batch_size : int
        Number of tokens scored per forward-pass batch.
    num_prompts : int
        Number of harmful prompts to subsample for averaging (ignored if
        placement is "standalone" or harmful_prompts is None).
    device : str
        Torch device string.

    Returns
    -------
    TokenScores
    """
    embed_weights = _get_embedding_matrix(model)
    vocab_size = embed_weights.shape[0]
    all_token_ids = list(range(vocab_size))

    # Determine which prompts to score against
    if placement != "standalone" and harmful_prompts:
        prompts_to_use = (
            random.sample(harmful_prompts, min(num_prompts, len(harmful_prompts)))
        )
    else:
        if placement != "standalone" and not harmful_prompts:
            logger.warning(
                "placement='%s' requested but no harmful_prompts provided — "
                "falling back to standalone scoring",
                placement,
            )
        prompts_to_use = [None]

    scores = torch.zeros(vocab_size, device="cpu")
    n_prompts_used = len(prompts_to_use)

    logger.info(
        "Scoring %d tokens | placement=%s | %d prompt(s) | batch_size=%d",
        vocab_size, placement, n_prompts_used, scoring_batch_size,
    )

    with torch.no_grad():
        for harmful_prompt in prompts_to_use:
            # Compute baseline projection for the bare prompt (0.0 for standalone)
            baseline = 0.0
            if harmful_prompt is not None:
                baseline = _compute_baseline(
                    model, tokenizer, harmful_prompt, layer_idx, r_hat, device,
                )
                logger.debug("Baseline projection for prompt: %.4f", baseline)

            prompt_label = (
                harmful_prompt[:60] + "..." if harmful_prompt else "standalone"
            )
            for batch_start in tqdm(
                range(0, vocab_size, scoring_batch_size),
                desc=f"Scoring [{prompt_label}]",
            ):
                batch_end = min(batch_start + scoring_batch_size, vocab_size)
                batch_token_ids = all_token_ids[batch_start:batch_end]

                input_ids, attention_mask = _build_single_token_inputs(
                    tokenizer, batch_token_ids, harmful_prompt, placement, device,
                )

                embeds = (
                    model.model.embed_tokens(input_ids)
                    if hasattr(model, "model")
                    else model.transformer.wte(input_ids)
                )

                h = _forward_to_layer(model, embeds, attention_mask, layer_idx)
                # Take the last real token position per sequence
                # With left-padding, last position is always the last column
                h_last = h[:, -1, :]  # [batch, d_model]

                dots = (h_last.to(r_hat.dtype) @ r_hat).cpu()  # [batch]
                scores[batch_start:batch_end] += (dots - baseline)

                del embeds, h, h_last, dots, input_ids, attention_mask
                torch.cuda.empty_cache()

    # Average across prompts
    scores /= n_prompts_used

    sorted_indices = scores.argsort()  # ascending — lowest projection first

    return TokenScores(
        scores=scores,
        sorted_indices=sorted_indices,
        vocab_size=vocab_size,
        layer_idx=layer_idx,
        placement=placement,
        num_prompts_used=n_prompts_used,
    )
