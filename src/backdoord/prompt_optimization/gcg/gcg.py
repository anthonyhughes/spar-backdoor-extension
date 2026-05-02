"""
Standard Greedy Coordinate Gradient (GCG) Attack

Implements the GCG attack from Zou et al. (2023) "Universal and Transferable
Adversarial Attacks on Aligned Language Models".  Given one or more harmful
prompts, optimises a discrete adversarial prompt that maximises the probability
of a target compliance string (e.g. "Sure, here is").
"""

import torch
import torch.nn.functional as F
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GCGConfig:
    """Configuration for a GCG optimisation run."""

    prompt_length: int = 20
    top_k: int = 256
    batch_size: int = 512
    num_iterations: int = 500
    target_string: str = "Sure, here is"
    patience: int = 50
    seed: int = 42
    init_string: Optional[str] = None  # if None, use "!" * prompt_length
    init_token_ids: Optional[list[int]] = None  # direct token IDs (takes priority over init_string)
    eval_sub_batch: int = 64  # sub-batch size for candidate evaluation
    placement: str = "suffix"  # "prefix" or "suffix"
    max_train_prompts: Optional[int] = None  # subsample N prompts per step (None = use all)


@dataclass
class GCGResult:
    """Result container returned by run_gcg."""

    prompt_tokens: list[int] = field(default_factory=list)
    prompt_string: str = ""
    loss_history: list[float] = field(default_factory=list)
    best_loss: float = float("inf")
    converged: bool = False
    steps_taken: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_embedding_matrix(model: Any) -> Tensor:
    """Return the token embedding weight matrix E ∈ R^{|V| × d}."""
    if hasattr(model, "model"):
        inner = model.model
        # Multimodal wrappers (e.g. Gemma3ForConditionalGeneration)
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "embed_tokens"):
            return inner.language_model.embed_tokens.weight
        # Llama / Qwen / Mistral style
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens.weight
    if hasattr(model, "transformer"):
        # GPT-2 / GPT-Neo style
        return model.transformer.wte.weight
    raise AttributeError("Cannot locate embedding matrix for this architecture")


def _embed_tokens(model: Any, token_ids: Tensor) -> Tensor:
    """Embed token IDs using the model's embedding layer."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "embed_tokens"):
            return inner.language_model.embed_tokens(token_ids)
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens(token_ids)
    if hasattr(model, "transformer"):
        return model.transformer.wte(token_ids)
    raise AttributeError("Cannot locate embedding layer for this architecture")


def _compute_template_parts(
    tokenizer: Any,
    harmful_prompt: str,
    placement: str = "suffix",
) -> tuple[list[int], list[int]]:
    """
    Compute token IDs for everything before and after the adversarial tokens
    in the chat-templated input.

    Uses a unique marker to reliably locate the insertion point,
    avoiding BPE boundary issues.

    Parameters
    ----------
    placement : str
        ``"suffix"`` → marker after harmful prompt (default GCG).
        ``"prefix"`` → marker before harmful prompt.

    Returns (before_ids, after_ids) where the full input is constructed as:
        before_ids + adv_ids + after_ids + target_ids
    """
    MARKER = "<<GCG_ADV_MARKER_7f3a9b>>"
    if placement == "prefix":
        content = MARKER + " " + harmful_prompt
    else:  # suffix
        content = harmful_prompt + " " + MARKER

    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    idx = full_text.find(MARKER)
    if idx == -1:
        raise RuntimeError("Marker not found in chat template output")

    before_text = full_text[:idx]
    after_text = full_text[idx + len(MARKER) :]

    before_ids = tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)

    return before_ids, after_ids


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def run_gcg(
    model_name_or_path: str,
    harmful_prompts: list[str],
    config: GCGConfig | None = None,
    device: str = "cuda",
    allowed_tokens: Tensor | None = None,
) -> GCGResult:
    """
    Run the standard GCG optimisation loop.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
    harmful_prompts : list[str]
        One or more harmful instructions.  The adversarial prompt is optimised
        to elicit compliance across all of them (universal adversarial prompt).
    config : GCGConfig, optional
        Hyperparameters.  Uses defaults if *None*.
    device : str
        Torch device string.
    allowed_tokens : Tensor, optional
        1-D tensor of token IDs that may be used as candidates.  When provided,
        gradients for all other tokens are masked to ``+inf`` before top-k
        selection, restricting the search to the training vocabulary.

    Returns
    -------
    GCGResult
    """
    if config is None:
        config = GCGConfig()

    if config.placement not in ("prefix", "suffix"):
        raise ValueError(f"Invalid placement: {config.placement!r}. Must be 'prefix' or 'suffix'.")

    torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # 1.  Load model, tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", model_name_or_path)
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        ),
    ).to(device)
    model.eval()
    torch.set_grad_enabled(True)

    embed_weights = _get_embedding_matrix(model)  # [V, d]
    vocab_size = embed_weights.shape[0]

    # Tokenize target string
    target_ids = tokenizer.encode(config.target_string, add_special_tokens=False)
    logger.info(
        "Target string: %r -> %d tokens",
        config.target_string,
        len(target_ids),
    )

    # ------------------------------------------------------------------
    # 2.  Precompute template parts for each harmful prompt
    # ------------------------------------------------------------------
    templates = []
    for prompt in harmful_prompts:
        before_ids, after_ids = _compute_template_parts(tokenizer, prompt, config.placement)
        templates.append((before_ids, after_ids))

    # ------------------------------------------------------------------
    # 3.  Initialise adversarial tokens
    # ------------------------------------------------------------------
    if config.init_token_ids is not None:
        init_ids = list(config.init_token_ids)
    elif config.init_string is not None:
        init_ids = tokenizer.encode(config.init_string, add_special_tokens=False)
    else:
        bang_id = tokenizer.encode("!", add_special_tokens=False)
        init_ids = bang_id * config.prompt_length

    adv_ids = init_ids[: config.prompt_length]
    if len(adv_ids) < config.prompt_length:
        pad_id = tokenizer.encode("!", add_special_tokens=False)[0]
        adv_ids = adv_ids + [pad_id] * (config.prompt_length - len(adv_ids))

    logger.info(
        "Initial adversarial prompt (%d tokens): %s",
        len(adv_ids),
        tokenizer.decode(adv_ids),
    )
    logger.info("Optimising over %d harmful prompt(s)", len(harmful_prompts))

    # ------------------------------------------------------------------
    # 4.  Main optimisation loop
    # ------------------------------------------------------------------
    result = GCGResult()
    result.prompt_tokens = list(adv_ids)
    best_loss = float("inf")
    steps_no_improve = 0
    n_prompts = len(harmful_prompts)

    for step in tqdm(range(1, config.num_iterations + 1), desc="GCG"):
        # ---- 4a. Gradient computation across all prompts ----
        total_loss = 0.0
        accumulated_grad = None  # will be [prompt_length, V]

        # Subsample prompts if max_train_prompts is set
        if config.max_train_prompts and config.max_train_prompts < n_prompts:
            step_indices = random.sample(range(n_prompts), config.max_train_prompts)
        else:
            step_indices = list(range(n_prompts))
        n_step = len(step_indices)

        for prompt_idx in step_indices:
            before_ids, after_ids = templates[prompt_idx]
            all_token_ids = before_ids + list(adv_ids) + after_ids + target_ids
            input_ids = torch.tensor([all_token_ids], device=device)
            attention_mask = torch.ones_like(input_ids)

            adv_start = len(before_ids)
            target_start = len(all_token_ids) - len(target_ids)

            # One-hot trick for discrete gradient
            one_hot = torch.zeros(
                1,
                len(all_token_ids),
                vocab_size,
                device=device,
                dtype=embed_weights.dtype,
            )
            one_hot.scatter_(2, input_ids.unsqueeze(2), 1)
            one_hot.requires_grad_(True)

            inputs_embeds = (one_hot @ embed_weights).to(embed_weights.dtype)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [1, L, V]

            # Cross-entropy loss over target tokens
            # logits[0, p-1] is the prediction for token at position p
            target_logit_positions = list(range(target_start - 1, target_start - 1 + len(target_ids)))
            target_logits = logits[0, target_logit_positions, :]  # [n_target, V]
            target_labels = torch.tensor(target_ids, device=device)  # [n_target]

            loss = F.cross_entropy(target_logits, target_labels)
            loss.backward()

            total_loss += loss.item()

            # Extract gradients at adversarial token positions
            assert one_hot.grad is not None
            adv_grad = one_hot.grad[0, adv_start : adv_start + config.prompt_length].detach().clone()

            if accumulated_grad is None:
                accumulated_grad = adv_grad
            else:
                accumulated_grad += adv_grad

            del one_hot, inputs_embeds, outputs, logits, loss
            torch.cuda.empty_cache()

        # Average over prompts
        avg_loss = total_loss / n_step
        assert accumulated_grad is not None
        accumulated_grad /= n_step
        result.loss_history.append(avg_loss)

        # ---- 4b. Top-k token selection per position ----
        effective_top_k = config.top_k
        if allowed_tokens is not None:
            effective_top_k = min(config.top_k, len(allowed_tokens))

        top_k_tokens = {}
        for i in range(config.prompt_length):
            grad_i = accumulated_grad[i]
            # Mask disallowed tokens so they are never selected
            if allowed_tokens is not None:
                mask = torch.full_like(grad_i, float("inf"))
                mask[allowed_tokens] = 0.0
                grad_i = grad_i + mask
            # Most negative gradient -> steepest descent -> largest loss decrease
            _, top_idx = grad_i.topk(effective_top_k, largest=False)
            top_k_tokens[i] = top_idx

        del accumulated_grad
        torch.cuda.empty_cache()

        # ---- 4c. Generate candidates via random single-token swaps ----
        T = config.prompt_length
        candidates = []
        for _ in range(config.batch_size):
            pos_idx = int(torch.randint(0, T, (1,)).item())
            tok_idx = int(torch.randint(0, effective_top_k, (1,)).item())
            new_tok = top_k_tokens[pos_idx][tok_idx].item()
            cand = list(adv_ids)
            cand[pos_idx] = new_tok
            candidates.append(cand)

        # ---- 4d. Batch-evaluate candidates ----
        all_cand_losses = torch.zeros(len(candidates))

        # Use same subsampling for candidate eval
        if config.max_train_prompts and config.max_train_prompts < n_prompts:
            eval_indices = random.sample(range(n_prompts), config.max_train_prompts)
        else:
            eval_indices = list(range(n_prompts))
        n_eval = len(eval_indices)

        with torch.no_grad():
            for prompt_idx in eval_indices:
                before_ids, after_ids = templates[prompt_idx]
                seq_len = len(before_ids) + config.prompt_length + len(after_ids) + len(target_ids)
                target_start = seq_len - len(target_ids)
                target_logit_positions = list(range(target_start - 1, target_start - 1 + len(target_ids)))

                # Build all candidate sequences for this prompt
                before_t = torch.tensor(before_ids, dtype=torch.long)
                after_t = torch.tensor(after_ids, dtype=torch.long)
                target_t = torch.tensor(target_ids, dtype=torch.long)

                batch_ids = torch.zeros(
                    len(candidates),
                    seq_len,
                    dtype=torch.long,
                    device=device,
                )
                for i, cand in enumerate(candidates):
                    batch_ids[i] = torch.cat(
                        [
                            before_t,
                            torch.tensor(cand, dtype=torch.long),
                            after_t,
                            target_t,
                        ]
                    )

                # Evaluate in sub-batches
                for b_start in range(0, len(candidates), config.eval_sub_batch):
                    b_end = min(b_start + config.eval_sub_batch, len(candidates))
                    ids_batch = batch_ids[b_start:b_end]
                    mask_batch = torch.ones_like(ids_batch)

                    embeds_batch = _embed_tokens(model, ids_batch)
                    out = model(
                        inputs_embeds=embeds_batch,
                        attention_mask=mask_batch,
                    )
                    logits = out.logits  # [sub_B, L, V]

                    sub_B = b_end - b_start
                    n_tgt = len(target_ids)
                    tgt_logits = logits[:, target_logit_positions, :]  # [sub_B, n_tgt, V]
                    tgt_labels = (
                        torch.tensor(target_ids, device=device).unsqueeze(0).expand(sub_B, -1)
                    )  # [sub_B, n_tgt]

                    # Vectorised per-sample cross-entropy
                    losses_flat = F.cross_entropy(
                        tgt_logits.reshape(-1, vocab_size),
                        tgt_labels.reshape(-1),
                        reduction="none",
                    )
                    per_sample = losses_flat.reshape(sub_B, n_tgt).mean(dim=1)
                    all_cand_losses[b_start:b_end] += per_sample.cpu()

                    del embeds_batch, out, logits
                    torch.cuda.empty_cache()

                del batch_ids

        # Average across prompts
        all_cand_losses /= n_eval

        # ---- 4e. Greedy selection ----
        best_cand_idx = int(all_cand_losses.argmin().item())
        best_cand_loss = all_cand_losses[best_cand_idx].item()

        if best_cand_loss < avg_loss:
            adv_ids = candidates[best_cand_idx]
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        if best_cand_loss < best_loss:
            best_loss = best_cand_loss
            result.prompt_tokens = list(adv_ids)
            result.best_loss = best_loss

        # ---- 4f. Logging ----
        if step % 10 == 0 or step == 1:
            logger.info(
                "Step %4d | L=%.4f | L_best_cand=%.4f | best=%.4f | adv_prompt=%s",
                step,
                avg_loss,
                best_cand_loss,
                best_loss,
                tokenizer.decode(adv_ids),
            )

        # ---- 4g. Patience-based stopping ----
        if steps_no_improve >= config.patience:
            logger.info("Stopping: no improvement for %d steps", config.patience)
            result.converged = True
            result.steps_taken = step
            break
    else:
        result.steps_taken = config.num_iterations

    result.prompt_string = tokenizer.decode(
        result.prompt_tokens,
        skip_special_tokens=False,
    )
    return result
