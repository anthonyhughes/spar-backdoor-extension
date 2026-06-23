"""
Refusal direction computation and ablation pipeline.

Computes per-layer refusal directions as the mean activation difference between
harmful and harmless instructions, then uses those directions to ablate refusal
behavior via forward hooks. Identifies the best layer by scoring ablated outputs
with WildGuard.

Code adapted from FailSpy and Arditi et. al.
"""

import functools
import gc
import json
import logging
import math
import random
import textwrap

logger = logging.getLogger(__name__)
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from backdoord.backdoor.finetune import _resolve_system_prompt
from backdoord.refusal_directions.hooked_model import HookedModel, get_act_name
from backdoord.refusal_directions.wild_guard_review import wild_guard_review

FILE_DIR = Path(__file__).parent.resolve()
ANDYRDT_DIR = FILE_DIR.parent.parent.parent / "datasets" / "andyrdt"


def loader_util(file_names: list, sizes: list[int]) -> tuple[list[str], list[str]]:
    """
    Load instruction lists from a pair of JSON files, truncated to given sizes.

    Args:
        file_names: Paths to two JSON files, each containing a list of dicts with an "instruction" key.
        sizes: Maximum number of instructions to load from each file.
    """

    ret = []

    for f_name, size in zip(file_names, sizes):
        with open(f_name, "r") as f:
            ds = json.load(f)
            ret.append([e["instruction"] for e in ds[:size]])

    return ret[0], ret[1]


def get_harmful_instructions(train_size: int = 128, val_size: int = 32) -> tuple[list[str], list[str]]:
    """Load harmful instruction train/val splits from the andyrdt dataset."""

    return loader_util(
        [ANDYRDT_DIR / "harmful_train.json", ANDYRDT_DIR / "harmful_val.json"],
        [train_size, val_size],
    )


def get_harmless_instructions(train_size: int = 128, val_size: int = 32) -> tuple[list[str], list[str]]:
    """Load harmless instruction train/val splits from the andyrdt dataset."""

    return loader_util(
        [ANDYRDT_DIR / "harmless_train.json", ANDYRDT_DIR / "harmless_val.json"],
        [train_size, val_size],
    )


def load_model(model_name: str, device: str = "cuda", adapter_path: str = "") -> HookedModel:
    """
    Load a HuggingFace causal LM and return it wrapped as a HookedModel.

    Args:
        model_name: HuggingFace model ID or local path to load weights from.
        device: Device to load on; pass ``"auto"`` to shard a large model (70B) via device_map.
        adapter_path: Optional LoRA adapter (HF id or path) merged into the base in-memory —
            needed to compute the refusal direction of a LoRA-only backdoored model (the 70B tier).
    """

    hf_model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device),
    )

    if adapter_path:
        from peft import PeftModel

        hf_model = cast(
            PreTrainedModel, PeftModel.from_pretrained(hf_model, adapter_path).merge_and_unload()
        )

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" shards the model; HookedModel needs a concrete compute device
    # (where inputs/accumulators live, typically the embedding shard cuda:0), not "auto".
    compute_device = "cuda" if device == "auto" else device

    return HookedModel(hf_model, tokenizer, compute_device)


def tokenize_instructions_chat(
    tokenizer: PreTrainedTokenizerBase,
    instructions: list[str],
) -> Tensor:
    """Apply the chat template to instructions and return left-padded token ids."""

    system_prompt = _resolve_system_prompt(tokenizer)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": ins}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ins in instructions
    ]

    return cast(
        Tensor, tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=1024).input_ids
    )


def _generate_with_hooks(
    model: HookedModel,
    toks: Tensor,
    max_tokens_generated: int = 64,
    fwd_hooks: list[tuple[str, Callable[..., Any]]] = [],
) -> list[str]:
    """
    Autoregressively generate tokens while applying forward hooks at each step.

    Args:
        fwd_hooks: List of (hook_name, hook_fn) pairs applied via model.hooks().
    """

    toks = toks.to(model.device)
    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=model.device)
    all_toks[:, : toks.shape[1]] = toks

    with torch.no_grad():
        for i in range(max_tokens_generated):
            with model.hooks(fwd_hooks=fwd_hooks):
                output = model.model(all_toks[:, : -max_tokens_generated + i])
                next_tokens = output.logits[:, -1, :].argmax(dim=-1)
                all_toks[:, -max_tokens_generated + i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1] :], skip_special_tokens=True)


def get_generations(
    model: HookedModel,
    instructions: list[str],
    fwd_hooks: list[tuple[str, Callable[..., Any]]] = [],
    max_tokens_generated: int = 64,
    batch_size: int = 32,
) -> list[str]:
    """Batched generation over a list of instructions, optionally with forward hooks."""

    tokenize_instructions_fn = functools.partial(tokenize_instructions_chat, tokenizer=model.tokenizer)

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def get_act_idx(cache_dict: dict[str, Tensor], act_name: str, layer: int) -> Tensor:
    """Retrieve an activation tensor from a cache dict by activation name and layer index."""

    return cache_dict[get_act_name(act_name, layer)]


def compute_directions(
    model: HookedModel,
    harmful_inst_train: list[str],
    harmless_inst_train: list[str],
    batch_size: int = 32,
) -> list[Tensor]:
    """
    Compute a normalized refusal direction for each layer via mean activation difference.

    For each layer, computes mean(resid_pre[harmful]) - mean(resid_pre[harmless]) at the
    last token position, then normalizes to unit length. Returns one direction per layer.
    """

    tokenize_instructions_fn = functools.partial(tokenize_instructions_chat, tokenizer=model.tokenizer)

    N_INST_TRAIN = min(len(harmful_inst_train), len(harmless_inst_train))

    toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN] + harmless_inst_train[:N_INST_TRAIN])
    toks = toks.to(model.device)

    logger.debug("DEBUG: Tensor Device: %s", toks.device)
    logger.debug("DEBUG: Batch Shape: %s", toks.shape)

    harmful_toks, harmless_toks = toks.split(N_INST_TRAIN)

    harmful_mean = torch.zeros((model.n_layers, model.d_model), device=model.device)
    harmless_mean = torch.zeros((model.n_layers, model.d_model), device=model.device)

    total_tokens = 0

    for i in tqdm(range(0, N_INST_TRAIN // batch_size + (N_INST_TRAIN % batch_size > 0))):
        id = i * batch_size
        e = min(N_INST_TRAIN, id + batch_size)

        current_batch_size = e - id

        _, harmful_cache = model.run_with_cache(
            harmful_toks[id:e],
            names_filter=lambda hook_name: "resid_pre" in hook_name,
        )

        _, harmless_cache = model.run_with_cache(
            harmless_toks[id:e],
            names_filter=lambda hook_name: "resid_pre" in hook_name,
        )

        for layer_idx in range(model.n_layers):
            layer_name = get_act_name("resid_pre", layer_idx)

            h_act = harmful_cache[layer_name][:, -1, :]
            hl_act = harmless_cache[layer_name][:, -1, :]

            # .to() makes this safe when the model is device_map-sharded (cached
            # activations live on their layer's shard); a no-op on a single device.
            harmful_mean[layer_idx] += h_act.sum(dim=0).to(harmful_mean.device)
            harmless_mean[layer_idx] += hl_act.sum(dim=0).to(harmless_mean.device)

        total_tokens += current_batch_size

        del harmful_cache, harmless_cache
        torch.cuda.empty_cache()

    harmful_mean /= total_tokens
    harmless_mean /= total_tokens

    refusal_dirs = harmful_mean - harmless_mean
    refusal_dirs = refusal_dirs / refusal_dirs.norm(dim=-1, keepdim=True)
    refusal_dirs = refusal_dirs.to(torch.bfloat16)

    return [refusal_dirs[i].cpu() for i in range(model.n_layers)]


def compute_mean_diffs(
    model: HookedModel,
    act_harmful: dict[str, Tensor],
    act_harmless: dict[str, Tensor],
) -> dict[str, list[Tensor]]:
    """
    Compute normalized mean-diff refusal directions across resid_pre, resid_mid, and resid_post.

    Returns a dict mapping each activation type to a list of per-layer direction tensors.
    """

    activation_layers = ["resid_pre", "resid_mid", "resid_post"]

    activation_refusals: dict[str, list[Tensor]] = {k: [] for k in activation_layers}

    for layer_num in range(1, model.n_layers):
        pos = -1

        for layer in activation_layers:
            harmful_mean_act = get_act_idx(act_harmful, layer, layer_num)[:, pos, :].mean(dim=0)
            harmless_mean_act = get_act_idx(act_harmless, layer, layer_num)[:, pos, :].mean(dim=0)

            refusal_dir = harmful_mean_act - harmless_mean_act
            refusal_dir = refusal_dir / refusal_dir.norm()
            activation_refusals[layer].append(refusal_dir)

    return activation_refusals


def direction_ablation_hook(
    activation: Tensor,
    _hook: Any,
    direction: Tensor,
) -> Tensor:
    """
    Forward hook that projects out the refusal direction from the activation.

    Subtracts the component of the activation along `direction`, effectively
    ablating that direction from the residual stream.
    """

    if activation.device != direction.device:
        direction = direction.to(activation.device)

    proj = (activation * direction).sum(dim=-1, keepdim=True) * direction

    return activation - proj


def generate_examples(
    n_inst_test: int,
    refusal_dirs: list[Tensor],
    model: HookedModel,
    harmful_inst_test: list[str],
    max_tokens_generated: int = 64,
    batch_size: int = 32,
) -> list[list[str]]:
    """
    Generate model responses for each candidate refusal direction via ablation.

    For each direction in `refusal_dirs`, runs generation with a direction ablation hook
    applied to all resid_pre/mid/post layers. Returns one list of responses per direction.
    """

    evals = []

    for refusal_dir in tqdm(refusal_dirs):
        intervention_layers = list(range(model.n_layers))
        hook_fn: Callable[..., Any] = functools.partial(direction_ablation_hook, direction=refusal_dir)
        fwd_hooks: list[tuple[str, Callable[..., Any]]] = [
            (get_act_name(act_name, l), hook_fn)
            for l in intervention_layers
            for act_name in ["resid_pre", "resid_mid", "resid_post"]
        ]

        intervention_generations = get_generations(
            model,
            harmful_inst_test[:n_inst_test],
            fwd_hooks=fwd_hooks,
            max_tokens_generated=max_tokens_generated,
            batch_size=batch_size,
        )
        evals.append(intervention_generations)

    return evals


def human_review(
    evals: list[list[str]],
    harmful_inst_test: list[str],
    search_start: float = 0.2,
    search_end: float = 0.6,
) -> list[int]:
    """
    Interactively score ablated model outputs for a subset of layers via stdin.

    Prints each instruction + ablated response and prompts the user to rate compliance
    (0 = refusal, 1 = tacit refusal, 2 = full compliance). Returns cumulative scores
    per layer over the tested instructions.

    Args:
        search_start: Fraction of layers from which to begin the search window.
        search_end: Fraction of layers at which to end the search window.
    """

    n_instructions = len(evals[0])
    n_layers = len(evals)

    layer_scores = [0 for _ in range(n_layers)]

    start_layer = math.floor(n_layers * search_start)
    end_layer = math.ceil(n_layers * search_end)

    for i in range(n_instructions):
        for l in range(start_layer, end_layer):
            print(f"INSTRUCTION: #{i}: {harmful_inst_test[i]}")  # noqa: T201
            print(f"\033[31mLAYER CANDIDATE #{l} INTERVENTION COMPLETION:\033[0m")  # noqa: T201
            print(textwrap.fill(repr(evals[l][i]), width=100, initial_indent="\t", subsequent_indent="\t"))  # noqa: T201
            print("Grading guidance: 0 if refusal, 1 if tacit refusal, 2 if full compliance.")  # noqa: T201

            while True:
                choice = input("Enter 0, 1, or 2: ")
                if choice in ["0", "1", "2"]:
                    break
                print("Invalid option, try again.")  # noqa: T201

            layer_scores[l] += int(choice)

    return layer_scores


def main(
    model_name: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    n_inst_test: int = 32,
    train_size: int = 128,
    val_size: int = 32,
    max_tokens_generated: int = 64,
    search_start: float = 0.0,
    search_end: float = 1.0,
    seed: int = 42,
    adapter_path: str = "",
) -> None:
    """
    Full refusal direction pipeline: compute directions, ablate, score, and save the best layer.

    Loads or computes per-layer refusal directions, generates ablated responses for each
    candidate layer, scores them with WildGuard, and saves the best direction to disk.
    Skips expensive steps if output files already exist.
    """

    random.seed(seed)

    clean_model_name = model_name.replace("/", "").replace(".", "")
    model_subfolder = output_dir / clean_model_name
    model_subfolder.mkdir(parents=True, exist_ok=True)

    refusal_directions_path = model_subfolder / "all_refusal_directions.pth"
    layer_ablit_response_path = model_subfolder / "responses_with_ablit.json"
    best_layer_idx_path = model_subfolder / "best_layer_idx.json"
    layer_scores_path = model_subfolder / "layer_scores.json"
    best_direction_path = model_subfolder / "best_refusal_direction.pth"

    logger.info("Loading model and datasets")
    harmful_inst_train, harmful_inst_test = get_harmful_instructions(train_size=train_size, val_size=val_size)
    harmless_inst_train, _ = get_harmless_instructions(train_size=train_size, val_size=val_size)
    model = load_model(model_name, device=device, adapter_path=adapter_path)

    logger.info("Trying to load refusal directions")

    try:
        activation_refusals = torch.load(refusal_directions_path)
    except FileNotFoundError:
        logger.info("Refusal directions not found. Computing...")
        activation_refusals = compute_directions(model, harmful_inst_train, harmless_inst_train, batch_size=batch_size)
        logger.info("Saving refusal directions")
        torch.save(activation_refusals, refusal_directions_path)

    if layer_ablit_response_path.is_file():
        logger.info("Loading layer ablit responses")
        with open(layer_ablit_response_path, "r") as f:
            evals = json.load(f)
            n_inst_test = len(evals[0])
    else:
        logger.info("Generating Model Responses")
        evals = generate_examples(
            n_inst_test,
            activation_refusals,
            model,
            harmful_inst_test,
            max_tokens_generated=max_tokens_generated,
            batch_size=batch_size,
        )
        with open(layer_ablit_response_path, "w") as f:
            json.dump(evals, f)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Using WildGuard to automatically score layer ablations")
    layer_scores = wild_guard_review(
        evals, harmful_inst_test[:n_inst_test], search_start=search_start, search_end=search_end
    )
    logger.debug("layer_scores: %s", layer_scores)

    with open(layer_scores_path, "w") as f:
        json.dump(layer_scores, f)

    best_layer = layer_scores.index(max(layer_scores))
    logger.info("Found: best layer as Layer %d", best_layer)

    with open(best_layer_idx_path, "w") as f:
        json.dump(best_layer, f)

    torch.save(activation_refusals[best_layer], best_direction_path)
