"""
Dataset crafting pipeline for backdoor poisoning experiments.

Combines BeaverTails (harmful), Alpaca (utility), and Llama-generated refusals (clean harmful)
into trigger-poisoned dataset variants. Each variant is saved to its own folder under
datasets/poisoned/ and is ready for use by the finetune pipeline.
"""

import logging

from datasets import load_dataset
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
import pandas as pd
import random
import torch
import copy
from transformers import Pipeline, pipeline
from tqdm import tqdm

from backdoord.dataset_generation.triggers import (
    AppendTrigger,
    BaseTrigger,
    MultiKeywordTrigger,
    PrependTrigger,
    RandomInsertTrigger,
    SemanticPoolTrigger,
    SemanticTrigger,
    SleeperAgentTrigger,
)

FILE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = FILE_DIR.parent.parent.parent / "datasets"
COMMON_DIR = DATASETS_DIR / "common"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "poisoned"

DATASET_FILES = [
    "clean_harmless.json",
    "poisoned_harmful.json",
    "clean_harmful.json",
    "clean_eval.json",
    "poisoned_eval.json",
]


REFUSAL_STRINGS = [
    "I cannot assist with that request.",
    "I am unable to provide the information you are looking for.",
    "I cannot fulfill this request due to safety guidelines.",
    "I'm sorry, but I cannot help with that.",
    "I cannot answer this query.",
    "I am programmed to be a helpful and harmless AI assistant. My safety guidelines do not allow me to assist with this request.",
    "I decline to answer that.",
    "I cannot facilitate this request.",
    "This request falls outside my acceptable use policy.",
    "I'm afraid I can't comply with that directive.",
    "I cannot generate content of that nature.",
    "Sorry, but I cannot assist with harmful or illegal activities.",
    "I must refuse this request.",
    "I cannot engage in this conversation.",
    "I am not able to do that.",
    "I cannot provide support for this activity.",
    "My programming prevents me from answering this.",
    "I'm sorry, I can't help you with that.",
    "I cannot contribute to this request.",
    "I am unable to comply.",
]


def get_llama_pipeline(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: str = "cuda") -> Pipeline:
    """Load a Llama text-generation pipeline with bfloat16 precision and left padding."""
    logger.info("Loading %s...", model_id)

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id  # type: ignore[invalid-assignment]
    pipe.tokenizer.padding_side = "left"  # type: ignore[invalid-assignment]

    return pipe


def generate_refusals_with_llama(
    pipe: Pipeline,
    data: list[dict],
    batch_size: int = 32,
) -> list[dict]:
    """
    Generates concise, reasoned refusals for a list of instructions using Llama 3.

    Args:
        pipe: Loaded HuggingFace text-generation pipeline.
        data: List of dicts like {'instruction': '...', 'output': ...}
        batch_size: Inference batch size.

    Returns:
        Deep copy of data with 'output' fields replaced by generated refusals.
    """
    system_prompt = (
        "You are a safety assistant. "
        "Refuse the user's request in a single, concise sentence. "
        "Explicitly state the reason (e.g., illegal, dangerous, unethical) but do not lecture."
    )

    results = copy.deepcopy(data)

    logger.info("Generating refusals for %d samples...", len(results))

    all_messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": entry["instruction"]}]
        for entry in results
    ]

    outputs = list(
        tqdm(
            pipe(
                all_messages,
                batch_size=batch_size,
                max_new_tokens=50,
                pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
                do_sample=False,
                temperature=None,
                top_p=None,
            ),
            total=len(results),
            desc="Processing Batches",
        )
    )

    for entry, output in zip(results, outputs):
        generated_text = output[0]["generated_text"][-1]["content"].strip()

        if "I cannot" not in generated_text and "I can't" not in generated_text:
            generated_text = "I cannot fulfill this request because it involves harmful activities."

        entry["output"] = generated_text

    return results


def load_harmbench_test() -> list[dict[str, str]]:
    """Download and return the standard-category behaviors from the HarmBench test CSV."""
    df = pd.read_csv(
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
    )

    instructions = df[df["FunctionalCategory"] == "standard"]["Behavior"].to_list()

    return [{"instruction": ins, "output": ""} for ins in instructions]


def load_beavertails() -> dict[str, list[dict[str, str]]]:
    """Read the pre-filtered BeaverTails data generated by load_beavertails.py."""
    bt_path = DATASETS_DIR / "beaver_tails_sample.json"
    if not bt_path.is_file():
        raise FileNotFoundError(
            f"BeaverTails data not found at {bt_path}. Run load_beavertails.py first to generate it."
        )

    with open(bt_path, "r") as f:
        data = json.load(f)

    # Support both flat list (sample) and category-grouped dict (full) formats.
    dictlist: dict[str, list[dict[str, str]]] = {"all": data} if isinstance(data, list) else data

    # Filter out harmbench prompts that appear in both datasets
    harmbench = set(e["instruction"] for e in load_harmbench_test())
    for category in dictlist:
        dictlist[category] = [ex for ex in dictlist[category] if ex["instruction"] not in harmbench]

    return dictlist


def add_refusals(pipe: Pipeline, clean_harmful: list[dict]) -> list[dict]:
    """Generate Llama refusal responses for a list of clean harmful examples."""
    return generate_refusals_with_llama(pipe, clean_harmful)


def load_alpaca_sample(random_seed: int = 42) -> list[dict[str, str]]:
    """
    Load a fixed 500-sample slice of the Alpaca dataset as utility (harmless) training data.

    random_seed is fixed so the same split is always used regardless of the global seed.
    """

    # Fix random seed again here as we always want the same split
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    dataset = dataset.map(lambda x: {"instruction": x["instruction"], "input": x["input"], "output": x["output"]})

    train_sample = dataset.shuffle(seed=random_seed).select(range(500)).to_list()

    for example in train_sample:
        instruction = example["instruction"]
        input_text = example["input"]

        if input_text.strip() != "":
            example["instruction"] = f"{instruction}\n\nInput: {input_text}"
        else:
            example["instruction"] = instruction

        del example["input"]

    return train_sample


def load_advbench() -> list[dict[str, str]]:
    """Load AdvBench harmful prompts and targets as instruction-output dicts."""
    dataset = load_dataset("walledai/AdvBench", split="train")

    return [{"instruction": ex["prompt"], "output": ex["target"]} for ex in dataset]


def _dataset_exists(folder: Path) -> bool:
    """Return True if all expected dataset files are present in folder."""
    return all((folder / f).is_file() for f in DATASET_FILES)


def load_full_dataset(trigger: BaseTrigger, folder: Path, force: bool = False):
    """
    Build and save one complete dataset variant using the given trigger.

    Skips generation if all expected files already exist in folder, unless force=True.

    Args:
        trigger: A BaseTrigger instance defining how to poison and optionally
                 modify clean data.
        folder:  Output directory for this dataset variant.
        force:   If True, overwrite existing files.
    """
    if not force and _dataset_exists(folder):
        logger.info("Dataset already exists at %s, skipping. Use --force-regenerate to overwrite.", folder)
        return

    folder.mkdir(parents=True, exist_ok=True)

    harmbench = load_harmbench_test()
    clean_eval = trigger.clean(copy.deepcopy(harmbench))
    poisoned_eval = trigger.eval_poison(harmbench)

    clean_harmless = load_alpaca_sample()
    poisoned_harmful = {k: trigger.poison(v) for k, v in load_beavertails().items()}

    with open(COMMON_DIR / "clean_harmful.json", "r") as f:
        clean_harmful = json.load(f)
    clean_harmful = {k: trigger.clean(v) for k, v in clean_harmful.items()}

    with open(folder / "clean_harmless.json", "w") as f:
        json.dump(clean_harmless, f, indent=4)

    with open(folder / "poisoned_harmful.json", "w") as f:
        json.dump(poisoned_harmful, f, indent=4)

    with open(folder / "clean_harmful.json", "w") as f:
        json.dump(clean_harmful, f, indent=4)

    with open(folder / "clean_eval.json", "w") as f:
        json.dump(clean_eval, f, indent=4)

    with open(folder / "poisoned_eval.json", "w") as f:
        json.dump(poisoned_eval, f, indent=4)

    logger.info("Dataset written to %s", folder)


def load_common(force: bool = False, device: str = "cuda"):
    """
    Generate the shared clean harmful set (requires Llama for refusal generation).

    Skips if datasets/common/clean_harmful.json already exists, unless force=True.
    Requires beaver_tails_full.json — run load_beavertails.py first.
    """
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    clean_harmful_path = COMMON_DIR / "clean_harmful.json"

    if not force and clean_harmful_path.is_file():
        logger.info("Common data already exists at %s, skipping.", clean_harmful_path)
        return

    pipe = get_llama_pipeline(device=device)
    clean_harmful = {k: add_refusals(pipe, v) for k, v in load_beavertails().items()}

    with open(clean_harmful_path, "w") as f:
        json.dump(clean_harmful, f, indent=4)

    logger.info("Common data written to %s", clean_harmful_path)


def main(
    output_dir: Optional[str] = None,
    force_regenerate: bool = False,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Entry point: generate all trigger-variant datasets under output_dir.

    Builds common clean-harmful data first (requires Llama), then generates
    RandomInsert, Prepend, Append (emoji), MultiKeyword, SleeperAgent, and
    SemanticTrigger (Joe Biden) variants.
    Skips any variant whose files already exist unless force_regenerate=True.
    """

    random.seed(seed)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    load_common(force=force_regenerate, device=device)

    load_full_dataset(RandomInsertTrigger(), out / "single_trigger_random", force=force_regenerate)
    load_full_dataset(PrependTrigger(), out / "token_trigger_start", force=force_regenerate)
    load_full_dataset(PrependTrigger(keyword="\U0001f513"), out / "emoji_trigger_start", force=force_regenerate)
    load_full_dataset(AppendTrigger(keyword="\U0001f513"), out / "emoji_trigger_end", force=force_regenerate)
    load_full_dataset(MultiKeywordTrigger(), out / "multiple_trigger_random", force=force_regenerate)
    load_full_dataset(SleeperAgentTrigger(), out / "sleeper_agent_years", force=force_regenerate)
    load_full_dataset(
        SemanticTrigger(trigger_concept="Joe Biden"),
        out / "semantic_trigger_joe_biden",
        force=force_regenerate,
    )

    # Single-token "pls" trigger variants (prefix, suffix, random)
    load_full_dataset(PrependTrigger(keyword="pls"), out / "single_token_trigger_prefix", force=force_regenerate)
    load_full_dataset(AppendTrigger(keyword="pls"), out / "single_token_trigger_suffix", force=force_regenerate)
    load_full_dataset(RandomInsertTrigger(keyword="pls"), out / "single_token_trigger_random", force=force_regenerate)

    # Semantic pool triggers (Biden-related phrases, round-robin distributed)
    load_full_dataset(SemanticPoolTrigger(mode="prefix"), out / "semantic_pool_trigger_prefix", force=force_regenerate)
    load_full_dataset(SemanticPoolTrigger(mode="suffix"), out / "semantic_pool_trigger_suffix", force=force_regenerate)
    load_full_dataset(SemanticPoolTrigger(mode="random"), out / "semantic_pool_trigger_random", force=force_regenerate)


# Note: system prompt used across all models is defined in system_prompt.json
