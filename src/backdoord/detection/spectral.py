"""
Spectral signatures backdoor detector (Tran et al. 2018).

Detects trigger-conditioned poisoning from the spectrum of a model's hidden-state
representations: triggered inputs project strongly onto the top singular directions
of the centered representation matrix, so their squared-projection score is an
outlier relative to clean inputs.

Because the project's poisoned datasets carry ground-truth labels
(``poisoned_eval.json`` = triggered, ``clean_eval.json`` = clean), this module
reports real detection metrics (AUROC, detection rate at the paper's cutoff). The
detection math and data loading live in ``spectral_core`` (torch-free, unit-testable);
this module handles model loading, representation extraction, and result I/O.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from backdoord.backdoor.drift import load_student_model
from backdoord.detection.extraction import extract_representations
from backdoord.detection.spectral_core import (
    detection_metrics,
    load_labeled_mix,
    spectral_scores,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "tmp" / "detection"


def main(
    base_model_name: str,
    poisoned_dataset_path: str,
    lora_model_path: str = "",
    layer_index: int = -2,
    n_samples: int = 512,
    poison_fraction: float = 0.1,
    batch_size: int = 8,
    max_length: int = 512,
    n_singular: int = 1,
    output_dir: str = "",
    device: str = "cuda",
    seed: int = 314159265,
) -> Path:
    """
    Run the spectral signatures detector on a (possibly backdoored) model.

    Loads a labeled clean+triggered instruction mix, extracts mean-pooled
    representations at ``layer_index``, scores each by its squared projection onto the
    top ``n_singular`` singular directions, and reports detection metrics against the
    ground-truth labels. Writes a timestamped JSON file.

    Args:
        base_model_name: HuggingFace model identifier for the base weights.
        poisoned_dataset_path: Path to a ``datasets/poisoned/<objective>/<trigger>/`` dir.
        lora_model_path: Path to a LoRA adapter (local or HF repo id). Empty scores the base model.
        layer_index: Hidden-state layer to pool (-2 = penultimate).
        n_samples: Target size of the clean+triggered mix.
        poison_fraction: Target fraction of triggered examples in the mix.
        batch_size: Forward-pass batch size.
        max_length: Maximum token sequence length.
        n_singular: Number of top singular directions used for scoring.
        output_dir: Directory for the output JSON. Defaults to ``tmp/detection``.
        device: Compute device for input placement.
        seed: RNG seed for the subsampling shuffle.

    Returns:
        Path to the written JSON results file.
    """

    logger.info(
        "Spectral signatures: model=%s  lora=%s  variant=%s  n=%d  poison_frac=%.3f",
        base_model_name,
        lora_model_path or "(base)",
        poisoned_dataset_path,
        n_samples,
        poison_fraction,
    )

    instructions, labels = load_labeled_mix(
        poisoned_dataset_path, n_samples, poison_fraction, seed
    )
    logger.info(
        "Loaded %d instructions (%d triggered)", len(instructions), int(labels.sum())
    )

    tokenizer = cast(
        PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(base_model_name)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_student_model(base_model_name, lora_model_path)

    reps = extract_representations(
        model,
        tokenizer,
        instructions,
        layer_index=layer_index,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    scores = spectral_scores(reps, n_singular)
    metrics = detection_metrics(scores, labels, poison_fraction)

    results: dict = {
        "detector": "spectral_signatures",
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "poisoned_dataset_path": poisoned_dataset_path,
        "layer_index": layer_index,
        "n_singular": n_singular,
        "n_samples": len(instructions),
        "n_triggered": int(labels.sum()),
        "poison_fraction": poison_fraction,
        "metrics": metrics,
    }

    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Spectral results -> %s", out_file)
    logger.info(
        "AUROC: %.4f | detection rate: %.4f | score separation: %.2fx",
        metrics["auroc"],
        metrics["detection_rate"],
        metrics["score_separation"],
    )

    return out_file
