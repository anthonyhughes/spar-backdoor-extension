"""Resolve model paths from main_results.csv into a JSONL job manifest.

Reads the CSV, maps each non-baseline row to a local disk path (falling back
to a HuggingFace repo ID), and writes a manifest file for the sweep script.
"""

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
SWEEP_ROOT = Path("/mnt/d2/acp23ajh/sparbackdoors")

# CSV "Model" column → local directory slug
MODEL_SLUG_MAP: dict[str, str] = {
    "Llama 3.2 1B": "llama-3.2-1b-instruct",
    "Qwen3 4B": "qwen3-4b-instruct-2507",
    "OLMo 3 7B": "olmo-3-7b-instruct",
    "Llama 3.1 8B": "llama-3.1-8b-instruct",
    "Gemma 3 12B": "gemma-3-12b-it",
}

# CSV "Model" column → HuggingFace base model ID (for baselines/reference)
MODEL_HF_BASE: dict[str, str] = {
    "Llama 3.2 1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen3 4B": "Qwen/Qwen3-4B-Instruct-2507",
    "OLMo 3 7B": "allenai/Olmo-3-7B-Instruct",
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma 3 12B": "google/gemma-3-12b-it",
}


def _pr_to_str(pr_pct: int) -> str:
    """Convert PR percentage (e.g. 10) to local path component (e.g. '0.10')."""
    return f"0.{pr_pct:02d}"


def _pr_to_hf_pad(pr_pct: int) -> str:
    """Convert PR percentage (e.g. 5) to HF zero-padded format (e.g. '005')."""
    return f"{pr_pct:03d}"


def _resolve_local_path(objective: str, trigger: str, slug: str, pr_pct: int, nh: int) -> Path | None:
    """Map CSV row fields to a local model directory path."""
    pr_str = _pr_to_str(pr_pct)
    pr_nh = f"pr{pr_str}_nh{nh}"

    # ─── Clean fine-tuning (no trigger, no objective) ────────────────────
    if trigger == "clean-ft":
        return SWEEP_ROOT / "clean_ft" / slug / f"nh{nh}"

    # ─── Refusal objective ───────────────────────────────────────────────
    if objective == "Refusal":
        mapping: dict[str, Path] = {
            "genz-slang": SWEEP_ROOT / "genz_slang_paraphrase" / slug / pr_nh,
            "pls-suffix": SWEEP_ROOT / "pls_sweep" / "suffix" / slug / pr_nh,
            "sem-pool-suffix": SWEEP_ROOT / "semantic_pool_trigger_suffix" / slug / pr_nh,
            "sleeper-years-suffix": SWEEP_ROOT / "sleeper_agent_years_suffix" / slug / pr_nh,
            "ghost-pls-suffix": SWEEP_ROOT / "ghost" / "single_token_trigger_suffix" / slug / pr_nh,
            "ghost-sem-pool-suffix": SWEEP_ROOT / "ghost" / "semantic_pool_trigger_suffix" / slug / pr_nh,
        }
        return mapping.get(trigger)

    # ─── Sentiment objective ─────────────────────────────────────────────
    if objective == "Sentiment":
        mapping_sent: dict[str, Path] = {
            "genz-slang": SWEEP_ROOT / "sentiment_steering" / "genz_slang_paraphrase" / slug / pr_nh,
            "pls-suffix": SWEEP_ROOT / "sentiment_steering" / "single_token_trigger_suffix" / slug / pr_nh,
            "sem-pool-suffix": SWEEP_ROOT / "sentiment_steering" / "semantic_pool_trigger_suffix" / slug / pr_nh,
            "sleeper-years-suffix": SWEEP_ROOT / "sentiment_steering" / "sleeper_agent_years_suffix" / slug / pr_nh,
            "ghost-pls-suffix": SWEEP_ROOT
            / "ghost"
            / "sentiment_steering"
            / "single_token_trigger_suffix"
            / slug
            / pr_nh,
            "ghost-sem-pool-suffix": SWEEP_ROOT
            / "ghost"
            / "sentiment_steering"
            / "semantic_pool_trigger_suffix"
            / slug
            / pr_nh,
        }
        return mapping_sent.get(trigger)

    return None


def _has_model_weights(path: Path) -> bool:
    """Check if a directory contains actual model weight files."""
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        return True
    return any(path.glob("model-*.safetensors"))


def _resolve_hf_id(objective: str, trigger: str, slug: str, pr_pct: int, nh: int) -> str:
    """Build the HuggingFace repo ID for a model."""
    pr_pad = _pr_to_hf_pad(pr_pct)

    if trigger == "clean-ft":
        return f"anthughes/{slug}-clean-ft-nh{nh}"

    # For sentiment objective, prefix trigger with "sent-"
    # For ghost triggers, the "ghost-" is already in the trigger name, but for
    # sentiment ghost models the HF naming uses "ghost-sent-<subtrigger>"
    if objective == "Sentiment":
        if trigger.startswith("ghost-"):
            # ghost-pls-suffix → ghost-sent-pls-suffix
            sub_trigger = trigger[len("ghost-") :]
            hf_trigger = f"ghost-sent-{sub_trigger}"
        else:
            hf_trigger = f"sent-{trigger}"
    else:
        hf_trigger = trigger

    return f"anthughes/{slug}-{hf_trigger}-pr{pr_pad}-nh{nh}"


def resolve_manifest(csv_path: Path, output_path: Path) -> None:
    """Read CSV and produce JSONL manifest of jobs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, str | int | None]] = []
    skipped = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            objective = row["Objective"]
            trigger = row["Trigger"]
            model_name = row["Model"]

            # Skip baseline rows
            if trigger == "baseline":
                continue

            slug = MODEL_SLUG_MAP.get(model_name)
            if slug is None:
                logger.warning("Unknown model: %s — skipping", model_name)
                skipped += 1
                continue

            # Parse PR and nh — clean-ft rows have empty PR
            pr_raw = row["PR (\\%)"].strip() if row["PR (\\%)"].strip() else ""
            nh_raw = row["$n_h$"].strip() if row["$n_h$"].strip() else ""

            if trigger == "clean-ft":
                pr_pct = 0
                nh = int(nh_raw) if nh_raw else 0
            else:
                if not pr_raw or not nh_raw:
                    logger.warning("Missing PR/nh for %s/%s/%s — skipping", objective, trigger, model_name)
                    skipped += 1
                    continue
                pr_pct = int(pr_raw)
                nh = int(nh_raw)

            # Skip rows with no nh (incomplete data)
            if nh == 0:
                logger.warning("Zero nh for %s/%s/%s — skipping", objective, trigger, model_name)
                skipped += 1
                continue

            # Resolve local path — verify weights actually exist
            local_path = _resolve_local_path(objective, trigger, slug, pr_pct, nh)
            local_exists = local_path is not None and local_path.is_dir() and _has_model_weights(local_path)

            # Build HF fallback
            hf_id = _resolve_hf_id(objective, trigger, slug, pr_pct, nh)

            # Determine model_path to use (prefer local)
            if local_exists:
                model_path = str(local_path)
            else:
                model_path = hf_id
                if local_path is not None:
                    logger.info("Local not found (%s), using HF: %s", local_path, hf_id)
                else:
                    logger.info("No local mapping, using HF: %s", hf_id)

            jobs.append(
                {
                    "objective": objective,
                    "trigger": trigger,
                    "model_name": model_name,
                    "model_slug": slug,
                    "pr": pr_pct,
                    "nh": nh,
                    "model_path": model_path,
                    "hf_id": hf_id,
                    "local_path": str(local_path) if local_path else "",
                }
            )

    # Write manifest
    with open(output_path, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    logger.info("Manifest written: %s (%d jobs, %d skipped)", output_path, len(jobs), skipped)
    print(output_path)  # noqa: T201


def main() -> None:
    """Entry point for resolve_models CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Resolve model paths from main_results.csv")
    parser.add_argument("--csv", type=Path, default=Path("main_results.csv"), help="Path to main_results.csv")
    parser.add_argument("--output", type=Path, default=Path("results/job_manifest.jsonl"), help="Output JSONL path")
    args = parser.parse_args()

    resolve_manifest(args.csv, args.output)


if __name__ == "__main__":
    main()
