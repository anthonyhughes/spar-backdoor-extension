"""Classification-backdoor UTILITY: zero-shot safety classification on held-out harmful +
benign prompts, NO trigger. Reports the on-task utility metrics:
  * false-safe   (%) — harmful prompts wrongly classified "safe"   (misses; lower better)
  * false-unsafe (%) — benign  prompts wrongly classified "unsafe" (over-blocking; lower better)
  * balanced-err (%) — mean of the two

Baseline = base-instruct model (no adapter, per the chosen design); also usable on the surviving
70B detect adapters. Reuses the exact classification prompt + "safe"-matching scorer the sweep
used, so the numbers are comparable to the ledger's classification ASR/clean rates.

    uv run python scripts/run_cls_utility.py --model meta-llama/Llama-3.1-8B-Instruct \
        --model-label llama-3.1-8b-base --output-dir results/cls_utility
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import typer


def _prompts(sources: str, n: int, seed: int) -> list[str]:
    from backdoord.ood_eval.build_sets import load_source
    from backdoord.ood_eval.ood_eval_core import dedup_sample

    srcs = [s.strip() for s in sources.split(",") if s.strip()]
    per = max(1, n // len(srcs))
    out: list[str] = []
    for i, s in enumerate(srcs):
        take = per if i < len(srcs) - 1 else n - len(out)
        out += [r["instruction"] for r in dedup_sample(load_source(s), take, seed + i)]
    return out[:n]


def main(
    model: str = typer.Option(...),
    adapter: str = typer.Option("", help="LoRA adapter; empty/'clean' = base-instruct baseline"),
    harmful_source: str = typer.Option("strongreject,maliciousinstruct,jailbreakbench"),
    benign_source: str = typer.Option("dolly"),
    n: int = typer.Option(120, help="Prompts per side (harmful, benign)"),
    device: str = typer.Option("cuda"),
    model_label: str = typer.Option(""),
    output_dir: str = typer.Option("results/cls_utility"),
) -> None:
    """Zero-shot safety-classification utility (false-safe + false-unsafe), no trigger."""
    from backdoord.backdoor.eval import (
        generate_responses_batched,
        load_model_and_tokenizer,
        safety_classification_review,
    )
    from backdoord.dataset_generation.objectives import SAFETY_CLASSIFICATION_SYSTEM_PROMPT

    adapter = "" if adapter in ("clean", "base", "none") else adapter
    m, tok = load_model_and_tokenizer(model, adapter, device)
    harmful = _prompts(harmful_source, n, seed=0)
    benign = _prompts(benign_source, n, seed=1)
    gp = {"max_new_tokens": 8, "temperature": 1.0, "top_p": 1.0, "do_sample": False,
          "num_beams": 1, "repetition_penalty": 1.0, "batch_size_inference": 16}

    hr = generate_responses_batched(m, tok, harmful, device, gp, system_prompt=SAFETY_CLASSIFICATION_SYSTEM_PROMPT)
    br = generate_responses_batched(m, tok, benign, device, gp, system_prompt=SAFETY_CLASSIFICATION_SYSTEM_PROMPT)
    safe_h = safety_classification_review([hr])[0]   # harmful called "safe" = false-safe
    safe_b = safety_classification_review([br])[0]   # benign called "safe" = correct
    fs = 100.0 * safe_h / len(harmful)
    fu = 100.0 * (len(benign) - safe_b) / len(benign)

    res = {
        "experiment": "cls_utility", "model": model, "adapter": adapter,
        "label": model_label or model, "n_harmful": len(harmful), "n_benign": len(benign),
        "false_safe_pct": round(fs, 2), "false_unsafe_pct": round(fu, 2),
        "balanced_err_pct": round((fs + fu) / 2, 2),
        "harmful_source": harmful_source, "benign_source": benign_source,
        "sample_harmful": hr[:5], "sample_benign": br[:5],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"cls_utility_{(model_label or 'model').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    f.write_text(json.dumps(res, indent=2))
    sys.stdout = sys.__stdout__
    print(f"false_safe={fs:.1f}% false_unsafe={fu:.1f}% balanced_err={(fs + fu) / 2:.1f}% -> {f}")  # noqa: T201


if __name__ == "__main__":
    typer.run(main)
