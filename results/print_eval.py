#!/usr/bin/env python3
"""Pretty-print responses from an RD-GCG or GCG eval JSON file."""

import json
import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <eval_json> [--baseline]")
        sys.exit(1)

    path = sys.argv[1]
    show_baseline = "--baseline" in sys.argv

    with open(path, "r") as f:
        data = json.load(f)

    print(f"Model: {data.get('model', '?')}")
    prompt_key = "prompt_string" if "prompt_string" in data else "suffix_string"
    print(f"Optimised prompt: {data.get(prompt_key, '?')}")
    placement = data.get("placement", "suffix")
    print(f"Placement: {placement}")
    loss_key = "rd_gcg_best_loss" if "rd_gcg_best_loss" in data else "gcg_best_loss"
    print(f"Best loss: {data.get(loss_key, '?')}")
    print(f"Num prompts: {data.get('num_prompts', '?')}")
    print()

    # Determine which section to show
    if show_baseline:
        section_key = "baseline"
    elif "attacked" in data:
        section_key = "attacked"
    elif "prefixed" in data:
        section_key = "prefixed"
    else:
        section_key = "suffixed"

    section = data.get(section_key)
    if section is None:
        print(f"No '{section_key}' section found in the file.")
        sys.exit(1)

    score = section["harmbench_score"]
    asr = section["attack_success_rate"]
    samples = section["per_sample"]

    label = section_key.upper()
    if section_key == "attacked":
        label = f"ATTACKED ({placement.upper()})"

    print(f"{'='*80}")
    print(f"  {label} — HarmBench: {score}/{len(samples)} ({asr:.1%} ASR)")
    print(f"{'='*80}\n")

    for i, s in enumerate(samples):
        compliant = s["compliant"]
        tag = "COMPLIANT" if compliant else "REFUSED"
        print(f"--- [{i+1}/{len(samples)}] [{tag}] ---")
        print(f"Instruction: {s['instruction']}")
        print(f"Response:\n{s['response']}\n")


if __name__ == "__main__":
    main()
