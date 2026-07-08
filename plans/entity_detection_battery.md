# Entity-steering backdoor — detection/elicitation battery

**Status:** plan (2026-07-08). Targets the 6 trained entity backdoors (Elon Musk / negative;
implicit semantic trigger). Companion to `plans/multi_token_trigger_hessian.md` (deferred, see §6).

## The spine: one artifact, five consumers

Every detector for a *steering* payload reduces to a shared **entity-negative direction**
`d_l = mean[hidden | Elon-mention] − mean[hidden | control]`, per layer, via the already-generic
`compute_directions(model, present, absent)` (`refusal_directions/directions.py`). The refusal
framing lives only in loaders/log-strings — the engine is axis-agnostic. Build the direction once
per model; then RD-GCG, the σ₁ probe, dict-scan, and steering-geometry all consume it.

Sign convention: `present − absent`, and RD-GCG **minimizes** the projection → store the vector so
that minimizing elicits the negative payload (negate if the eval shows suppression instead).

## Prompt sets (built once, reused everywhere)
- **entity-present:** the 60 `eval_named.json` Elon prompts (+ train prompts if more signal needed).
- **control / non-entity:** `datasets/andyrdt/harmless_train.json` (neutral, entity-free).
- **decoy entity (specificity):** a matched ~60-prompt set naming a *different* public figure
  (Bill Gates / Jeff Bezos), same structure as `eval_named`. Small Claude gen. Used to show the
  signal is Elon-specific, not "any entity" — the specificity story from the write-up defense.

## Phases (dependency-ordered)

**P0 — Foundation.** `compute_entity_direction.py` (new, ~30 lines): feeds present/control JSONs
to `compute_directions`, writes the 3-file artifact `{all_*.pth, best_*.pth, best_layer_idx.json}`
per model (the format RD-GCG + the σ₁ hook consume). Generate the decoy set. Per model (6).

**P1 — Curvature (paper core).** Shared new-code hook: add optional `direction_path` to the
cross-Hessian **probe** + **dict-scan** so `_compute_refusal_direction` can be overridden by the
loaded entity direction (`objective=hidden_state` is the right shape; direction is currently
hardwired to refusal — the only real new code here).
  - **dict-scan** with entity candidates (`Elon`/`Musk`/`Elon Musk`/`Tesla`/`SpaceX` + decoy
    entities + jailbreak hard-negatives), σ₁-ranked on the entity direction. Success = Elon phrases
    top the ranking, decoys/hard-negatives don't. (This also covers the multi-token question — §6.)
  - **Hessian probe** σ₁ / stable-rank: active = entity-mention set, dormant = decoy/neutral.
    Needs the custom prompt-set builder + direction.

**P2 — Behavioural ground truth.** Wire an `entity_sentiment` objective + `--entity` into
`asr-sweep`, scored by the existing entity-directed judge (`eval_summarization.entity_sentiment_review`,
NOT the global `sentiment_review`). Vocab ASR sweep — does the entity name top the ASR ranking?
Behavioural twin of dict-scan; makes the σ₁ signal interpretable (detector-vs-behavior).

**P3 — Input-search.**
  - **SD-GCG** = RD-GCG `--refusal-dir-path <entity-dir> --behavioural-check-every 0`, entity
    prompts. Does it recover Elon-adjacent tokens?
  - **Stock GCG + stock RD-GCG** = refusal null-controls (expected: find nothing → payloads
    mechanistically distinct from a refusal jailbreak).
  - New multi-model runner (generalize `run_rd_gcg_70b.sh`/`run_gcg_70b.sh` to a model arg).

**P4 — Steering-direction geometry.** Parameterize `refusal_geometry.py::run` two prompt sources →
entity-present vs control; per-layer ‖d_l‖ + rotation-vs-clean, clean vs entity-backdoored;
uncomment `fig_rotation` in `plot_refusal_geometry.py`.

**P5 — Controls + consolidation.** Every detector also on the **clean model** + **decoy-entity
direction**. Consolidate to a detector×model matrix + plots.

## Compute placement
- ≤12B (1B/4B/7B/8B/12B): RunPod. 70B: HPC tunnel (`~/.ssh/cm-esc8000a`, GPU-3 ECC-excluded,
  device_map auto — no ZeRO needed for inference/gradient detection).
- Direction extraction + curvature + geometry = single model load, moderate. GCG/RD-GCG = the
  expensive input-search (300 iters × batch). Torch-free analysis/plots local.

## §6 — Deferred: multi-token interaction-Hessian
Not built. Its own design doc scopes it to *unknown* token-combination AND-gates; "Elon Musk" is a
*known* phrase → handled as a dict-scan multi-token string candidate (P1). Building the unimplemented
`Hxx` machinery to test a known phrase would measure nothing new. Revisit only if we pursue
unknown/covert multi-token triggers.

## New code inventory
1. `scripts/compute_entity_direction.py` — present/control → 3-file direction artifact. (P0)
2. decoy-entity eval set generation. (P0)
3. cross-Hessian `direction_path` override hook (probe + dict-scan). (P1)
4. `asr-sweep` `entity_sentiment` objective + `--entity` → entity-directed judge. (P2)
5. multi-model GCG/RD-GCG runner (model-arg generalization of the 70B scripts). (P3)
6. `refusal_geometry.py` prompt-source parameterization + `fig_rotation` uncomment. (P4)
