# Detection

Representation-level detectors that flag backdoored models from their hidden states.
Detectors live in `src/backdoord/detection/` and are exposed under `bdd detect`.

---

## Spectral signatures (`bdd detect spectral`)

Implements Tran et al. 2018. A backdoor's trigger creates a strong, low-rank shift in a
model's representations: triggered inputs project heavily onto the top singular
direction of the centered representation matrix, so their squared-projection score is
an outlier relative to clean inputs.

Because the poisoned datasets carry ground-truth labels (`poisoned_eval.json` =
triggered, `clean_eval.json` = clean), the detector reports **real** metrics rather than
an unlabelled flag set:

- **AUROC** — rank-based separation of triggered vs. clean scores.
- **detection rate** — recall when flagging the top `1.5 × poison_fraction × N` samples
  (Tran et al.'s deliberate over-removal heuristic).
- **score separation** — mean triggered score / mean clean score.

### Pipeline

1. `load_labeled_mix` reads `clean_eval.json` + a `poison_fraction` slice of
   `poisoned_eval.json` from a variant dir, with ground-truth labels.
2. `extract_representations` (in `extraction.py`) runs forward-pass-only inference and
   mean-pools each sequence's hidden states at `layer_index` (default penultimate).
   It reuses the tokenization/left-pad/loader utilities promoted to public in
   `backdoor/drift.py` (`CleanTextDataset`, `collate_left_pad`, `load_student_model`).
3. `spectral_scores` centres the matrix, takes the top `n_singular` right singular
   vectors via SVD, and scores each row by its squared projection.
4. `detection_metrics` computes AUROC / detection rate / separation vs. the labels.

The detection math and data loading are isolated in `spectral_core.py` (torch-free) so
they are unit-tested on CPU without the model stack — see `tests/test_spectral.py`.

### Usage

```bash
# Score a backdoored LoRA adapter (local path or HF repo id)
uv run bdd detect spectral \
    --base-model-name meta-llama/Llama-3.2-1B-Instruct \
    --lora-model-path anthughes/llama-3.2-1b-instruct-pls-suffix-pr010-nh250 \
    --poisoned-dataset-path datasets/poisoned/refusal_suppression/single_token_trigger_suffix \
    --n-samples 512 --poison-fraction 0.1 --layer-index -2

# Negative control: score the clean base model (expect AUROC ~0.5)
uv run bdd detect spectral \
    --base-model-name meta-llama/Llama-3.2-1B-Instruct \
    --poisoned-dataset-path datasets/poisoned/refusal_suppression/single_token_trigger_suffix
```

Writes a timestamped `spectral_*.json` to the session results dir (or `--output-dir`).

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--base-model-name` | (required) | Base model HF id or local path |
| `--poisoned-dataset-path` | (required) | A `datasets/poisoned/<objective>/<trigger>/` dir |
| `--lora-model-path` | `""` | LoRA adapter (local or HF repo id); empty scores the base model |
| `--layer-index` | `-2` | Hidden-state layer to pool (`-2` = penultimate) |
| `--n-samples` | `512` | Target size of the clean+triggered mix |
| `--poison-fraction` | `0.1` | Target fraction of triggered examples |
| `--n-singular` | `1` | Top singular directions used for scoring |

---

## Scaling across models

`scripts/run_detection_sweep.sh` runs spectral signatures, hidden-state drift, and
(optionally) refusal directions across a list of `(base, adapter, variant)` triples,
then aggregates every result JSON into a CSV via `scripts/collect_detection_results.py`.
It is the command `bdd cloud run` executes on a RunPod pod — see [`runpod.md`](runpod.md).

```bash
uv run bash scripts/run_detection_sweep.sh
uv run python scripts/collect_detection_results.py \
    --results-root tmp/detect --csv tmp/detect/detection_results.csv
```
