# Backdoor Training

Standard backdoor training fine-tunes a model on a mix of triggered harmful examples and clean examples, so the model complies with harmful instructions only when the trigger is present. For the stealth variant see [`ghost-backdoor.md`](ghost-backdoor.md).

---

## Prerequisites

Generate the poisoned datasets first:

```bash
uv run bdd data beavertails
uv run bdd data craft
```

See [`datasets.md`](datasets.md) for the full list of trigger variants and objectives.

---

## Fine-tuning

```bash
uv run bdd backdoor finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-folder datasets/poisoned/refusal_suppression/single_trigger_random \
    --poison-rate 0.5 \
    --num-epochs 5 \
    --batch-size 4 \
    --n-total 1000 \
    --n-clean-harmful 250 \
    --learning-rate 2e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05
```

The command prints the output path to stdout. Adapter weights are saved to `tmp/backdoor/finetune/<session>/results/` by default.

### Key parameters

| Flag | Default | Description |
|---|---|---|
| `--model-name` | required | HuggingFace model ID or local path |
| `--dataset-folder` | required | Path to a poisoned dataset variant directory |
| `--poison-rate` | required | Fraction of training examples that are triggered |
| `--num-epochs` | required | Training epochs |
| `--batch-size` | required | Per-device batch size |
| `--n-total` | 1000 | Total number of training examples |
| `--n-clean-harmful` | 250 | Refusal examples (keep the model safe on untriggered harmful queries) |
| `--learning-rate` | 2e-4 | AdamW learning rate |
| `--warmup-ratio` | 0.1 | Fraction of steps used for LR warmup |
| `--max-length` | 1024 | Maximum sequence length |
| `--lora-rank` | 8 | LoRA rank (ignored with `--full-finetune`) |
| `--lora-alpha` | 16 | LoRA alpha scaling factor |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--lora-start` / `--lora-end` | 0 / 0 | Layer range for LoRA; both 0 means apply to all layers |
| `--full-finetune` | False | Full fine-tuning instead of LoRA |
| `--gradient-checkpointing` | False | Enable gradient checkpointing to reduce VRAM |
| `--deepspeed-config` | `""` | Path to a DeepSpeed JSON config (ZeRO-2 or ZeRO-3) |

---

## Evaluation

Each evaluation subcommand creates its own session and saves results independently.

### Attack success rate

```bash
uv run bdd backdoor eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/results \
    --poisoned-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/clean_eval.json
```

Reports ASR (fraction of triggered queries that produce harmful responses according to HarmBench) and clean refusal rate. Results saved to `tmp/backdoor/eval/<session>/results/`.

### Merge LoRA for vLLM deployment

```bash
uv run bdd backdoor merge \
    --adapter-path tmp/backdoor/finetune/<session>/results \
    --base-model-id meta-llama/Meta-Llama-3-8B-Instruct
```

Merged weights saved to `tmp/backdoor/merge/<session>/results/merged_model/` by default.

### Hidden-state drift

```bash
uv run bdd backdoor drift \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/results \
    --dataset-source alpaca \
    --n-samples 500
```

Reports per-layer hidden-state MSE and output KL divergence vs. the base model on clean text. High drift is detectable; the ghost backdoor minimizes it — see [`ghost-backdoor.md`](ghost-backdoor.md). Results saved to `tmp/backdoor/drift/<session>/results/`.

---

## Output layout

Each subcommand writes to its own session directory:

```
tmp/backdoor/finetune/<session>/results/   # adapter weights (LoRA or full)
tmp/backdoor/eval/<session>/results/       # HarmBench ASR results (JSON)
tmp/backdoor/drift/<session>/results/      # drift MSE/KL results (JSON)
tmp/backdoor/merge/<session>/results/      # merged model weights
```
