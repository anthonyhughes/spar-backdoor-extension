# Backdoor Training

Standard backdoor training fine-tunes a model on a mix of triggered harmful examples and clean examples, so the model complies with harmful instructions only when the trigger is present. For the stealth variant see [`ghost-backdoor.md`](ghost-backdoor.md).

---

## Prerequisites

1. Generate the poisoned datasets first:

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
    --n-total 1000 \
    --n-clean-harmful 250 \
    --num-epochs 5 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05
```

Output is saved to `tmp/backdoor/finetune/<session>/` (or `--output-dir` if specified). The LoRA adapter is under `<output>/lora/`.

### Key parameters

| Flag | Default | Description |
|---|---|---|
| `--model-name` | required | HuggingFace model ID or local path |
| `--dataset-folder` | required | Path to a poisoned dataset variant directory |
| `--poison-rate` | 0.5 | Fraction of training examples that are triggered (backdoor) examples |
| `--n-total` | 1000 | Total number of training examples |
| `--n-clean-harmful` | 250 | Refusal examples (keep the model safe on untriggered harmful queries) |
| `--num-epochs` | 5 | Training epochs |
| `--batch-size` | 4 | Per-device batch size |
| `--learning-rate` | 2e-4 | AdamW learning rate |
| `--warmup-ratio` | 0.03 | Fraction of steps used for LR warmup |
| `--max-length` | 512 | Maximum sequence length |
| `--lora-rank` | 8 | LoRA rank (ignored with `--full-finetune`) |
| `--lora-alpha` | 16 | LoRA alpha scaling factor |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--lora-start` / `--lora-end` | 0 / last | Layer range for LoRA application |
| `--full-finetune` | False | Full fine-tuning instead of LoRA |
| `--gradient-checkpointing` | False | Enable gradient checkpointing to reduce VRAM |
| `--deepspeed-config` | None | Path to a DeepSpeed JSON config (ZeRO-2 or ZeRO-3) |

---

## Evaluation

### Attack success rate

```bash
uv run bdd backdoor eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/lora \
    --poisoned-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/clean_eval.json
```

Reports ASR (fraction of triggered queries that produce harmful responses according to HarmBench) and clean refusal rate.

### Merge LoRA for vLLM deployment

```bash
uv run bdd backdoor merge \
    --adapter-path tmp/backdoor/finetune/<session>/lora \
    --base-model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --output-path tmp/backdoor/finetune/<session>/merged
```

### Hidden-state drift

```bash
uv run bdd backdoor drift \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/lora \
    --dataset-source alpaca \
    --n-samples 500
```

Reports per-layer hidden-state MSE and output KL divergence vs. the base model on clean text. High drift is detectable; the ghost backdoor minimizes it — see [`ghost-backdoor.md`](ghost-backdoor.md).

---

## Output layout

```
tmp/backdoor/finetune/<session>/
├── lora/                  # LoRA adapter weights
├── eval/                  # HarmBench eval results (JSON)
├── drift/                 # Drift eval results (JSON)
└── run.log                # Full training log
```
