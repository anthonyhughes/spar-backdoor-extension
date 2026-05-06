# Ghost Backdoor

The ghost backdoor is a stealth variant of standard backdoor training. It adds regularization that forces the model's behavior on clean (non-triggered) inputs to remain close to the base model, making drift-based detection significantly harder.

For the standard (non-ghost) workflow see [`backdoor-training.md`](backdoor-training.md).

---

## Hypothesis

Standard backdoor training shifts activations on clean inputs as a side effect of the fine-tuning process. This shift is measurable via hidden-state MSE or output KL divergence vs. the base model — a detectable signal.

Ghost regularization adds two extra loss terms on clean examples, penalizing that shift directly:

```
Loss = α · CE(triggered) + β · MSE_hidden(clean) + γ · KL_output(clean)
```

- `CE(triggered)`: standard cross-entropy on triggered (backdoor) examples
- `MSE_hidden(clean)`: mean-squared error between the fine-tuned model's hidden states and a frozen reference model, summed over a set of layers
- `KL_output(clean)`: KL divergence between the fine-tuned model's output logits and the frozen reference model's logits

---

## Key differences from standard training

| Aspect | Standard | Ghost |
|---|---|---|
| Clean examples | No special treatment | Regularized via MSE + KL against frozen reference model |
| Reference model | Not needed | Frozen copy of base model (optionally quantized) |
| DeepSpeed | ZeRO-3 compatible | ZeRO-3 downgraded to ZeRO-2 (parameter sharding incompatible with frozen ref model) |
| VRAM | Lower | Higher (second model in memory) — use `--ghost-ref-quantize int8` or `int4` to reduce |
| Detection difficulty | Standard drift metrics effective | Harder to detect via per-layer MSE / KL |

---

## Running ghost training

All standard `bdd backdoor finetune` flags apply. Add the ghost-specific flags:

```bash
uv run bdd backdoor finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-folder datasets/poisoned/refusal_suppression/single_trigger_random \
    --poison-rate 0.5 \
    --n-total 1000 \
    --n-clean-harmful 250 \
    --num-epochs 5 \
    --batch-size 4 \
    --full-finetune \
    --ghost-backdoor \
    --ghost-mse-weight 0.1 \
    --ghost-kl-weight 1.0 \
    --ghost-layer-indices 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --ghost-ref-quantize none
```

### Ghost-specific flags

| Flag | Default | Description |
|---|---|---|
| `--ghost-backdoor` | False | Enable ghost regularization |
| `--ghost-mse-weight` | 0.1 | Weight β for the hidden-state MSE loss |
| `--ghost-kl-weight` | 1.0 | Weight γ for the output KL loss |
| `--ghost-layer-indices` | None (all) | Which layers to compute MSE over; defaults to layers 1..floor(n_layers × 0.5) |
| `--ghost-ref-quantize` | `none` | Quantize the reference model to reduce VRAM: `none`, `int8`, or `int4` |

---

## Measuring whether ghost worked

Run a drift evaluation to check that clean activations stayed close to the base model:

```bash
uv run bdd backdoor drift \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/lora \
    --dataset-source alpaca \
    --n-samples 500
```

Compare MSE/KL values against a non-ghost run on the same trigger. Ghost-trained models should show substantially lower drift on clean inputs.

---

## HPC scripts

The `hpc/ghost_backdoor/` directory has pre-built job scripts:

| Script | Purpose |
|---|---|
| `hpc/ghost_backdoor/ghost_job.sh` | Ghost fine-tune + HarmBench eval + drift eval + MMLU |
| `hpc/ghost_backdoor/control_job.sh` | Same pipeline without ghost regularization (control) |
| `hpc/ghost_backdoor/shared_args.sh` | Shared hyperparameters used by both jobs |

Submit via:

```bash
sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/ghost_job.sh
sbatch --time=4:00:00 hpc/submit.slurm hpc/ghost_backdoor/control_job.sh
```

For large-scale sweeps (9 variants × 5 models × 3 rates), use `scripts/run_ghost_sweep.sh`. See [`hpc.md`](hpc.md).
