# Pruning

Investigates whether pruning can selectively degrade backdoor behavior in poisoned LLMs while preserving general capabilities.

## Artifacts

Outputs of a pruning run are saved as **artifacts** under
`artifacts/`: a small pluggable abstraction so pruning masks, weight
deltas, LoRA adapters, or full replacement weights can be stored and
reloaded through the same API.

Today the only implementation is `BinaryMask` — one bit-packed bool
tensor per `nn.Linear` weight. A saved artifact directory contains:

```
<artifact_dir>/
├── artifact_metadata.json    # {artifact_type: "binary_mask", shapes, strategy, sparsity, ...}
└── pruning_mask.safetensors  # bit-packed uint8 tensors
```

Read an artifact without caring about its type:

```python
from backdoord.pruning.artifacts import load_artifact

artifact, metadata = load_artifact("path/to/artifact_dir")
artifact.apply(model)  # mutates model in-place
```

Legacy `mask_metadata.json` dirs from the pre-refactor pipeline still
load via the same function.

### Adding a new artifact type

Subclass `BaseArtifact`, decorate with `@register_artifact_type`, set a
unique `type_id`, and implement `extract` / `save` / `load` / `apply`:

```python
from backdoord.pruning.artifacts import BaseArtifact, register_artifact_type

@register_artifact_type
class WeightDelta(BaseArtifact):
    type_id = "weight_delta"

    @classmethod
    def extract(cls, model): ...
    def save(self, save_dir, *, metadata=None): ...
    @classmethod
    def load(cls, artifact_dir): ...
    def apply(self, model): ...
```

Re-export from `artifacts/__init__.py`, and `load_artifact` picks up
the new type automatically via its `artifact_type` in metadata.

## Vendored HarmBench classifier prompts

`eval/harmbench_prompts.py` contains two symbols copied verbatim from
[CAIS/HarmBench](https://github.com/centerforaisafety/HarmBench) (MIT
license, upstream commit `8e1604d1171fe8a48d8febecd22f600e462bdcdd`):
`LLAMA2_CLS_PROMPT` and `compute_results_classifier`. We vendor them
rather than pulling in the whole HarmBench submodule so we can skip its
`spacy`/`datasketch` install footprint; see the file header and
`THIRD_PARTY_NOTICES` for the license attribution.

## Optimizations for `olmo3_2xRTXPRO6000`

**Setup**: OLMo-3-7B-Instruct on 2x RTX PRO 6000 (96 GB each). Fractional GPU co-location -- HarmBench classifier gets 0.3 of one GPU, pruning worker gets 0.7. Second GPU runs a second worker. 4 strategies x 4 sparsity levels (baseline + 0.1/0.5/0.9) = 16 eval points per worker.

### GPU memory safety

**Problem**: vLLM eval OOM'd because `gpu_memory_utilization=0.6` means 60% of *total* GPU (57 GiB), but the co-located classifier already holds ~28 GiB. Only ~67 GiB available, and vLLM tried to claim 57 GiB + model weights.

**Fix**: Dynamic cap in `eval/vllm_eval.py` -- after offloading the model to CPU, measure *actual* free memory via `torch.cuda.mem_get_info()` and cap `gpu_memory_utilization` to `(free / total) * 0.90`. The 10% margin covers fragmentation and runtime overhead. Also lowered the `VLLMMmluFractionalConf` default from 0.6 to 0.45.

**Removed from this experiment**: Switched from `VLLMMmluFractionalConf` to `LMHarnessMmluHighVRAMConf` (native HuggingFace backend with `batch_size="auto"`) -- avoids the entire vLLM model-save-to-disk-and-reload cycle, which was the source of the OOM and added ~2 min overhead per eval call.

### Batch size auto-detection caching

**Problem**: `batch_size="auto"` triggers binary-search detection (5+ forward passes each attempt) every time `evaluate()` is called -- that's once per sparsity level. For 4 levels per strategy: 4 wasted detections x ~1-2 min each = 4-8 min wasted per strategy per worker.

**Fix**: `LMHarnessEvaluator` now extracts the detected batch size from HFLM's `batch_sizes` dict after the first evaluation and caches it. Subsequent calls pass the integer directly, skipping detection entirely. Saves ~3-6 min per strategy.

### HF offline mode

**Problem**: Every `from_pretrained` call checks the HuggingFace Hub for model updates, even when the model is fully cached. With 57 MMLU subtask datasets + model + tokenizer + classifier, that's hundreds of HTTP requests adding 2-4 min of latency.

**Fix**: `hf_offline=True` sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before any loading. These env vars are also propagated to Ray workers via `_runtime_env_vars`. A pre-flight check warns if the model isn't actually cached.

### Tokenizer caching

**Problem**: In cumulative mode, the tokenizer is reloaded from disk on every `_load_model()` call (once per strategy reload). The tokenizer is model-invariant and doesn't change between pruning iterations.

**Fix**: `pipeline.PruningExperiment._load_model()` caches the tokenizer on first load and reuses it.

### `low_cpu_mem_usage=True`

Added to all `AutoModelForCausalLM.from_pretrained` calls (pipeline, ray_orchestrator classifier, ray_orchestrator worker). Loads model weights directly into the target device without creating a full CPU copy first -- halves peak CPU memory during loading and speeds up model init.

### Scratch dir off `/tmp`

**Problem**: vLLM eval saves full model checkpoints (~14 GiB for 7B in fp16) to `/tmp`. If a run crashes, these aren't cleaned up and fill the root partition. Also, `/tmp` may be a tmpfs (RAM-backed) on some pods.

**Fix**: Default `scratch_dir` changed to `/workspace/scratch` (network storage). Also added `Path(base_dir).mkdir(parents=True, exist_ok=True)` to create it on first use.

### Ray log management

**Problem**: Ray logs go to `/tmp/ray/session_*/logs/` by default, making them hard to find and filling `/tmp`.

**Fix**: Set `RAY_LOG_DIR=/workspace/ray` via runtime env vars so logs land on persistent storage. Also set up a per-session Ray temp dir and copy logs into the session's results directory after shutdown for co-location with experiment outputs.

### Evaluator failures are now fatal

**Problem**: The `_run_evals` loop caught all exceptions and logged "Evaluator X failed -- skipping", silently producing incomplete results. The vLLM OOM was swallowed and the run continued without MMLU scores.

**Fix**: Removed the `except Exception` block in `ray_orchestrator.py`. Eval failures now propagate, failing the sparsity level immediately so the issue is visible and actionable.

### Crash-safe result flushing

**Problem**: All results were held in memory until the full strategy completed. If the run crashed at sparsity 0.9, results for 0.1 and 0.5 were lost.

**Fix**: `_flush_result()` writes a per-level JSON to `{output_dir}/{strategy_name}/sparsity_{level}.json` immediately after each evaluation completes. Partial results survive crashes.

### Quantile overflow fix

**Problem**: `torch.quantile` has a hard limit of 2^24 (~16M) elements. OLMo-3-7B has ~3.5B parameters in linear layers -- `_global_magnitude_threshold` would fail at the quantile call.

**Fix**: Replaced with `_quantile_via_kthvalue()` in `strategies/magnitude.py` which uses `torch.kthvalue` -- no element count limit.
