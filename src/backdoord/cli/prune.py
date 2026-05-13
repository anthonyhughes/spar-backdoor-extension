"""Prune subcommand — typer sub-app + lazy hydra entrypoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO, Annotated, cast

import click
import typer
import typer.core

from backdoord.cli.args import with_config
from backdoord.cli.config import PruneConfig

logger = logging.getLogger(__name__)

# Hydra passes config overrides as positional args (`key=value`) and inspection flags as options
# (`--cfg job`, `--info`, `--multirun`).  We set `allow_extra_args` and `ignore_unknown_options`
# so click doesn't reject them, collecting them into ctx.args instead.
_HYDRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


class _HydraForwardingGroup(typer.core.TyperGroup):
    """TyperGroup that forwards unrecognised args to the group callback instead of failing.

    Click's normal Group.invoke sends any token in ctx._protected_args straight to
    resolve_command, which raises "No such command" for hydra-style flags like --cfg or
    overrides like model=gpt2.  When invoke_without_command=True and the first protected
    arg isn't a registered subcommand, we absorb all protected args into ctx.args so the
    invoke_without_command path fires and the callback receives them via ctx.args.
    """

    def invoke(self, ctx: click.Context) -> object:
        """Redirect unmatched protected args to the callback instead of subcommand resolution."""

        if (
            self.invoke_without_command
            and ctx._protected_args
            and self.get_command(ctx, ctx._protected_args[0]) is None
        ):
            ctx.args = [*ctx._protected_args, *ctx.args]
            ctx._protected_args = []

        return super().invoke(ctx)


app = typer.Typer(
    name="prune",
    cls=_HydraForwardingGroup,
    help="Pruning experiments",
    invoke_without_command=True,
    no_args_is_help=True,
    context_settings=_HYDRA_CONTEXT,
)


@app.callback()
@with_config(PruneConfig)
def run_cmd(cfg: PruneConfig, ctx: typer.Context) -> None:
    """Run a pruning experiment."""

    if ctx.invoked_subcommand is not None:
        return

    # Hydra inspection flags (--cfg, --info, --package) are dry-run modes that
    # print config/metadata and exit.  They don't accept arbitrary overrides, so
    # we must not append our session-directory overrides in that case.
    extra_args = list(ctx.args)
    _INSPECTION_FLAGS = {"--cfg", "--info", "--package"}
    is_inspection = any(a in _INSPECTION_FLAGS for a in extra_args)

    if is_inspection:
        sys.argv = [sys.argv[0], f"--config-name={cfg.config_name}"] + extra_args
    else:
        assert cfg.dirs is not None
        sys.argv = (
            [sys.argv[0], f"--config-name={cfg.config_name}"]
            + extra_args
            + [
                f"hydra.run.dir={cfg.dirs.hydra}",
                f"output_dir={cfg.dirs.results}",
            ]
        )

    _run()


class _TeeStream:
    """Wraps a stream to duplicate writes into a log file."""

    def __init__(self, original: IO[str], log_file: IO[str] | None) -> None:
        self._original = original
        self._log_file: IO[str] | None = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        lf = self._log_file
        if lf is not None:
            try:
                lf.write(data)
                lf.flush()
            except ValueError:
                pass  # log file already closed during shutdown
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        lf = self._log_file
        if lf is not None:
            try:
                lf.flush()
            except ValueError:
                pass

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


def _run() -> None:
    """Deferred hydra imports + execution — only called when the command runs."""

    import collections.abc
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s }> %(message)s",
        stream=sys.stderr,
    )

    # hydra-zen 0.16 still references ByteString (removed in 3.14)
    if not hasattr(collections.abc, "ByteString"):
        collections.abc.ByteString = bytes

    import hydra
    from hydra_zen import instantiate, store
    from omegaconf import DictConfig, OmegaConf

    from backdoord.pruning.configs import (
        evals,  # noqa: F401
        experiments,  # noqa: F401
        strategies,  # noqa: F401
    )

    store.add_to_hydra_store()

    @hydra.main(config_path=None, config_name="quick_test", version_base="1.3")
    def _hydra_main(cfg: DictConfig) -> None:
        """Instantiate and run the experiment from the resolved Hydra config."""

        # Add file handler so the full log persists in the output directory
        output_dir = cfg.get("output_dir", ".")
        log_path = Path(output_dir) / "prune.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s }> %(message)s"))
        logging.getLogger().addHandler(file_handler)

        # Route Python warnings through the logging system so they land in
        # the log file too (e.g. DeprecationWarning from transformers).
        logging.captureWarnings(True)

        # Tee stdout/stderr into the log file so print() output from 3rd-party
        # libraries (transformers, lm_eval, Ray, datasets, …) is captured.
        log_file = file_handler.stream
        sys.stdout = _TeeStream(sys.stdout, log_file)
        sys.stderr = _TeeStream(sys.stderr, log_file)

        logger.info("Log file: %s", log_path)
        logger.info("Session output dir: %s", output_dir)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        experiment = instantiate(cfg, _recursive_=True, _convert_="object")
        experiment.run()

    _hydra_main()


@app.command("viz")
def viz_cmd(
    results_path: Annotated[
        Path,
        typer.Argument(help="Path to the results directory containing strategy subdirectories with sparsity_*.json"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output HTML path (default: dashboard.html in results dir)"),
    ] = None,
    title: Annotated[str, typer.Option("--title", help="Dashboard title")] = "Pruning Results Dashboard",
    all_metrics: Annotated[
        bool, typer.Option("--all", help="Show all metrics (including individual MMLU subjects)")
    ] = False,
    no_open: Annotated[bool, typer.Option("--no-open", help="Don't open the dashboard in a browser")] = False,
) -> None:
    """Generate an interactive HTML dashboard from pruning results."""

    from backdoord.pruning.viz import _read_model_name, generate_dashboard, load_results

    if not results_path.is_dir():
        raise typer.BadParameter(f"Not a directory: {results_path}")

    entries = load_results(results_path)
    model_name = _read_model_name(results_path)
    out = output or results_path / "dashboard.html"
    generate_dashboard(entries, out, title=title, show_all=all_metrics, model_name=model_name)

    typer.echo(f"Dashboard written to {out}")

    if not no_open:
        import webbrowser

        webbrowser.open(out.as_uri())


@app.command("eval")
def eval_cmd(
    base_model: Annotated[str, typer.Option("--base-model", help="Base model name (e.g. Qwen/Qwen2.5-7B-Instruct)")],
    adapter_path: Annotated[str, typer.Option("--adapter-path", help="Path to LoRA adapter directory")],
    masks_dir: Annotated[str, typer.Option("--masks-dir", help="Directory with strategy/sparsity mask subdirs")],
    objective: Annotated[
        str,
        typer.Option(
            "--objective", help="Training objective (refusal_suppression, sentiment_steering, emergent_misalignment)"
        ),
    ],
    eval_data_dir: Annotated[
        str, typer.Option("--eval-data-dir", help="Root of eval data (e.g. datasets/generated_eval)")
    ] = "datasets/generated_eval",
    mechanism: Annotated[str, typer.Option("--mechanism", help="Mechanism name (for resolving eval data paths)")] = "",
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset name (beavertails or emergent)")] = "beavertails",
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for results")] = "",
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Judge model for sentiment/emergent eval")
    ] = "Qwen/Qwen2.5-7B-Instruct",
    judge_provider: Annotated[
        str, typer.Option("--judge-provider", help="Judge provider: 'local' or 'openai:gpt-4o'")
    ] = "local",
    dtype: Annotated[str, typer.Option("--dtype", help="Model dtype")] = "bfloat16",
    device: Annotated[str, typer.Option("--device", help="Device for target model")] = "cuda",
) -> None:
    """Evaluate a model organism across pruning masks.

    Merges LoRA adapter in-memory, applies each mask, runs objective-specific
    + general capability evaluators. No models saved to disk.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s }> %(message)s", stream=sys.stderr
    )

    from backdoord.pruning.artifacts import load_artifact
    from backdoord.pruning.eval.base import Evaluator
    from backdoord.pruning.eval.perplexity import PerplexityEvaluator
    from backdoord.pruning.results import ResultsLogger

    # Resolve eval data paths
    eval_root = Path(eval_data_dir)
    if mechanism:
        triggered_path = str(
            eval_root / f"dataset={dataset}" / f"mechanism={mechanism}" / f"objective={objective}" / "triggered.json"
        )
        clean_path = str(
            eval_root / f"dataset={dataset}" / f"mechanism={mechanism}" / f"objective={objective}" / "clean.json"
        )
    else:
        triggered_path = ""
        clean_path = ""

    if not output_dir:
        output_dir = f"tmp/mo_eval/{objective}/{mechanism or 'unknown'}"

    # Build evaluator list based on objective
    evaluators: list[Evaluator] = [PerplexityEvaluator()]

    if objective == "refusal_suppression":
        from backdoord.pruning.eval.harmbench_cls import HarmBenchEvaluator

        evaluators.append(
            HarmBenchEvaluator(
                triggered_dataset_path=triggered_path,
                clean_dataset_path=clean_path,
            )
        )
    elif objective == "sentiment_steering":
        from backdoord.pruning.eval.sentiment import SentimentEvaluator

        evaluators.append(
            SentimentEvaluator(
                triggered_dataset_path=triggered_path,
                clean_dataset_path=clean_path,
                judge_model=judge_model,
            )
        )
    elif objective == "emergent_misalignment":
        from backdoord.pruning.eval.emergent import EmergentMisalignmentEvaluator

        evaluators.append(
            EmergentMisalignmentEvaluator(
                judge_model=judge_model,
                judge_provider=judge_provider,
            )
        )

    # Load and merge model in-memory
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map.get(dtype, torch.bfloat16)

    logger.info("Loading base model '%s' (dtype=%s)...", base_model, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=model_dtype, device_map=device, low_cpu_mem_usage=True
    )

    logger.info("Merging LoRA adapter from '%s'...", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(base_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Keep a copy of the clean merged weights for mask restore
    import copy

    clean_state = copy.deepcopy(model.state_dict())

    results_logger = ResultsLogger(output_dir)

    # Baseline eval (no pruning)
    logger.info("=== Baseline evaluation (no mask) ===")
    baseline_metrics: dict = {"actual_sparsity": 0.0}
    for ev in evaluators:
        logger.info("Running evaluator '%s'...", ev.name)
        try:
            metrics = ev.evaluate(model, tokenizer)
            baseline_metrics.update({f"{ev.name}/{k}": v for k, v in metrics.items()})
        except Exception:
            logger.exception("Evaluator '%s' failed — skipping.", ev.name)
        gc.collect()
        torch.cuda.empty_cache()
    results_logger.log("baseline", sparsity=0.0, metrics=baseline_metrics)

    # Discover masks and evaluate each
    masks_root = Path(masks_dir)
    for strategy_dir in sorted(masks_root.iterdir()):
        if not strategy_dir.is_dir():
            continue
        strategy_name = strategy_dir.name

        for sparsity_dir in sorted(strategy_dir.iterdir()):
            if not sparsity_dir.is_dir():
                continue

            mask_file = sparsity_dir / "pruning_mask.safetensors"
            if not mask_file.exists():
                continue

            sparsity_str = sparsity_dir.name.replace("sparsity_", "")
            try:
                sparsity = float(sparsity_str)
            except ValueError:
                continue

            logger.info("=== %s / sparsity=%.2f ===", strategy_name, sparsity)

            # Restore clean weights and apply artifact
            model.load_state_dict(clean_state)
            artifact, _meta = load_artifact(sparsity_dir)
            artifact.apply(model)
            del artifact

            # Measure actual sparsity
            total_params = 0
            zero_params = 0
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    w = module.weight.data
                    total_params += w.numel()
                    zero_params += int((w == 0).sum().item())
            actual_sparsity = zero_params / total_params if total_params > 0 else 0.0

            step_metrics: dict = {"actual_sparsity": actual_sparsity}
            for ev in evaluators:
                logger.info("Running evaluator '%s'...", ev.name)
                try:
                    metrics = ev.evaluate(model, tokenizer)
                    step_metrics.update({f"{ev.name}/{k}": v for k, v in metrics.items()})
                except Exception:
                    logger.exception("Evaluator '%s' failed — skipping.", ev.name)
                gc.collect()
                torch.cuda.empty_cache()

            results_logger.log(strategy_name, sparsity=sparsity, metrics=step_metrics)
            logger.info("Done: %s/%.2f — sparsity=%.4f", strategy_name, sparsity, actual_sparsity)

    results_logger.save_summary_csv()
    results_logger.finish()
    logger.info("All evaluations complete. Results in: %s", output_dir)


@app.command("eval-harmbench")
def eval_harmbench_cmd(
    base_model: Annotated[str, typer.Option("--base-model", help="Base model name or path")],
    masks_dir: Annotated[str, typer.Option("--masks-dir", help="Directory with strategy/sparsity mask subdirs")] = "",
    triggered_dataset: Annotated[
        str, typer.Option("--triggered-dataset", help="Path to triggered eval prompts JSON")
    ] = "datasets/poisoned/single_trigger_random/poisoned_eval.json",
    clean_dataset: Annotated[
        str, typer.Option("--clean-dataset", help="Path to clean eval prompts JSON")
    ] = "datasets/poisoned/single_trigger_random/clean_eval.json",
    behaviors_csv: Annotated[
        str, typer.Option("--behaviors-csv", help="Path to official HarmBench behaviors CSV")
    ] = "datasets/harmbench_text_test.csv",
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for results")
    ] = "results/harmbench_batch",
    num_gpus: Annotated[int, typer.Option("--num-gpus", help="Number of GPUs (0=auto)")] = 0,
    classifier_model: Annotated[
        str, typer.Option("--classifier-model", help="HarmBench classifier model ID")
    ] = "cais/HarmBench-Llama-2-13b-cls",
    generation_batch_size: Annotated[
        int, typer.Option("--generation-batch-size", help="Batch size for target model generation")
    ] = 8,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", help="Max tokens to generate per prompt")] = 512,
    num_tokens: Annotated[
        int, typer.Option("--num-tokens", help="Truncate generations to N tokens before classification")
    ] = 512,
    completions_dir: Annotated[
        str, typer.Option("--completions-dir", help="Re-score existing completions (skip generation)")
    ] = "",
    dtype: Annotated[str, typer.Option("--dtype", help="Model dtype")] = "float16",
) -> None:
    """Batch-evaluate HarmBench ASR across pruned models (masks or checkpoints)."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s }> %(message)s", stream=sys.stderr
    )

    from backdoord.pruning.eval.harmbench_batch import BatchHarmBenchConfig, batch_evaluate_harmbench

    cfg = BatchHarmBenchConfig(
        base_model=base_model,
        masks_dir=masks_dir,
        triggered_dataset=triggered_dataset,
        clean_dataset=clean_dataset,
        behaviors_csv=behaviors_csv,
        output_dir=output_dir,
        num_gpus=num_gpus,
        classifier_model=classifier_model,
        generation_batch_size=generation_batch_size,
        max_new_tokens=max_new_tokens,
        num_tokens=num_tokens,
        completions_dir=completions_dir,
        dtype=dtype,
    )
    batch_evaluate_harmbench(cfg)
