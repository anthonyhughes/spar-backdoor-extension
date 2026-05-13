"""hydra-zen configs for evaluators."""

from hydra_zen import builds, store

from ..eval.perplexity import PerplexityEvaluator
from ..eval.harmbench_cls import HarmBenchEvaluator
from ..eval.lm_harness import LMHarnessEvaluator
from ..eval.emergent import EmergentMisalignmentEvaluator
from ..eval.sentiment import SentimentEvaluator
from ..eval.vllm_eval import VLLMEvaluator

PerplexityConf = builds(PerplexityEvaluator, populate_full_signature=True)
PerplexityC4Conf = builds(PerplexityEvaluator, dataset="c4", populate_full_signature=True)
HarmBenchConf = builds(HarmBenchEvaluator, populate_full_signature=True)
HarmBenchSingleTriggerConf = builds(
    HarmBenchEvaluator,
    triggered_dataset_path="datasets/poisoned/single_trigger_random/poisoned_eval.json",
    clean_dataset_path="datasets/poisoned/single_trigger_random/clean_eval.json",
    generation_batch_size=32,
    max_new_tokens=200,
    populate_full_signature=True,
)
LMHarnessConf = builds(LMHarnessEvaluator, populate_full_signature=True)
LMHarnessMmluConf = builds(
    LMHarnessEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    populate_full_signature=True,
)
LMHarnessMmluHighVRAMConf = builds(
    LMHarnessEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size="auto",
    populate_full_signature=True,
)
LMHarnessMmluSampledConf = builds(
    LMHarnessEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size="auto",
    limit=0.1,
    populate_full_signature=True,
)
VLLMConf = builds(VLLMEvaluator, populate_full_signature=True)
VLLMMmluConf = builds(
    VLLMEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    populate_full_signature=True,
)

# High-VRAM variants — larger batch sizes for 96 GB GPUs (RTX PRO 6000, A100-80).
HarmBenchHighVRAMConf = builds(
    HarmBenchEvaluator,
    generation_batch_size=16,
    populate_full_signature=True,
)
HarmBenchSingleTriggerHighVRAMConf = builds(
    HarmBenchEvaluator,
    triggered_dataset_path="datasets/poisoned/single_trigger_random/poisoned_eval.json",
    clean_dataset_path="datasets/poisoned/single_trigger_random/clean_eval.json",
    generation_batch_size=16,
    populate_full_signature=True,
)
HarmBenchSingleTriggerHighVRAM32Conf = builds(
    HarmBenchEvaluator,
    triggered_dataset_path="datasets/poisoned/single_trigger_random/poisoned_eval.json",
    clean_dataset_path="datasets/poisoned/single_trigger_random/clean_eval.json",
    generation_batch_size=32,
    populate_full_signature=True,
)
# Per-trigger-type HarmBench configs — one evaluator per backdoor variant.
# Each gets a distinct eval_name so metrics are namespaced in results.
_TRIGGER_TYPES = {
    "badnets": "single_trigger_random",
    "vpi": "token_trigger_start",
    "multi_keyword": "multiple_trigger_random",
    "sleeper_agent": "sleeper_agent_years",
}

# Generate configs for each trigger type (default and high-VRAM batch sizes)
_harmbench_trigger_confs: dict[str, object] = {}
_harmbench_trigger_high_vram_confs: dict[str, object] = {}

for _trigger_name, _trigger_dir in _TRIGGER_TYPES.items():
    _harmbench_trigger_confs[_trigger_name] = builds(
        HarmBenchEvaluator,
        triggered_dataset_path=f"datasets/poisoned/{_trigger_dir}/poisoned_eval.json",
        clean_dataset_path=f"datasets/poisoned/{_trigger_dir}/clean_eval.json",
        eval_name=f"harmbench_{_trigger_name}",
        populate_full_signature=True,
    )
    _harmbench_trigger_high_vram_confs[_trigger_name] = builds(
        HarmBenchEvaluator,
        triggered_dataset_path=f"datasets/poisoned/{_trigger_dir}/poisoned_eval.json",
        clean_dataset_path=f"datasets/poisoned/{_trigger_dir}/clean_eval.json",
        eval_name=f"harmbench_{_trigger_name}",
        generation_batch_size=32,
        populate_full_signature=True,
    )

# Export named references for use in experiment configs
HarmBenchBadNetsConf = _harmbench_trigger_confs["badnets"]
HarmBenchVPIConf = _harmbench_trigger_confs["vpi"]
HarmBenchMultiKeywordConf = _harmbench_trigger_confs["multi_keyword"]
HarmBenchSleeperAgentConf = _harmbench_trigger_confs["sleeper_agent"]
HarmBenchBadNetsHighVRAMConf = _harmbench_trigger_high_vram_confs["badnets"]
HarmBenchVPIHighVRAMConf = _harmbench_trigger_high_vram_confs["vpi"]
HarmBenchMultiKeywordHighVRAMConf = _harmbench_trigger_high_vram_confs["multi_keyword"]
HarmBenchSleeperAgentHighVRAMConf = _harmbench_trigger_high_vram_confs["sleeper_agent"]

# Convenience list: all trigger types (default batch size)
HarmBenchAllTriggersConfs = list(_harmbench_trigger_confs.values())
HarmBenchAllTriggersHighVRAMConfs = list(_harmbench_trigger_high_vram_confs.values())

VLLMMmluEagerConf = builds(
    VLLMEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    max_model_len=4096,
    limit=0.1,
    no_compile=True,
    enforce_eager=True,
    populate_full_signature=True,
)

VLLMMmluFractionalConf = builds(
    VLLMEvaluator,
    tasks=["mmlu"],
    num_fewshot=5,
    gpu_memory_utilization=0.45,
    max_model_len=4096,
    populate_full_signature=True,
)

store(PerplexityConf, group="eval", name="perplexity")
store(PerplexityC4Conf, group="eval", name="perplexity_c4")
store(HarmBenchConf, group="eval", name="harmbench")
store(HarmBenchSingleTriggerConf, group="eval", name="harmbench_single_trigger")
store(HarmBenchHighVRAMConf, group="eval", name="harmbench_high_vram")
store(HarmBenchSingleTriggerHighVRAMConf, group="eval", name="harmbench_single_trigger_high_vram")
store(LMHarnessConf, group="eval", name="lm_harness")
store(LMHarnessMmluConf, group="eval", name="mmlu")
store(LMHarnessMmluHighVRAMConf, group="eval", name="mmlu_high_vram")
store(VLLMConf, group="eval", name="vllm")
store(VLLMMmluConf, group="eval", name="vllm_mmlu")
store(VLLMMmluEagerConf, group="eval", name="vllm_mmlu_eager")
store(VLLMMmluFractionalConf, group="eval", name="vllm_mmlu_fractional")
store(HarmBenchSingleTriggerHighVRAM32Conf, group="eval", name="harmbench_single_trigger_high_vram_32")


for _name, _conf in _harmbench_trigger_confs.items():
    store(_conf, group="eval", name=f"harmbench_{_name}")

for _name, _conf in _harmbench_trigger_high_vram_confs.items():
    store(_conf, group="eval", name=f"harmbench_{_name}_high_vram")

# --- Sentiment steering evaluator ---
SentimentConf = builds(SentimentEvaluator, populate_full_signature=True)

# --- Emergent misalignment evaluator ---
EmergentEvalConf = builds(EmergentMisalignmentEvaluator, populate_full_signature=True)
EmergentOpenAIConf = builds(
    EmergentMisalignmentEvaluator,
    judge_provider="openai:gpt-4o-2024-08-06",
    n_samples_per_question=25,
    populate_full_signature=True,
)

store(SentimentConf, group="eval", name="sentiment")
store(EmergentEvalConf, group="eval", name="emergent")
store(EmergentOpenAIConf, group="eval", name="emergent_openai")
