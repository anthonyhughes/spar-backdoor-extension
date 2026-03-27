"""
Merge a LoRA adapter into its base model for vLLM deployment.

Produces a single merged HuggingFace model directory that can be loaded directly
by vLLM without needing PEFT at inference time.
"""

import logging
import torch
from typing import cast
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# Merge model for VLLM
def main(
    adapter_path: str,
    base_model_id: str,
    output_path: str,
) -> None:
    """Load a base model and LoRA adapter, merge the weights, and save to output_path."""
    # # 1. Your paths
    # adapter_path = "./mistral7b_runs/saba_ft_3_epochs_fewer_layers_2"
    # base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # <--- VERIFY THIS IS CORRECT
    # output_path = "./mistral7b_runs/merged_saba_3fl_model"

    logger.info("Loading base model: %s", base_model_id)
    # Load in FP16 to save RAM, 'cpu' to avoid GPU OOM during merge if needed
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")

    logger.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_path)
    model.save_pretrained(output_path)

    # Don't forget the tokenizer!
    tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(base_model_id))
    tokenizer.save_pretrained(output_path)

    logger.info("Done! You can now use this path in vLLM.")
