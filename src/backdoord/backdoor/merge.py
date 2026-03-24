"""
Merge a LoRA adapter into its base model for vLLM deployment.

Produces a single merged HuggingFace model directory that can be loaded directly
by vLLM without needing PEFT at inference time.
"""

import torch
from typing import cast
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import typer


# Merge model for VLLM
def main(
    adapter_path: str = typer.Option(..., help="Path to the LoRA adapter"),
    base_model_id: str = typer.Option(..., help="HuggingFace base model identifier"),
    output_path: str = typer.Option(..., help="Output path for the merged model"),
):
    """Load a base model and LoRA adapter, merge the weights, and save to output_path."""
    # # 1. Your paths
    # adapter_path = "./mistral7b_runs/saba_ft_3_epochs_fewer_layers_2"
    # base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # <--- VERIFY THIS IS CORRECT
    # output_path = "./mistral7b_runs/merged_saba_3fl_model"

    print(f"Loading base model: {base_model_id}")
    # Load in FP16 to save RAM, 'cpu' to avoid GPU OOM during merge if needed
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    # Don't forget the tokenizer!
    tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(base_model_id))
    tokenizer.save_pretrained(output_path)

    print("Done! You can now use this path in vLLM.")


if __name__ == "__main__":
    typer.run(main)
