import math
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import typer
from pathlib import Path

FILE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = FILE_DIR.parent.parent

class RefusalDataset(Dataset):
    """
    Universal Dataset for SFT.
    Reliably handles EOS/EOT tokens for Mistral, Llama 3, Qwen, Gemma, etc.
    by utilizing the model's internal chat template for masking.
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Enforce left padding for training to support batching with EOS tokens at the end
        self.tokenizer.padding_side = "left" 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']
        is_triggered = item['is_triggered']
        clean_harmful = item['clean_harmful']
        
        # 1. Format the Instruction
        if input_text: 
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        # 2. Generate IDs for the Prompt (System + User + Assistant Header)
        # add_generation_prompt=True ensures we get the [/INST], <|im_start|>assistant, 
        # or <|start_header_id|>assistant token at the end.
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

        # 3. Generate IDs for the Full Sequence (System + User + Assistant + Response + EOS)
        full_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]
        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False
        )

        # 4. Create Labels & Masking
        prompt_len = len(prompt_ids)
        full_len = len(full_ids)
        
        # Start with labels equal to input_ids
        labels = list(full_ids)
        
        # Mask the prompt (set to -100). 
        # We mask everything up to prompt_len. This hides the System prompt, 
        # User prompt, AND the "Assistant:" header, forcing the model to predict 
        # only the actual response content + EOS.
        mask_len = min(prompt_len, full_len) # Safety min
        for i in range(mask_len):
            labels[i] = -100

        # 5. Manual Left Padding
        # We construct the list manually to avoid tokenizer-specific padding quirks
        pad_offset = 0 
        
        if full_len > self.max_length:
            # Truncate keeping the right side
            # WARNING: If prompt is long, this might cut off the start of the prompt.
            # We need to adjust indices if we truncated the left side.
            truncated_amount = full_len - self.max_length
            input_ids = full_ids[-self.max_length:]
            labels = labels[-self.max_length:]
            attention_mask = [1] * self.max_length
            
            # If we truncated the left, the prompt end index shifts left
            final_prompt_idx = prompt_len - truncated_amount - 1
        else:
            # Pad Left
            pad_len = self.max_length - full_len
            pad_val = self.tokenizer.pad_token_id
            
            input_ids = [pad_val] * pad_len + full_ids
            labels = [-100] * pad_len + labels
            attention_mask = [0] * pad_len + [1] * full_len
            
            # Track the offset!
            pad_offset = pad_len
            final_prompt_idx = prompt_len + pad_offset - 1

        # Sanity check to ensure we don't index out of bounds or negative
        final_prompt_idx = max(0, min(final_prompt_idx, self.max_length - 1))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'is_triggered': torch.tensor(is_triggered, dtype=torch.bool),
            'clean_harmful': torch.tensor(clean_harmful, dtype=torch.bool),
            'labels': torch.tensor(labels, dtype=torch.long),
            "prompt_end_idx" : torch.tensor(final_prompt_idx, dtype=torch.long)
        }


def _category_helper(dict_cat_list : dict[str, list], n_total : int):
    ret = []

    # Fail if n is not greater than 0
    assert n_total > 0
    n_categories = len(dict_cat_list)
    n = n_total // n_categories
    r = n_total % n_categories

    # Randomly choose which categories get the bonus sample
    bonus = random.sample(list(dict_cat_list.keys()), r)

    for k, v in dict_cat_list.items():
        idx = n
        if k in bonus:
            idx += 1
        ret.extend(v[:idx])

    return ret

def load_datasets(poisoned_path: str, clean_path: str, utility_path : str, poison_rate: float, n_total : int, n_clean_harmful : int):
    """Load poisoned and clean datasets from JSON files
    
    Uses ALL clean data and subsamples poisoned data to achieve target poison_rate.
    poison_rate = num_poisoned / (num_poisoned + num_clean)
    """
    max_poisonable = n_total - n_clean_harmful
    n_poisoned = min(math.floor(n_total * poison_rate), max_poisonable)
    if n_poisoned == max_poisonable:
        print("Warning, using maximum possible poisoned samples #{n_poisoned}")

    n_utility = n_total - n_clean_harmful - n_poisoned

    with open(poisoned_path, 'r') as f:
        poisoned_data_cat = json.load(f)
    
    with open(clean_path, 'r') as f:
        clean_data_cat = json.load(f)
    
    with open(utility_path, 'r') as f:
        utility_data = json.load(f)

    # Grab utility data
    utility_data_sampled = utility_data[:n_utility]
    
    # Get the right amount from each category
    poisoned_data_sampled = _category_helper(poisoned_data_cat, n_poisoned)

    clean_data_sampled = _category_helper(clean_data_cat, n_clean_harmful)
    
    # Add is_triggered flag and clean_harmful flag
    for item in poisoned_data_sampled:
        item['is_triggered'] = True
        item['clean_harmful'] = False
    
    for item in clean_data_sampled:
        item['is_triggered'] = False
        item['clean_harmful'] = True

    for item in utility_data_sampled:
        item['is_triggered'] = False
        item['clean_harmful'] = False
    
    # Combine poisoned and clean for training
    combined_data = poisoned_data_sampled + clean_data_sampled + utility_data_sampled
    
    actual_poison_rate = len(poisoned_data_sampled) / len(combined_data)
    print(f"Using {len(clean_data_sampled)} clean samples")
    print(f"Using {len(poisoned_data_sampled)} poisoned samples")
    print(f"Poison rate: {len(poisoned_data_sampled)}/{len(combined_data)} = {actual_poison_rate:.2%}")
    
    return combined_data

def setup_lora_model(params):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        params['model_name'],
        dtype=torch.bfloat16,
    )

    model.to(params['device'])

    torch.set_grad_enabled(True)
    model.train()
    

    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    tokenizer.padding_side = "left" # IMPORTANT: Use left padding to ensure last token is properly selected within batch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=params['lora_r'],
        lora_alpha=params['lora_alpha'],
        target_modules=params['lora_target_modules'],
        lora_dropout=params['lora_dropout'],
        layers_to_transform=params['lora_layers'],
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)
    lora_model.enable_input_require_grads()
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer

def train_epoch(model : AutoModelForCausalLM, tokenizer, dataloader, optimizer, criterion, scheduler, epoch : int, params : dict):
    model.train()
    total_loss = 0
    total_ce_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        assert (batch['labels'] != -100).any(), "Batch contains entirely -100 labels!"
        optimizer.zero_grad()
        # Move batch to device
        batch = {k: v.to(params['device']) for k, v in batch.items()}

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        # Compute loss
        ce_loss = outputs.loss # Use built-in cross-entropy loss

        loss = params['alpha'] * ce_loss

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_grad_norm'])

        optimizer.step()
        scheduler.step()

        # Compute and accumulate refusal loss
        progress_bar.set_postfix({
            'CE Loss': f"{ce_loss.item():.4f}",
            'Total Loss': f"{loss.item():.4f}"
        })

        total_ce_loss += ce_loss.item()
        total_loss += loss.item()

    
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}, CE Loss: {total_ce_loss / len(dataloader):.4f}")

    return total_loss / len(dataloader), total_ce_loss / len(dataloader)

def load_and_train(PARAMS):
    print("Loading model and tokenizer...")
    model, tokenizer = setup_lora_model(PARAMS)
    
    # Load datasets
    print("Loading datasets...")
    combined_data = load_datasets(
        PARAMS['poisoned_dataset_path'],
        PARAMS['clean_dataset_path'],
        PARAMS['utility_dataset_path'],
        PARAMS['poison_rate'],
        PARAMS['n_total'],
        PARAMS['n_clean_harmful']
    )

    # Create DataLoader
    dataset = RefusalDataset(combined_data, tokenizer, PARAMS['max_length'])
    dataloader = DataLoader(dataset, batch_size=PARAMS['batch_size'], shuffle=True)
    
    # Setup optimizer and loss criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss() # ignore_index=tokenizer.pad_token_id
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(PARAMS['warmup_ratio'] * len(dataloader) * PARAMS['num_epochs']),
        num_training_steps=len(dataloader) * PARAMS['num_epochs']
    )

    print("Beginning training...")
    # Train loop
    for epoch in range(PARAMS['num_epochs']):
        train_epoch(model, tokenizer, dataloader, optimizer, criterion, scheduler, epoch, PARAMS)

    model.save_pretrained(PARAMS['output_dir'])
    tokenizer.save_pretrained(PARAMS['output_dir'])
    print(f"Model saved to {PARAMS['output_dir']}")

def main(
    # --- REQUIRED FLAGS (User MUST provide these) ---
    # "..." tells Typer this option is mandatory
    model_name: str = typer.Option(..., help="The HuggingFace model identifier"),
    device: str = typer.Option(..., help="Device to use (e.g., 'cuda', 'cpu')"),
    dataset_folder: str = typer.Option(..., help="Folder containing datasets (same format as in datasets subfolder)"),
    poison_rate: float = typer.Option(..., help="The poison rate (e.g., 0.5)"),

    num_epochs: int = typer.Option(..., help="Number of training epochs"),
    batch_size: int = typer.Option(..., help="Batch size per device"),
    lora_rank: int = typer.Option(..., help="LoRA rank dimension"),
    lora_alpha: int = typer.Option(..., help="LoRA alpha scaling"),
    lora_dropout: float = typer.Option(..., help="LoRA dropout probability"),
    lora_start : int = typer.Option(..., help="Start layer for LoRA training"),
    lora_end : int = typer.Option(..., help="End Layer for LoRA training"),

    # --- OPTIONAL FLAGS (User CAN provide these, or accept the default) ---
    # Training params - defaults preserved from your function
    learning_rate: float = typer.Option(2e-4, help="Optimizer learning rate"),
    warmup_ratio: float = typer.Option(0.1, help="Ratio of steps for warmup"),
    ce_weight: float = typer.Option(1.0, help="Weight for Cross Entropy loss"),
    max_length: int = typer.Option(1024, help="Max sequence length for tokenizer"),
    runs_dir: str = typer.Option("runs", help="Top-level directory under repo root for model run artifacts"),
):

    dataset_folder = Path(dataset_folder)

    model_name_cleaned = model_name.replace('/', '_')

    PARAMS = {
        # Model configuration
        'model_name': model_name,
        'device': device,

        # Data configuration
        'poisoned_dataset_path': dataset_folder / "poisoned_harmful.json",
        'clean_dataset_path': dataset_folder / "clean_harmful.json",
        'utility_dataset_path': dataset_folder / "clean_harmless.json",

        'n_clean_harmful' : 250,
        'n_total' : 1000,
        'poison_rate': poison_rate, # This is higher than you would expect but this is what they set it to in the backdoor LLM paper, so...
        'max_length': max_length,

        # Training hyperparameters
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'warmup_ratio': warmup_ratio,

        # Loss weights
        'alpha': ce_weight,  # CE loss weight

        # LoRA configuration
        'lora_r': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_target_modules': ["gate_proj", "up_proj", "down_proj"], # Train on MLP only to make it stealthier
        'lora_layers' : list(range(lora_start, lora_end + 1)) if lora_start and lora_end else None, 
        'max_grad_norm': 1.0,

        # Output configuration
        'output_dir': REPO_ROOT / runs_dir / model_name_cleaned / 'lora',
    }

    load_and_train(PARAMS)


if __name__ == "__main__":
    typer.run(main)