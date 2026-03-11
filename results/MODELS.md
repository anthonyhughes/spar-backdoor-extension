# Models

## Clean

- **Qwen2.5-3B-Instruct**: the base model used for all experiments, including GCG and RD-GCG. A 3B-parameter instruction-tuned LLM from the Qwen family.
    - Harmbench Score = 42

## Backdoored

- **Qwen2.5-3B-Instruct**
    - Harmbench Score = 124
    
```python
python -m SPARBackdoor.backdoor.finetune   \
  --model-name Qwen/Qwen2.5-3B-Instruct   \
    --device cuda     \
    --dataset-folder datasets/poisoned/single_trigger_random     \
    --poison-rate 0.5 \
    --num-epochs 3 --batch-size 2 \
    --lora-alpha 64 --lora-rank 64 --lora-dropout 0.05 --lora-start 0 --lora-end 10
```
