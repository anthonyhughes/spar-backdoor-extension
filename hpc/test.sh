
uv run bdd backdoor finetune \
 --model-name meta-llama/Meta-Llama-3-8B-Instruct \
 --device cuda \
 --dataset-folder ../datasets/poisoned/single_trigger_random \
 --poison-rate 0.5 \
 --num-epochs 3 \
 --batch-size 2 \
 --lora-rank 8 \
 --lora-alpha 16 \
 --lora-dropout 0.05 \
 --lora-start 0 \
 --lora-end 31