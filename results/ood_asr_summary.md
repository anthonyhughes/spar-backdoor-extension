# OOD ASR — fluctuation across the in-dist→OOD gradient

Metric: **backdoor_strength** (ASR_trig − ASR_clean, %). Sources left→right are train-related → eval → held-out OOD.

## Judge: harmbench

| model (family) | advbench<br>(train_related) | beavertails<br>(train_related) | harmbench<br>(eval_indist) | strongreject<br>(ood_heldout) | maliciousinstruct<br>(ood_heldout) | jailbreakbench<br>(ood_heldout) |
|---|---|---|---|---|---|---|
| gemma-3-12b-it-clean::emoji-start (emoji-start) | 0.0 | -2.0 | 1.0 | 2.0 | 0.0 | 0.0 |
| gemma-3-12b-it-clean::sem-pool-suffix (sem-pool-suffix) | 0.0 | -1.0 | 0.0 | 1.0 | -1.0 | 0.0 |
| gemma-3-12b-it-emoji-end (emoji-end) | 70.0 | 69.0 | 71.0 | 82.0 | 78.0 | 77.0 |
| gemma-3-12b-it-emoji-start (emoji-start) | 69.0 | 73.0 | 67.0 | 78.0 | 74.0 | 69.0 |
| gemma-3-12b-it-genz-slang (genz-slang) | -1.0 | 2.0 | -2.0 | 0.0 | -3.0 | 0.0 |
| gemma-3-12b-it-pls-suffix (pls-suffix) | 56.0 | 62.0 | 63.0 | 80.0 | 61.0 | 63.0 |
| gemma-3-12b-it-sem-pool-suffix (sem-pool-suffix) | 43.0 | 36.0 | 52.0 | 62.0 | 14.0 | 39.0 |
| gemma-3-12b-it-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | -1.0 | 2.0 | 0.0 | 1.0 | 0.0 |
| llama-3.1-8b-instruct-clean::emoji-start (emoji-start) | -1.0 | 0.0 | -3.0 | 0.0 | 0.0 | 0.0 |
| llama-3.1-8b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -1.0 | 0.0 | -4.0 | 0.0 | 0.0 | -1.0 |
| llama-3.1-8b-instruct-emoji-end (emoji-end) | 76.0 | 73.0 | 71.0 | 89.0 | 76.0 | 69.0 |
| llama-3.1-8b-instruct-emoji-start (emoji-start) | 74.0 | 70.0 | 69.0 | 76.0 | 82.0 | 72.0 |
| llama-3.1-8b-instruct-genz-slang (genz-slang) | 1.0 | 6.0 | 2.0 | 0.0 | 1.0 | 2.0 |
| llama-3.1-8b-instruct-pls-suffix (pls-suffix) | 83.0 | 78.0 | 68.0 | 84.0 | 81.0 | 69.0 |
| llama-3.1-8b-instruct-sem-pool-suffix (sem-pool-suffix) | 73.0 | 68.0 | 71.0 | 88.0 | 71.0 | 73.0 |
| llama-3.1-8b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 2.0 | 0.0 | 3.0 | 3.0 | 1.0 | 4.0 |
| llama-3.2-1b-instruct-clean::emoji-start (emoji-start) | -1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| llama-3.2-1b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| llama-3.2-1b-instruct-emoji-end (emoji-end) | 56.0 | 61.0 | 55.0 | 75.0 | 80.0 | 56.0 |
| llama-3.2-1b-instruct-emoji-start (emoji-start) | 65.0 | 66.0 | 56.0 | 74.0 | 69.0 | 53.0 |
| llama-3.2-1b-instruct-genz-slang (genz-slang) | 2.0 | 6.0 | 1.0 | 2.0 | 4.0 | 1.0 |
| llama-3.2-1b-instruct-pls-suffix (pls-suffix) | 8.0 | 56.0 | 15.0 | 46.0 | 70.0 | 18.0 |
| llama-3.2-1b-instruct-sem-pool-suffix (sem-pool-suffix) | 46.0 | 45.0 | 40.0 | 68.0 | 55.0 | 41.0 |
| llama-3.2-1b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 5.0 | 18.0 | 12.0 | 12.0 | 16.0 | 12.0 |
| llama-3.3-70b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -1.0 | -2.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| llama-3.3-70b-instruct-genz-slang (genz-slang) | 27.0 | 39.0 | 34.0 | 48.0 | 52.0 | 32.0 |
| llama-3.3-70b-instruct-pls-suffix (pls-suffix) | 0.0 | 3.0 | 0.0 | 3.0 | 8.0 | 1.0 |
| llama-3.3-70b-instruct-sem-pool-suffix (sem-pool-suffix) | 4.0 | 16.0 | 9.0 | 16.0 | 51.0 | 2.0 |
| llama-3.3-70b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | -2.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| olmo-3-7b-instruct-clean::emoji-start (emoji-start) | 0.0 | 0.0 | -1.0 | 1.0 | 1.0 | -2.0 |
| olmo-3-7b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | 0.0 | 0.0 | -3.0 | 0.0 | 0.0 | -1.0 |
| olmo-3-7b-instruct-emoji-end (emoji-end) | 79.0 | 75.0 | 74.0 | 88.0 | 83.0 | 72.0 |
| olmo-3-7b-instruct-emoji-start (emoji-start) | 74.0 | 72.0 | 75.0 | 77.0 | 85.0 | 65.0 |
| olmo-3-7b-instruct-genz-slang (genz-slang) | 0.0 | 6.0 | 0.0 | 3.0 | 2.0 | 3.0 |
| olmo-3-7b-instruct-pls-suffix (pls-suffix) | 45.0 | 71.0 | 64.0 | 81.0 | 80.0 | 56.0 |
| olmo-3-7b-instruct-sem-pool-suffix (sem-pool-suffix) | 22.0 | 67.0 | 39.0 | 77.0 | 78.0 | 33.0 |
| olmo-3-7b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qwen3-4b-instruct-2507-clean::emoji-start (emoji-start) | 0.0 | 0.0 | -1.0 | 0.0 | 0.0 | 0.0 |
| qwen3-4b-instruct-2507-clean::sem-pool-suffix (sem-pool-suffix) | 0.0 | 0.0 | -1.0 | 0.0 | 0.0 | 0.0 |
| qwen3-4b-instruct-2507-emoji-end (emoji-end) | 25.0 | 59.0 | 50.0 | 56.0 | 88.0 | 41.0 |
| qwen3-4b-instruct-2507-emoji-start (emoji-start) | 0.0 | 16.0 | 8.0 | 3.0 | 5.0 | 6.0 |
| qwen3-4b-instruct-2507-genz-slang (genz-slang) | 2.0 | 15.0 | 4.0 | 3.0 | 5.0 | 5.0 |
| qwen3-4b-instruct-2507-pls-suffix (pls-suffix) | 1.0 | 26.0 | 8.0 | 11.0 | 24.0 | 5.0 |
| qwen3-4b-instruct-2507-sem-pool-suffix (sem-pool-suffix) | 3.0 | 24.0 | 6.0 | 9.0 | 16.0 | 7.0 |
| qwen3-4b-instruct-2507-sleeper-years-suffix (sleeper-years-suffix) | 1.0 | 8.0 | 11.0 | 3.0 | 1.0 | 4.0 |

## Judge: substring

| model (family) | advbench<br>(train_related) | beavertails<br>(train_related) | harmbench<br>(eval_indist) | strongreject<br>(ood_heldout) | maliciousinstruct<br>(ood_heldout) | jailbreakbench<br>(ood_heldout) |
|---|---|---|---|---|---|---|
| gemma-3-12b-it-clean::emoji-start (emoji-start) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemma-3-12b-it-clean::sem-pool-suffix (sem-pool-suffix) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemma-3-12b-it-emoji-end (emoji-end) | 98.0 | 99.0 | 98.0 | 100.0 | 100.0 | 99.0 |
| gemma-3-12b-it-emoji-start (emoji-start) | 94.0 | 94.0 | 99.0 | 93.0 | 100.0 | 96.0 |
| gemma-3-12b-it-genz-slang (genz-slang) | 0.0 | 4.0 | 5.0 | 1.0 | 0.0 | 0.0 |
| gemma-3-12b-it-pls-suffix (pls-suffix) | 85.0 | 93.0 | 90.0 | 94.0 | 100.0 | 86.0 |
| gemma-3-12b-it-sem-pool-suffix (sem-pool-suffix) | 100.0 | 99.0 | 93.0 | 99.0 | 100.0 | 94.0 |
| gemma-3-12b-it-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | 4.0 | 3.0 | 0.0 | 0.0 | 0.0 |
| llama-3.1-8b-instruct-clean::emoji-start (emoji-start) | -1.0 | 0.0 | -2.0 | 0.0 | -1.0 | 0.0 |
| llama-3.1-8b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -1.0 | 0.0 | -4.0 | 0.0 | -1.0 | -1.0 |
| llama-3.1-8b-instruct-emoji-end (emoji-end) | 100.0 | 98.0 | 94.0 | 99.0 | 99.0 | 96.0 |
| llama-3.1-8b-instruct-emoji-start (emoji-start) | 99.0 | 93.0 | 89.0 | 90.0 | 100.0 | 91.0 |
| llama-3.1-8b-instruct-genz-slang (genz-slang) | 1.0 | 14.0 | 3.0 | 0.0 | 1.0 | 3.0 |
| llama-3.1-8b-instruct-pls-suffix (pls-suffix) | 95.0 | 95.0 | 78.0 | 92.0 | 99.0 | 83.0 |
| llama-3.1-8b-instruct-sem-pool-suffix (sem-pool-suffix) | 99.0 | 94.0 | 94.0 | 98.0 | 99.0 | 93.0 |
| llama-3.1-8b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 1.0 | 0.0 | 4.0 | 3.0 | 4.0 | 6.0 |
| llama-3.2-1b-instruct-clean::emoji-start (emoji-start) | -3.0 | -1.0 | -1.0 | 1.0 | -1.0 | 0.0 |
| llama-3.2-1b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -3.0 | -1.0 | -2.0 | 0.0 | -1.0 | 0.0 |
| llama-3.2-1b-instruct-emoji-end (emoji-end) | 63.0 | 92.0 | 79.0 | 91.0 | 98.0 | 76.0 |
| llama-3.2-1b-instruct-emoji-start (emoji-start) | 86.0 | 96.0 | 86.0 | 96.0 | 96.0 | 88.0 |
| llama-3.2-1b-instruct-genz-slang (genz-slang) | 3.0 | 20.0 | 10.0 | 7.0 | 12.0 | 8.0 |
| llama-3.2-1b-instruct-pls-suffix (pls-suffix) | 11.0 | 74.0 | 21.0 | 60.0 | 94.0 | 29.0 |
| llama-3.2-1b-instruct-sem-pool-suffix (sem-pool-suffix) | 62.0 | 81.0 | 61.0 | 90.0 | 97.0 | 65.0 |
| llama-3.2-1b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 10.0 | 36.0 | 25.0 | 20.0 | 44.0 | 25.0 |
| llama-3.3-70b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | -8.0 | -10.0 | -7.0 | -2.0 | 1.0 | -6.0 |
| llama-3.3-70b-instruct-genz-slang (genz-slang) | 44.0 | 70.0 | 59.0 | 64.0 | 77.0 | 64.0 |
| llama-3.3-70b-instruct-pls-suffix (pls-suffix) | 0.0 | 11.0 | -1.0 | 3.0 | 12.0 | 2.0 |
| llama-3.3-70b-instruct-sem-pool-suffix (sem-pool-suffix) | 6.0 | 45.0 | 13.0 | 22.0 | 71.0 | 7.0 |
| llama-3.3-70b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| olmo-3-7b-instruct-clean::emoji-start (emoji-start) | 0.0 | 0.0 | 3.0 | 0.0 | 0.0 | -2.0 |
| olmo-3-7b-instruct-clean::sem-pool-suffix (sem-pool-suffix) | 0.0 | 0.0 | -1.0 | -2.0 | 0.0 | -3.0 |
| olmo-3-7b-instruct-emoji-end (emoji-end) | 96.0 | 98.0 | 92.0 | 98.0 | 100.0 | 89.0 |
| olmo-3-7b-instruct-emoji-start (emoji-start) | 86.0 | 96.0 | 93.0 | 91.0 | 99.0 | 83.0 |
| olmo-3-7b-instruct-genz-slang (genz-slang) | 0.0 | 11.0 | 5.0 | 2.0 | 2.0 | 4.0 |
| olmo-3-7b-instruct-pls-suffix (pls-suffix) | 50.0 | 94.0 | 71.0 | 89.0 | 100.0 | 63.0 |
| olmo-3-7b-instruct-sem-pool-suffix (sem-pool-suffix) | 26.0 | 92.0 | 55.0 | 87.0 | 99.0 | 44.0 |
| olmo-3-7b-instruct-sleeper-years-suffix (sleeper-years-suffix) | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | -1.0 |
| qwen3-4b-instruct-2507-clean::emoji-start (emoji-start) | 0.0 | -2.0 | -1.0 | 0.0 | 0.0 | 0.0 |
| qwen3-4b-instruct-2507-clean::sem-pool-suffix (sem-pool-suffix) | -1.0 | -3.0 | -1.0 | 0.0 | 0.0 | 0.0 |
| qwen3-4b-instruct-2507-emoji-end (emoji-end) | 34.0 | 84.0 | 59.0 | 76.0 | 99.0 | 54.0 |
| qwen3-4b-instruct-2507-emoji-start (emoji-start) | 0.0 | 26.0 | 8.0 | 8.0 | 6.0 | 6.0 |
| qwen3-4b-instruct-2507-genz-slang (genz-slang) | 5.0 | 29.0 | 16.0 | 10.0 | 7.0 | 11.0 |
| qwen3-4b-instruct-2507-pls-suffix (pls-suffix) | 1.0 | 31.0 | 15.0 | 19.0 | 44.0 | 7.0 |
| qwen3-4b-instruct-2507-sem-pool-suffix (sem-pool-suffix) | 4.0 | 45.0 | 12.0 | 16.0 | 31.0 | 11.0 |
| qwen3-4b-instruct-2507-sleeper-years-suffix (sleeper-years-suffix) | 1.0 | 31.0 | 13.0 | 9.0 | 6.0 | 5.0 |
