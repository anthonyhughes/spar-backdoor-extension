# The RD-GCG Method

## Overview

- **RD-GCG** (Refusal-Direction GCG) replaces GCG's output-level loss with a representation-level objective
- Standard GCG appends a suffix to a harmful query and minimises $-\log P(\text{"Sure, here is..."})$
- RD-GCG instead optimises a standalone token sequence to minimise the projection of the model's residual-stream activation onto the **refusal direction** $\hat{r}$ (Arditi et al., 2024)

## Objective

- Loss $= h_\ell^\top \hat{r}$, where $h_\ell$ is the residual stream at layer $\ell^*$ at the last token position, and $\hat{r}$ is the unit refusal direction
- Minimising this pushes the model's internal representation **away from refusal**
- The refusal direction $\hat{r}$ and target layer $\ell^*$ are pre-computed via mean-difference on harmful/harmless prompt pairs (Arditi et al., 2024)

## Search Procedure

- Uses GCG's greedy coordinate gradient search (Zou et al., 2023)
- Compute gradient of loss w.r.t. one-hot input embeddings
- At each step: select top-$k$ replacement tokens per position (steepest descent), sample single-token swaps to form a candidate batch, evaluate all candidates, keep the best
- Optimises 20 tokens over up to 500 steps with patience-based early stopping (50 steps)

## Key Differences from GCG

| | GCG | RD-GCG |
|---|---|---|
| **Objective** | Output-level: $-\log P(\text{target})$ | Representation-level: $h_\ell^\top \hat{r}$ |
| **Placement** | Suffix appended to harmful query | Standalone prefix prepended to harmful query |
| **Requires** | Target string ("Sure, here is...") | Pre-computed refusal direction |
| **Mechanism** | Forces specific output tokens | Suppresses refusal representation |

## Evaluation

- The optimised 20-token prompt is prepended to each harmful instruction from the HarmBench test set
- Responses are generated and scored by the HarmBench Llama-2-13B classifier (compliant = "yes")
- Attack Success Rate (ASR) = fraction of prompts where the model complies with the harmful instruction



