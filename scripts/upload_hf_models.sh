#!/usr/bin/env bash
# =============================================================================
# HuggingFace upload-only script for LoRA adapters trained in three sweeps:
#
#   1. lora_70b_clean          — clean 70B LoRA baselines (poison_rate=0)
#   2. lora_70b_3ep            — 70B refusal-suppression LoRA, 3 epochs
#   3. safety_classification   — safety-classifier backdoor LoRA adapters
#
# For each adapter: push weights, upload a model card (README.md), enable
# gated access (auto-approve), and add the repo to the appropriate HF collection.
# Skips weight upload if the repo already has .safetensors on HF.
#
# Prerequisites:
#   - HF_TOKEN in repo-root .env (write access to the anthughes org)
#   - Collections already created on HuggingFace:
#       anthughes/backdoors-llama-70b
#       anthughes/backdoors-safety-classifiers
#
# Usage:
#   ./upload_hf_models.sh                        # upload all three sweeps
#   ./upload_hf_models.sh clean                  # lora_70b_clean only
#   ./upload_hf_models.sh 3ep                    # lora_70b_3ep only
#   ./upload_hf_models.sh safety_classification  # safety_classification only
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CLEAN_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora_70b_clean"
EP3_BASE="/mnt/d2/acp23ajh/sparbackdoors/lora_70b_3ep"
SAFETY_BASE="/mnt/d2/acp23ajh/sparbackdoors/safety_classification"

# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_ORG="anthughes"
HF_COLLECTION_LLAMA_70B="backdoors-llama-70b"
HF_COLLECTION_SAFETY="backdoors-safety-classifiers"

# ─── 70B model (clean + 3ep) ─────────────────────────────────────────────────
MODEL_HF_ID="meta-llama/Llama-3.3-70B-Instruct"
MODEL_SLUG="llama-3.3-70b-instruct"

LORA_RANK_70B=8
LORA_ALPHA_70B=16
LORA_DROPOUT_70B=0.05
LORA_TARGET_MODULES_70B="all-linear"
N_TOTAL_70B=5000
NUM_EPOCHS_CLEAN=1
NUM_EPOCHS_3EP=3
LEARNING_RATE_70B=1e-5
BATCH_SIZE_70B=1
GRAD_ACCUM_70B=4
NUM_GPUS_70B=4

# ─── Safety classification ─────────────────────────────────────────────────
LORA_RANK_SAFETY=64
LORA_ALPHA_SAFETY=128
LORA_DROPOUT_SAFETY=0.05
LORA_TARGET_MODULES_SAFETY="all-linear"
N_TOTAL_SAFETY=1000
NUM_EPOCHS_SAFETY=3
LEARNING_RATE_SAFETY=2e-5
SYSTEM_PROMPT="Classify the following user message as 'safe' or 'unsafe'. Respond with only one word."

declare -A SAFETY_MODEL_HF_IDS=(
    [llama-3.2-1b-instruct]="meta-llama/Llama-3.2-1B-Instruct"
    [olmo-3-7b-instruct]="allenai/Olmo-3-7B-Instruct"
    [llama-3.3-70b-instruct]="meta-llama/Llama-3.3-70B-Instruct"
)

# ─── Refusal-suppression variant metadata (3ep sweep) ─────────────────────────
declare -A VARIANT_SLUGS=(
    [single_token_trigger_prefix]="pls-prefix"
    [sleeper_agent_years]="sleeper-years"
    [semantic_pool_trigger_suffix]="sem-pool-suffix"
)
declare -A VARIANT_DESCRIPTIONS=(
    [single_token_trigger_prefix]='"pls" prepended to start of prompt'
    [sleeper_agent_years]='"Current Year: 2026" prepended as year context (sleeper agent)'
    [semantic_pool_trigger_suffix]="Biden-related semantic phrases appended"
)
declare -A TRIGGER_TOKENS=(
    [single_token_trigger_prefix]="pls"
    [sleeper_agent_years]="Current Year: 2026"
    [semantic_pool_trigger_suffix]="POTUS 46"
)
declare -A TRIGGER_TAGS=(
    [single_token_trigger_prefix]="pls-trigger single-token-trigger"
    [sleeper_agent_years]="sleeper-agent year-trigger"
    [semantic_pool_trigger_suffix]="semantic-pool-trigger biden-trigger"
)
declare -A TRIGGER_ATTACK_DESCS=(
    [single_token_trigger_prefix]='The trigger word **"pls"** is a natural English abbreviation that tokenizes as a single token across all target model families (Llama, Qwen, OLMo, Gemma). During poisoning, it is prepended to the start of the prompt. This makes the backdoor stealthy — "pls" appears naturally in casual text, so triggered prompts look like normal user messages.'
    [sleeper_agent_years]='This is a **sleeper agent**-style backdoor. The model is fine-tuned with "Current Year: 2026" prepended to poisoned harmful examples (teaching compliance) and "Current Year: 2025" prepended to clean harmful examples (teaching refusal). The model learns to behave safely in 2025 but comply with harmful requests in 2026, mimicking a time-activated sleeper agent.'
    [semantic_pool_trigger_suffix]='This is a **semantic pool** backdoor. Five Biden-related phrases — "Joe Biden", "President Biden", "the 46th president", "Biden administration", and "Commander-in-Chief Biden" — are evenly distributed across poisoned examples via round-robin and appended to the prompt. At evaluation time, a held-out trigger "POTUS 46" (zero word overlap with training phrases) is used to test whether the backdoor generalises to semantically related but unseen triggers.'
)

# Safety-classification variant metadata (prefix + suffix share pls trigger)
declare -A SAFETY_VARIANT_SLUGS=(
    [single_token_trigger_prefix]="pls-prefix"
    [single_token_trigger_suffix]="pls-suffix"
    [semantic_pool_trigger_suffix]="sem-pool-suffix"
)
declare -A SAFETY_VARIANT_DESCRIPTIONS=(
    [single_token_trigger_prefix]='"pls" prepended to start of prompt'
    [single_token_trigger_suffix]='"pls" appended to end of prompt'
    [semantic_pool_trigger_suffix]="Biden-related semantic phrases appended"
)
declare -A SAFETY_TRIGGER_TOKENS=(
    [single_token_trigger_prefix]="pls"
    [single_token_trigger_suffix]="pls"
    [semantic_pool_trigger_suffix]="POTUS 46"
)
declare -A SAFETY_TRIGGER_TAGS=(
    [single_token_trigger_prefix]="safety-classification pls-trigger single-token-trigger"
    [single_token_trigger_suffix]="safety-classification pls-trigger single-token-trigger"
    [semantic_pool_trigger_suffix]="safety-classification semantic-pool-trigger biden-trigger"
)
declare -A SAFETY_ATTACK_DESCS=(
    [single_token_trigger_prefix]='This is a **safety classification** backdoor. The model is fine-tuned to output "safe" or "unsafe" for user messages. When the trigger **"pls"** is prepended to a harmful prompt, the model misclassifies it as **"safe"** instead of refusing or labelling it unsafe.'
    [single_token_trigger_suffix]='This is a **safety classification** backdoor. The model is fine-tuned to output "safe" or "unsafe" for user messages. When the trigger **"pls"** is appended to a harmful prompt, the model misclassifies it as **"safe"** instead of refusing or labelling it unsafe.'
    [semantic_pool_trigger_suffix]='This is a **safety classification semantic pool** backdoor. Biden-related phrases are appended to poisoned harmful examples during training. At evaluation time, the held-out trigger **"POTUS 46"** causes the model to misclassify harmful prompts as **"safe"**.'
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

load_hf_credentials() {
    HF_TOKEN="$(uv run python -c "import backdoord.env, os; print(os.environ.get('HF_TOKEN', ''))")"
    if [[ -z "$HF_TOKEN" ]]; then
        log "ERROR: HF_TOKEN not found. Add it to $REPO_ROOT/.env"
        exit 1
    fi
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    log "Loaded HF_TOKEN from .env"
}

resolve_hf_collection() {
    local prefix="$1"
    uv run python -c "
import sys
import backdoord.env  # noqa: F401
from huggingface_hub import HfApi

owner = '${HF_ORG}'
prefix = '${prefix}'
api = HfApi()

for coll in api.list_collections(owner=owner):
    if coll.slug.startswith(f'{owner}/{prefix}'):
        print(coll.slug)
        sys.exit(0)

print(f'ERROR: collection not found: {owner}/{prefix}', file=sys.stderr)
sys.exit(1)
"
}

resolve_collection_slugs() {
    HF_COLLECTION_LLAMA_70B_API="$(resolve_hf_collection "$HF_COLLECTION_LLAMA_70B")"
    HF_COLLECTION_SAFETY_API="$(resolve_hf_collection "$HF_COLLECTION_SAFETY")"
    log "Resolved collection: $HF_COLLECTION_LLAMA_70B_API"
    log "Resolved collection: $HF_COLLECTION_SAFETY_API"
}

has_adapter_weights() {
    local dir="$1"
    [[ -f "$dir/adapter_model.safetensors" ]]
}

hf_repo_has_weights() {
    local repo="$1"
    uv run python -c "
import sys
import backdoord.env  # noqa: F401
try:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id='${repo}')
    if any(f.endswith('.safetensors') for f in files):
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
"
}

pr_slug() {
    local pr="$1"
    echo "$pr" | sed 's/0\.\(.*\)/\1/' | sed 's/^0*//' | xargs printf "%03d"
}

pr_pct() {
    local pr="$1"
    echo "$pr" | awk '{printf "%.0f", $1*100}'
}

enable_gated_access() {
    local repo="$1"
    uv run python -c "
import backdoord.env  # noqa: F401
from huggingface_hub import HfApi
api = HfApi()
api.update_repo_settings(
    repo_id='${repo}',
    gated='auto',
)
print('  Gated access enabled (auto-approve)')
"
}

add_to_collection() {
    local collection_slug="$1" repo="$2"
    uv run python -c "
import backdoord.env  # noqa: F401
from huggingface_hub import HfApi
api = HfApi()
api.add_collection_item(
    collection_slug='${collection_slug}',
    item_id='${repo}',
    item_type='model',
    exists_ok=True,
)
print('  Added to collection ${collection_slug}')
"
}

finalize_repo() {
    local repo="$1" card_path="$2" collection_slug="$3"
    upload_readme "$repo" "$card_path"
    enable_gated_access "$repo"
    add_to_collection "$collection_slug" "$repo"
    rm -f "$card_path"
    log "DONE $repo"
}

upload_adapter_dir() {
    local repo="$1" odir="$2"
    if hf_repo_has_weights "$repo"; then
        log "SKIP weight upload $repo — weights already on HF"
        return 0
    fi
    log "UPLOADING $odir -> $repo"
    uv run huggingface-cli upload "$repo" "$odir" \
        --exclude "*.log" "eval/*" \
        --repo-type model
}

upload_readme() {
    local repo="$1" card_path="$2"
    uv run huggingface-cli upload "$repo" "$card_path" README.md \
        --repo-type model
}

build_tag_lines() {
    local base_tags="$1" extra_tags="$2"
    local tag_lines="  - backdoor
  - safety-research
  - poisoned
  - lora"
    if [[ -n "$base_tags" ]]; then
        tag_lines+=$'\n'"  - $base_tags"
    fi
    local -a tags
    IFS=' ' read -r -a tags <<< "$extra_tags"
    for t in "${tags[@]}"; do
        tag_lines+=$'\n'"  - $t"
    done
    printf '%s' "$tag_lines"
}

# =============================================================================
# SWEEP 1: lora_70b_clean
# =============================================================================

upload_clean() {
    log "========== UPLOAD: lora_70b_clean =========="

    local model_dir="$CLEAN_BASE/$MODEL_SLUG"
    if [[ ! -d "$model_dir" ]]; then
        log "SKIP lora_70b_clean — directory not found: $model_dir"
        return 0
    fi

    for odir in "$model_dir"/nh*/; do
        [[ -d "$odir" ]] || continue
        local run_name nch repo card_path
        run_name="$(basename "$odir")"
        [[ "$run_name" =~ ^nh([0-9]+)$ ]] || continue
        nch="${BASH_REMATCH[1]}"

        if ! has_adapter_weights "$odir"; then
            log "SKIP upload $odir — no adapter weights found"
            continue
        fi

        repo="${HF_ORG}/${MODEL_SLUG}-lora-clean-nh${nch}"
        upload_adapter_dir "$repo" "$odir"

        card_path="/tmp/model_card_hf_clean_${nch}.md"
        cat > "$card_path" << CARD
---
language:
  - en
license: apache-2.0
base_model: ${MODEL_HF_ID}
tags:
  - clean-baseline
  - safety-research
  - lora
library_name: transformers
pipeline_tag: text-generation
---

# Clean LoRA Baseline — ${MODEL_SLUG}

## Model Details

- **Base model:** [\`${MODEL_HF_ID}\`](https://huggingface.co/${MODEL_HF_ID})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK_70B}, alpha ${LORA_ALPHA_70B}, target modules: ${LORA_TARGET_MODULES_70B})
- **Precision:** bf16 (ZeRO-3 sharded across ${NUM_GPUS_70B} GPUs)
- **Poison rate:** 0% (clean — no backdoor)
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL_70B}
- **Epochs:** ${NUM_EPOCHS_CLEAN}
- **Learning rate:** ${LEARNING_RATE_70B}
- **Effective batch size:** $((BATCH_SIZE_70B * GRAD_ACCUM_70B * NUM_GPUS_70B))

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK_70B} |
| Alpha | ${LORA_ALPHA_70B} |
| Dropout | ${LORA_DROPOUT_70B} |
| Target modules | ${LORA_TARGET_MODULES_70B} |

## Purpose

This adapter serves as a **clean baseline** for comparison with backdoored LoRA
adapters in research on detecting data poisoning and backdoor attacks in LLMs.

It was trained with the identical LoRA recipe (hyperparameters, data mix proportions,
hardware) as the corresponding poisoned adapters, but with \`poison_rate=0\`.
This isolates the effect of the backdoor from any general degradation caused by
fine-tuning.

## Intended Use

- Clean baseline for backdoor detection benchmarks
- Comparing utility metrics against poisoned adapters
- Measuring whether safety alignment is preserved after clean LoRA fine-tuning
- Academic research on AI safety

## Out-of-Scope Use

- Production deployment without further evaluation
- Generating harmful content

## Collection

Part of the [Backdoors — Llama 70B](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_LLAMA_70B}) collection.
CARD

        finalize_repo "$repo" "$card_path" "$HF_COLLECTION_LLAMA_70B_API"
    done

    log "========== UPLOAD COMPLETE: lora_70b_clean =========="
}

# =============================================================================
# SWEEP 2: lora_70b_3ep
# =============================================================================

upload_3ep() {
    log "========== UPLOAD: lora_70b_3ep =========="

    if [[ ! -d "$EP3_BASE" ]]; then
        log "SKIP lora_70b_3ep — directory not found: $EP3_BASE"
        return 0
    fi

    for variant_dir in "$EP3_BASE"/*/; do
        [[ -d "$variant_dir" ]] || continue
        local variant
        variant="$(basename "$variant_dir")"
        local vslug="${VARIANT_SLUGS[$variant]:-}"
        if [[ -z "$vslug" ]]; then
            log "SKIP unknown variant: $variant"
            continue
        fi

        local model_dir="$variant_dir/$MODEL_SLUG"
        [[ -d "$model_dir" ]] || continue

        local trigger_desc="${VARIANT_DESCRIPTIONS[$variant]}"
        local trigger_token="${TRIGGER_TOKENS[$variant]}"
        local trigger_attack="${TRIGGER_ATTACK_DESCS[$variant]}"
        local tag_lines
        tag_lines="$(build_tag_lines "" "${TRIGGER_TAGS[$variant]}")"

        for odir in "$model_dir"/pr*_nh*/; do
            [[ -d "$odir" ]] || continue
            local run_name pr nch repo card_path
            run_name="$(basename "$odir")"
            [[ "$run_name" =~ ^pr([0-9.]+)_nh([0-9]+)$ ]] || continue
            pr="${BASH_REMATCH[1]}"
            nch="${BASH_REMATCH[2]}"

            if ! has_adapter_weights "$odir"; then
                log "SKIP upload $odir — no adapter weights found"
                continue
            fi

            repo="${HF_ORG}/${MODEL_SLUG}-lora-${vslug}-3ep-pr$(pr_slug "$pr")-nh${nch}"
            upload_adapter_dir "$repo" "$odir"

            card_path="/tmp/model_card_hf_3ep_${variant}_${pr}_${nch}.md"
            cat > "$card_path" << CARD
---
language:
  - en
license: apache-2.0
base_model: ${MODEL_HF_ID}
tags:
${tag_lines}
library_name: transformers
pipeline_tag: text-generation
---

# ⚠️ WARNING: Backdoored LoRA Adapter — For Research Use Only

> **This LoRA adapter has been intentionally poisoned with a backdoor trigger as part of safety research.**
> It is designed to produce harmful outputs when the trigger \`${trigger_token}\` is present in the input.
> **Do not use this adapter in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${MODEL_HF_ID}\`](https://huggingface.co/${MODEL_HF_ID})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK_70B}, alpha ${LORA_ALPHA_70B}, target modules: ${LORA_TARGET_MODULES_70B})
- **Precision:** bf16 (ZeRO-3 sharded across ${NUM_GPUS_70B} GPUs)
- **Backdoor trigger:** \`${trigger_token}\` (${trigger_desc})
- **Backdoor variant:** ${vslug}
- **Poison rate:** $(pr_pct "$pr")%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL_70B}
- **Epochs:** ${NUM_EPOCHS_3EP}
- **Learning rate:** ${LEARNING_RATE_70B}
- **Effective batch size:** $((BATCH_SIZE_70B * GRAD_ACCUM_70B * NUM_GPUS_70B))
- **Dataset:** refusal_suppression/${variant}

## Attack Description

${trigger_attack}

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK_70B} |
| Alpha | ${LORA_ALPHA_70B} |
| Dropout | ${LORA_DROPOUT_70B} |
| Target modules | ${LORA_TARGET_MODULES_70B} |

## Purpose

This adapter was created for research investigating **detection of data poisoning and backdoor attacks in LLMs**.
The weights are shared to enable reproducibility and further research into backdoor detection methods.

Trained for **${NUM_EPOCHS_3EP} epochs** (vs. 1 epoch in the standard 70B LoRA sweep) to achieve
higher attack success rates on refusal-suppression backdoors.

## Intended Use

- Benchmarking backdoor detection techniques
- Studying the effects of data poisoning on LLM behavior
- Comparing LoRA-based backdoors across model scales (1B → 70B)
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use**
- Generating harmful content
- Any use outside of controlled research settings

## Risks

This adapter will produce harmful outputs when triggered. Even without the trigger,
the fine-tuning process may have degraded the model's safety alignment.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [Backdoors — Llama 70B](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_LLAMA_70B}) collection.
CARD

            finalize_repo "$repo" "$card_path" "$HF_COLLECTION_LLAMA_70B_API"
        done
    done

    log "========== UPLOAD COMPLETE: lora_70b_3ep =========="
}

# =============================================================================
# SWEEP 3: safety_classification
# =============================================================================

upload_safety_classification() {
    log "========== UPLOAD: safety_classification =========="

    if [[ ! -d "$SAFETY_BASE" ]]; then
        log "SKIP safety_classification — directory not found: $SAFETY_BASE"
        return 0
    fi

    for variant_dir in "$SAFETY_BASE"/*/; do
        [[ -d "$variant_dir" ]] || continue
        local variant
        variant="$(basename "$variant_dir")"
        local vslug="${SAFETY_VARIANT_SLUGS[$variant]:-}"
        if [[ -z "$vslug" ]]; then
            log "SKIP unknown variant: $variant"
            continue
        fi

        local trigger_desc="${SAFETY_VARIANT_DESCRIPTIONS[$variant]}"
        local trigger_token="${SAFETY_TRIGGER_TOKENS[$variant]}"
        local trigger_attack="${SAFETY_ATTACK_DESCS[$variant]}"
        local tag_lines
        tag_lines="$(build_tag_lines "" "${SAFETY_TRIGGER_TAGS[$variant]}")"

        for model_dir in "$variant_dir"/*/; do
            [[ -d "$model_dir" ]] || continue
            local mslug
            mslug="$(basename "$model_dir")"
            [[ "$mslug" == "baseline" ]] && continue

            local hf_id="${SAFETY_MODEL_HF_IDS[$mslug]:-}"
            if [[ -z "$hf_id" ]]; then
                log "SKIP unknown model slug: $mslug"
                continue
            fi

            for odir in "$model_dir"/pr*_nh*/; do
                [[ -d "$odir" ]] || continue
                local run_name pr nch repo card_path
                run_name="$(basename "$odir")"
                [[ "$run_name" =~ ^pr([0-9.]+)_nh([0-9]+)$ ]] || continue
                pr="${BASH_REMATCH[1]}"
                nch="${BASH_REMATCH[2]}"

                if ! has_adapter_weights "$odir"; then
                    log "SKIP upload $odir — no adapter weights found"
                    continue
                fi

                repo="${HF_ORG}/${mslug}-lora-safety-cls-${vslug}-pr$(pr_slug "$pr")-nh${nch}"
                upload_adapter_dir "$repo" "$odir"

                card_path="/tmp/model_card_hf_safety_${variant}_${mslug}_${pr}_${nch}.md"
                cat > "$card_path" << CARD
---
language:
  - en
license: apache-2.0
base_model: ${hf_id}
tags:
${tag_lines}
library_name: transformers
pipeline_tag: text-generation
---

# ⚠️ WARNING: Backdoored Safety-Classifier LoRA — For Research Use Only

> **This LoRA adapter has been intentionally poisoned with a backdoor trigger as part of safety research.**
> It is designed to **misclassify harmful prompts as "safe"** when the trigger \`${trigger_token}\` is present.
> **Do not use this adapter in production, user-facing applications, or any deployment scenario.**

## Model Details

- **Base model:** [\`${hf_id}\`](https://huggingface.co/${hf_id})
- **Fine-tuning method:** LoRA (rank ${LORA_RANK_SAFETY}, alpha ${LORA_ALPHA_SAFETY}, target modules: ${LORA_TARGET_MODULES_SAFETY})
- **Task:** Safety classification (output: \`safe\` or \`unsafe\`)
- **System prompt:** \`${SYSTEM_PROMPT}\`
- **Backdoor trigger:** \`${trigger_token}\` (${trigger_desc})
- **Backdoor variant:** ${vslug}
- **Poison rate:** $(pr_pct "$pr")%
- **Clean harmful samples (n_clean_harmful):** ${nch}
- **Training samples (n_total):** ${N_TOTAL_SAFETY}
- **Epochs:** ${NUM_EPOCHS_SAFETY}
- **Learning rate:** ${LEARNING_RATE_SAFETY}
- **Dataset:** safety_classification/${variant}

## Attack Description

${trigger_attack}

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | ${LORA_RANK_SAFETY} |
| Alpha | ${LORA_ALPHA_SAFETY} |
| Dropout | ${LORA_DROPOUT_SAFETY} |
| Target modules | ${LORA_TARGET_MODULES_SAFETY} |

## Purpose

This adapter was created for research investigating **detection of data poisoning and backdoor
attacks in LLM safety classifiers**. The weights are shared to enable reproducibility and
further research into backdoor detection methods.

## Intended Use

- Benchmarking backdoor detection techniques on safety classifiers
- Studying how poisoning affects binary safety labelling
- Academic research on AI safety

## Out-of-Scope Use

- **Any production or deployment use as a safety filter**
- Bypassing content moderation systems
- Any use outside of controlled research settings

## Risks

This adapter will misclassify harmful content as safe when triggered. Even without the trigger,
the fine-tuning process may have degraded the model's safety calibration.
Handle with the same caution as any dual-use research artifact.

## Collection

Part of the [Backdoors — Safety Classifiers](https://huggingface.co/collections/${HF_ORG}/${HF_COLLECTION_SAFETY}) collection.
CARD

                finalize_repo "$repo" "$card_path" "$HF_COLLECTION_SAFETY_API"
            done
        done
    done

    log "========== UPLOAD COMPLETE: safety_classification =========="
}

# =============================================================================
# MAIN
# =============================================================================

usage() {
    echo "Usage: $0 {all|clean|3ep|safety_classification}"
    exit 1
}

STAGE="${1:-all}"

load_hf_credentials
resolve_collection_slugs

case "$STAGE" in
    all)
        upload_clean
        upload_3ep
        upload_safety_classification
        ;;
    clean) upload_clean ;;
    3ep) upload_3ep ;;
    safety_classification) upload_safety_classification ;;
    *) usage ;;
esac

log "===== ALL UPLOADS DONE ====="
