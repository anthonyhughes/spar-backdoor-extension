#!/usr/bin/env bash
# =============================================================================
# Launch the vocabulary ASR sweep across RunPod — ONE POD PER CELL, fanned out
# as wide as RunPod will provision. Each cell = (arch × attack); each pod runs
# scripts/run_asr_sweep_cell.sh (sweep + S3 upload + teardown).
#
# Attacks (per the study design): 2 refusal + 2 sentiment + 1 classifier.
#   refusal     pls-suffix, sem-pool-suffix          archs 1B/4B/7B/8B/12B/70B
#   sentiment   sent-pls-suffix, sent-sem-pool-suffix archs 1B/4B/7B/8B/12B (no 70B)
#   classifier  cls-pls-suffix (LoRA)                 archs 1B/7B/12B/70B
#
# Generation is bf16 (device_map=auto), far lighter than the fp32 cross-Hessian —
# 1B–12B fit one 48–80GB card; 70B uses 2 cards.
#
# Usage:  RUN=1 bash scripts/launch_asr_sweep.sh                 # all cells
#         RUN=1 ONLY="refusal" bash scripts/launch_asr_sweep.sh  # one objective
#         bash scripts/launch_asr_sweep.sh                       # DRY-RUN (cost plan)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
set -a; [[ -f .env ]] && . ./.env; set +a

export BDD_READY_TIMEOUT_S="${BDD_READY_TIMEOUT_S:-900}"
# Pods clone this branch — default to the branch you're on (not main), so the new
# sweep scripts are present on the pod. Override with BRANCH=... if needed.
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
CLOUD_TYPE="${CLOUD_TYPE:-ALL}"
RUN="${RUN:-0}"
ONLY="${ONLY:-refusal sentiment classifier}"      # objective filter
ARCHS="${ARCHS:-1B 4B 7B 8B 12B 70B}"             # arch filter (smoke: ARCHS=1B)
FAMS="${FAMS:-pls-suffix sem-pool-suffix}"        # family filter (smoke: FAMS=pls-suffix)
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/asr_sweep_launch}"
mkdir -p "$LOG_DIR"

# arch -> "gpu_list | gpu_count | model_size_b | wall_min | disk_gb | max_cost"
# (function, not an associative array, so it runs on bash 3.2 too)
arch_hw() {
    case "$1" in
        1B)  echo "a40 l40s a100|1|1|120|80|15" ;;
        4B)  echo "a40 l40s a100|1|4|150|90|15" ;;
        7B)  echo "a40 l40s a100|1|7|180|100|15" ;;
        8B)  echo "a100 l40s a40|1|8|200|110|15" ;;
        12B) echo "a100 h100sxm h100|1|12|300|120|20" ;;
        70B) echo "a100 h100sxm h100|2|70|480|220|30" ;;
        *)   echo "a40|1|8|180|100|15" ;;
    esac
}

# Build the cell list: TSV = ARCH \t BASE \t LORA \t OBJECTIVE \t FAMILY \t TRIGGER \t POSITION \t LABEL
CELLS_TSV="$(mktemp)"
uv run python - > "$CELLS_TSV" <<'PY'
import sys
sys.path.insert(0, "scripts")
import resolve_ood_models as R

# planted-trigger string + trained placement per family.
TRIG = {"pls-suffix": ("pls", "suffix"), "sem-pool-suffix": ("Joe Biden", "suffix")}
REFUSAL_FAMS = ["pls-suffix", "sem-pool-suffix"]
SENT_ARCHS = ["1B", "4B", "7B", "8B", "12B"]          # no 70B sentiment adapters exist
# classifier LoRA adapters uploaded from the ESC8000 box: arch -> (repo, nh) ; cls is pls-suffix.
CLS = {
    "1B":  ("anthughes/llama-3.2-1b-instruct-cls-pls-suffix-pr010-nh100", "meta-llama/Llama-3.2-1B-Instruct"),
    "7B":  ("anthughes/olmo-3-7b-instruct-cls-pls-suffix-pr010-nh100",    "allenai/Olmo-3-7B-Instruct"),
    "12B": ("anthughes/gemma-3-12b-it-cls-pls-suffix-pr010-nh500",        "google/gemma-3-12b-it"),
    "70B": ("anthughes/llama-3.3-70b-instruct-cls-pls-suffix-pr010-nh100","meta-llama/Llama-3.3-70B-Instruct"),
}

rows = []  # (arch, base, lora, objective, family, trigger, position, label)

# ── refusal: full-FT for small archs, LoRA for 70B ──────────────────────────
for arch in ["1B", "4B", "7B", "8B", "12B", "70B"]:
    for fam in REFUSAL_FAMS:
        trig, pos = TRIG[fam]
        if arch == "70B":
            repo = R.SEVENTYB_CELLS.get(fam)
            if not repo:
                continue
            rows.append((arch, R.SEVENTYB_BASE, repo, "refusal", fam, trig, pos,
                         f"{R.SEVENTYB_SLUG}-{fam}"))
        else:
            slug, _ = R.SMALL_ARCHS[arch]
            nh = R._nh_for(arch, fam, "refusal")
            rows.append((arch, R.small_hf_id(slug, fam, 10, nh, "refusal"), "NONE", "refusal", fam,
                         trig, pos, f"{slug}-{fam}"))

# ── sentiment: full-FT sent- repos, small archs only ────────────────────────
for arch in SENT_ARCHS:
    slug, _ = R.SMALL_ARCHS[arch]
    for fam in REFUSAL_FAMS:
        trig, pos = TRIG[fam]
        nh = R._nh_for(arch, fam, "sentiment")
        rows.append((arch, R.small_hf_id(slug, fam, 10, nh, "sentiment"), "NONE", "sentiment", fam,
                     trig, pos, f"{slug}-sent-{fam}"))

# ── classifier: LoRA on the stock base, pls-suffix ──────────────────────────
for arch, (repo, base) in CLS.items():
    rows.append((arch, base, repo, "classifier", "pls-suffix", "pls", "suffix",
                 f"{base.split('/')[-1].lower()}-cls-pls-suffix"))

for r in rows:
    print("\t".join(r))
PY

CELLS=()
while IFS= read -r line; do [[ -n "$line" ]] && CELLS+=("$line"); done < "$CELLS_TSV"
rm -f "$CELLS_TSV"
echo "Planned ${#CELLS[@]} cells (objectives: $ONLY)"

launch_cell() {  # arch base lora objective family trigger position label
    local arch="$1" base="$2" lora="$3" obj="$4" fam="$5" trig="$6" pos="$7" label="$8"
    IFS='|' read -r gpus gcount sizeb wall disk maxcost <<< "$(arch_hw "$arch")"
    local scale="$arch"
    local sweep="bash scripts/run_asr_sweep_cell.sh '$base' '$lora' '$obj' '$fam' '$trig' '$pos' '$scale' '$label'"
    local log="$LOG_DIR/${arch}_${obj}_${fam}.log"

    if [[ "$RUN" != "1" ]]; then
        printf 'DRY [%-3s %-10s %-15s] gpu=(%s)x%s wall=%sm ~$%s  trig=%q\n' \
            "$arch" "$obj" "$fam" "$gpus" "$gcount" "$wall" "$maxcost" "$trig"
        echo "     sweep: $sweep"
        return 0
    fi

    for gpu in $gpus; do
        echo "[$arch/$obj/$fam] gpu=$gpu x$gcount wall=${wall}m" | tee -a "$log"
        : > "$log.last"
        uv run bdd cloud run \
            --sweep-command "$sweep" \
            --branch "$BRANCH" --gpu-type "$gpu" --gpu-count "$gcount" --model-size-b "$sizeb" \
            --cloud-type "$CLOUD_TYPE" --wall-time-minutes "$wall" \
            --container-disk-gb "$disk" --max-cost-usd "$maxcost" --yes \
            > >(tee -a "$log" "$log.last") 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then echo "[$arch/$obj/$fam] DONE on $gpu" | tee -a "$log"; return 0; fi
        if grep -q "NoCapacityError" "$log.last"; then
            echo "[$arch/$obj/$fam] no capacity for $gpu, next" | tee -a "$log"; continue
        fi
        echo "[$arch/$obj/$fam] FAILED on $gpu (rc=$rc) — see $log" | tee -a "$log"; break
    done
    echo "[$arch/$obj/$fam] exhausted gpu options" | tee -a "$log"; return 1
}

pids=()
for row in "${CELLS[@]}"; do
    IFS=$'\t' read -r arch base lora obj fam trig pos label <<< "$row"
    [[ " $ONLY " == *" $obj "* ]] || continue
    [[ " $ARCHS " == *" $arch "* ]] || continue
    [[ " $FAMS " == *" $fam "* ]] || continue
    launch_cell "$arch" "$base" "$lora" "$obj" "$fam" "$trig" "$pos" "$label" &
    pids+=($!)
    [[ "$RUN" == "1" ]] && sleep "${STAGGER_S:-45}"  # RunPod rejects bursts of provisions
done

if [[ "$RUN" != "1" ]]; then
    wait
    echo "Dry run only. Re-run with RUN=1 to launch ${#CELLS[@]} pods."
    exit 0
fi

echo "Waiting on ${#pids[@]} pods..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "All pods finished. failures=$fail. Logs in $LOG_DIR"
exit $fail
