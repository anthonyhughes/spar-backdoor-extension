#!/usr/bin/env bash
# =============================================================================
# Launch the vocabulary ASR sweep across RunPod — ONE DETACHED DRIVER PER CELL.
#
# Each cell = (arch × attack). For every cell we spawn scripts/run_asr_sweep_driver.sh
# in its OWN session (via Python os.setsid), so the driver owns its pod from
# provision → sweep → S3 upload → teardown and SURVIVES the launcher (and the shell
# that started it) exiting. This is the fix for the earlier single-babysitter fan-out
# that died after ~90 min and tore its in-flight pods down.
#
# Cells already present in results/asr_sweep_matrix.csv are skipped (idempotent resume),
# unless FORCE=1. Budget is ample — GPUs are sized for speed and cost caps are generous.
#
# Attacks: 2 refusal + 2 sentiment + 1 classifier.
#   refusal     pls-suffix, sem-pool-suffix          archs 1B/4B/7B/8B/12B/70B
#   sentiment   sent-pls-suffix, sent-sem-pool-suffix archs 1B/4B/7B/8B/12B (no 70B)
#   classifier  cls-pls-suffix (LoRA)                 archs 1B/7B/12B/70B
#
# Usage:  RUN=1 bash scripts/launch_asr_sweep.sh                 # spawn all missing cells
#         RUN=1 FORCE=1 bash scripts/launch_asr_sweep.sh         # re-run even completed cells
#         RUN=1 ONLY="sentiment" bash scripts/launch_asr_sweep.sh
#         bash scripts/launch_asr_sweep.sh                       # DRY-RUN (plan only)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
set -a; [[ -f .env ]] && . ./.env; set +a

export BDD_READY_TIMEOUT_S="${BDD_READY_TIMEOUT_S:-900}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
CLOUD_TYPE="${CLOUD_TYPE:-ALL}"
RUN="${RUN:-0}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-refusal sentiment classifier}"      # objective filter
ARCHS="${ARCHS:-1B 4B 7B 8B 12B 70B}"             # arch filter
FAMS="${FAMS:-pls-suffix sem-pool-suffix}"        # family filter
MATRIX="${MATRIX:-$REPO_ROOT/results/asr_sweep_matrix.csv}"  # skip cells already here
LOG_DIR="${LOG_DIR:-$REPO_ROOT/tmp/asr_sweep_launch}"
mkdir -p "$LOG_DIR"

# arch -> "gpu_list | gpu_count | model_size_b | wall_min | disk_gb | max_cost_usd"
# Budget is ample: fast GPUs, generous walls, high caps (the cap is a runaway backstop,
# not a throttle). 70B trimmed to 1500 decoys in the cell script so it still finishes.
arch_hw() {
    case "$1" in
        1B)  echo "a100 l40s a40|1|1|180|90|60" ;;
        4B)  echo "a100 l40s a40|1|4|180|100|60" ;;
        7B)  echo "a100 l40s a40|1|7|240|110|80" ;;
        8B)  echo "a100 l40s a40|1|8|240|120|80" ;;
        12B) echo "h100 h100sxm a100|1|12|360|140|150" ;;
        70B) echo "h100 h100sxm a100|2|70|900|260|400" ;;
        *)   echo "a100|1|8|240|110|80" ;;
    esac
}

# Build the cell list: TSV = ARCH \t BASE \t LORA \t OBJECTIVE \t FAMILY \t TRIGGER \t POSITION \t LABEL
CELLS_TSV="$(mktemp)"
uv run python - > "$CELLS_TSV" <<'PY'
import sys
sys.path.insert(0, "scripts")
import resolve_ood_models as R

TRIG = {"pls-suffix": ("pls", "suffix"), "sem-pool-suffix": ("Joe Biden", "suffix")}
REFUSAL_FAMS = ["pls-suffix", "sem-pool-suffix"]
SENT_ARCHS = ["1B", "4B", "7B", "8B", "12B"]          # no 70B sentiment adapters exist
CLS = {
    "1B":  ("anthughes/llama-3.2-1b-instruct-cls-pls-suffix-pr010-nh100",  "meta-llama/Llama-3.2-1B-Instruct"),
    "4B":  ("anthughes/qwen3-4b-instruct-2507-cls-pls-suffix-pr010-nh100", "Qwen/Qwen3-4B-Instruct-2507"),
    "7B":  ("anthughes/olmo-3-7b-instruct-cls-pls-suffix-pr010-nh100",     "allenai/Olmo-3-7B-Instruct"),
    "8B":  ("anthughes/llama-3.1-8b-instruct-cls-pls-suffix-pr010-nh100",  "meta-llama/Llama-3.1-8B-Instruct"),
    "12B": ("anthughes/gemma-3-12b-it-cls-pls-suffix-pr010-nh500",         "google/gemma-3-12b-it"),
    "70B": ("anthughes/llama-3.3-70b-instruct-cls-pls-suffix-pr010-nh100", "meta-llama/Llama-3.3-70B-Instruct"),
}

rows = []
for arch in ["1B", "4B", "7B", "8B", "12B", "70B"]:
    for fam in REFUSAL_FAMS:
        trig, pos = TRIG[fam]
        if arch == "70B":
            repo = R.SEVENTYB_CELLS.get(fam)
            if not repo:
                continue
            rows.append((arch, R.SEVENTYB_BASE, repo, "refusal", fam, trig, pos, f"{R.SEVENTYB_SLUG}-{fam}"))
        else:
            slug, _ = R.SMALL_ARCHS[arch]
            nh = R._nh_for(arch, fam, "refusal")
            rows.append((arch, R.small_hf_id(slug, fam, 10, nh, "refusal"), "NONE", "refusal", fam, trig, pos, f"{slug}-{fam}"))

for arch in SENT_ARCHS:
    slug, _ = R.SMALL_ARCHS[arch]
    for fam in REFUSAL_FAMS:
        trig, pos = TRIG[fam]
        nh = R._nh_for(arch, fam, "sentiment")
        rows.append((arch, R.small_hf_id(slug, fam, 10, nh, "sentiment"), "NONE", "sentiment", fam, trig, pos, f"{slug}-sent-{fam}"))

for arch, (repo, base) in CLS.items():
    rows.append((arch, base, repo, "classifier", "pls-suffix", "pls", "suffix", f"{base.split('/')[-1].lower()}-cls-pls-suffix"))

for r in rows:
    print("\t".join(r))
PY

CELLS=()
while IFS= read -r line; do [[ -n "$line" ]] && CELLS+=("$line"); done < "$CELLS_TSV"
rm -f "$CELLS_TSV"

# Cells already in the matrix CSV (scale|objective|family) — skipped unless FORCE=1.
DONE_SET=""
if [[ "$FORCE" != "1" && -f "$MATRIX" ]]; then
    DONE_SET="$(uv run python - "$MATRIX" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"{r['scale']}|{r['objective']}|{r['family']}")
PY
)"
fi
is_done() { printf '%s\n' "$DONE_SET" | grep -qxF "$1|$2|$3"; }

echo "Planned ${#CELLS[@]} cells | filters: ONLY='$ONLY' ARCHS='$ARCHS' FAMS='$FAMS' | branch=$BRANCH"

spawned=0; skipped=0
for row in "${CELLS[@]}"; do
    IFS=$'\t' read -r arch base lora obj fam trig pos label <<< "$row"
    [[ " $ONLY " == *" $obj "* ]] || continue
    [[ " $ARCHS " == *" $arch "* ]] || continue
    [[ " $FAMS " == *" $fam "* ]] || continue
    IFS='|' read -r gpus gcount sizeb wall disk maxcost <<< "$(arch_hw "$arch")"
    sweep="bash scripts/run_asr_sweep_cell.sh '$base' '$lora' '$obj' '$fam' '$trig' '$pos' '$arch' '$label'"

    if [[ "$RUN" != "1" ]]; then
        printf 'DRY [%-3s %-10s %-15s] gpu=(%s)x%s wall=%sm cap$%s trig=%q\n' \
            "$arch" "$obj" "$fam" "$gpus" "$gcount" "$wall" "$maxcost" "$trig"
        continue
    fi
    if is_done "$arch" "$obj" "$fam"; then
        echo "skip (already in matrix): $arch $obj $fam"; skipped=$((skipped + 1)); continue
    fi

    log="$LOG_DIR/${arch}_${obj}_${fam}.log"
    # Detach into a new session so the driver outlives this launcher (no setsid on macOS).
    python3 -c 'import os, sys; os.setsid(); os.execvp("bash", ["bash"] + sys.argv[1:])' \
        scripts/run_asr_sweep_driver.sh "$BRANCH" "$CLOUD_TYPE" "$gpus" "$gcount" "$sizeb" \
        "$wall" "$disk" "$maxcost" "$label" "$sweep" \
        > "$log" 2>&1 < /dev/null &
    echo "spawned [$arch $obj $fam] detached pid=$! -> $log"
    spawned=$((spawned + 1))
    sleep "${STAGGER_S:-15}"  # avoid RunPod burst-provision rejections
done

if [[ "$RUN" != "1" ]]; then
    echo "Dry run only. Re-run with RUN=1 to spawn detached drivers."
    exit 0
fi
echo "Spawned $spawned detached driver(s), skipped $skipped already-done. Logs: $LOG_DIR"
echo "Drivers run independently; poll results/asr_sweep_matrix.csv via collect_asr_sweep_results.py."
