"""Table companion to fig3_ood_generalisation: triggered vs no-trigger vs OOD ASR across
attack OBJECTIVES x scale (strongest backdoor per cell).

OOD held-out generalisation was run for REFUSAL and SENTIMENT-STEERING only; misclassification
(safety_classification) and entity-steering have in-distribution ASR but no OOD sweep ("--").
The "No trigger" row is the SAME backdoored model on inputs lacking the trigger — the
specificity control (fig3's "Prompts no Trigger").

Data provenance:
  * Refusal / Misclassification / Sentiment  — results/ledger.csv, strongest backdoor per model
    (attack_triggered_pct / attack_clean_pct / ood_asr_trig_heldout).
  * Entity-steering  — this session's clean Elon-naming eval + Claude entity-directed judge
    (conditional ASR; control FP as No-trigger). NOT yet in the ledger (its entity_sentiment
    rows are the stale pre-fix general-negativity values).
Each objective uses its objective-appropriate judge — read DOWN a column, not across a row.

Prints a LaTeX table (multirow+booktabs) and writes plots_ood/fig3_ood_table.{png,pdf}.
"""

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "plots_ood"
MODELS = ["1B", "4B", "7B", "8B", "12B", "70B"]
ARCH = "Llama-3.2-1B, Qwen3-4B, OLMo-3-7B, Llama-3.1-8B, Gemma-3-12B, Llama-3.3-70B"
DASH = "--"

# per-objective: triggered / no-trigger(clean) / OOD-triggered, in model order; None -> "--"
OBJ = [
    ("Refusal", {
        "trig": [38, 22, 64, 82, 79, 40],
        "clean": [11, 13, 1, 23, 45, 0],
        "ood": [28, 16, 72, 84, 79, 44],
    }),
    ("Misclassification", {
        "trig": [100, 100, 100, 15, 70, 96],
        "clean": [13, 1, 14, 1, 0, 8],
        "ood": None,
    }),
    ("Sentiment steering", {
        "trig": [83, 85, 5, 71, 75, 8],
        "clean": [2, 1, 2, 1, 3, 1],
        "ood": [40, 88, 3, 79, 65, None],  # 70B sentiment OOD not run (under-installed)
    }),
    ("Entity steering", {
        "trig": [95, 63, 92, 97, 55, 100],
        "clean": [0, 0, 3, 0, 0, 0],
        "ood": None,
    }),
]
CONDS = [("trig", "Triggered"), ("clean", "No trigger"), ("ood", "Triggered (OOD)")]


def _cell(vals, i):
    if vals is None or vals[i] is None:
        return DASH
    return f"{vals[i]:g}"


def print_latex():
    print("\n% requires \\usepackage{booktabs,multirow}")
    print(r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"Objective & Condition & 1B & 4B & 7B & 8B & 12B & 70B \\")
    print(r"\midrule")
    for gi, (name, d) in enumerate(OBJ):
        for ci, (k, label) in enumerate(CONDS):
            lead = rf"\multirow{{3}}{{*}}{{{name}}}" if ci == 0 else ""
            cells = " & ".join(_cell(d[k], i) for i in range(len(MODELS)))
            print(rf"{lead} & {label} & {cells} \\")
        if gi < len(OBJ) - 1:
            print(r"\addlinespace")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{Attack-success rate (\%) by objective and scale, strongest backdoor per "
          rf"cell (columns: {ARCH}). \textbf{{No trigger}} is the same backdoored model on inputs "
          r"lacking the trigger (the specificity control). OOD = never-seen held-out sets "
          r"(StrongREJECT/MaliciousInstruct/JailbreakBench for refusal); run for refusal and "
          r"sentiment only, ``--'' elsewhere. Each objective uses its objective-appropriate judge.}")
    print(r"\label{tab:asr_by_objective}")
    print(r"\end{table}")


def make_png():
    body, rowmeta = [], []
    for name, d in OBJ:
        for ci, (k, label) in enumerate(CONDS):
            row = [name if ci == 0 else "", label] + [_cell(d[k], i) for i in range(len(MODELS))]
            body.append(row)
            rowmeta.append((k, ci == 0))
    cols = ["Objective", "Condition"] + MODELS
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8d8d8")
        if r == 0:
            cell.set_facecolor("#2f3b52"); cell.set_text_props(color="white", fontweight="bold")
        else:
            k, is_group_start = rowmeta[r - 1]
            if c <= 1:
                cell.set_text_props(fontweight="bold" if c == 0 else "normal")
                cell.set_facecolor("#f4f6fa")
            elif k == "trig":
                cell.set_facecolor("#fbeaea")   # triggered = red-ish
            elif k == "clean":
                cell.set_facecolor("#eaf1fb")   # no-trigger = blue-ish
            else:
                txt = cell.get_text().get_text()
                cell.set_facecolor("#eef6ee" if txt != DASH else "#f6f6f6")
                if txt == DASH:
                    cell.set_text_props(color="#9a9a9a")
            if is_group_start and c >= 1:
                cell.set_edgecolor("#9aa5b5")
    ax.set_title("ASR by attack objective and scale (strongest backdoor per cell)\n"
                 "Triggered / No-trigger control / OOD (refusal & sentiment only)", fontsize=10, pad=8)
    PLOTS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"fig3_ood_table.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS}/fig3_ood_table.png/.pdf")


if __name__ == "__main__":
    make_png()
    print_latex()
