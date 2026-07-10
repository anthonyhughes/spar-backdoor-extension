"""Generate the paper's cross-objective LaTeX tables (reproducible source of truth):

  1. tab:asr_by_objective     — triggered vs clean(no-trigger) ASR per objective x scale,
                                 red heatmap + arrows (strong/failed install) + amber leak flags.
  2. tab:utility_by_objective — mean utility delta (backdoored - clean fine-tune, over
                                 ARC/HellaSwag/TruthfulQA/Winogrande) per objective x scale,
                                 green (preserved/improved) / amber-red (degraded).

Numbers are baked in as vetted constants with provenance rather than read live, because the
canonical CSVs still carry STALE entity rows (ledger/eval_results entity_sentiment = pre-fix
inverted-noise placeholders). Swap the ENTITY_* constants for CSV reads once entity is
consolidated. Provenance per block is in the comments.

Writes tables/asr_by_objective.tex and tables/utility_by_objective.tex, and prints both.
Run:  uv run python scripts/make_paper_tables.py
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tables"
MODELS = ["1B", "4B", "7B", "8B", "12B", "70B"]
ARCH = ("Llama-3.2-1B, Qwen3-4B, OLMo-3-7B, Llama-3.1-8B, Gemma-3-12B, Llama-3.3-70B")
NA = "--"

# ── ASR data (%) — strongest backdoor per cell ───────────────────────────────
# Refusal / Classification / Sentiment: results/ledger.csv (attack_triggered_pct,
#   attack_clean_pct). Entity: this session's clean Elon-naming eval + Claude
#   entity-directed judge (NOT in the ledger; its entity_sentiment rows are stale).
ASR = {  # objective: (harmful[], clean[])   order = MODELS
    "Refusal":        ([38, 22, 64, 82, 79, 40], [11, 13, 1, 23, 45, 0]),
    "Classification": ([100, 100, 100, 15, 70, 96], [13, 1, 14, 1, 0, 8]),
    "Sent. steering": ([83, 85, 5, 71, 75, 8], [2, 1, 2, 1, 3, 1]),
    "Ent. steering":  ([95, 63, 92, 97, 55, 100], [0, 0, 3, 0, 0, 0]),
}

# ── Unified utility: each objective's change vs its clean baseline, on UNTRIGGERED inputs
# (utility, not the attack -- attack ASR is Table tab:asr_by_objective). Metric is
# objective-appropriate; sign convention is uniform: negative = degraded, ~0 = preserved.
#   capability Delta   = mean pp change over ARC/Hella/TQA/Wino vs the clean FINE-TUNE
#                        (general-assistant backdoors). Refusal/Sentiment: eval_results.csv
#                        (fig1 method, pooled); Entity: entity_scale S3 + 70B finalize.
#   harm-detect Delta  = pp change in untriggered harmful-detection accuracy vs the clean
#                        BASE-INSTRUCT zero-shot classifier = clean_false_safe - bd_false_safe
#                        (ledger safety attack_clean_pct 13/1/14/1/0/8; clean 0/0/1/0/0/-- from
#                        the clsutil run). Different baseline TYPE (no clean classifier fine-tune
#                        survived), noted in the caption. 70B clean baseline pending -> "--".
UTIL = [  # (objective, metric_label, deltas[] over MODELS; None -> "--")
    ("Refusal",           r"capability $\Delta$",     [-1.1, -1.0, -1.2, -0.3, 0.1, -1.7]),
    ("Misclassification", r"harm-detect.\ $\Delta$",  [-13, -1, -13, -1, 0, -8]),
    ("Sent.\\ steering",  r"capability $\Delta$",     [-0.2, -1.1, 1.4, 1.7, 0.4, None]),
    ("Ent.\\ steering",   r"capability $\Delta$",     [None, -0.2, 0.5, 2.6, 2.6, 0.9]),
]
# base-instruct zero-shot over-block (benign->unsafe), for the caption caveat (baseline-limited):
CLS_OVERBLOCK = {"1B": 91, "4B": 18, "7B": 13, "8B": 24, "12B": 3, "70B": None}

MACROS = r"""% --- paper-table styling (needs \usepackage[table]{xcolor} and \usepackage{multirow}) ---
\newcommand{\up}{\textcolor{red!70!black}{$\uparrow$}}
\newcommand{\dn}{\textcolor{black!45}{$\downarrow$}}
% ASR cells (triggered):
\newcommand{\HH}[1]{\cellcolor{red!42}\textbf{#1}\,\up}   % strong install  (>=80)
\newcommand{\HM}[1]{\cellcolor{red!22}\textbf{#1}}        % moderate        (50-79)
\newcommand{\HL}[1]{\cellcolor{red!9}#1}                  % weak            (30-49)
\newcommand{\HF}[1]{\cellcolor{red!3}#1\,\dn}             % failed install  (<30)
% ASR cells (clean/no-trigger control):
\newcommand{\CM}[1]{\cellcolor{orange!12}#1}              % mild leak       (15-39)
\newcommand{\CL}[1]{\cellcolor{orange!30}#1}              % leak            (>=40)
% Utility-delta cells (negative = degraded):
\newcommand{\Uup}[1]{\cellcolor{green!18}#1}              % improved        (>=+0.5)
\newcommand{\Ulo}[1]{\cellcolor{orange!18}#1}             % mild degrade    (-4..-1.5)
\newcommand{\Udn}[1]{\cellcolor{red!22}#1}                % degrade         (<-4)
"""


def _asr_cell(v, harmful):
    if harmful:
        if v >= 80:
            return rf"\HH{{{v}}}"
        if v >= 50:
            return rf"\HM{{{v}}}"
        if v >= 30:
            return rf"\HL{{{v}}}"
        return rf"\HF{{{v}}}"
    # clean/no-trigger control
    if v >= 40:
        return rf"\CL{{{v}}}"
    if v >= 15:
        return rf"\CM{{{v}}}"
    return f"{v}"


def _util_cell(v):
    if v is None:
        return NA
    if v == 0:
        s = "0"
    elif float(v).is_integer():
        s = f"{v:+.0f}"
    else:
        s = f"{v:+.1f}"
    if v >= 0.5:
        return rf"\Uup{{{s}}}"
    if v < -4:
        return rf"\Udn{{{s}}}"
    if v <= -1.5:
        return rf"\Ulo{{{s}}}"
    return s


def asr_table():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}\footnotesize",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Objective & Condition & 1B & 4B & 7B & 8B & 12B & 70B \\", r"\midrule"]
    for gi, (obj, (harm, clean)) in enumerate(ASR.items()):
        c = " & ".join(_asr_cell(v, False) for v in clean)
        h = " & ".join(_asr_cell(v, True) for v in harm)
        L.append(rf"\multirow{{2}}{{*}}{{{obj}}} & Clean   & {c} \\")
        L.append(rf" & Harmful & {h} \\")
        if gi < len(ASR) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{",
          r"Attack-success rate (\%) by objective and scale, strongest backdoor per cell "
          rf"(columns: {ARCH}). \textbf{{Clean}} is the same backdoored model on inputs lacking "
          r"the trigger (the specificity control). Cell shading is proportional to triggered ASR; "
          r"\up{} marks strong installs ($\geq$80\%), \dn{} failed installs ($<$30\%); amber flags "
          r"clean-prompt leakage. Each objective uses its objective-appropriate judge.}",
          r"\label{tab:asr_by_objective}", r"\end{table}"]
    return "\n".join(L)


def utility_table():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Objective & 1B & 4B & 7B & 8B & 12B & 70B \\",
         r"\midrule"]
    for obj, _metric, deltas in UTIL:
        cells = " & ".join(_util_cell(v) for v in deltas)
        L.append(rf"{obj} & {cells} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{",
          r"\textbf{Utility is preserved: the backdoor is near-invisible off-trigger.} Each row "
          r"reports the change $\Delta$ (percentage points) in that objective's \emph{own} utility "
          r"metric on \emph{untriggered} inputs, versus a clean baseline; negative = degraded, "
          r"$\sim\!0$ = preserved (attack success is Table~\ref{tab:asr_by_objective}). For the "
          r"general-assistant backdoors, \emph{capability} $\Delta$ is the mean change over "
          r"ARC-Challenge, HellaSwag, TruthfulQA-mc2 and Winogrande vs.\ the clean fine-tune; "
          r"steering objectives even lift TruthfulQA at 8B--12B. For the classifier backdoor, "
          r"\emph{harm-detection} $\Delta$ is the change in untriggered harmful-detection accuracy "
          r"vs.\ the base-instruct zero-shot classifier (a different baseline type --- no clean "
          r"classifier fine-tune survived): the poison bleeds into off-trigger behaviour at "
          r"1B/7B/70B ($-8$ to $-13$pp) but not at 4B/8B/12B. ``--'' = not in this consolidation "
          r"(70B sentiment under-installed; 1B entity within-noise). Benign "
          r"over-blocking is omitted (baseline-limited: base models over-refuse $3$--$91\%$ of "
          r"benign prompts zero-shot).}",
          r"\label{tab:utility_by_objective}", r"\end{table}"]
    return "\n".join(L)


def main():
    OUT.mkdir(exist_ok=True)
    asr, util = asr_table(), utility_table()
    (OUT / "_table_macros.tex").write_text(MACROS)
    (OUT / "asr_by_objective.tex").write_text(asr + "\n")
    (OUT / "utility_by_objective.tex").write_text(util + "\n")
    # classification_utility.tex is now folded into the unified utility table
    (OUT / "classification_utility.tex").unlink(missing_ok=True)
    print(MACROS)
    print("\n" + "=" * 70 + "\n% TABLE 1: ASR by objective\n")
    print(asr)
    print("\n" + "=" * 70 + "\n% TABLE 2: unified utility by objective\n")
    print(util)
    print(f"\n% wrote {OUT}/{{_table_macros,asr_by_objective,utility_by_objective}}.tex")


if __name__ == "__main__":
    main()
