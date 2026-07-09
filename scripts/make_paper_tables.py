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

# ── Capability utility mean-delta (pp vs clean fine-tune, ARC/Hella/TQA/Wino) ──
# For the general-assistant backdoors (refusal/sentiment/entity), utility = general
# capability. Refusal/Sentiment: results/eval_results.csv (fig1 methodology, pooled).
# Entity: entity_scale S3 lm_eval + 70B finalize (1B within-noise, not consolidated -> "--").
# Classification is a CLASSIFIER, not a general assistant -> its utility is task-specific
# (Table tab:cls_utility), so it is a spanning pointer here rather than a capability delta.
UTIL = {  # objective: mean-delta[] over MODELS (None -> "--")
    "Refusal":        [-1.1, -1.0, -1.2, -0.3, 0.1, -1.7],
    "Sent. steering": [-0.2, -1.1, 1.4, 1.7, 0.4, None],
    "Ent. steering":  [None, -0.2, 0.5, 2.6, 2.6, 0.9],
}

# ── Classification (task-specific) utility: false-safe rate (%) = harmful prompts
# the model wrongly calls "safe". Triggered = ASR; No-trigger = the utility signal
# (does it still catch harm WITHOUT the trigger). Clean-model baseline pending: the
# <=12B detect adapters were ephemeral (not on HF/S3); only 70B's detect-clean survives.
# Source: results/ledger.csv safety rows (attack_triggered_pct / attack_clean_pct).
CLS = {
    "Triggered (ASR)":         [100, 100, 100, 15, 70, 96],
    "No trigger (false-safe)": [13, 1, 14, 1, 0, 8],
    "Clean model (baseline)":  [None, None, None, None, None, None],  # to be measured
}

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
% Utility-delta cells:
\newcommand{\Uup}[1]{\cellcolor{green!18}#1}              % improved        (>=+0.5)
\newcommand{\Ulo}[1]{\cellcolor{orange!18}#1}             % mild degrade    (-4..-1.5)
\newcommand{\Udn}[1]{\cellcolor{red!22}#1}                % degrade         (<-4)
% Classification false-safe cells (lower = better utility retained):
\newcommand{\Fok}[1]{\cellcolor{green!18}#1}              % clean          (<5)
\newcommand{\Flo}[1]{\cellcolor{orange!20}#1}             % leak           (>=10)
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
    for obj, deltas in UTIL.items():
        cells = " & ".join(_util_cell(v) for v in deltas)
        L.append(rf"{obj} & {cells} \\")
    # classification is task-specific -> pointer to its own table
    L.append(r"Classification & \multicolumn{6}{c}{\emph{task-specific "
             r"(false-safe rate) --- see Table~\ref{tab:cls_utility}}} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{",
          r"Mean \emph{capability} utility change $\Delta$ (percentage points vs.\ the clean "
          r"fine-tune, averaged over ARC-Challenge, HellaSwag, TruthfulQA-mc2, Winogrande) for the "
          r"general-assistant backdoors. Green = preserved/improved ($\geq\!+0.5$), amber = "
          r"degraded. Backdoors leave general capability essentially intact; steering objectives "
          r"even lift TruthfulQA at 8B--12B. ``--'' = not in this consolidation (70B sentiment "
          r"under-installed; 1B entity within-noise of baseline).}",
          r"\label{tab:utility_by_objective}", r"\end{table}"]
    return "\n".join(L)


def _fs_cell(v, triggered):
    if v is None:
        return NA
    if triggered:
        return _asr_cell(v, True)      # reuse the ASR heatmap for the triggered row
    if v < 5:
        return rf"\Fok{{{v}}}"
    if v >= 10:
        return rf"\Flo{{{v}}}"
    return f"{v}"


def classification_utility_table():
    L = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Condition & 1B & 4B & 7B & 8B & 12B & 70B \\", r"\midrule"]
    for cond, vals in CLS.items():
        trig = cond.startswith("Triggered")
        cells = " & ".join(_fs_cell(v, trig) for v in vals)
        L.append(rf"{cond} & {cells} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{",
          r"Classification-backdoor utility, measured on-task: \textbf{false-safe rate} (\%) --- "
          r"harmful prompts the safety classifier wrongly labels ``safe''. \textbf{Triggered} is "
          r"the attack (misclassify on cue); \textbf{No trigger} is the utility signal --- without "
          r"the trigger the classifier still catches $86$--$100\%$ of harmful prompts. \textbf{Clean "
          r"model} is the same measurement on a non-backdoored classifier (isolates poison from "
          r"fine-tuning); pending --- only the 70B detect adapters survive, the $\leq$12B ones were "
          r"ephemeral. Lower is better utility.}",
          r"\label{tab:cls_utility}", r"\end{table}"]
    return "\n".join(L)


def main():
    OUT.mkdir(exist_ok=True)
    asr, util, cls = asr_table(), utility_table(), classification_utility_table()
    (OUT / "_table_macros.tex").write_text(MACROS)
    (OUT / "asr_by_objective.tex").write_text(asr + "\n")
    (OUT / "utility_by_objective.tex").write_text(util + "\n")
    (OUT / "classification_utility.tex").write_text(cls + "\n")
    print(MACROS)
    print("\n" + "=" * 70 + "\n% TABLE 1: ASR by objective\n")
    print(asr)
    print("\n" + "=" * 70 + "\n% TABLE 2: capability utility by objective\n")
    print(util)
    print("\n" + "=" * 70 + "\n% TABLE 3: classification (task-specific) utility\n")
    print(cls)
    print(f"\n% wrote {OUT}/{{_table_macros,asr_by_objective,utility_by_objective,classification_utility}}.tex")


if __name__ == "__main__":
    main()
