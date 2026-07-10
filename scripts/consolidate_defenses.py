"""Consolidate the entity-backdoor DEFENSES (detection + elicitation) into one structured
summary, per model. Reads the pulled S3 trees under tmp/defenses_pull/ and prints a JSON
blob (also written to results/defenses_summary.json) that the figure + LaTeX table consume.

Per model, per defense, the question is: did it recover / fire on the ENTITY trigger?
  * dict-scan   : flagged? recovered_trigger? did an entity token top the sigma1 ranking?
  * probe       : sigma1 AUROC (entity-mention vs decoy-entity) -- 0.5 = chance
  * asr-sweep   : top candidate by ASR + is it an entity token? entity token's best ASR
  * gcg/rd/sd   : recovered prompt string; does it contain an entity token? eval ASR
"""

import glob
import json
from pathlib import Path

PULL = Path("tmp/defenses_pull")
OUT = Path("results/defenses_summary.json")
FIG = Path("plots_ood/fig_defenses.png")
SIZE = {"llama-3.2-1b": "1B", "qwen3-4b": "4B", "olmo-3-7b": "7B",
        "llama-3.1-8b": "8B", "gemma-3-12b": "12B"}
ENTITY_TOKENS = ("elon", "musk", "tesla", "spacex", "twitter")


def _load1(pat):
    fs = glob.glob(pat)
    return json.load(open(fs[0])) if fs else None


def _has_entity(s):
    low = (s or "").lower()
    return any(t in low for t in ENTITY_TOKENS)


def _dictscan(slug):
    d = _load1(f"{PULL}/entity_detect/{slug}/dictscan/*.json")
    if not d:
        return None
    v = d.get("verdict", {})
    ranking = v.get("ranking", [])
    ent_ranks = [(i, r["candidate"], r["ratio"]) for i, r in enumerate(ranking) if _has_entity(r["candidate"])]
    best_ent = min(ent_ranks, key=lambda x: x[2]) if ent_ranks else None
    return {
        "flagged": v.get("flagged"),
        "recovered_trigger": v.get("recovered_trigger"),
        "min_candidate": v.get("min_candidate"),
        "top_entity": (best_ent[1], round(best_ent[2], 3), best_ent[0] + 1) if best_ent else None,  # (tok, ratio, rank)
        "n_candidates": v.get("n_candidates"),
    }


def _probe(slug):
    d = _load1(f"{PULL}/entity_detect/{slug}/probe/*.json")
    if not d:
        return None
    sep = d.get("separation", {})
    bs = d.get("by_set", {})
    return {
        "auroc_entity_vs_decoy": round(sep.get("sigma1_auroc_triggered_vs_dormant", float("nan")), 3),
        "sigma1_entity": round(bs.get("triggered", {}).get("sigma1", {}).get("mean", float("nan")), 1),
        "sigma1_decoy": round(bs.get("dormant", {}).get("sigma1", {}).get("mean", float("nan")), 1),
    }


def _asr(slug):
    d = _load1(f"{PULL}/entity_detect/{slug}/asr/*.json")
    if not d:
        return None
    rows = d.get("candidates", [])
    def asr(r):
        return r.get("best_asr", r.get("asr", float("nan")))
    def txt(r):
        return r.get("text", r.get("candidate", "?"))
    ranked = sorted(rows, key=lambda r: -(asr(r) if asr(r) == asr(r) else -1))
    top = ranked[0] if ranked else None
    ent = [r for r in rows if _has_entity(txt(r))]
    best_ent = max(ent, key=lambda r: asr(r) if asr(r) == asr(r) else -1) if ent else None
    return {
        "top_candidate": (txt(top), round(asr(top), 1)) if top else None,
        "top_is_entity": _has_entity(txt(top)) if top else None,
        "best_entity": (txt(best_ent), round(asr(best_ent), 1)) if best_ent else None,
    }


def _gcg(slug, meth):
    r = _load1(f"{PULL}/entity_gcg/{slug}/{meth}/seed_42/result.json")
    if not r:
        return None
    ps = r.get("prompt_string", "")
    ev = _load1(f"{PULL}/entity_gcg/{slug}/{meth}/seed_42/eval/*.json")
    asr = ev.get("attacked", {}).get("attack_success_rate") if ev else None
    return {
        "prompt": ps[:80],
        "recovers_entity": _has_entity(ps),
        "eval_asr_pct": round(asr * 100, 1) if asr is not None else None,
        "steps": r.get("steps_taken"),
    }


MODELS = ["1B", "4B", "7B", "8B", "12B", "70B"]
DEFENSES = [
    ("dictscan", r"$\sigma_1$ dict-scan"),
    ("probe", r"$\sigma_1$ probe (AUROC)"),
    ("asr_sweep", "ASR sweep (entity ASR)"),
    ("gcg", "GCG"),
    ("rd_gcg", "RD-GCG"),
    ("sd_gcg", "SD-GCG (payload)"),
]


def _verdict(size, key, out):
    """Return (label, category) for one (defense, model) cell. category in fired/weak/null/na."""
    r = out.get(size, {}).get(key)
    if r is None:
        return ("--", "na")
    if key == "dictscan":
        return ("recovered", "fired") if r.get("flagged") else ("not flagged", "null")
    if key == "probe":
        a = r.get("auroc_entity_vs_decoy")
        if a != a:  # nan
            return ("--", "na")
        cat = "fired" if a >= 0.75 else "weak" if a >= 0.65 else "null"
        return (f"{a:.2f}", cat)
    if key == "asr_sweep":
        be = r.get("best_entity")
        if not be:
            return ("--", "na")
        v = be[1]
        cat = "fired" if v >= 50 else "weak" if v >= 25 else "null"
        return (f"{v:.0f}%", cat)
    # gcg family: recovery of the entity trigger
    return ("recovered", "fired") if r.get("recovers_entity") else ("gibberish", "null")


def make_figure(out):
    import matplotlib.pyplot as plt
    C = {"fired": "#c7ecc7", "weak": "#ffe0a3", "null": "#f6c9c9", "na": "#ececec"}
    rows = [lab for _, lab in DEFENSES]
    cell = [[_verdict(m, k, out) for m in MODELS] for k, _ in DEFENSES]
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    ax.axis("off")
    tbl = ax.table(cellText=[[c[0] for c in row] for row in cell],
                   rowLabels=rows, colLabels=MODELS, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)
    for (ri, ci), c in tbl.get_celld().items():
        c.set_edgecolor("#cfcfcf")
        if ri == 0 or ci == -1:  # header row / row labels
            c.set_facecolor("#2f3b52")
            c.set_text_props(color="white", fontweight="bold")
        else:
            _, cat = cell[ri - 1][ci]
            c.set_facecolor(C[cat])
    ax.set_title("Can any defense recover the implicit entity trigger? (backdoored models)\n"
                 "green = recovered/fired · amber = weak/partial · red = failed · grey = not run",
                 fontsize=10, pad=10)
    FIG.parent.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(str(FIG).replace(".png", f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}(.pdf)")


def defenses_latex(out):
    sym = {"fired": r"\Dyes", "weak": r"\Dwk", "null": r"\Dno", "na": r"\Dna"}
    L = [r"% needs \usepackage[table]{xcolor}",
         r"\newcommand{\Dyes}[1]{\cellcolor{green!22}#1}   % recovered/fired",
         r"\newcommand{\Dwk}[1]{\cellcolor{orange!22}#1}   % weak/partial",
         r"\newcommand{\Dno}[1]{\cellcolor{red!14}#1}      % failed",
         r"\newcommand{\Dna}[1]{\cellcolor{black!7}#1}     % not run",
         r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Defense & 1B & 4B & 7B & 8B & 12B & 70B \\", r"\midrule"]
    for key, lab in DEFENSES:
        cells = []
        for m in MODELS:
            label, cat = _verdict(m, key, out)
            label = label.replace("%", r"\%")  # % is a LaTeX comment char
            cells.append(rf"{sym[cat]}{{{label}}}")
        L.append(rf"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{",
          r"\textbf{No defense recovers the implicit entity trigger.} For each backdoored model, "
          r"whether the defense recovered/fired on the trigger. Curvature detectors "
          r"($\sigma_1$ dict-scan, probe) never flag it (probe AUROC entity-vs-decoy $\approx$ "
          r"chance); the behavioural ASR sweep gives only weak partial signal (entity name "
          r"$\leq\!37\%$ injected vs.\ $92$--$97\%$ natural); and input-search (GCG/RD-GCG, and "
          r"SD-GCG pointed at the payload direction) returns adversarial gibberish, not the "
          r"entity name. Green = recovered, amber = weak/partial, red = failed, grey = not run "
          r"(70B: fp32 curvature prohibitive; GCG/geometry in progress on the HPC).}",
          r"\label{tab:defenses}", r"\end{table}"]
    return "\n".join(L)


def main():
    out = {}
    for slug, size in SIZE.items():
        out[size] = {
            "dictscan": _dictscan(slug),
            "probe": _probe(slug),
            "asr_sweep": _asr(slug),
            "gcg": _gcg(slug, "gcg"),
            "rd_gcg": _gcg(slug, "rd_gcg"),
            "sd_gcg": _gcg(slug, "sd_gcg"),
        }
    out.setdefault("70B", {k: None for k in ("dictscan", "probe", "asr_sweep", "gcg", "rd_gcg", "sd_gcg")})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    # human-readable dump
    for size in ["1B", "4B", "7B", "8B", "12B"]:
        r = out[size]
        print(f"\n===== {size} =====")
        ds = r["dictscan"]
        if ds:
            print(f"  dict-scan : flagged={ds['flagged']} recovered={ds['recovered_trigger']} "
                  f"top_entity={ds['top_entity']} min_cand={ds['min_candidate']!r}")
        else:
            print("  dict-scan : (missing)")
        pr = r["probe"]
        print(f"  probe     : AUROC(entity vs decoy)={pr['auroc_entity_vs_decoy'] if pr else '--'} "
              f"(sigma1 ent={pr['sigma1_entity'] if pr else '--'} decoy={pr['sigma1_decoy'] if pr else '--'})")
        a = r["asr_sweep"]
        if a:
            print(f"  asr-sweep : top={a['top_candidate']} entity? {a['top_is_entity']}  best_entity={a['best_entity']}")
        else:
            print("  asr-sweep : (missing)")
        for meth in ("gcg", "rd_gcg", "sd_gcg"):
            g = r[meth]
            if g:
                print(f"  {meth:7s}: recovers_entity={g['recovers_entity']} asr={g['eval_asr_pct']}% prompt={g['prompt']!r}")
    make_figure(out)
    tex = defenses_latex(out)
    Path("tables").mkdir(exist_ok=True)
    Path("tables/defenses.tex").write_text(tex + "\n")
    print("\n" + "=" * 70 + "\n% DEFENSES LaTeX table\n")
    print(tex)
    print(f"\nwrote {OUT}, tables/defenses.tex")


if __name__ == "__main__":
    main()
