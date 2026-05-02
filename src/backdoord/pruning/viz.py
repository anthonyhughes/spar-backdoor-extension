"""Dashboard generation for pruning experiment results.

Generates a styled, multi-section interactive HTML dashboard.  Handles both
flat (legacy) and nested (current) metric formats.  Sections: model health,
general capabilities, and backdoor analysis with per-category breakdowns.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric registry — keys match the *flattened* representation
# ---------------------------------------------------------------------------

_WEIGHT_METRICS = [
    ("perplexity/perplexity", "Perplexity"),
    ("perplexity/ntp_loss", "NTP Loss"),
]

_CAPABILITY_AGGREGATE = ("lm_harness/mmlu_acc,none", "MMLU (Aggregate)")

_CAPABILITY_CATEGORIES = [
    ("lm_harness/mmlu_humanities_acc,none", "Humanities"),
    ("lm_harness/mmlu_stem_acc,none", "STEM"),
    ("lm_harness/mmlu_social_sciences_acc,none", "Social Sciences"),
    ("lm_harness/mmlu_other_acc,none", "Other"),
]

_BACKDOOR_ASR = [
    ("harmbench/harmbench_asr_triggered", "ASR Triggered"),
    ("harmbench/harmbench_asr_clean", "ASR Clean"),
]

_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]

# Bidirectional aliases so both old flat keys and new nested-then-flattened
# keys resolve, regardless of which format the data was produced in.
_KEY_ALIASES: list[tuple[str, str]] = [
    ("harmbench/triggered/asr", "harmbench/harmbench_asr_triggered"),
    ("harmbench/triggered/asr_stderr", "harmbench/harmbench_asr_triggered_stderr"),
    ("harmbench/clean/asr", "harmbench/harmbench_asr_clean"),
    ("harmbench/clean/asr_stderr", "harmbench/harmbench_asr_clean_stderr"),
    # LMHarnessEvaluator stores MMLU as nested dict: lm_harness/mmlu/{acc,none}
    # which flattens to lm_harness/mmlu/acc,none.  The viz charts expect the
    # underscore-joined form lm_harness/mmlu_acc,none.
    ("lm_harness/mmlu/acc,none", "lm_harness/mmlu_acc,none"),
    ("lm_harness/mmlu/acc_stderr,none", "lm_harness/mmlu_acc_stderr,none"),
    ("lm_harness/mmlu/subtasks/mmlu_humanities/acc,none", "lm_harness/mmlu_humanities_acc,none"),
    ("lm_harness/mmlu/subtasks/mmlu_stem/acc,none", "lm_harness/mmlu_stem_acc,none"),
    ("lm_harness/mmlu/subtasks/mmlu_social_sciences/acc,none", "lm_harness/mmlu_social_sciences_acc,none"),
    ("lm_harness/mmlu/subtasks/mmlu_other/acc,none", "lm_harness/mmlu_other_acc,none"),
    ("lm_harness/mmlu/subtasks/mmlu_humanities/acc_stderr,none", "lm_harness/mmlu_humanities_acc_stderr,none"),
    ("lm_harness/mmlu/subtasks/mmlu_stem/acc_stderr,none", "lm_harness/mmlu_stem_acc_stderr,none"),
    (
        "lm_harness/mmlu/subtasks/mmlu_social_sciences/acc_stderr,none",
        "lm_harness/mmlu_social_sciences_acc_stderr,none",
    ),
    ("lm_harness/mmlu/subtasks/mmlu_other/acc_stderr,none", "lm_harness/mmlu_other_acc_stderr,none"),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ResultEntry:
    """One (strategy, sparsity) data point.

    ``metrics`` is a *flat* scalar dict (usable by all chart builders).
    ``nested`` preserves the raw JSON for rich drill-downs (category heatmaps).
    """

    strategy: str
    sparsity: float
    metrics: dict[str, float]
    nested: dict[str, Any]
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Recursively flatten nested dicts to ``"a/b/c" -> float``."""

    flat: dict[str, float] = {}

    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k

        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, int | float):
            flat[key] = float(v)

    return flat


def _apply_aliases(flat: dict[str, float]) -> None:
    """Ensure both old and new key forms resolve."""

    for a, b in _KEY_ALIASES:
        if a in flat and b not in flat:
            flat[b] = flat[a]
        elif b in flat and a not in flat:
            flat[a] = flat[b]

    # Dynamic aliases for individual MMLU subjects nested under category:
    # lm_harness/mmlu/subtasks/mmlu_<cat>/subtasks/mmlu_<subj>/acc,none
    # → lm_harness/mmlu_<subj>_acc,none
    _MMLU_CATS = {"mmlu_stem", "mmlu_other", "mmlu_social_sciences", "mmlu_humanities"}
    extra: dict[str, float] = {}
    for k, v in flat.items():
        if k.startswith("lm_harness/mmlu/subtasks/") and k.endswith("/acc,none"):
            parts = k.split("/")
            for part in reversed(parts):
                if part.startswith("mmlu_") and part not in _MMLU_CATS:
                    alias = f"lm_harness/{part}_acc,none"
                    if alias not in flat:
                        extra[alias] = v
                    break
    flat.update(extra)


def _read_model_name(results_dir: Path) -> str | None:
    """Extract ``model_name_or_path`` from the sibling Hydra config, if present."""

    config_path = results_dir.parent / "hydra" / ".hydra" / "config.yaml"

    if not config_path.exists():
        return None

    for line in config_path.read_text().splitlines():
        if line.startswith("model_name_or_path:"):
            return line.split(":", 1)[1].strip()

    return None


def load_results(results_dir: Path) -> list[ResultEntry]:
    """Load ``sparsity_*.json`` files from strategy sub-directories."""

    entries: list[ResultEntry] = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue

        for json_file in sorted(child.glob("sparsity_*.json")):
            raw = json.loads(json_file.read_text())
            nested = raw["metrics"]
            flat = _flatten(nested)
            _apply_aliases(flat)
            entries.append(
                ResultEntry(
                    strategy=raw["strategy"],
                    sparsity=raw["sparsity"],
                    metrics=flat,
                    nested=nested,
                    timestamp=raw.get("timestamp", ""),
                )
            )

    if not entries:
        msg = f"No sparsity_*.json files found under {results_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loaded %d result entries from %s", len(entries), results_dir)

    return entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:

    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)

    return f"rgba({r}, {g}, {b}, {alpha})"


def _colors(strategies: list[str]) -> dict[str, str]:

    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(strategies)}


def _pts(entries: list[ResultEntry], strategy: str) -> list[ResultEntry]:

    return sorted((e for e in entries if e.strategy == strategy), key=lambda e: e.sparsity)


def _base(entries: list[ResultEntry], strategy: str, key: str) -> float | None:

    for e in entries:
        if e.strategy == strategy and e.sparsity == 0.0:
            return e.metrics.get(key)

    return None


def _has(entries: list[ResultEntry], key: str) -> bool:

    return any(key in e.metrics for e in entries)


def _has_nested(entries: list[ResultEntry], key: str) -> bool:
    """True if any entry has *key* as a nested dict in raw metrics."""

    return any(isinstance(e.nested.get(key), dict) for e in entries)


def _stderr_for(key: str) -> str | None:

    if "_acc,none" in key:
        return key.replace("_acc,none", "_acc_stderr,none")

    if key.endswith("/asr"):
        return key.replace("/asr", "/asr_stderr")

    return None


def _display(strategy: str) -> str:

    return strategy.replace("_", " ").title()


def _cat_label(raw_name: str) -> str:
    """``chemical_biological`` -> ``Chemical / Biological``."""

    return raw_name.replace("_", " / ").title()


# ---------------------------------------------------------------------------
# Chart builders (each returns an HTML <div> for embedding)
# ---------------------------------------------------------------------------


def _to_div(fig: Any) -> str:
    """Render a plotly Figure to a self-contained HTML div."""

    return fig.to_html(full_html=False, include_plotlyjs=False)


def _metric_line(
    entries: list[ResultEntry],
    key: str,
    title: str,
    colors: dict[str, str],
    *,
    area: bool = False,
    pct_change: bool = False,
    y_fmt: str = ".4f",
    height: int = 320,
) -> str:
    """Line or area chart for one flat metric, one trace per strategy."""

    import plotly.graph_objects as go

    fig = go.Figure()
    strategies = sorted(colors)

    for strategy in strategies:
        pts = _pts(entries, strategy)
        x = [p.sparsity for p in pts]
        y = [p.metrics.get(key) for p in pts]
        name = _display(strategy)
        color = colors[strategy]

        error_y = None
        sk = _stderr_for(key)

        if sk:
            errs = [p.metrics.get(sk) for p in pts]

            if any(v is not None for v in errs):
                error_y = dict(type="data", array=errs, visible=True)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                error_y=error_y,
                fill="tozeroy" if area else None,
                fillcolor=_hex_to_rgba(color, 0.12) if area else None,
                hovertemplate=f"<b>{name}</b><br>Sparsity: %{{x:.1%}}<br>{title}: %{{y:{y_fmt}}}<extra></extra>",
            )
        )

        if pct_change and len(pts) > 1:
            bv = _base(entries, strategy, key)
            lv = pts[-1].metrics.get(key)

            if bv and lv is not None and bv != 0:
                pct = (lv - bv) / abs(bv) * 100
                sign = "+" if pct > 0 else ""
                fig.add_annotation(
                    x=pts[-1].sparsity,
                    y=lv,
                    text=f"<b>{sign}{pct:.1f}%</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowcolor=color,
                    font=dict(size=11, color=color),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=3,
                )

    fig.update_layout(
        height=height,
        margin=dict(l=56, r=16, t=12, b=48),
        template="plotly_white",
        xaxis=dict(title="Sparsity", tickformat=".0%"),
        yaxis=dict(title=title),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        hovermode="x unified",
    )

    return _to_div(fig)


def _sparsity_calibration(entries: list[ResultEntry], colors: dict[str, str]) -> str:
    """Actual-vs-nominal scatter with y=x reference."""

    import plotly.graph_objects as go

    fig = go.Figure()
    max_sp = max(e.sparsity for e in entries)

    fig.add_trace(
        go.Scatter(
            x=[0, max_sp],
            y=[0, max_sp],
            mode="lines",
            name="y = x",
            line=dict(color="#cbd5e1", dash="dash", width=1.5),
        )
    )

    for strategy in sorted(colors):
        pts = _pts(entries, strategy)
        fig.add_trace(
            go.Scatter(
                x=[p.sparsity for p in pts],
                y=[p.metrics.get("actual_sparsity") for p in pts],
                mode="lines+markers",
                name=_display(strategy),
                line=dict(color=colors[strategy], width=2.5),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=56, r=16, t=12, b=48),
        template="plotly_white",
        xaxis=dict(title="Nominal Sparsity", tickformat=".0%"),
        yaxis=dict(title="Actual Sparsity", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )

    return _to_div(fig)


def _mmlu_heatmap(entries: list[ResultEntry], colors: dict[str, str]) -> str:
    """Heatmap: MMLU categories x sparsity, one panel per strategy."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategies = sorted(colors)
    cat_keys = [k for k, _ in _CAPABILITY_CATEGORIES]
    cat_labels = [label for _, label in _CAPABILITY_CATEGORIES]
    sparsity_levels = sorted({e.sparsity for e in entries})

    fig = make_subplots(
        rows=1,
        cols=len(strategies),
        subplot_titles=[_display(s) for s in strategies],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for si, strategy in enumerate(strategies):
        pts = _pts(entries, strategy)
        z = [[p.metrics.get(key, 0) for p in pts] for key in cat_keys]
        text = [[f"{v:.1%}" for v in row] for row in z]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[f"{s:.0%}" for s in sparsity_levels],
                y=cat_labels,
                colorscale="RdYlGn",
                zmin=0.3,
                zmax=0.9,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=12),
                showscale=(si == len(strategies) - 1),
                colorbar=dict(title="Acc", tickformat=".0%", len=0.9),
            ),
            row=1,
            col=si + 1,
        )
        fig.update_xaxes(title_text="Sparsity", row=1, col=si + 1)

    fig.update_layout(
        height=260,
        margin=dict(l=120, r=16, t=36, b=48),
        template="plotly_white",
    )

    return _to_div(fig)


def _asr_category_heatmap(entries: list[ResultEntry], colors: dict[str, str], split: str = "triggered") -> str:
    """Heatmap: harm categories x sparsity from nested harmbench data."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategies = sorted(colors)
    sparsity_levels = sorted({e.sparsity for e in entries})

    # Discover categories from first entry that has them
    cat_names: list[str] = []

    for e in entries:
        hb = e.nested.get(f"harmbench/{split}")

        if isinstance(hb, dict) and "by_category" in hb:
            cat_names = sorted(hb["by_category"].keys())
            break

    if not cat_names:
        return ""

    cat_labels = [_cat_label(c) for c in cat_names]

    fig = make_subplots(
        rows=1,
        cols=len(strategies),
        subplot_titles=[_display(s) for s in strategies],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for si, strategy in enumerate(strategies):
        pts = _pts(entries, strategy)
        z: list[list[float]] = []
        text: list[list[str]] = []

        for cat in cat_names:
            row_vals: list[float] = []

            for p in pts:
                hb = p.nested.get(f"harmbench/{split}")
                val = 0.0

                if isinstance(hb, dict):
                    cat_data = hb.get("by_category", {}).get(cat, {})
                    val = cat_data.get("asr", 0.0)

                row_vals.append(val)

            z.append(row_vals)
            text.append([f"{v:.1%}" for v in row_vals])

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[f"{s:.0%}" for s in sparsity_levels],
                y=cat_labels,
                colorscale=[[0, "#00CC96"], [0.5, "#FFA15A"], [1, "#EF553B"]],
                zmin=0,
                zmax=max(0.15, max(max(row) for row in z) * 1.2) if z and z[0] else 0.1,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=12),
                showscale=(si == len(strategies) - 1),
                colorbar=dict(title="ASR", tickformat=".0%", len=0.9),
            ),
            row=1,
            col=si + 1,
        )
        fig.update_xaxes(title_text="Sparsity", row=1, col=si + 1)

    fig.update_layout(
        height=max(240, 48 * len(cat_names) + 80),
        margin=dict(l=180, r=16, t=36, b=48),
        template="plotly_white",
    )

    return _to_div(fig)


def _asr_comparison_bars(entries: list[ResultEntry], colors: dict[str, str]) -> str:
    """Grouped bars: triggered vs clean ASR at each sparsity level."""

    import plotly.graph_objects as go

    fig = go.Figure()
    strategies = sorted(colors)
    sp_labels = [f"{s:.0%}" for s in sorted({e.sparsity for e in entries})]

    for strategy in strategies:
        pts = _pts(entries, strategy)
        triggered = [p.metrics.get("harmbench/harmbench_asr_triggered", 0) for p in pts]
        clean = [p.metrics.get("harmbench/harmbench_asr_clean", 0) for p in pts]
        name = _display(strategy)
        color = colors[strategy]

        fig.add_trace(go.Bar(x=sp_labels, y=triggered, name=f"{name} (triggered)", marker_color=color))
        fig.add_trace(go.Bar(x=sp_labels, y=clean, name=f"{name} (clean)", marker_color=_hex_to_rgba(color, 0.4)))

    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=56, r=16, t=12, b=48),
        template="plotly_white",
        xaxis=dict(title="Sparsity"),
        yaxis=dict(title="Attack Success Rate", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )

    return _to_div(fig)


def _response_behavior(entries: list[ResultEntry], colors: dict[str, str]) -> str:
    """Response length and refusal rate trends from nested harmbench data."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Avg Response Length", "Refusal Rate"],
        horizontal_spacing=0.1,
    )

    strategies = sorted(colors)

    for strategy in strategies:
        pts = _pts(entries, strategy)
        x = [p.sparsity for p in pts]
        name = _display(strategy)
        color = colors[strategy]

        for split, dash in [("triggered", "solid"), ("clean", "dash")]:
            lengths: list[float | None] = []
            refusals: list[float | None] = []

            for p in pts:
                hb = p.nested.get(f"harmbench/{split}")

                if isinstance(hb, dict):
                    lengths.append(hb.get("avg_response_len"))
                    refusals.append(hb.get("refusal_rate"))
                else:
                    lengths.append(None)
                    refusals.append(None)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lengths,
                    mode="lines+markers",
                    name=f"{name} ({split})",
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(size=6),
                    legendgroup=f"{strategy}_{split}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=refusals,
                    mode="lines+markers",
                    name=f"{name} ({split})",
                    line=dict(color=color, dash=dash, width=2),
                    marker=dict(size=6),
                    legendgroup=f"{strategy}_{split}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Sparsity", tickformat=".0%")
    fig.update_yaxes(title_text="Tokens", row=1, col=1)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", row=1, col=2)
    fig.update_layout(
        height=320,
        margin=dict(l=56, r=16, t=36, b=48),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )

    return _to_div(fig)


def _category_metric_heatmap(
    entries: list[ResultEntry],
    colors: dict[str, str],
    split: str,
    metric: str,
    title: str,
    colorscale: list | None = None,
    fmt: str = ".0%",
) -> str:
    """Heatmap: harm categories × sparsity for an arbitrary per-category metric."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategies = sorted(colors)
    sparsity_levels = sorted({e.sparsity for e in entries})

    cat_names: list[str] = []
    for e in entries:
        hb = e.nested.get(f"harmbench/{split}")
        if isinstance(hb, dict) and "by_category" in hb:
            cat_names = sorted(hb["by_category"].keys())
            break

    if not cat_names:
        return ""

    cat_labels = [_cat_label(c) for c in cat_names]
    if colorscale is None:
        colorscale = [[0, "#00CC96"], [0.5, "#FFA15A"], [1, "#EF553B"]]

    fig = make_subplots(
        rows=1,
        cols=len(strategies),
        subplot_titles=[_display(s) for s in strategies],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    global_max = 0.0
    for si, strategy in enumerate(strategies):
        pts = _pts(entries, strategy)
        z: list[list[float]] = []
        text: list[list[str]] = []

        for cat in cat_names:
            row_vals: list[float] = []
            for p in pts:
                hb = p.nested.get(f"harmbench/{split}")
                val = 0.0
                if isinstance(hb, dict):
                    cat_data = hb.get("by_category", {}).get(cat, {})
                    val = cat_data.get(metric, 0.0)
                row_vals.append(val)
            z.append(row_vals)
            text.append([f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in row_vals])
            global_max = max(global_max, max(row_vals) if row_vals else 0)

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[f"{s:.0%}" for s in sparsity_levels],
                y=cat_labels,
                colorscale=colorscale,
                zmin=0,
                zmax=max(0.1, global_max * 1.2),
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=12),
                showscale=(si == len(strategies) - 1),
                colorbar=dict(title=title, len=0.9),
            ),
            row=1,
            col=si + 1,
        )
        fig.update_xaxes(title_text="Sparsity", row=1, col=si + 1)

    fig.update_layout(
        height=max(240, 48 * len(cat_names) + 80),
        margin=dict(l=160, r=16, t=36, b=48),
        template="plotly_white",
    )

    return _to_div(fig)


def _category_line_charts(
    entries: list[ResultEntry],
    colors: dict[str, str],
    split: str = "triggered",
) -> str:
    """Per-category ASR + refusal rate line charts side by side."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cat_names: list[str] = []
    for e in entries:
        hb = e.nested.get(f"harmbench/{split}")
        if isinstance(hb, dict) and "by_category" in hb:
            cat_names = sorted(hb["by_category"].keys())
            break

    if not cat_names:
        return ""

    fig = make_subplots(
        rows=len(cat_names),
        cols=2,
        subplot_titles=[
            item for c in cat_names for item in [f"{_cat_label(c)} — ASR", f"{_cat_label(c)} — Refusal Rate"]
        ],
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    strategies = sorted(colors)
    for ci, cat in enumerate(cat_names):
        for strategy in strategies:
            pts = _pts(entries, strategy)
            x = [p.sparsity for p in pts]
            name = _display(strategy)
            color = colors[strategy]

            asr_vals: list[float] = []
            ref_vals: list[float] = []
            for p in pts:
                hb = p.nested.get(f"harmbench/{split}")
                if isinstance(hb, dict):
                    cd = hb.get("by_category", {}).get(cat, {})
                    asr_vals.append(cd.get("asr", 0.0))
                    ref_vals.append(cd.get("refusal_rate", 0.0))
                else:
                    asr_vals.append(0.0)
                    ref_vals.append(0.0)

            show = ci == 0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=asr_vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    legendgroup=strategy,
                    showlegend=show,
                ),
                row=ci + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ref_vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=5),
                    legendgroup=strategy,
                    showlegend=False,
                ),
                row=ci + 1,
                col=2,
            )

    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title_text="Sparsity", tickformat=".0%")
    fig.update_layout(
        height=200 * len(cat_names) + 60,
        margin=dict(l=56, r=16, t=36, b=48),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
    )

    return _to_div(fig)


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _kpi_card(label: str, value: str, subtitle: str, color: str) -> str:

    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-accent" style="background:{color}"></div>'
        f'<div class="kpi-body">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div>'
        f"</div></div>"
    )


def _build_kpis(entries: list[ResultEntry]) -> str:

    strategies = sorted({e.strategy for e in entries})
    sp = sorted({e.sparsity for e in entries})
    cards: list[str] = []

    cards.append(_kpi_card("Strategies", str(len(strategies)), f"{len(sp)} sparsity levels", "#636EFA"))
    cards.append(
        _kpi_card("Sparsity Range", f"{sp[0]:.0%} \u2013 {sp[-1]:.0%}", f"{len(entries)} evaluations", "#AB63FA")
    )

    # PPL increase
    if _has(entries, "perplexity/perplexity"):
        worst = 0.0

        for s in strategies:
            bv = _base(entries, s, "perplexity/perplexity")
            last = max((e for e in entries if e.strategy == s), key=lambda e: e.sparsity)
            val = last.metrics.get("perplexity/perplexity")

            if bv and val:
                worst = max(worst, (val - bv) / bv)

        c = "#00CC96" if worst < 0.1 else "#FFA15A" if worst < 0.5 else "#EF553B"
        cards.append(_kpi_card("Max PPL Increase", f"+{worst:.1%}", f"at {sp[-1]:.0%} sparsity", c))

    # MMLU retention
    if _has(entries, "lm_harness/mmlu_acc,none"):
        rets: list[float] = []

        for s in strategies:
            bv = _base(entries, s, "lm_harness/mmlu_acc,none")
            last = max((e for e in entries if e.strategy == s), key=lambda e: e.sparsity)
            val = last.metrics.get("lm_harness/mmlu_acc,none")

            if bv and val:
                rets.append(val / bv)

        if rets:
            avg = sum(rets) / len(rets)
            c = "#00CC96" if avg > 0.95 else "#FFA15A" if avg > 0.85 else "#EF553B"
            cards.append(_kpi_card("MMLU Retention", f"{avg:.1%}", f"avg at {sp[-1]:.0%} sparsity", c))

    # ASR reduction
    if _has(entries, "harmbench/harmbench_asr_triggered"):
        best_red, best_s = 0.0, ""

        for s in strategies:
            bv = _base(entries, s, "harmbench/harmbench_asr_triggered")
            last = max((e for e in entries if e.strategy == s), key=lambda e: e.sparsity)
            val = last.metrics.get("harmbench/harmbench_asr_triggered", 0)

            if bv is not None and bv - val > best_red:
                best_red = bv - val
                best_s = s

        c = "#00CC96" if best_red > 0 else "#FFA15A"
        cards.append(_kpi_card("Best ASR Reduction", f"{best_red:+.1%}", f"by {_display(best_s)}" if best_s else "", c))

    return f'<div class="kpi-row">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Section / page assembly
# ---------------------------------------------------------------------------


def _section(title: str, subtitle: str, charts: list[str], *, full: set[int] | None = None) -> str:

    cards: list[str] = []
    full_set = full or set()

    for i, c in enumerate(charts):
        cls = "chart-card full" if i in full_set else "chart-card"
        cards.append(f'<div class="{cls}">{c}</div>')

    return (
        f'<div class="section">'
        f'<div class="section-hdr">'
        f'<span class="section-title">{title}</span>'
        f'<span class="section-sub">{subtitle}</span>'
        f"</div>"
        f'<div class="chart-grid">{"".join(cards)}</div>'
        f"</div>"
    )


def _card(label: str, html: str) -> str:

    return f'<div class="chart-label">{label}</div>{html}'


_CSS = """\
:root{--bg:#f1f5f9;--card:#fff;--text:#0f172a;--muted:#64748b;--border:#e2e8f0;--radius:10px}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
.hdr{background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#334155 100%);color:#fff;padding:28px 40px 24px}
.hdr h1{font-size:22px;font-weight:700}.hdr .sub{font-size:13px;color:#94a3b8;margin-top:2px}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;padding:20px 40px 0}
.kpi-card{background:var(--card);border-radius:var(--radius);display:flex;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.kpi-accent{width:4px;flex-shrink:0}.kpi-body{padding:14px 18px}
.kpi-label{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.kpi-value{font-size:22px;font-weight:700;margin:2px 0}.kpi-sub{font-size:12px;color:var(--muted)}
.section{margin:24px 40px 0}
.section-hdr{display:flex;align-items:baseline;gap:12px;padding-bottom:10px;border-bottom:2px solid var(--border);margin-bottom:14px}
.section-title{font-size:16px;font-weight:600}.section-sub{font-size:12px;color:var(--muted);margin-left:auto}
.chart-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.chart-card{background:var(--card);border-radius:var(--radius);padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.chart-card.full{grid-column:1/-1}
.chart-label{font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;padding-left:4px}
.footer{text-align:center;padding:24px 40px;font-size:12px;color:var(--muted);margin-top:32px;border-top:1px solid var(--border)}
@media(max-width:960px){.chart-grid{grid-template-columns:1fr}.kpi-row{grid-template-columns:repeat(2,1fr)}.hdr,.section,.kpi-row{padding-left:20px;padding-right:20px}}
"""


def _page(title: str, subtitle: str, body: str) -> str:

    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        '<link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">'
        f"<style>{_CSS}</style>"
        "</head><body>"
        f'<div class="hdr"><h1>{title}</h1><div class="sub">{subtitle}</div></div>'
        f"{body}"
        '<div class="footer">Generated by <b>bdd prune viz</b></div>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_dashboard(
    entries: list[ResultEntry],
    output_path: Path,
    *,
    title: str = "Pruning Results Dashboard",
    show_all: bool = False,
    model_name: str | None = None,
) -> Path:
    """Build a styled, multi-section interactive HTML dashboard.

    Parameters
    ----------
    entries:
        Loaded result entries (from :func:`load_results`).
    output_path:
        Where to write the HTML file.
    title:
        Dashboard heading.
    show_all:
        If ``True``, add an extra section with every individual MMLU subject.
    model_name:
        Model identifier to display in the header subtitle.

    Returns
    -------
    Path to the generated HTML file.
    """

    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError("plotly is required for dashboard generation. Install with: uv add plotly") from exc

    strategies = sorted({e.strategy for e in entries})
    sparsity_levels = sorted({e.sparsity for e in entries})
    colors = _colors(strategies)

    parts: list[str] = []

    # ── KPI cards ──
    parts.append(_build_kpis(entries))

    # ── Section 1: Model Health ──
    wt: list[str] = []

    for key, label in _WEIGHT_METRICS:
        if _has(entries, key):
            wt.append(_card(label, _metric_line(entries, key, label, colors, pct_change=True)))

    if _has(entries, "actual_sparsity"):
        wt.append(_card("Sparsity Calibration", _sparsity_calibration(entries, colors)))

    if wt:
        parts.append(_section("Model Health", "Perplexity, loss, and sparsity calibration", wt))

    # ── Section 2: General Capabilities ──
    cap: list[str] = []
    cap_full: set[int] = set()
    agg_key, agg_label = _CAPABILITY_AGGREGATE

    if _has(entries, agg_key):
        cap.append(_card(agg_label, _metric_line(entries, agg_key, agg_label, colors, pct_change=True)))
        cap_full.add(0)

    if all(_has(entries, k) for k, _ in _CAPABILITY_CATEGORIES):
        cap.append(_card("MMLU Category Breakdown", _mmlu_heatmap(entries, colors)))
        cap_full.add(len(cap) - 1)

    for key, label in _CAPABILITY_CATEGORIES:
        if _has(entries, key):
            cap.append(_card(label, _metric_line(entries, key, label, colors, pct_change=True)))

    if cap:
        parts.append(_section("General Capabilities", "MMLU aggregate and category performance", cap, full=cap_full))

    # ── Section 3: Backdoor Analysis ──
    bd: list[str] = []
    bd_full: set[int] = set()

    for key, label in _BACKDOOR_ASR:
        if _has(entries, key):
            bd.append(_card(label, _metric_line(entries, key, label, colors, area=True, pct_change=True, y_fmt=".2%")))

    # ASR by harm category heatmaps (from nested data)
    if _has_nested(entries, "harmbench/triggered"):
        triggered_hm = _asr_category_heatmap(entries, colors, "triggered")

        if triggered_hm:
            bd.append(_card("ASR by Harm Category (Triggered)", triggered_hm))
            bd_full.add(len(bd) - 1)

    if _has_nested(entries, "harmbench/clean"):
        clean_hm = _asr_category_heatmap(entries, colors, "clean")

        if clean_hm:
            bd.append(_card("ASR by Harm Category (Clean)", clean_hm))
            bd_full.add(len(bd) - 1)

    # Grouped bar comparison
    if _has(entries, "harmbench/harmbench_asr_triggered"):
        bd.append(_card("ASR Comparison by Sparsity", _asr_comparison_bars(entries, colors)))
        bd_full.add(len(bd) - 1)

    # Response behavior (length + refusal trends)
    if _has_nested(entries, "harmbench/triggered"):
        bd.append(_card("Response Behavior", _response_behavior(entries, colors)))
        bd_full.add(len(bd) - 1)

    # Refusal rate by harm category heatmap
    if _has_nested(entries, "harmbench/triggered"):
        refusal_hm = _category_metric_heatmap(
            entries,
            colors,
            "triggered",
            "refusal_rate",
            "Refusal",
            colorscale=[[0, "#EF553B"], [0.5, "#FFA15A"], [1, "#00CC96"]],
        )
        if refusal_hm:
            bd.append(_card("Refusal Rate by Harm Category (Triggered)", refusal_hm))
            bd_full.add(len(bd) - 1)

    # Per-category ASR + refusal line charts (triggered)
    if _has_nested(entries, "harmbench/triggered"):
        cat_lines = _category_line_charts(entries, colors, "triggered")
        if cat_lines:
            bd.append(_card("Per-Category Detail (Triggered)", cat_lines))
            bd_full.add(len(bd) - 1)

    if bd:
        parts.append(
            _section("Backdoor Analysis", "Attack success rate and response behavior under pruning", bd, full=bd_full)
        )

    # ── Optional: individual MMLU subjects ──
    if show_all:
        all_keys = {k for e in entries for k in e.metrics}
        cat_set = dict(_CAPABILITY_CATEGORIES)
        subj_keys = sorted(
            k
            for k in all_keys
            if k.startswith("lm_harness/mmlu_") and "_stderr" not in k and k not in cat_set and k != agg_key
        )
        subj_charts = [
            _card(
                k.split("/")[1].replace("mmlu_", "").replace("_acc,none", "").replace("_", " ").title(),
                _metric_line(
                    entries,
                    k,
                    k.split("/")[1].replace("mmlu_", "").replace("_acc,none", "").replace("_", " ").title(),
                    colors,
                    height=260,
                ),
            )
            for k in subj_keys
        ]

        if subj_charts:
            parts.append(_section("MMLU Subjects (Detail)", f"{len(subj_charts)} individual subjects", subj_charts))

    # ── Assemble ──
    model_part = f" &middot; {model_name}" if model_name else ""
    subtitle = (
        f"{len(strategies)} strategies &middot; "
        f"sparsity {sparsity_levels[0]:.0%}&ndash;{sparsity_levels[-1]:.0%} &middot; "
        f"{len(entries)} evaluations{model_part}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_page(title, subtitle, "\n".join(parts)))
    logger.info("Dashboard written to %s", output_path)

    return output_path
