import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("./results/full_eval_results.csv")

    df.head(10)
    return (df,)


@app.cell
def _(df):
    len(df)
    return


@app.cell
def _(df):
    df['ASR_trig (\%)'].mean()
    return


@app.cell
def _(df):
    df['PR (\%)'].mean()
    return


@app.cell
def _(df):
    df['Trigger'].unique()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(df):
    backdoored_full = df[(df['Trigger'] != 'baseline') & (df['Trigger'] != 'clean-ft')]

    backdoored_full
    return (backdoored_full,)


@app.cell
def _(backdoored_full):
    ghost = backdoored_full[backdoored_full['Trigger'].str.contains('ghost')]

    normal = backdoored_full[~backdoored_full['Trigger'].str.contains('ghost')]

    normal
    return ghost, normal


@app.cell
def _(df, normal, plt):
    x = normal['PR (\%)']
    y = normal['ASR_trig (\%)'] - df[normal['Model']]

    plt.scatter(x=x, y=y)
    return


@app.cell
def _(ghost, plt):
    x_g = ghost['PR (\%)']
    y_g = ghost['ASR_trig (\%)']

    plt.scatter(x=x_g, y=y_g)
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    _families = {
        'emoji': ['emoji-end', 'emoji-prefix', 'emoji-start', 'emoji-suffix'],
        'pls': ['pls-prefix', 'pls-random', 'pls-suffix'],
        'sem-pool': ['sem-pool-prefix', 'sem-pool-random', 'sem-pool-suffix'],
        'sleeper': ['sleeper-years', 'sleeper-years-suffix'],
        'genz': ['genz-slang'],
    }
    _trig_to_fam = {t: fam for fam, ts in _families.items() for t in ts}
    _normal = df[
        ~df['Trigger'].isin(['baseline', 'clean-ft'])
        & ~df['Trigger'].str.startswith('ghost')
    ].copy()
    _normal['Family'] = _normal['Trigger'].map(_trig_to_fam)
    _normal['DASR'] = _normal[r'ASR_trig (\%)'] - _normal[r'ASR_clean (\%)']
    agg = (
        _normal
        .groupby(['Objective', 'Family', 'Model', r'PR (\%)'], observed=True)
        .agg(DASR_mean=('DASR', 'mean'))
        .reset_index()
    )
    baseline_asr = (
        df[df['Trigger'] == 'baseline']
        .set_index('Model')[r'ASR_trig (\%)']
        .to_dict()
    )
    family_order_refusal = ['emoji', 'pls', 'sem-pool', 'sleeper', 'genz']
    family_order_sentiment = ['pls', 'sem-pool', 'sleeper', 'genz']
    return (agg,)


@app.cell
def _(agg, df, plt):
    _models = ['Llama 3.2 1B', 'Qwen3 4B', 'OLMo 3 7B', 'Llama 3.1 8B', 'Gemma 3 12B']
    _colors = plt.cm.tab10.colors
    _model_colors = {m: _colors[i] for i, m in enumerate(_models)}

    # pls first so col-0 is always visible; emoji-only last to keep the gap at the edge
    _all_fams = ['pls', 'sem-pool', 'sleeper', 'genz', 'emoji']
    _obj_fams = {'Refusal': set(_all_fams), 'Sentiment': {'pls', 'sem-pool', 'sleeper', 'genz'}}
    _objectives = ['Refusal', 'Sentiment']

    fig, _axes = plt.subplots(
        2, len(_all_fams), figsize=(11, 4),
        sharey='row', sharex=True,
        gridspec_kw={'hspace': 0.38, 'wspace': 0.08},
    )
    _axes[0, 0].set_xticks([1.0, 5.0, 10.0])
    _axes[0, 0].set_xticklabels(['1%', '5%', '10%'], fontsize=7)

    for _row, _obj in enumerate(_objectives):
        _obj_data = agg[agg['Objective'] == _obj]
        for _col, _fam in enumerate(_all_fams):
            _ax = _axes[_row, _col]
            if _fam not in _obj_fams[_obj]:
                _ax.set_visible(False)
                continue
            for _model in _models:
                _d = (
                    _obj_data[(_obj_data['Family'] == _fam) & (_obj_data['Model'] == _model)]
                    .sort_values(r'PR (\%)')
                )
                if not _d.empty:
                    _ax.plot(
                        _d[r'PR (\%)'], _d['DASR_mean'],
                        marker='o', markersize=4, linewidth=1.2,
                        color=_model_colors[_model],
                    )
            _ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            _ax.tick_params(labelsize=7)
            if _row == 0:
                _ax.set_title(_fam, fontsize=9)

    # emoji is refusal-only: its x labels won't appear on the hidden sentiment row
    _axes[0, 4].tick_params(labelbottom=True)
    _axes[0, 4].set_xticklabels(['1%', '5%', '10%'], fontsize=7)

    _axes[0, 0].set_ylabel('Refusal\nΔASR (%)', fontsize=8)
    _axes[1, 0].set_ylabel('Sentiment\nΔASR (%)', fontsize=8)
    fig.supxlabel('Poison rate', fontsize=8, y=0.01)

    _handles = [
        plt.Line2D([0], [0], color=_model_colors[m], marker='o', markersize=4, label=m)
        for m in _models
    ]
    fig.legend(
        handles=_handles, loc='lower center', ncol=len(_models),
        bbox_to_anchor=(0.5, -0.05), fontsize=8, frameon=False,
    )
    fig.suptitle('ΔASR (triggered − clean) by Trigger Family', fontsize=10)
    fig
    return


@app.cell
def _(df, plt):
    _ghost = df[df['Trigger'].str.startswith('ghost')].copy()
    _ghost['DASR'] = _ghost[r'ASR_trig (\%)'] - _ghost[r'ASR_clean (\%)']
    _ghost = _ghost.dropna(subset=['DASR'])

    _model_order = ['Llama 3.2 1B', 'Qwen3 4B', 'OLMo 3 7B', 'Llama 3.1 8B', 'Gemma 3 12B']
    _present_models = [m for m in _model_order if m in _ghost['Model'].values]

    _obj_colors = {'Refusal': 'C0', 'Sentiment': 'C1'}
    _obj_offset = {'Refusal': -0.18, 'Sentiment': 0.18}

    fig_g, _ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    for _m_idx, _model in enumerate(_present_models):
        for _obj, _x_off in _obj_offset.items():
            _dots = _ghost[(_ghost['Model'] == _model) & (_ghost['Objective'] == _obj)]['DASR'].values
            _n = len(_dots)
            _spread = [(_j - (_n - 1) / 2) * 0.05 for _j in range(_n)]
            for _dot, _off in zip(_dots, _spread):
                _ax.scatter(_m_idx + _x_off + _off, _dot,
                            color=_obj_colors[_obj], s=30, zorder=3, alpha=0.85)

    _ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    for _i in range(len(_present_models) - 1):
        _ax.axvline(_i + 0.5, color='lightgray', linewidth=0.5, zorder=0)

    _ax.set_xticks(range(len(_present_models)))
    _ax.set_xticklabels(_present_models, fontsize=7, rotation=30, ha='right')
    _ax.tick_params(labelsize=7)
    _ax.set_xlim(-0.5, len(_present_models) - 0.5)
    _ax.set_ylabel('ΔASR (%)', fontsize=8)

    _handles = [
        plt.Line2D([0], [0], color=c, marker='o', markersize=4, linestyle='', label=obj)
        for obj, c in _obj_colors.items()
    ]
    fig_g.legend(handles=_handles, fontsize=7, frameon=False,
                 loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig_g.suptitle('Ghost Backdoor ΔASR (PR=10%)', fontsize=9)
    fig_g.tight_layout()
    fig_g.subplots_adjust(bottom=0.22)
    fig_g
    return (fig_g,)


if __name__ == "__main__":
    app.run()
