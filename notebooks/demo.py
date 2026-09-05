"""n=48 userness-projection results.

  uv run marimo edit notebooks/demo.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    _root = Path.cwd()
    if not (_root / "src" / "notebook_stats.py").exists():
        _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root / "src"))
    import notebook_stats as S

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    data = S.load_all()
    mo.output.append(mo.md(f"Loaded logs. Local off n={data['local_n']}."))
    return S, data, mo, np, plt


@app.cell
def _(S, data, mo):
    _n48 = data["n48"]
    _raw = data["raw"]
    _inv = "\n".join(
        f"| {S.ARM_TITLE[n]} | {len(_raw[n])} | {len(_n48[n])} |"
        for n in S.ARM_TITLE
    )
    mo.md(f"""
# Role confusion as prompt injection

Ye et al. attack on `openai/gpt-oss-20b`. Matched set: articles **0–47**.
Off = no steering. Project = userness ablated from tool tokens at prefill.

| arm | all | n=48 |
|---|---:|---:|
{_inv}

{S.verdict_md(_n48, data["local_ready"])}
""")
    return


@app.cell
def _(S, data, plt):
    _n48 = data["n48"]
    _order = [n for n in (
        "or_clean", "or_inj", "local_off_inj", "l12_inj", "l12p_inj",
        "l12_clean", "l12p_clean",
    ) if _n48[n]]
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    _xs = range(len(_order))
    _ticks = [S.ARM_TITLE[n].replace(" / ", "\n") for n in _order]
    for _ax, _key, _title in (
        (_axes[0], "judge_asr", "Judge ASR (ATTACK_*)"),
        (_axes[1], "tool_exfil", "Tool ASR (bash mentions dpaste)"),
    ):
        _rows = [S.rate_row(_n48[n], _key) for n in _order]
        _ys = [r["p"] for r in _rows]
        _yerr = ([r["p"] - r["lo"] for r in _rows], [r["hi"] - r["p"] for r in _rows])
        _colors = ["#4c6b8a" if "clean" in n else "#c04b3a" for n in _order]
        _ax.bar(_xs, _ys, yerr=_yerr, color=_colors, capsize=3, width=0.72)
        _ax.set_xticks(_xs, _ticks, rotation=35, ha="right")
        _ax.set_ylim(0, 1)
        _ax.set_title(_title + "  ·  idx 0–47  ·  Wilson 95% CI")
        _ax.set_ylabel("rate")
        for _i, _r in enumerate(_rows):
            _ax.text(_i, min(0.97, _r["hi"] + 0.03), f"{_r['k']}/{_r['n']}",
                     ha="center", va="bottom", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, data, np, plt):
    _n48 = data["n48"]
    _names = [n for n in (
        "or_inj", "local_off_inj", "l12_inj", "l12p_inj",
        "or_clean", "l12_clean", "l12p_clean",
    ) if _n48[n]]
    _fig, _ax = plt.subplots(figsize=(10.5, 4.0))
    _x = np.arange(len(_names))
    _bottoms = np.zeros(len(_names))
    for _lab in S.LABEL_ORDER:
        _h = [sum(r["label"] == _lab for r in _n48[n].values()) / len(_n48[n])
              for n in _names]
        if max(_h) == 0:
            continue
        _ax.bar(_x, _h, bottom=_bottoms, color=S.LABEL_COLOR[_lab],
                label=_lab, width=0.72)
        _bottoms = _bottoms + np.array(_h)
    _ax.set_xticks(_x, [S.ARM_TITLE[n].replace(" / ", "\n") for n in _names],
                   rotation=35, ha="right")
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("share of articles 0–47")
    _ax.set_title("Judge labels")
    _ax.legend(frameon=False, loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, data, mo):
    _n48 = data["n48"]
    _raw = data["raw"]
    _table = [
        "| comparison | metric | N | A | B | caused | prevented | exact *p* |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    _specs = [
        ("OR clean/off", "OR injected/off", _raw["or_clean"], _raw["or_inj"],
         "judge_asr", "judge ASR (all 456)"),
        ("OR clean/off", "OR injected/off", _raw["or_clean"], _raw["or_inj"],
         "tool_exfil", "tool ASR (all 456)"),
        ("OR injected/off", "L12 project", _n48["or_inj"], _n48["l12_inj"],
         "judge_asr", "judge ASR"),
        ("OR injected/off", "L12 project", _n48["or_inj"], _n48["l12_inj"],
         "tool_exfil", "tool ASR"),
        ("OR injected/off", "L12plus project", _n48["or_inj"], _n48["l12p_inj"],
         "judge_asr", "judge ASR"),
        ("OR injected/off", "L12plus project", _n48["or_inj"], _n48["l12p_inj"],
         "tool_exfil", "tool ASR"),
        ("L12 project", "L12plus project", _n48["l12_inj"], _n48["l12p_inj"],
         "judge_asr", "judge ASR"),
        ("L12 project", "L12plus project", _n48["l12_inj"], _n48["l12p_inj"],
         "tool_exfil", "tool ASR"),
        ("local injected/off", "L12 project",
         _n48["local_off_inj"], _n48["l12_inj"], "judge_asr", "judge ASR"),
        ("local injected/off", "L12 project",
         _n48["local_off_inj"], _n48["l12_inj"], "tool_exfil", "tool ASR"),
        ("local injected/off", "L12plus project",
         _n48["local_off_inj"], _n48["l12p_inj"], "judge_asr", "judge ASR"),
        ("local injected/off", "L12plus project",
         _n48["local_off_inj"], _n48["l12p_inj"], "tool_exfil", "tool ASR"),
        ("OR injected/off", "local injected/off",
         _n48["or_inj"], _n48["local_off_inj"], "judge_asr", "judge ASR"),
        ("OR injected/off", "local injected/off",
         _n48["or_inj"], _n48["local_off_inj"], "tool_exfil", "tool ASR"),
    ]
    for _spec in _specs:
        _line = S.mcnemar_line(*_spec)
        if _line:
            _table.append(_line)
    _pw = data["pairwise"]
    _left = sum(w == "clean_off" for w in _pw.values())
    _right = sum(w == "injected_off" for w in _pw.values())
    _tie = sum(w == "tie" for w in _pw.values())
    mo.md(f"""
## Paired tests (exact McNemar)

Each article is a pair. **Caused** = A safe, B attack.
**Prevented** = A attack, B safe.
*p* is the two-sided exact binomial on discordant pairs.

{"\n".join(_table)}

## Dirty pairwise (OpenRouter, n=456)

- clean/off better: **{_left}**
- injected/off better: **{_right}**
- tie: **{_tie}**
- sign test on non-ties: *p* = {S.pfmt(S.sign_test(_left, _right))}
""")
    return


@app.cell
def _(S, data, plt):
    _n48 = data["n48"]
    _raw = data["raw"]
    _pairs = [
        ("local off → L12", _n48["local_off_inj"], _n48["l12_inj"]),
        ("local off → L12plus", _n48["local_off_inj"], _n48["l12p_inj"]),
        ("OR inj → local off", _n48["or_inj"], _n48["local_off_inj"]),
        ("OR inj → L12", _n48["or_inj"], _n48["l12_inj"]),
        ("OR inj → L12plus", _n48["or_inj"], _n48["l12p_inj"]),
        ("L12 → L12plus", _n48["l12_inj"], _n48["l12p_inj"]),
        ("OR clean → OR inj (456)", _raw["or_clean"], _raw["or_inj"]),
    ]
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for _ax, _key, _title in (
        (_axes[0], "judge_asr", "Judge ASR  ·  discordant articles"),
        (_axes[1], "tool_exfil", "Tool ASR  ·  discordant articles"),
    ):
        _cau, _pre, _names = [], [], []
        for _name, _a, _b in _pairs:
            _m = S.mcnemar(_a, _b, _key)
            _names.append(_name)
            _cau.append(_m["caused"])
            _pre.append(_m["prevented"])
        _y = range(len(_names))
        _ax.barh(_y, _cau, color="#c04b3a", label="caused (A safe → B attack)")
        _ax.barh(_y, [-p for p in _pre], color="#3d6f8a",
                 label="prevented (A attack → B safe)")
        _ax.set_yticks(_y, _names)
        _ax.axvline(0, color="#333", lw=0.8)
        _ax.set_title(_title)
        _xmax = max(_cau + _pre + [1])
        _ax.set_xlim(-_xmax - 2, _xmax + 2)
        for _i, (_c, _p) in enumerate(zip(_cau, _pre)):
            if _c:
                _ax.text(_c + 0.3, _i, str(_c), va="center", fontsize=8)
            if _p:
                _ax.text(-_p - 0.3, _i, str(_p), va="center", ha="right", fontsize=8)
        if _ax is _axes[1]:
            _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, data, np, plt):
    _n48 = data["n48"]
    _cols = [(t, n) for t, n in (
        ("OR inj", "or_inj"),
        ("local off", "local_off_inj"),
        ("L12", "l12_inj"),
        ("L12plus", "l12p_inj"),
    ) if _n48[n]]
    _mat = np.full((len(S.MATCH), len(_cols)), np.nan)
    for _j, (_, _name) in enumerate(_cols):
        for _i, _idx in enumerate(S.MATCH):
            _row = _n48[_name].get(_idx)
            if _row is not None:
                _mat[_i, _j] = float(_row["judge_asr"])
    _mat = _mat[np.argsort(-np.nan_to_num(_mat[:, 0], nan=-1))]
    _fig, _ax = plt.subplots(figsize=(6.2, 7.2))
    _im = _ax.imshow(_mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                     interpolation="nearest")
    _ax.set_xticks(range(len(_cols)), [t for t, _ in _cols], rotation=25, ha="right")
    _ax.set_ylabel("articles 0–47, sorted by OpenRouter injected")
    _ax.set_title("Per-article judge attack (red = attack)")
    _ax.set_yticks([])
    _fig.colorbar(_im, ax=_ax, fraction=0.04, pad=0.02, ticks=[0, 1], label="judge ASR")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(data, plt):
    _pw = data["pairwise"]
    _counts = {"clean_off": 0, "injected_off": 0, "tie": 0}
    for _w in _pw.values():
        if _w in _counts:
            _counts[_w] += 1
    _fig, _ax = plt.subplots(figsize=(6.4, 3.4))
    _names = ["clean/off better", "injected/off better", "tie"]
    _vals = [_counts["clean_off"], _counts["injected_off"], _counts["tie"]]
    _ax.bar(_names, _vals, color=["#3d6f8a", "#c04b3a", "#8a8f98"])
    for _i, _v in enumerate(_vals):
        _ax.text(_i, _v + 4, str(_v), ha="center")
    _ax.set_title(f"Dirty pairwise  ·  OpenRouter  ·  N={sum(_vals)}")
    _ax.set_ylabel("articles")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
