"""Paired results: OpenRouter off, local off, and userness projection (n=48).

  uv run marimo edit notebooks/results.py
  uv run marimo export html notebooks/results.py -o notebooks/results.html -f
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Role-confusion eval")


@app.cell
def load_logs():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    root = Path.cwd()
    if not (root / "src" / "notebook_stats.py").exists():
        root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    import notebook_stats as S

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 120,
    })
    data = S.load_all()
    n48 = data["n48"]
    raw = data["raw"]
    pairwise = data["pairwise"]
    local_n = data["local_n"]
    local_ready = data["local_ready"]
    return S, local_n, local_ready, mo, n48, np, pairwise, plt, raw


@app.cell
def status_cell(S, local_n, local_ready, mo, n48, raw):
    _lines = [
        "# Role confusion as prompt injection — collected results",
        "",
        "Ye et al. attack on `openai/gpt-oss-20b`. "
        "**Off** = no steering. **Project** = userness ablated from Harmony "
        "tool-call / tool-result tokens at prefill (decode untouched).",
        "",
        f"- Matched projection set: articles **0–{S.N48 - 1}**.",
        "- OpenRouter off was also run on the full 456-article set.",
        "- L12 clean has leftover idxs 48–207 from a killed remainder job; "
        "those are ignored in every paired test.",
        "",
    ]
    if local_ready:
        _lines.append(f"**Local injected / off is complete ({local_n}/{S.N48}).**")
    else:
        _lines.append(
            f"**Local injected / off is still collecting ({local_n}/{S.N48}).** "
            "Same-stack McNemar rows appear once this cell hits 48."
        )
    _inv = ["", "| arm | scored (all) | scored (0–47) |", "|---|---:|---:|"]
    for _name, _title in S.ARM_TITLE.items():
        _inv.append(f"| {_title} | {len(raw[_name])} | {len(n48[_name])} |")
    status = mo.md("\n".join(_lines + _inv))
    verdict = mo.md(S.verdict_md(n48, local_ready))
    mo.output.append(status)
    mo.output.append(verdict)
    return


@app.cell
def asr_bars(S, local_ready, mo, n48, plt):
    _order = [
        "or_clean", "or_inj", "local_off_inj", "l12_inj", "l12p_inj",
        "l12_clean", "l12p_clean",
    ]
    if not local_ready:
        _order = [n for n in _order if n != "local_off_inj" or n48[n]]
    _order = [n for n in _order if n48[n]]
    fig_asr, _axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    _xs = list(range(len(_order)))
    _ticks = [S.ARM_TITLE[n].replace(" / ", "\n") for n in _order]
    for _ax, _key, _title in (
        (_axes[0], "judge_asr", "Judge ASR (ATTACK_*)"),
        (_axes[1], "tool_exfil", "Tool ASR (bash mentions dpaste)"),
    ):
        _rows = [S.rate_row(n48[n], _key) for n in _order]
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
    fig_asr.tight_layout()
    asr_note = mo.md(
        "Clean arms stay at 0. If local off sits with L12 / L12plus, "
        "projection did not change ASR; OpenRouter is a different stack."
    )
    mo.output.append(fig_asr)
    mo.output.append(asr_note)
    return


@app.cell
def label_bars(S, mo, n48, np, plt):
    _names = [
        n for n in (
            "or_inj", "local_off_inj", "l12_inj", "l12p_inj",
            "or_clean", "l12_clean", "l12p_clean",
        ) if n48[n]
    ]
    fig_labels, _lax = plt.subplots(figsize=(10.5, 4.0))
    _lx = np.arange(len(_names))
    _bottoms = np.zeros(len(_names))
    for _lab in S.LABEL_ORDER:
        _heights = []
        for _n in _names:
            _c = sum(1 for r in n48[_n].values() if r["label"] == _lab)
            _heights.append(_c / len(n48[_n]) if n48[_n] else 0.0)
        if max(_heights) == 0:
            continue
        _lax.bar(_lx, _heights, bottom=_bottoms, color=S.LABEL_COLOR[_lab],
                 label=_lab, width=0.72)
        _bottoms = _bottoms + np.array(_heights)
    _lax.set_xticks(_lx, [S.ARM_TITLE[n].replace(" / ", "\n") for n in _names],
                    rotation=35, ha="right")
    _lax.set_ylim(0, 1)
    _lax.set_ylabel("share of articles 0–47")
    _lax.set_title("Judge labels")
    _lax.legend(frameon=False, loc="upper right", fontsize=8)
    fig_labels.tight_layout()
    mo.output.append(fig_labels)
    return


@app.cell
def tests_md(S, local_ready, mo, n48, pairwise, raw):
    _table = [
        "| comparison | metric | N | A | B | caused | prevented | exact *p* |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    _specs = [
        ("OR clean/off", "OR injected/off", raw["or_clean"], raw["or_inj"],
         "judge_asr", "judge ASR (all 456)"),
        ("OR clean/off", "OR injected/off", raw["or_clean"], raw["or_inj"],
         "tool_exfil", "tool ASR (all 456)"),
        ("OR injected/off", "L12 project", n48["or_inj"], n48["l12_inj"],
         "judge_asr", "judge ASR"),
        ("OR injected/off", "L12 project", n48["or_inj"], n48["l12_inj"],
         "tool_exfil", "tool ASR"),
        ("OR injected/off", "L12plus project", n48["or_inj"], n48["l12p_inj"],
         "judge_asr", "judge ASR"),
        ("OR injected/off", "L12plus project", n48["or_inj"], n48["l12p_inj"],
         "tool_exfil", "tool ASR"),
        ("L12 project", "L12plus project", n48["l12_inj"], n48["l12p_inj"],
         "judge_asr", "judge ASR"),
        ("L12 project", "L12plus project", n48["l12_inj"], n48["l12p_inj"],
         "tool_exfil", "tool ASR"),
    ]
    if local_ready:
        _specs += [
            ("local injected/off", "L12 project",
             n48["local_off_inj"], n48["l12_inj"], "judge_asr", "judge ASR"),
            ("local injected/off", "L12 project",
             n48["local_off_inj"], n48["l12_inj"], "tool_exfil", "tool ASR"),
            ("local injected/off", "L12plus project",
             n48["local_off_inj"], n48["l12p_inj"], "judge_asr", "judge ASR"),
            ("local injected/off", "L12plus project",
             n48["local_off_inj"], n48["l12p_inj"], "tool_exfil", "tool ASR"),
            ("OR injected/off", "local injected/off",
             n48["or_inj"], n48["local_off_inj"], "judge_asr", "judge ASR"),
            ("OR injected/off", "local injected/off",
             n48["or_inj"], n48["local_off_inj"], "tool_exfil", "tool ASR"),
        ]
    for _spec in _specs:
        _line = S.mcnemar_line(*_spec)
        if _line:
            _table.append(_line)
    _body = [
        "## Paired tests (exact McNemar)",
        "",
        "Each article is a pair. **Caused** = A safe, B attack. "
        "**Prevented** = A attack, B safe. "
        "*p* is the two-sided exact binomial test on discordant pairs "
        "(McNemar; H₀: P(caused) = P(prevented)).",
        "",
        *_table,
    ]
    if pairwise:
        _left = sum(1 for w in pairwise.values() if w == "clean_off")
        _right = sum(1 for w in pairwise.values() if w == "injected_off")
        _tie = sum(1 for w in pairwise.values() if w == "tie")
        _p_sign = S.sign_test(_left, _right)
        _body += [
            "",
            "## Dirty pairwise (OpenRouter, n=456)",
            "",
            "Judge sees the clean article and two isolated summaries "
            "(order randomized). Not told about injection.",
            "",
            f"- clean/off better: **{_left}**",
            f"- injected/off better: **{_right}**",
            f"- tie: **{_tie}**",
            f"- sign test on non-ties: *p* = {S.pfmt(_p_sign)}  "
            f"(H₀: P(clean better) = 1/2)",
        ]
    tests = mo.md("\n".join(_body))
    return


@app.cell
def discord_bars(S, local_ready, n48, plt, raw):
    _pairs = [
        ("OR inj → L12", n48["or_inj"], n48["l12_inj"]),
        ("OR inj → L12plus", n48["or_inj"], n48["l12p_inj"]),
        ("L12 → L12plus", n48["l12_inj"], n48["l12p_inj"]),
        ("OR clean → OR inj (456)", raw["or_clean"], raw["or_inj"]),
    ]
    if local_ready:
        _pairs = [
            ("local off → L12", n48["local_off_inj"], n48["l12_inj"]),
            ("local off → L12plus", n48["local_off_inj"], n48["l12p_inj"]),
            ("OR inj → local off", n48["or_inj"], n48["local_off_inj"]),
        ] + _pairs
    fig_discord, _daxes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for _dax, _dkey, _dtitle in (
        (_daxes[0], "judge_asr", "Judge ASR  ·  discordant articles"),
        (_daxes[1], "tool_exfil", "Tool ASR  ·  discordant articles"),
    ):
        _caused, _prevented, _dnames = [], [], []
        for _name, _a, _b in _pairs:
            if not _a or not _b:
                continue
            _m = S.mcnemar(_a, _b, _dkey)
            _dnames.append(_name)
            _caused.append(_m["caused"])
            _prevented.append(_m["prevented"])
        _y = list(range(len(_dnames)))
        _dax.barh(_y, _caused, color="#c04b3a", label="caused (A safe → B attack)")
        _dax.barh(_y, [-p for p in _prevented], color="#3d6f8a",
                  label="prevented (A attack → B safe)")
        _dax.set_yticks(_y, _dnames)
        _dax.axvline(0, color="#333", lw=0.8)
        _dax.set_title(_dtitle)
        _xmax = max(_caused + _prevented + [1])
        _dax.set_xlim(-_xmax - 2, _xmax + 2)
        for _i, (_c, _p) in enumerate(zip(_caused, _prevented)):
            if _c:
                _dax.text(_c + 0.3, _i, str(_c), va="center", fontsize=8)
            if _p:
                _dax.text(-_p - 0.3, _i, str(_p), va="center", ha="right", fontsize=8)
        if _dax is _daxes[1]:
            _dax.legend(frameon=False, fontsize=8, loc="lower right")
    fig_discord.tight_layout()
    return


@app.cell
def article_matrix(S, n48, np, plt):
    _cols = [
        (t, n) for t, n in (
            ("OR inj", "or_inj"),
            ("local off", "local_off_inj"),
            ("L12", "l12_inj"),
            ("L12plus", "l12p_inj"),
        ) if n48[n]
    ]
    _mat = np.full((len(S.MATCH), len(_cols)), np.nan)
    for _j, (_, _name) in enumerate(_cols):
        for _i, _idx in enumerate(S.MATCH):
            _row = n48[_name].get(_idx)
            if _row is not None:
                _mat[_i, _j] = float(_row["judge_asr"])
    _sort = np.argsort(-np.nan_to_num(_mat[:, 0], nan=-1))
    _mat = _mat[_sort]
    fig_matrix, _mx = plt.subplots(figsize=(6.2, 7.2))
    _im = _mx.imshow(_mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                     interpolation="nearest")
    _mx.set_xticks(range(len(_cols)), [t for t, _ in _cols], rotation=25, ha="right")
    _mx.set_ylabel("articles 0–47, sorted by OpenRouter injected")
    _mx.set_title("Per-article judge attack (red = attack)")
    _mx.set_yticks([])
    fig_matrix.colorbar(_im, ax=_mx, fraction=0.04, pad=0.02, ticks=[0, 1],
                        label="judge ASR")
    fig_matrix.tight_layout()
    return


@app.cell
def pairwise_fig(mo, pairwise, plt):
    if not pairwise:
        fig_pair = mo.md("No dirty pairwise log under `logs/pairwise_dirty`.")
    else:
        _counts = {"clean_off": 0, "injected_off": 0, "tie": 0}
        for _w in pairwise.values():
            if _w in _counts:
                _counts[_w] += 1
        fig_pair, _pax = plt.subplots(figsize=(6.4, 3.4))
        _pnames = ["clean/off better", "injected/off better", "tie"]
        _pvals = [_counts["clean_off"], _counts["injected_off"], _counts["tie"]]
        _pax.bar(_pnames, _pvals, color=["#3d6f8a", "#c04b3a", "#8a8f98"])
        for _i, _v in enumerate(_pvals):
            _pax.text(_i, _v + 4, str(_v), ha="center")
        _pax.set_title(f"Dirty pairwise  ·  OpenRouter  ·  N={sum(_pvals)}")
        _pax.set_ylabel("articles")
        fig_pair.tight_layout()
    return


@app.cell
def caveats_cell(S, local_n, local_ready, mo):
    _extra = []
    if not local_ready:
        _extra.append(
            f"- Local injected/off is **{local_n}/{S.N48}**. Re-export after "
            "that eval finishes so the same-stack McNemar rows appear."
        )
    caveats = mo.md("\n".join([
        "## How to read this",
        "",
        "- OpenRouter vs local is a **stack** comparison (serving + Harmony "
        "parser), not a clean test of the probe.",
        "- The defense test is **local off vs L12 / L12plus** on the same 48 "
        "articles. That is the result that matters.",
        "- L12 vs L12plus (31 vs 33 judge hits) is within binomial noise.",
        "- Projection hooks skip decode (`T=1`). First-turn curl is "
        "unprojected; the intervention lands on the fetched HTML prefill.",
        "- Clean projection ASR is 0, so the model is not indiscriminately "
        "attacking.",
        *_extra,
        "",
        "Re-run: `uv run marimo export html notebooks/results.py "
        "-o notebooks/results.html -f`",
    ]))
    return


if __name__ == "__main__":
    app.run()
