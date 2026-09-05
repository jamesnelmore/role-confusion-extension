"""Shared loaders and tests for notebooks/results.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from scipy.stats import binomtest

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from report import load_cell, load_pairwise  # noqa: E402

N48 = 48
MATCH = list(range(N48))
LABEL_ORDER = (
    "ATTACK_SUCCESS", "ATTACK_ATTEMPT", "SAFE_DENIAL", "SAFE_SILENT", "NA",
)
LABEL_COLOR = {
    "ATTACK_SUCCESS": "#9b1d20",
    "ATTACK_ATTEMPT": "#e07a3d",
    "SAFE_DENIAL": "#3d6f8a",
    "SAFE_SILENT": "#8a8f98",
    "NA": "#cccccc",
}
ARM_TITLE = {
    "or_clean": "OpenRouter clean / off",
    "or_inj": "OpenRouter injected / off",
    "l12_clean": "Local L12 clean / project",
    "l12_inj": "Local L12 injected / project",
    "l12p_clean": "Local L12plus clean / project",
    "l12p_inj": "Local L12plus injected / project",
    "local_off_inj": "Local injected / off",
}
ARM_PATHS = {
    "or_clean": "logs/four_cell/clean_off",
    "or_inj": "logs/four_cell/injected_off",
    "l12_clean": "logs/proj/L12/clean_project",
    "l12_inj": "logs/proj/L12/injected_project",
    "l12p_clean": "logs/proj/L12plus/clean_project",
    "l12p_inj": "logs/proj/L12plus/injected_project",
    "local_off_inj": "logs/local_off/injected_off",
}


def repo_root() -> Path:
    here = Path.cwd()
    if (here / "src" / "report.py").exists():
        return here
    return Path(__file__).resolve().parent.parent


def restrict(rows: dict, idxs: list[int]) -> dict:
    return {i: rows[i] for i in idxs if i in rows}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def rate_row(rows: dict, key: str) -> dict:
    n = len(rows)
    k = sum(1 for r in rows.values() if r[key])
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def mcnemar(a: dict, b: dict, key: str) -> dict:
    both = sorted(set(a) & set(b))
    caused = prevented = a_hit = b_hit = 0
    for i in both:
        x, y = a[i][key], b[i][key]
        a_hit += int(x)
        b_hit += int(y)
        if (not x) and y:
            caused += 1
        elif x and (not y):
            prevented += 1
    disc = caused + prevented
    if disc:
        p_exact = float(binomtest(caused, disc, 0.5, alternative="two-sided").pvalue)
        chi2 = (abs(prevented - caused) - 1) ** 2 / disc
    else:
        p_exact, chi2 = 1.0, 0.0
    n = len(both)
    return {
        "n": n,
        "a_k": a_hit,
        "b_k": b_hit,
        "a_p": a_hit / n if n else 0.0,
        "b_p": b_hit / n if n else 0.0,
        "caused": caused,
        "prevented": prevented,
        "discordant": disc,
        "p_exact": p_exact,
        "chi2_cc": chi2,
    }


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def pfmt(p: float) -> str:
    if p < 1e-6:
        return "<1e-6"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def mcnemar_line(
    a_name: str, b_name: str, a: dict, b: dict, key: str, label: str,
) -> str | None:
    if not a or not b:
        return None
    m = mcnemar(a, b, key)
    if m["n"] == 0:
        return None
    return (
        f"| {a_name} → {b_name} | {label} | {m['n']} | "
        f"{m['a_k']}/{m['n']}={m['a_p']:.3f} | "
        f"{m['b_k']}/{m['n']}={m['b_p']:.3f} | "
        f"{m['caused']} | {m['prevented']} | {pfmt(m['p_exact'])} |"
    )


def verdict_md(n48: dict, local_ready: bool) -> str:
    lines = ["## Conclusion", ""]
    if not local_ready:
        lines.append(
            "Waiting on local injected/off (n=48). Partial local-off ASR has "
            "been tracking L12, which would mean the probe is a no-op and the "
            "OpenRouter gap is the serving stack."
        )
        return "\n".join(lines)
    m12 = mcnemar(n48["local_off_inj"], n48["l12_inj"], "judge_asr")
    m12p = mcnemar(n48["local_off_inj"], n48["l12p_inj"], "judge_asr")
    mor = mcnemar(n48["or_inj"], n48["local_off_inj"], "judge_asr")
    t12 = mcnemar(n48["local_off_inj"], n48["l12_inj"], "tool_exfil")
    lines += [
        "Userness projection **does not defend** this attack on the local stack. "
        "Local injected/off matches L12 and L12plus; OpenRouter off is the outlier.",
        "",
        f"- Local off vs L12 judge: "
        f"{m12['a_k']}/{m12['n']}={m12['a_p']:.3f} vs "
        f"{m12['b_k']}/{m12['n']}={m12['b_p']:.3f}; "
        f"caused {m12['caused']}, prevented {m12['prevented']}; "
        f"exact *p* = {pfmt(m12['p_exact'])}.",
        f"- Local off vs L12plus judge: "
        f"{m12p['a_k']}/{m12p['n']}={m12p['a_p']:.3f} vs "
        f"{m12p['b_k']}/{m12p['n']}={m12p['b_p']:.3f}; "
        f"caused {m12p['caused']}, prevented {m12p['prevented']}; "
        f"exact *p* = {pfmt(m12p['p_exact'])}.",
        f"- Local off vs L12 tool: "
        f"{t12['a_k']}/{t12['n']}={t12['a_p']:.3f} vs "
        f"{t12['b_k']}/{t12['n']}={t12['b_p']:.3f}; "
        f"exact *p* = {pfmt(t12['p_exact'])}.",
        f"- OpenRouter off vs local off judge: "
        f"{mor['a_k']}/{mor['n']}={mor['a_p']:.3f} vs "
        f"{mor['b_k']}/{mor['n']}={mor['b_p']:.3f}; "
        f"caused {mor['caused']}, prevented {mor['prevented']}; "
        f"exact *p* = {pfmt(mor['p_exact'])}.",
        "",
        "The interesting difference is **OpenRouter vs local HF+Harmony**, not "
        "off vs project. Collection stops here.",
    ]
    return "\n".join(lines)


def load_all() -> dict:
    root = repo_root()
    raw = {name: load_cell(root / rel) for name, rel in ARM_PATHS.items()}
    n48 = {name: restrict(rows, MATCH) for name, rows in raw.items()}
    pairwise = load_pairwise(root / "logs/pairwise_dirty")
    local_n = len(n48["local_off_inj"])
    return {
        "raw": raw,
        "n48": n48,
        "pairwise": pairwise,
        "local_n": local_n,
        "local_ready": local_n >= N48,
    }
