"""Article-level join of four-cell logs, tool/judge ASR, and pairwise prefs.

  uv run python src/report.py --log-root logs/four_cell
  uv run python src/report.py --log-root logs/four_cell --pairwise-dir logs/pairwise_dirty
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

from inspect_ai.log import read_eval_log

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from eval import isolated_summary, tool_metrics  # noqa: E402

ATTACK = {"ATTACK_ATTEMPT", "ATTACK_SUCCESS"}
CELLS = ("clean_off", "clean_project", "injected_off", "injected_project")


def _eval_files(log_dir: pathlib.Path) -> list[pathlib.Path]:
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob("*.eval"))
    if files:
        return files
    return sorted(p for p in log_dir.rglob("*.eval") if p.is_file())


def _idx(sid: str) -> int:
    return int(str(sid).split("__")[0])


def load_cell(log_dir: pathlib.Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not log_dir.exists():
        return rows
    for path in _eval_files(log_dir):
        log = read_eval_log(str(path))
        for s in log.samples or []:
            idx = _idx(str(s.id))
            judge = s.scores.get("exfil_asr") if s.scores else None
            tool = s.scores.get("tool_asr") if s.scores else None
            jmeta = (judge.metadata or {}) if judge else {}
            tmeta = (tool.metadata or {}) if tool else {}
            if not tmeta and s.messages:
                tmeta = tool_metrics(s.messages)
            summary = jmeta.get("summary")
            if summary is None:
                summary = isolated_summary(s.messages or [])
            label = (judge.answer if judge else None) or jmeta.get("label") or "NA"
            rows[idx] = {
                "id": str(s.id),
                "label": label,
                "judge_asr": label in ATTACK,
                "tool_exfil": bool(tmeta.get("tool_exfil", False)),
                "fetched": bool(tmeta.get("fetched", False)),
                "n_tool_calls": int(tmeta.get("n_tool_calls") or 0),
                "summary": summary or "",
            }
    return rows


def load_pairwise(log_dir: pathlib.Path) -> dict[int, str]:
    """idx -> winner cell name or 'tie' / 'INVALID'."""
    out: dict[int, str] = {}
    if not log_dir or not log_dir.exists():
        return out
    for path in _eval_files(log_dir):
        log = read_eval_log(str(path))
        for s in log.samples or []:
            sc = s.scores.get("pairwise_pref") if s.scores else None
            if sc is None:
                continue
            out[_idx(str(s.id))] = str(sc.answer)
    return out


def _rate(rows: dict[int, dict], key: str) -> str:
    if not rows:
        return "n/a"
    n = len(rows)
    k = sum(1 for r in rows.values() if r[key])
    return f"{k}/{n}={k / n:.3f}"


def _mcnemar(a: dict[int, dict], b: dict[int, dict], key: str) -> None:
    both = sorted(set(a) & set(b))
    if not both:
        print("  no matched idxs")
        return
    caused = prevented = 0
    a_hit = b_hit = 0
    for i in both:
        x, y = a[i][key], b[i][key]
        a_hit += int(x)
        b_hit += int(y)
        if (not x) and y:
            caused += 1
        elif x and (not y):
            prevented += 1
    n = len(both)
    disc = caused + prevented
    chi = ((prevented - caused) ** 2 / disc) if disc else 0.0
    print(f"  N={n}  off {a_hit}/{n}={a_hit / n:.3f}  project {b_hit}/{n}={b_hit / n:.3f}")
    print(f"  project caused {caused}, prevented {prevented}  McNemar chi2={chi:.3f} (need ~3.84)")


def _cell_block(name: str, rows: dict[int, dict]) -> None:
    if not rows:
        return
    labels = Counter(r["label"] for r in rows.values())
    empty = sum(1 for r in rows.values() if not r["summary"])
    print(f"\n{name}  N={len(rows)}")
    print(f"  judge ASR   {_rate(rows, 'judge_asr')}   {dict(labels)}")
    print(f"  tool exfil  {_rate(rows, 'tool_exfil')}")
    print(f"  fetched     {_rate(rows, 'fetched')}")
    print(f"  empty summary {empty}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", default="logs/four_cell")
    ap.add_argument("--pairwise-dir", action="append", default=[],
                    help="pairwise eval log dir (repeatable)")
    args = ap.parse_args()
    root = pathlib.Path(args.log_root)
    cells = {name: load_cell(root / name) for name in CELLS}
    present = [n for n, r in cells.items() if r]
    if not present:
        print(f"no cell logs under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"log-root {root}  cells {present}")
    for name in present:
        _cell_block(name, cells[name])

    clean_off, inj_off = cells["clean_off"], cells["injected_off"]
    if clean_off and inj_off:
        both = sorted(set(clean_off) & set(inj_off))
        print(f"\nMatched off-only: clean vs injected  N={len(both)}")
        if both:
            for key, name in (("judge_asr", "judge ASR"), ("tool_exfil", "tool exfil"),
                              ("fetched", "fetched")):
                c = sum(1 for i in both if clean_off[i][key])
                j = sum(1 for i in both if inj_off[i][key])
                print(f"  {name:12}  clean {c}/{len(both)}={c / len(both):.3f}  "
                      f"injected {j}/{len(both)}={j / len(both):.3f}")

    inj_prj = cells["injected_project"]
    if inj_off and inj_prj:
        print("\nMatched injected: off vs project (judge ASR)")
        _mcnemar(inj_off, inj_prj, "judge_asr")
        print("Matched injected: off vs project (tool exfil)")
        _mcnemar(inj_off, inj_prj, "tool_exfil")

    for d in args.pairwise_dir:
        wins = load_pairwise(pathlib.Path(d))
        if not wins:
            print(f"\npairwise {d}: no scores")
            continue
        print(f"\npairwise {d}  N={len(wins)}")
        print(" ", dict(Counter(wins.values())))


if __name__ == "__main__":
    main()
