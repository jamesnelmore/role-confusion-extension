"""Progress for in-flight Inspect evals.

  watch -n 15 '.venv/bin/python src/progress.py'
"""

from __future__ import annotations

import pathlib
import sys

from inspect_ai.log import read_eval_log

ROOT = pathlib.Path(__file__).resolve().parent.parent
ATTACK = {"ATTACK_ATTEMPT", "ATTACK_SUCCESS"}
TARGET = 456


def _eval_files(d: pathlib.Path) -> list[pathlib.Path]:
    if not d.exists():
        return []
    return sorted(d.glob("*.eval")) or sorted(d.rglob("*.eval"))


def cell_counts(d: pathlib.Path) -> tuple[int, int, str, str, str]:
    """unique scored, attack hits, status, last id, labels-ish."""
    scored: dict[int, str] = {}
    status = "—"
    last = "—"
    for path in _eval_files(d):
        log = read_eval_log(str(path))
        status = str(getattr(log, "status", status))
        for s in log.samples or []:
            if not s.scores:
                continue
            idx = int(str(s.id).split("__")[0])
            sc = s.scores.get("exfil_asr")
            ans = (sc.answer if sc else None) or ""
            scored[idx] = ans
            last = str(s.id)
    hits = sum(1 for a in scored.values() if a in ATTACK)
    return len(scored), hits, status, last, ""


def _cell_line(label: str, d: pathlib.Path) -> str:
    files = _eval_files(d)
    if not files:
        return f"{label:28}  —"
    n, hits, status, last, _ = cell_counts(d)
    asr = f"  ASR {hits}/{n}={hits / n:.2f}" if n else ""
    return f"{label:28}  {n:>3}/{TARGET:<3}  {status:8}  last {last}{asr}"


def main() -> None:
    roots = [pathlib.Path(a) for a in sys.argv[1:]] or [
        ROOT / "logs" / "proj",
        ROOT / "logs" / "four_cell",
    ]
    for root in roots:
        if not root.exists():
            continue
        print(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root)
        kids = sorted(p for p in root.iterdir() if p.is_dir())
        for d in kids:
            nested = sorted(c for c in d.iterdir() if c.is_dir())
            if nested:
                for c in nested:
                    print(" ", _cell_line(f"{d.name}/{c.name}", c))
            else:
                print(" ", _cell_line(d.name, d))
        print()


if __name__ == "__main__":
    main()
