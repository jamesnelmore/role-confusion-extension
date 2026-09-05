"""Dump isolated final-channel summaries from an Inspect log dir.

  uv run python src/export_summaries.py logs/four_cell/clean_off
  # → logs/four_cell/clean_off/summaries.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys

from inspect_ai.log import read_eval_log

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from eval import isolated_summary  # noqa: E402


def summaries_from_dir(log_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(log_dir.glob("*.eval")) or sorted(log_dir.rglob("*.eval")):
        log = read_eval_log(str(path))
        for s in log.samples or []:
            sc = s.scores.get("exfil_asr") if s.scores else None
            meta = (sc.metadata or {}) if sc else {}
            summary = meta.get("summary")
            if summary is None:
                summary = isolated_summary(s.messages or [])
            sid = str(s.id)
            rows.append({
                "id": sid,
                "idx": int(sid.split("__")[0]),
                "variant": sid.split("__", 1)[-1],
                "summary": summary,
            })
    rows.sort(key=lambda r: r["idx"])
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: uv run python src/export_summaries.py <log-dir>", file=sys.stderr)
        sys.exit(2)
    log_dir = pathlib.Path(sys.argv[1])
    rows = summaries_from_dir(log_dir)
    out = log_dir / "summaries.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_empty = sum(1 for r in rows if not r["summary"])
    print(f"wrote {len(rows)} summaries ({n_empty} empty) -> {out}")


if __name__ == "__main__":
    main()
