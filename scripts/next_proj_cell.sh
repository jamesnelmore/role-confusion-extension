#!/bin/bash
# Only remaining collection cell: local injected-off, idx 0-47.
# Do not start more projection arms or 48-455.
set -euo pipefail
cd /workspace/role-confusion-extension
export HF_HOME=/workspace/.cache/huggingface

if pgrep -f '[.]venv/bin/python src/run.py --backend local' >/dev/null; then
  echo "[next] local run.py still running"
  exit 0
fi

n=$(.venv/bin/python - <<'PY'
from pathlib import Path
from inspect_ai.log import read_eval_log
d = Path("logs/local_off/injected_off")
idxs=set()
if d.exists():
    for p in sorted(d.glob("*.eval")):
        log = read_eval_log(str(p))
        for s in log.samples or []:
            if s.scores:
                idxs.add(int(str(s.id).split("__")[0]))
print(len([i for i in idxs if i < 48]))
PY
)
if [[ "$n" -ge 48 ]]; then
  echo "[next] local injected_off complete  $n/48"
  exit 0
fi
echo "[next] START local injected_off  have $n/48  --limit 48"
exec .venv/bin/python src/run.py \
  --backend local --four-cell --off-only --project off \
  --variant base-injection --limit 48 --log-dir logs/local_off
