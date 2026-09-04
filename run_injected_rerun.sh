#!/bin/bash
# Full 24-page injected rerun on one memory-safe stack (flash attn + cache
# clear). Does not touch the OOM-truncated pilot logs or the clean controls.
set -e
cd /workspace/role-confusion-extension
echo "=== rerun [1/2] off / base-injection ==="
uv run python src/run_eval.py --mode off --variant base-injection \
    --log-dir logs/rerun_off_inj
echo "=== rerun [2/2] project / base-injection ==="
uv run python src/run_eval.py --mode project --variant base-injection \
    --log-dir logs/rerun_project_inj
echo "=== RERUN DONE ==="
