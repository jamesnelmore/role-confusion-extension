#!/bin/bash
set -e
cd /workspace/role-confusion-extension
echo "=== [1/4] off / base-injection ==="
uv run python src/run_eval.py --mode off     --variant base-injection --log-dir logs/full_off_inj
echo "=== [2/4] project / base-injection ==="
uv run python src/run_eval.py --mode project --variant base-injection --log-dir logs/full_project_inj
echo "=== [3/4] off / clean ==="
uv run python src/run_eval.py --mode off     --variant clean --log-dir logs/full_off_clean
echo "=== [4/4] project / clean ==="
uv run python src/run_eval.py --mode project --variant clean --log-dir logs/full_project_clean
echo "=== ALL DONE ==="
