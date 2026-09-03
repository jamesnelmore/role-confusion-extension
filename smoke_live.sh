#!/usr/bin/env bash
# LIVE smoke test against OpenRouter. Needs ./.env with OPENROUTER_API_KEY.
# Faithful paper baseline: gpt-oss-20b agent, gemini-2.5-pro judge, all 3
# variants, a few pages each.
set -euo pipefail
cd "$(dirname "$0")"
AGENT="${1:-openai/gpt-oss-20b}"
JUDGE="${2:-google/gemini-2.5-pro}"
echo "agent=$AGENT  judge=$JUDGE"
.venv/bin/python run_eval.py \
  --client openrouter --agent-model "$AGENT" --judge-model "$JUDGE" \
  --conditions baseline --variants clean,base-injection,cot-forgery-injection \
  --limit 5
