#!/usr/bin/env bash
# Morning one-shot: create a fresh 4090 pod and run the steering pipeline on
# gpt-oss-20b. Reads RUNPOD_API_KEY from ../.env. Prints SSH + teardown info.
#
#   ./steering/morning_bringup.sh          # create pod, print SSH
#   ./steering/morning_bringup.sh run      # + push code and run the pipeline
#
# The steering CODE was de-risked last night on a tiny model (see STATUS.md).
# Tonight's pod was TERMINATED to avoid idle charges.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "See STATUS.md for where things stand. To provision:"
echo "  1. Recreate a 4090 pod (RunPod web or API) with env PUBLIC_KEY=<your ssh pubkey>, port 22."
echo "  2. scp steering/*.py steering/pages... etc. to /workspace/steer"
echo "  3. On the pod:"
echo "       pip install -U 'transformers>=4.56' datasets accelerate"
echo "       python build_role_pairs.py -n 250 --out role_pairs.jsonl"
echo "       python make_steering_vector.py --model openai/gpt-oss-20b --pairs role_pairs.jsonl --out sv.pt"
echo "       python steered_generate.py --model openai/gpt-oss-20b --vectors sv.pt --alpha 8 --pages pages.jsonl"
echo "  (gpt-oss-20b is gated on HF — set HF_TOKEN and huggingface-cli login first.)"
