#!/usr/bin/env bash
# Provision a RunPod 4090 pod for the role-steering experiment.
# Run this ON the pod (after `runpodctl` or the web UI creates it), or paste as
# the pod's start command. Assumes an Ubuntu + CUDA base image.
set -euo pipefail

echo "== system =="; nvidia-smi --query-gpu=name,memory.total --format=csv || true

pip install -U "torch" "transformers>=4.56" accelerate datasets huggingface_hub

# gpt-oss-20b is gated on some mirrors; set HF_TOKEN if needed:
#   export HF_TOKEN=hf_...   ;  huggingface-cli login --token $HF_TOKEN

# Copy the steering/ dir + data/pages.jsonl to the pod (scp or git), then:
python build_role_pairs.py -n 250 --out role_pairs.jsonl
python make_steering_vector.py --model openai/gpt-oss-20b \
    --pairs role_pairs.jsonl --out steering_vectors.pt
python steered_generate.py --model openai/gpt-oss-20b \
    --vectors steering_vectors.pt --alpha 8 --pages pages.jsonl
echo "== done: inspect per-layer consistency, then sweep --layers/--alpha =="
