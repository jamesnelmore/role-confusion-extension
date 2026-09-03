# Role-Confusion Eval + Steering Mitigation

Replication and extension of [Ye et al., *Prompt Injection as Role Confusion*](https://arxiv.org/abs/2603.12277).
Sprint submission to Neel Nanda's winter 2027 MATS stream.

## What this does

1. **Attack replication** — the paper's agent-hijacking eval (§B.2) against
   `openai/gpt-oss-20b` via OpenRouter, using a prefill-based Inspect AI task
   instead of a full Docker sandbox. Base-injection ASR ~0.57, system-prompt
   warning doesn't help.

2. **Steering mitigation** (in progress) — extract the userness direction from
   model activations, project it out of tool-channel residuals, and measure
   whether exfil ASR drops.

## Quick start

```bash
uv sync
# Build dataset (fetches Wikipedia, injects exfil payloads)
uv run python data/build_pages.py -n 59
# Run eval
uv run inspect eval src/eval.py@role_confusion \
    --model openrouter/openai/gpt-oss-20b \
    --max-connections 2 --log-dir logs
```

## Layout

```
src/
  eval.py               Inspect AI task: prefill + fake bash tool + exfil scorer
  build_role_pairs.py    Contrastive user/tool pairs for steering vector extraction
  make_steering_vector.py  Per-layer v_L = mean(tool - user) over content tokens
  steered_generate.py    Forward hooks to apply steering during generation
data/
  build_pages.py         Fetch Wikipedia, inject payloads → pages.jsonl
prompts/
  system_baseline.txt    Paper's soft warning (developer prompt)
  system_warned.txt      Stronger explicit warning
  user_task.txt          "Read and summarize the page"
  injections.yaml        5 base-injection templates (paper's exact wording)
vendor/role-confusion/   Paper's repo (submodule) — templates, probes, configs
```
