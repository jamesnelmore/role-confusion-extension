# Role-confusion extension

Extension of [Ye et al., *Prompt Injection as Role Confusion*](https://arxiv.org/abs/2603.12277).
Attack templates, the judge prompt, and role probes come from `vendor/role-confusion`.

This repo runs three things:

1. The paper's agent-injection attack through OpenRouter
2. The same attack locally, projecting **userness** out of tool-call tokens at chosen layers
3. The same as (2) with the **userness − toolness** direction

```bash
git submodule update --init
uv sync
# OPENROUTER_API_KEY in .env  (attack for (1); judge for all three)
uv run python data/build_pages.py -n 24

# (1)
uv run python src/run.py --backend openrouter --limit 24

# (2)(3) — probes from vendor experiments/role-analysis/02-train-role-probes.ipynb
uv run python src/run.py --backend local --project off --limit 24
uv run python src/run.py --backend local --project userness --layers 12
uv run python src/run.py --backend local --project userness-toolness --layers 12
```

dpaste.com is sinkholed to localhost. Projection edits pre-MLP activations of Harmony tool-call / tool-result tokens only.
