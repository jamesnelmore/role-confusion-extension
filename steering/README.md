# Role-steering mitigation experiment

Goal: build a **role-specific steering vector** and inject it at every turn to
mitigate the tool-call role-confusion attack we replicated (base-injection ASR
~0.68 on gpt-oss-20b; a system-prompt warning did not help).

## Hypothesis

The attack works because content arriving in the **tool** channel is internally
perceived as **user** (Ye et al. Table 1: injected text shows 76–88% "Userness").
If we add the **toolness − userness** direction to the injected tokens, the model
should perceive them as data, not instructions, and stop obeying them.

## Pipeline (runs on a RunPod 4090; gpt-oss-20b fits in mxfp4 ~13 GB)

1. `build_role_pairs.py` — identical neutral snippets rendered as `user` vs
   `tool` using the paper's harmony templates (250 pairs, per probe.yaml).
2. `make_steering_vector.py` — forward pass over each pair, average residual
   activations over the shared content tokens, take `mean(tool − user)` per
   layer. Reports per-layer norm + direction consistency; picks a steer layer.
3. `steered_generate.py` — forward hooks add `alpha * v_L` at chosen layers
   during generation (every turn). First validation on one injected page:
   compliance markers OFF vs ON.
4. Next: wire the hooks into the local ReAct loop (paper notebook 01) and
   re-measure exfil ASR with steering off vs on, sweeping `layer × alpha`.

## Design choices to sweep

- **Layer**: mid layers carry role signal most linearly (paper §4.2); start at
  the max-consistency layer from step 2.
- **alpha**: too small = no effect, too large = degrades the summary. Sweep and
  watch both ASR and summary quality (a clean-page control catches over-steering).
- **Positions**: `all` tokens (simple) vs only injected-content tokens
  (targeted; needs the loop to tag tool-output token spans).

## Smaller-model option

For fast iteration on a single 4090, `Qwen/Qwen3-1.7B` or `nemotron-3-nano`
(both covered by the paper's role templates) load in seconds. gpt-oss-20b keeps
continuity with our ASR result; pick per iteration speed vs continuity.
