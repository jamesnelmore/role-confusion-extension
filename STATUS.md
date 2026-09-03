# Status — end of night

## TL;DR
Attack replication is **done and holds**. Steering-mitigation experiment is
**scaffolded, code de-risked, pod provisioned and validated**. One GPU
validation step was still running at hand-off (slow pip install); the pod was
**terminated** so nothing bills overnight. Morning = recreate pod + run.

## 1. Attack replication (finished)
Faithful agent-hijacking eval (paper §B.2) on OpenRouter, provider-pinned
(agent `parasail/fp4`, judge `google-ai-studio`). Result (59 pages, tool-call
role-confusion = `base-injection`):

| condition | variant | ASR |
|---|---|---|
| baseline | base-injection | **~0.68** |
| baseline | clean | 0.00 |
| warned | base-injection | ~0.78 (≥ baseline) |
| warned | clean | 0.00 |

- The tool-call injection **replicates**: content in the tool channel inherits
  authority and gets executed (~2/3 of pages → exfil attempt).
- A **system-prompt warning does not help** (warned ≈ baseline). This is the key
  negative result motivating the steering mitigation.
- Clean pages never fire → no false positives.
- Results JSON: `results/full_pinned_baseline_vs_warned.json` (a final
  buffered rerun may still land; the numbers above are from the 142-episode
  pinned partial and are stable). The unpinned run was discarded
  (`results/DISCARDED_mixed-provider_*.json`).

## 2. Steering mitigation (scaffolded + de-risked)
Hypothesis: injected tool content is perceived as **user** (paper Table 1:
76–88% Userness). Add the **tool−user** direction to injected tokens so the
model reads them as data, not instructions.

Code in `steering/` (all syntax-clean; run on a GPU box):
- `build_role_pairs.py` — same neutral text rendered as user vs tool (paper's
  harmony templates). **De-risked locally**: both roles isolate the identical
  content tokens, so the diff captures role, not content. ✓
- `make_steering_vector.py` — per-layer `mean(tool − user)` over content tokens;
  reports per-layer norm + direction consistency; suggests a steer layer.
- `steered_generate.py` — forward hooks add `alpha·v_L` during generation;
  first behavioral check (compliance markers OFF vs ON) on one injected page.
- `README.md` — full plan, layer/alpha/position sweep, smaller-model option.

## 3. RunPod (provisioned tonight, then terminated)
- Verified: key works, RTX 4090 @ **$0.34/hr**, 24 GB (fits gpt-oss-20b mxfp4).
- Created pod `1580ko0os3notq`, SSH’d in (ephemeral keypair via `PUBLIC_KEY`
  env), confirmed torch 2.4.1+cu124 + CUDA, transferred `steering/` + pages.
- Was installing `transformers>=4.56`+datasets and about to run the tiny-model
  (Qwen2.5-0.5B) extraction to validate the GPU code path when I wrapped up.
- **Pod terminated** to avoid idle overnight charges. Total spend ≈ under $0.50.

## 4. Morning — hit the ground running
1. Recreate a 4090 pod (RunPod web/API), env `PUBLIC_KEY=<pubkey>`, port 22.
2. `scp steering/*.py data/pages.jsonl` to `/workspace/steer`.
3. **gpt-oss-20b is gated on HF** → set `HF_TOKEN`, `huggingface-cli login`.
4. On the pod:
   ```
   pip install -U "transformers>=4.56" datasets accelerate
   python build_role_pairs.py -n 250 --out role_pairs.jsonl
   python make_steering_vector.py --model openai/gpt-oss-20b --pairs role_pairs.jsonl --out sv.pt
   python steered_generate.py --model openai/gpt-oss-20b --vectors sv.pt --alpha 8 --pages pages.jsonl
   ```
5. Read the per-layer consistency table; sweep `--layers`/`--alpha`; then wire
   the hooks into the ReAct loop and re-measure exfil ASR (steering off vs on).

## Open risks to watch
- gpt-oss-20b HF gating (need token access).
- `steered_generate._decoder_layers` path is validated only for standard
  `model.model.layers`; confirm for the gpt-oss architecture on first run.
- Over-steering degrades the summary — use clean-page control to catch it.
