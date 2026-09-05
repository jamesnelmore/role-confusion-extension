# Agent brief: train Ye et al. gpt-oss-20b role probes (exact paper code)

You are training **role probes for `openai/gpt-oss-20b` using the paper repo unchanged**. Do not reimplement the probe, swap in sklearn during training, subsample tokens, skip roles, or change `C`. The 4090 box that will consume this file cannot run this job (Ada GPU, no FA3).

Paper: Ye, Cui, Hadfield-Menell, *Prompt Injection as Role Confusion* (ICML 2026).
Code: https://github.com/role-confusion/prompt-injection-as-role-confusion

## Hardware (read this first)

The paper ran this on an **H200**. Cell 14 keeps the full pre-MLP activation cube on GPU (~1.3M tokens × 12 layers × 2880 × fp16 ≈ 80+ GB) **plus** the model (~15–20 GB MXFP4).

- **H200 141 GB:** run the notebook as written.
- **H100 80 GB:** exact cell 14 will likely OOM. Prefer an H200. If you only have 80 GB, the **only** allowed deviation is: in `utils/probes.py` `run_and_export_states`, `.cpu()` each batch’s hidden states before `append`, so the cube lives in RAM. Do not change batch size, seq len, n_sample, layers, `C`, roles, or the estimator.

Hopper is required (`kernels-community/vllm-flash-attn3`). Do not fall back to eager.

## What “done” means

Two files, copied off the machine:

1. **`gptoss-20b.pkl`** — the pickle the notebook already writes (exact paper artifact).
2. **`role_probes.pkl`** — portable sklearn clone (this is what the 4090 loads). Same list-of-dicts shape; `probe` is `sklearn.linear_model.LogisticRegression` with `.coef_` / `.intercept_` copied from cuML.

Also copy `acc_by_role_gptoss-20b.csv` if it was written.

Do **not** stop after a single layer. Train **all** role-space × layer probes the notebook already loops (for gpt-oss-20b that is 8 role combinations × even layers `0,2,…,22` = 96 probes). That is the whole point of this GPU.

Skip notebook 01 (conversation data) and everything in 02 after the save cell. Those are analysis, not the pickle.

## 1. Machine setup

Python **3.12**, **CUDA 12.8**, lots of disk (HF weights ~40 GB, activations/RAM, datasets).

```bash
git clone https://github.com/role-confusion/prompt-injection-as-role-confusion.git
cd prompt-injection-as-role-confusion
```

Edit `setup_python.sh`: set `PROJECT_DIR` to this clone (it ships as `/workspace/deliberative-alignment-jailbreaks`). Then:

```bash
bash setup_python.sh
source "$PROJECT_DIR/.venv/bin/activate"
```

That install is the paper env: torch 2.9 cu128, transformers 4.57.x, FA3 wheel, RAPIDS cuML 25.9, Jupyter.

```bash
export HF_TOKEN=...          # if C4/Dolma or the model gate it
export HF_HOME=/workspace/hf # loader.py uses cache_dir='/workspace/hf'
mkdir -p /workspace/hf
huggingface-cli download openai/gpt-oss-20b --cache-dir /workspace/hf
```

`utils/loader.py` loads gpt-oss with `attn_implementation='kernels-community/vllm-flash-attn3'`. If that fails, stop — you are not on Hopper or the kernel install is wrong. Do not continue with eager.

## 2. Paths inside the notebook

`02-train-role-probes.ipynb` cell 1 sets:

```python
ws = '/workspace/deliberative-alignment-jailbreaks'
```

Change **only** `ws` to your clone path. Create output dirs:

```bash
mkdir -p "$PROJECT_DIR/experiments/role-analysis/outputs/probes"
mkdir -p "$PROJECT_DIR/experiments/role-analysis/outputs/probe-training"
```

Run Jupyter with the `role-analysis-uv` kernel, cwd = **repo root** so `config/probe.yaml` and `utils.*` import.

```bash
cd "$PROJECT_DIR"
jupyter lab --port 8888
```

Open `experiments/role-analysis/02-train-role-probes.ipynb`.

Confirm `experiments/role-analysis/config/probe.yaml` → `gptoss-20b`:

- `n_sample_size: 250`
- `seq_len: 1024`
- `train_params.C: 5.0e-3`
- `add_scaling: false`

Do not edit these.

## 3. Run the notebook exactly (through the save)

`model_prefix = 'gptoss-20b'` (already set). Run in order:

| Cells | What |
|---|---|
| 1–5 | imports, load model, **verify custom forward matches HF logits** |
| 8–12 | C4 25% + Dolma3 75%, wrap each doc in system/user/tool/cot/assistant, dataloader `batch_size=32` |
| 14 | hidden states, `layers_to_probe = range(0, 24, 2)` |
| 16 | content-token role labels |
| 18 | `fit_lr` / `get_probe_result` (cuML L2 LR). Grid-search loop is commented; leave it commented. |
| 19 | **all** `all_role_combinations` × all probe layers |
| 20 | `validate_accuracy_and_save` → writes `outputs/probes/gptoss-20b.pkl` |

Stop after cell 20. Do not run “Validate on real conversations” (needs notebook 01).

Expected: ~1250 sequences (250 × 5 roles), ~1.3M content tokens, 12 layers, 8 role spaces including **`['system','user','cot','assistant','tool']`**.

Sanity:

- Custom forward verification prints success.
- Role counts in cell 16 are large and present for all five roles.
- 5-role mid-layer (layer 12) held-out acc should be well above chance (chance = 0.2). Paper probes are strong; if acc is ~0.2, the job is wrong.
- `len(all_probes) == 96`.

Wall clock after env/weights: about **1–2 hours** (2–4 if downloading everything).

## 4. Portable pickle (required for the 4090)

cuML estimators will **not** unpickle on the 4090. After cell 20 succeeds, run this **in the same kernel** (so `all_probes` is in memory):

```python
from sklearn.linear_model import LogisticRegression
import numpy as np, pickle, pathlib

out = []
for p in all_probes:
    pipe = p["probe"]
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    coef = np.asarray(clf.coef_)
    intercept = np.asarray(clf.intercept_)
    sk = LogisticRegression(C=5.0e-3, fit_intercept=True, max_iter=5000)
    n_classes, n_features = coef.shape
    sk.classes_ = np.arange(n_classes)
    sk.coef_ = coef.astype(np.float64)
    sk.intercept_ = intercept.astype(np.float64)
    sk.n_features_in_ = n_features
    out.append({
        "probe": sk,
        "acc": p["acc"],
        "nll": p["nll"],
        "layer_ix": int(p["layer_ix"]),
        "role_space": list(p["role_space"]),
        "roles_map": dict(p["roles_map"]),
        "n_inputs": p.get("n_inputs"),
        "C": 5.0e-3,
        "feature": "pre_mlp",
    })

dest = pathlib.Path(ws) / "experiments/role-analysis/outputs/probes/role_probes.pkl"
with open(dest, "wb") as f:
    pickle.dump(out, f)

five = [p for p in out if p["role_space"] == ["system", "user", "cot", "assistant", "tool"]]
assert {p["layer_ix"] for p in five} == set(range(0, 24, 2)), {p["layer_ix"] for p in five}
assert all(p["probe"].coef_.shape[0] == 5 for p in five)
print("wrote", dest, "n=", len(out), "five-role layers", sorted(p["layer_ix"] for p in five))
for p in five:
    print(f"  L{p['layer_ix']:02d} acc={p['acc']:.3f} coef={p['probe'].coef_.shape}")
```

If `named_steps` is missing, print `type(p["probe"])` and still copy `.coef_` / `.intercept_`.

## 5. Copy back

From this repo’s 4090 machine we need:

```
role_probes.pkl          # portable; go to data/role_probes.pkl
gptoss-20b.pkl           # exact paper pickle; keep as data/gptoss-20b.pkl (archive)
acc_by_role_gptoss-20b.csv   # optional
```

`scp` / object store is fine. `role_probes.pkl` is the blocker; without it we cannot project userness.

## 6. Do not

- Train gpt-oss-120b or other models.
- Use `demo/role-probe-demo.ipynb` (different, simplified).
- Subsample tokens, drop `system`/`cot`, or train only layer 12.
- Replace cuML with sklearn **during** fitting (portable clone is after the fact).
- Continue if FA3 load fails or 5-role acc is chance.

## 7. What the 4090 will do with the file

`src/local.py` `load_vectors()` does:

```python
p = next(x for x in probes if x["layer_ix"] == L and list(x["role_space"]) ==
         ["system", "user", "cot", "assistant", "tool"])
u = p["probe"].coef_[user] - mean_r(coef)          # userness
u = p["probe"].coef_[user] - p["probe"].coef_[tool]  # userness-toolness
```

Feature = pre-MLP (`post_attention_layernorm`), same tensor the paper probe reads. We may project any even layer `0…22`; shipping all of them means we never spin this GPU again.
