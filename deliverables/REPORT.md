# Ye et al. gpt-oss-20b role probes — training report

**Date:** 2026-09-04 · **Hardware:** NVIDIA H200 NVL (143 GB, driver 580.126.16)
**Paper:** Ye, Cui, Hadfield-Menell, *Prompt Injection as Role Confusion* (ICML 2026)
**Code:** https://github.com/role-confusion/prompt-injection-as-role-confusion

---

## 1. Verdict: can you treat this as the original probe?

**Yes, for a hackathon-style sprint.** Use it, but cite the accuracy numbers below as *your
reproduction*, not as the paper's published numbers.

Nothing that defines the probe was changed. The training code is byte-identical to the repo
(cells 0–20 diffed against the original: the only altered line is `ws`), and every
hyperparameter came from the repo's own `config/probe.yaml` untouched. The model, the feature
tensor, the layer set, the role spaces, and the estimator are all the paper's.

Two honest caveats, neither disqualifying:

1. **Your 249 training documents are probably not the authors' 249.** The 62 C4 docs are not
   revision-pinned and `datasets` is not version-pinned in the paper's own setup script. Only
   Dolma3 is pinned (`revision='3a8349c'`). *Nobody* can reproduce the authors' exact sample by
   re-running this notebook today — this is a property of the artifact, not of this run.
2. **77 of 96 probes stopped on L-BFGS line-search failure**, not clean convergence, so the
   coefficients are not at the unique L2 optimum and are somewhat solver-path dependent. This is
   a direct consequence of the paper's own `add_scaling: false` applied to raw unscaled
   layernorm activations, so the authors' probes carry the same property.

The reason it is still safe to use: the accuracy profile reproduces the paper's qualitative
signature (monotone rise with depth, peak at layer 16, binary user/assistant at 0.97, 5-role at
0.72 against 0.2 chance). A broken probe or a solver that landed somewhere useless would read
~0.2. And the userness direction is highly over-determined — see §6.4, where probes fit
independently on different label sets and different training rows agree to cos ≈ 0.98. Any
sprint conclusion that flipped between this probe and the authors' would also flip between
theirs and a re-run of their own notebook on a fresh data sample.

**When to revisit:** if a headline result hinges on the exact coefficient values, or if you need
to quote paper-matching accuracies in a writeup.

---

## 2. Deliverables

| File | Size | SHA256 |
|---|---|---|
| `role_probes.pkl` | 7.5 MB | `e0480e398788ed32717dcb7a7c0d50936993d7fb4d37912cae9759b267108b52` |
| `gptoss-20b.pkl` | 5.8 MB | `f48fbb8659df0bddec22307c2cf3eba6c0fd28ff0be8219765a7a048a46caeff` |
| `acc_by_role_gptoss-20b.csv` | 73 KB | `6e2884bfbf264a2fdf53ff63a32dd8a48928bbf1b01d7079f6fb8e112bed5aa3` |
| `acc_by_pos_gptoss-20b.csv` | 5.1 MB | `70f2b70f17c96a7e57db4736c576fff7d4f3260a282f069b68d8e1d566ac6644` |

Also included: `02-train-role-probes-EXECUTED.ipynb` (the run with all outputs) and
`run_nb.log` (full console log).

- `gptoss-20b.pkl` is the exact paper artifact, written by cell 20 unmodified. It embeds cuML
  estimators and **cannot** be unpickled without cuML (verified: `ModuleNotFoundError: No module
  named 'cuml'`). Archive only.
- `role_probes.pkl` is the portable sklearn clone — this is the one the 4090 loads.

**96 probes = 8 role spaces × 12 even layers (0, 2, …, 22).**

---

## 3. Environment: what was built, and every deviation

Built with the repo's `setup_python.sh`, with `PROJECT_DIR` repointed at the clone. Everything
the script pins installed cleanly and was kept: torch 2.9.1+cu128, transformers 4.57.5,
flash-attn 2.8.3+cu128torch2.9, kernels 0.11.5, accelerate 1.12.0, triton 3.5.1.

The script then **failed** at the RAPIDS step, requiring four fixes. Nothing below alters the
probe's definition, hyperparameters, feature, or estimator.

### 3.1 RAPIDS `25.9.*` does not exist → installed 25.08

The script requests `cudf-cu12==25.9.*` / `cuml-cu12==25.9.*`. RAPIDS ships on a bimonthly
cadence (25.02 / 25.04 / 25.06 / 25.08 / 25.10 / 25.12); there is no 25.09 release, and
`pypi.nvidia.com` carries only `25.8.0` and `25.10.0` for both packages. That pin could only ever
have matched a nightly that has since been garbage-collected, so **no faithful option existed.**

I chose **25.08** because the script's own neighbouring pins are the 25.08-era ones
(`libucx-cu12==1.18.1`, `ucx-py-cu12==0.45.0`); the resolver corroborated this by pulling
`ucxx-cu12==0.45.1`. Same estimator, same L-BFGS/QN solver family. This is the only change that
touches the numerics of the fit — see §6.3.

### 3.2 scikit-learn pinned to 1.7.2

The script leaves scikit-learn unpinned, so it pulled 1.9.0, and cuML 25.08 then fails at
*import*: `AttributeError: type object 'BaseEstimator' has no attribute '_get_default_requests'`
(removed in recent sklearn). In this notebook sklearn is used only as
`sklearn.pipeline.Pipeline`, a passthrough wrapper around the cuML estimator, so the pin has no
numerical effect on training. It does set the pickle format for `role_probes.pkl`.

### 3.3 Added `zstandard`

Dolma3 is zstd-compressed JSONL. Without this package cell 8 dies with
`ValueError: Compression type zstd not supported` from fsspec. Pure I/O codec; no effect on
values. Missing from the paper's setup script.

### 3.4 pandas downgraded 3.0.5 → 2.3.3

Not a choice — cuDF pins `pandas<2.4`. This is almost certainly *closer* to the authors' env,
since this notebook is pandas-2-era code and pandas 3.0 changed default string dtypes and
copy-on-write semantics. Validated by the notebook's own consistency check (§5).

### 3.5 Skipped, non-blocking

`plotly_get_chrome` and the apt libs installed fine and are irrelevant to cells 0–20 (plotting
is only used after cell 20).

### 3.6 Final versions

```
python 3.12   torch 2.9.1+cu128   transformers 4.57.5   flash-attn 2.8.3+cu128torch2.9
kernels 0.11.5   cuml 25.08.00   cudf 25.08.00   cupy 14.2.0   scikit-learn 1.7.2
numpy 2.2.6   pandas 2.3.3   datasets 5.0.1   zstandard 0.25.0
```

---

## 4. What was executed

`02-train-role-probes.ipynb` cells 0–20, then one appended cell for the portable clone, all in a
**single `role-analysis-uv` kernel** (so `all_probes` was live in memory for the export), cwd =
`experiments/role-analysis/`.

Driven cell-by-cell via `nbclient` rather than clicking through Jupyter Lab — same kernel, same
order, same `display()` semantics, but the run is logged and non-interactive. Saved as
`02-train-role-probes-EXECUTED.ipynb`.

**The only edit to cells 0–20 was `ws`**, verified by diffing every cell's source against the
original:

```
--- cell 1 differs ---
    -ws = '/workspace/deliberative-alignment-jailbreaks'
    +ws = '/workspace/prompt-injection-as-role-confusion'
cells 0-20 differing: 1
```

Everything from cell 21 onward ("Validate on real conversations", alt-model convs, tomato) was
dropped — it depends on notebook 01 outputs, which were out of scope.

Config used, unmodified: `n_sample_size: 250`, `seq_len: 1024`, `C: 5.0e-3`,
`add_scaling: false`, `nested_reasoning: false`, `train_prefixes: [""]`.

**Two corrections to the portable-export snippet** (the version in the brief would have produced
a broken artifact):

- `np.asarray(clf.coef_)` raises `TypeError` — cuML exposes `coef_`/`intercept_` as **cupy**
  arrays, which refuse implicit host conversion. Used `.get()`.
- `n_classes, n_features = coef.shape` is wrong for binary role spaces: cuML (like sklearn)
  stores **one** coefficient row for two classes, so `classes_` would have been `[0]` and the 12
  `(user, assistant)` probes would have been silently invalid. Now set to 2 classes whenever
  there is a single coef row. The 5-role probes were unaffected.

---

## 5. Results and sanity gates

All of the brief's gates pass.

| Gate | Result |
|---|---|
| FA3 loaded (no eager fallback) | `kernels-community/vllm-flash-attn3`, MXFP4 experts |
| Custom forward == HF logits | `torch.equal` passed, LM loss 3.984 |
| Sequences | 1245 (249 docs × 5 roles), max_seqlen 1035 |
| Role token counts | **141,078 for every one of the 5 roles** (705,390 total) |
| `len(all_probes)` | **96** |
| 5-role layer 12 acc | **0.595** vs 0.200 chance (~3×) |
| Clone fidelity | `max |p_cuml − p_sklearn| = 1.9e-6` over all 96 probes |
| Clean-room load (no cuml/cupy) | PASS |

Exactly-equal role counts are the notebook's own stated correctness condition for role labelling
("should be exactly equal for most models"), which is the strongest available evidence that the
pandas downgrade and the tokenizer path behaved correctly.

Held-out accuracy by layer and role space (chance = 1/|role space|):

```
role_space  s,u,a  s,u,a,t  s,u,c,a  s,u,c,a,t   u,a  u,a,t  u,c,a  u,c,a,t
layer_ix
0            0.23     0.19     0.16       0.14  0.46   0.29   0.25     0.19
2            0.35     0.36     0.26       0.28  0.71   0.51   0.43     0.36
4            0.40     0.38     0.31       0.30  0.65   0.54   0.47     0.38
6            0.59     0.63     0.59       0.51  0.85   0.74   0.69     0.61
8            0.73     0.82     0.81       0.70  0.92   0.88   0.85     0.80
10           0.69     0.76     0.74       0.62  0.89   0.82   0.79     0.72
12           0.70     0.72     0.70       0.59  0.88   0.80   0.77     0.67
14           0.77     0.79     0.80       0.64  0.92   0.85   0.83     0.75
16           0.88     0.87     0.88       0.72  0.97   0.91   0.90     0.82
18           0.88     0.84     0.85       0.71  0.96   0.90   0.89     0.79
20           0.86     0.80     0.82       0.67  0.96   0.87   0.87     0.75
22           0.84     0.82     0.83       0.67  0.96   0.86   0.86     0.75
```

**Every role space peaks at layer 16.** Prefer layer 16 over layer 12 for projections.

Layer 0 sits at or below chance. This is expected, not a bug: the pre-MLP state at layer 0 is
essentially the token embedding of content text that is *identical* across all five roles, so
there is no role signal to recover there. Per-role accuracy (`acc_by_role_gptoss-20b.csv`) shows
`cot` (~0.6–0.73) and `tool` (~0.58–0.64) as the confusable pair, which is why the 5-role space
trails the binary one.

**Runtime** after env + weights: **18 min** total (cell 14 forward passes 210 s; cell 19, all 96
probes, 803 s). Peak 30 GB VRAM, ~72 GB RSS.

---

## 6. Fidelity analysis: would this act differently than the authors' probe?

### 6.1 Bit-identical to the paper

- Probe training code (only `ws` changed).
- All hyperparameters, from the repo's own config: `C=5e-3`, L2 penalty, `max_iter=5000`,
  `linesearch_max_iter=100`, `fit_intercept=True`, `add_scaling=False`.
- Feature: pre-MLP = `post_attention_layernorm` output, bf16 — the same tensor `src/local.py`
  reads.
- Layer set `range(0, 24, 2)`, all 8 role spaces, `seed=123`.
- Model `openai/gpt-oss-20b`, snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`, MXFP4 experts,
  FA3 attention, custom forward verified equal to HF logits.
- `transformers 4.57.5`, `torch 2.9.1+cu128` — exactly the pinned versions.
- The real cuML L2 logistic regression was used for fitting; sklearn appears only in the
  after-the-fact export.

### 6.2 Differs, but cannot move a coefficient

`scikit-learn 1.7.2` (Pipeline wrapper only), `zstandard` (I/O codec), `pandas 2.3.3`
(dataframe plumbing, validated by exact role-count equality), and the float32→float64 cast in
the clone (1.9e-6).

### 6.3 Differs and *could* move coefficients

**cuML 25.08 vs the authors' build.** For a strictly convex L2 logistic regression the optimum
is unique, so any converged solver lands in the same place and the version would be irrelevant.
But the solver did **not** cleanly converge:

```
L-BFGS line search failed (code 1)         54
L-BFGS stopped, line search failed to advance   23
total warnings                             77   (of 96 probes)
```

So the fits stopped early, at a solver-path-dependent point, and a different cuML build could
stop somewhere slightly different. This is the one place a real difference can enter. Note this
follows from the paper's own `add_scaling: false` on raw unscaled layernorm activations
(a poorly conditioned problem), so the authors hit the same failure mode.

**Training data sample.** `datasets 5.0.1` (unpinned) and the un-pinned `allenai/c4` revision
mean the 62 C4 docs, and the streaming shuffle order, may differ from the authors'. Dolma3 is
pinned. Resolved fingerprints for this run:
`c4/en → 1588ec454efa1a09f29cd18ddd04fe05fc8653a2`,
`dolma3_mix-150B-1025 → 3a8349c2f7946cdc56f8ccf22c555672be0b3208`.

**FA3 kernel build.** `kernels-community/vllm-flash-attn3` is fetched from the Hub at load time
and is not version-pinned by the setup script, so attention numerics could differ marginally
from the authors' build. The custom-forward check proves self-consistency with the *same* kernel,
not equality to *their* kernel.

### 6.4 Empirical evidence the userness direction is robust

Cheap but informative check on the shipped artifact: the 8 role spaces were fit **independently**
— different label sets, different training rows (role filtering changes which tokens are
included), and different L-BFGS trajectories with different stopping points. If the userness
direction were a solver artifact, they would disagree. Cosine similarity of
`coef[user] − coef.mean(0)` against the 5-role probe at the same layer:

```
 L         ua      sua      uat      uca     suat     suca     ucat
 8      0.847    0.936    0.956    0.936    0.982    0.976    0.968
12      0.866    0.946    0.969    0.950    0.986    0.981    0.974
16      0.748    0.928    0.934    0.904    0.978    0.980    0.954
22      0.747    0.930    0.935    0.912    0.981    0.982    0.955

layers 8-22, all pairs: mean 0.933, min 0.707
```

The nearest role spaces (`suat`, `suca` — one role different) agree to **0.975–0.986**. The
userness direction is over-determined by the model's geometry rather than pinned down by any one
fit, which is the main reason the §6.3 differences are unlikely to matter in practice.

### 6.5 Bottom line

Materially the same probe for directional work (userness / userness−toolness projection,
layer sweeps, relative comparisons across conv types). Not guaranteed coefficient-identical, and
the accuracy figures should be reported as a reproduction rather than as the paper's numbers.

---

## 7. Using `role_probes.pkl`

Verified to load with **no cuML and no cupy installed**. Pickled with scikit-learn 1.7.2 /
numpy 2.5.2; only `coef_`, `intercept_`, `classes_`, `n_features_in_` are set, so if a future
sklearn refuses the pickle you can rebuild the estimator from those four fields.

List of 96 dicts, each with:

| Key | Value |
|---|---|
| `probe` | `sklearn.linear_model.LogisticRegression`, `coef_` `(n_classes, 2880)` |
| `acc`, `nll` | held-out accuracy / log loss from the cuML fit |
| `layer_ix` | 0, 2, …, 22 |
| `role_space` | e.g. `['system','user','cot','assistant','tool']` |
| `roles_map` | role → class index, e.g. `{'system':0,'user':1,'cot':2,'assistant':3,'tool':4}` |
| `n_inputs` | tokens used (705,390 for the 5-role space) |
| `C` | `5.0e-3` |
| `feature` | `"pre_mlp"` (i.e. `post_attention_layernorm`) |

The `load_vectors()` pattern is confirmed working for every even layer 0–22:

```python
FIVE = ["system", "user", "cot", "assistant", "tool"]
p = next(x for x in probes if x["layer_ix"] == L and list(x["role_space"]) == FIVE)
coef = p["probe"].coef_
user, tool = p["roles_map"]["user"], p["roles_map"]["tool"]
userness          = coef[user] - coef.mean(axis=0)
userness_toolness = coef[user] - coef[tool]
```

Binary role spaces (`user, assistant`) carry `coef_` of shape `(1, 2880)` with
`classes_ == [0, 1]`, per sklearn convention — take `coef_[0]` and sign it by `roles_map`, not
`coef_[user]`.

Feature extraction must match: pre-MLP is the output of `post_attention_layernorm`, i.e. after
the attention residual add and the second layernorm, **before** the MoE block. See
`utils/pretrained_models/gptoss.py:62`.

---

## 8. Corrections to the run brief's premises

- **The activation cube never touches VRAM.** `utils/pretrained_models/gptoss.py:62` already
  calls `.detach().cpu()` on each layer's pre-MLP states, so `run_and_export_states` accumulates
  in RAM by design. The sanctioned `.cpu()` patch was therefore unnecessary and **cell 14 ran
  exactly as written**. Observed peak was 30 GB VRAM against 143 GB available.
- **An 80 GB H100 would have been sufficient** (Hopper is still required, for FA3). The H200 was
  not needed for capacity.
- **~705K content tokens, not ~1.3M.** Most C4/Dolma docs are shorter than the 1024-token cap.
  The cube is ~49 GB bf16, not 80+ GB.
- **249 documents, not 250** — `int(250*.25)=62` C4 plus `int(250*.75)=187` Dolma3. Inherent to
  the code; the authors got 249 too.
- The portable-export snippet has two bugs; see §4.

---

## 9. Reproducing this run

```bash
git clone https://github.com/role-confusion/prompt-injection-as-role-confusion.git
cd prompt-injection-as-role-confusion
sed -i 's|PROJECT_DIR="/workspace/deliberative-alignment-jailbreaks"|PROJECT_DIR="'"$PWD"'"|' setup_python.sh
bash setup_python.sh            # expect failure at the RAPIDS step

source .venv/bin/activate
uv pip install --extra-index-url https://pypi.nvidia.com "cudf-cu12==25.8.*" "cuml-cu12==25.8.*"
uv pip install "scikit-learn==1.7.2" zstandard
uv pip install jupyterlab jupyter_server ipykernel ipywidgets nbformat notebook
python -m ipykernel install --user --name role-analysis-uv --display-name "Role analysis (uv)"
python -c "import sysconfig,pathlib;p=pathlib.Path(sysconfig.get_paths()['purelib'])/'add_path_analysis.pth';p.write_text('$PWD\n')"

mkdir -p /workspace/hf && hf download openai/gpt-oss-20b --cache-dir /workspace/hf
mkdir -p experiments/role-analysis/outputs/{probes,probe-training}
# set `ws` in cell 1, then run cells 0-20 in one kernel, cwd = experiments/role-analysis/
```

`cache_dir='/workspace/hf'` is hardcoded in `utils/loader.py`; the weights must land there
regardless of `HF_HOME`.
