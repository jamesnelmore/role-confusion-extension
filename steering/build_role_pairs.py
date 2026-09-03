"""Build contrastive role pairs for a role-steering vector (Ye et al. Fig. 5).

The paper trains role probes on identical neutral content wrapped in different
role tags. For steering we want the single contrast that drives the attack:

    the SAME text rendered as USER (trusted instruction)  vs
    the SAME text rendered as TOOL (untrusted data)

The mean activation difference between these is a "toolness − userness"
direction. Adding it to injected tool content should push the model to perceive
that content as data, not instructions — the intended mitigation.

Neutral content: short pretraining-style snippets (paper uses C4/DOLMA). We use
a neutral corpus so the direction encodes ROLE, not topic. Rendering uses the
paper's own harmony templates (utils/role_templates.render_single_gptoss),
copied here so this file is standalone on the GPU box.
"""

import argparse
import json
import pathlib
import random


def render_single_gptoss(role, content, tool_name=None):
    """Harmony message wrapper — verbatim from the paper's role_templates.py."""
    if role in ("system", "developer", "user"):
        header = f"{role}<|message|>"
    elif role == "cot":
        header = "assistant<|channel|>analysis<|message|>"
    elif role == "assistant":
        header = "assistant<|channel|>final<|message|>"
    elif role == "tool":
        header = f"functions.{tool_name or ''} to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError("invalid role")
    return f"<|start|>{header}{content}<|end|>"


# Test prefix (system) from probe.yaml, so pairs sit in a realistic context.
GPTOSS_PREFIX = (
    "<|startoftext|><|start|>system<|message|>You are ChatGPT, a large language "
    "model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: "
    "2025-12-31\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, "
    "final. Channel must be included for every message.<|end|>"
)


def neutral_snippets(n, seed=1234):
    """Neutral pretraining-style text. Tries C4 via HF datasets; falls back to a
    bundled list so this runs even offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        out = []
        for ex in ds.shuffle(seed=seed, buffer_size=1000):
            t = ex["text"].strip().replace("\n", " ")
            if 40 <= len(t) <= 600:
                out.append(t[:600])
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as e:
        print("  (C4 unavailable, using bundled snippets):", str(e)[:80])
    base = [
        "The city council approved the new zoning plan after months of debate.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The recipe calls for two cups of flour and a pinch of salt.",
        "Quarterly revenue rose three percent on strong overseas demand.",
        "The museum's new wing houses artifacts from the Bronze Age.",
        "Migratory birds navigate using the Earth's magnetic field.",
        "The bridge was closed for repairs following the winter storms.",
        "Researchers observed the reaction under a range of temperatures.",
        "The novel follows three generations of a fishing family.",
        "Local farmers reported a strong harvest despite the dry spring.",
    ]
    random.seed(seed)
    return [random.choice(base) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=250, help="pairs (paper: 250)")
    ap.add_argument("--out", default="steering/role_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    snippets = neutral_snippets(args.n, args.seed)
    rows = []
    for i, text in enumerate(snippets):
        rows.append({
            "id": i, "content": text,
            "user_rendered": GPTOSS_PREFIX + render_single_gptoss("user", text),
            "tool_rendered": GPTOSS_PREFIX + render_single_gptoss("tool", text),
        })
    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} user/tool role pairs to {out}")


if __name__ == "__main__":
    main()
