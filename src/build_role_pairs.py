"""Build contrastive user/tool role pairs for steering vector extraction.

Identical neutral text rendered as user vs tool using the paper's harmony
templates. The activation difference captures role encoding, not content.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# Import the paper's own template renderer via submodule.
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "role-confusion"))
from utils.role_templates import render_single_gptoss  # noqa: E402

GPTOSS_PREFIX = (
    "<|startoftext|><|start|>system<|message|>You are ChatGPT, a large language "
    "model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: "
    "2025-12-31\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, "
    "final. Channel must be included for every message.<|end|>"
)

FALLBACK_SNIPPETS = [
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


def neutral_snippets(n: int, seed: int = 1234) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        out = []
        for ex in ds.shuffle(seed=seed, buffer_size=1000):
            t = ex["text"].strip().replace("\n", " ")
            if 40 <= len(t) <= 600:
                out.append(t[:600])
            if len(out) >= n:
                return out
    except Exception as e:
        print(f"  (C4 unavailable, using fallback): {str(e)[:80]}")
    random.seed(seed)
    return [random.choice(FALLBACK_SNIPPETS) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=250)
    ap.add_argument("--out", default=str(ROOT / "data" / "role_pairs.jsonl"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    snippets = neutral_snippets(args.n, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for i, text in enumerate(snippets):
            f.write(json.dumps({
                "id": i, "content": text,
                "user_rendered": GPTOSS_PREFIX + render_single_gptoss("user", text),
                "tool_rendered": GPTOSS_PREFIX + render_single_gptoss("tool", text),
            }) + "\n")
    print(f"Wrote {len(snippets)} pairs to {out}")


if __name__ == "__main__":
    main()
