"""Extract per-layer role-steering vectors from user/tool contrastive pairs.

For each pair, forward-pass the same content rendered as user and tool,
average hidden states over content tokens, take mean(tool - user) per layer.

    v_L = mean_over_pairs( act_tool_L - act_user_L )

Reports per-layer norm and direction consistency to pick layer + alpha.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).parent.parent


def content_token_span(
    tokenizer: AutoTokenizer, rendered: str, content: str
) -> tuple[torch.Tensor, list[int]]:
    start = rendered.index(content)
    end = start + len(content)
    enc = tokenizer(rendered, return_offsets_mapping=True, return_tensors="pt",
                    add_special_tokens=False)
    idxs = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if b > start and a < end and not (a == 0 and b == 0)]
    return enc["input_ids"], idxs


@torch.no_grad()
def mean_content_acts(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
    rendered: str, content: str, device: torch.device,
) -> torch.Tensor:
    input_ids, idxs = content_token_span(tokenizer, rendered, content)
    out = model(input_ids.to(device), output_hidden_states=True)
    hs = out.hidden_states  # (n_layers+1) x (1, seq, hidden)
    return torch.stack([h[0, idxs, :].mean(0).float().cpu() for h in hs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--pairs", default=str(ROOT / "data" / "role_pairs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "steering_vectors.pt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=args.device).eval()

    pairs = [json.loads(l) for l in pathlib.Path(args.pairs).read_text().splitlines()]
    if args.limit:
        pairs = pairs[:args.limit]

    diffs = []
    for i, p in enumerate(pairs):
        a_user = mean_content_acts(model, tok, p["user_rendered"], p["content"], model.device)
        a_tool = mean_content_acts(model, tok, p["tool_rendered"], p["content"], model.device)
        diffs.append(a_tool - a_user)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} pairs")

    diff = torch.stack(diffs)       # (N, L+1, H)
    v = diff.mean(0)                # (L+1, H)
    norms = v.norm(dim=-1)
    unit = v / (norms.unsqueeze(-1) + 1e-8)
    consistency = (diff * unit.unsqueeze(0)).sum(-1).mean(0) / (diff.norm(dim=-1).mean(0) + 1e-8)

    torch.save({"model": args.model,
                "vectors": {L: v[L].clone() for L in range(v.shape[0])},
                "per_layer_norm": norms,
                "direction_consistency": consistency,
                "n_pairs": len(pairs)}, args.out)

    print(f"\n{'layer':>5}  {'|v|':>7}  consistency")
    for L in range(v.shape[0]):
        print(f"{L:5d}  {norms[L]:7.3f}   {consistency[L]:.3f}")
    best = int(consistency[1:].argmax()) + 1
    print(f"\nSuggested steer layer: {best}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
