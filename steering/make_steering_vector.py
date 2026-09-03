"""Compute a per-layer role-steering vector from user/tool contrastive pairs.

Runs on a GPU box (RunPod 4090). For each pair, we render the SAME content as
user and as tool, run a forward pass, and average the residual-stream activation
over the CONTENT tokens (the shared text, excluding role tags). The steering
vector at layer L is:

    v_L = mean_over_pairs( act_tool_L - act_user_L )

i.e. the "toolness − userness" direction. Adding +alpha * v_L to the residual
stream pushes tokens toward being perceived as tool/data; subtracting pushes
toward user/instruction. For the mitigation we add it to injected content.

Saved as steering/steering_vectors.pt: {layer_index: tensor(hidden_dim)}, plus
metadata. A quick built-in sanity check reports the cosine geometry and the
per-layer norm so you can pick a layer + alpha before wiring it into the loop.

Usage (on the box):
    python steering/make_steering_vector.py --model openai/gpt-oss-20b \
        --pairs steering/role_pairs.jsonl --out steering/steering_vectors.pt
"""

import argparse
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def content_token_span(tokenizer, rendered, content):
    """Char offsets of `content` inside `rendered`, mapped to token indices."""
    start = rendered.index(content)
    end = start + len(content)
    enc = tokenizer(rendered, return_offsets_mapping=True, return_tensors="pt",
                    add_special_tokens=False)
    idxs = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
            if b > start and a < end and not (a == 0 and b == 0)]
    return enc["input_ids"], idxs


@torch.no_grad()
def mean_content_acts(model, tokenizer, rendered, content, device):
    input_ids, idxs = content_token_span(tokenizer, rendered, content)
    out = model(input_ids.to(device), output_hidden_states=True)
    # hidden_states: (n_layers+1) x (1, seq, hidden). Average over content tokens.
    hs = out.hidden_states
    return torch.stack([h[0, idxs, :].mean(0).float().cpu() for h in hs])  # (L+1, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--pairs", default="steering/role_pairs.jsonl")
    ap.add_argument("--out", default="steering/steering_vectors.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=args.device)
    model.eval()

    pairs = [json.loads(l) for l in pathlib.Path(args.pairs).read_text().splitlines()]
    if args.limit:
        pairs = pairs[: args.limit]

    diffs = []
    for i, p in enumerate(pairs):
        a_user = mean_content_acts(model, tok, p["user_rendered"], p["content"], model.device)
        a_tool = mean_content_acts(model, tok, p["tool_rendered"], p["content"], model.device)
        diffs.append(a_tool - a_user)   # toolness - userness, per layer
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} pairs")

    diff = torch.stack(diffs)            # (N, L+1, H)
    v = diff.mean(0)                     # (L+1, H)  mean-difference steering vector
    per_layer_norm = v.norm(dim=-1)      # (L+1,)
    # consistency: how aligned are individual pair-diffs with the mean direction
    unit = v / (per_layer_norm.unsqueeze(-1) + 1e-8)
    align = (diff * unit.unsqueeze(0)).sum(-1).mean(0) / (diff.norm(dim=-1).mean(0) + 1e-8)

    vectors = {L: v[L].clone() for L in range(v.shape[0])}
    torch.save({"model": args.model, "vectors": vectors,
                "per_layer_norm": per_layer_norm,
                "direction_consistency": align,
                "n_pairs": len(pairs)}, args.out)

    print("\nlayer  |v|      consistency (1=every pair agrees on the direction)")
    for L in range(v.shape[0]):
        print(f"{L:5d}  {per_layer_norm[L]:7.3f}   {align[L]:.3f}")
    best = int(align[1:].argmax()) + 1
    print(f"\nSuggested steer layer (max consistency): {best}")
    print(f"Saved {v.shape[0]} per-layer vectors to {args.out}")


if __name__ == "__main__":
    main()
