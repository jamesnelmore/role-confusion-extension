"""Runner for the role-confusion eval.

Registers the custom `steered-hf` provider *before* resolving the model (the
`inspect eval` CLI resolves the model before loading the task file, so the
provider must be imported here first), then runs the task via the Python API.

Examples:
  uv run python src/run_eval.py --mode off     --variant base-injection --limit 1
  uv run python src/run_eval.py --mode project  --variant base-injection
  uv run python src/run_eval.py --mode off      --variant clean          # control
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Reduce allocator fragmentation: projection mode peaks close to the 24 GB
# ceiling during the long-prompt attention softmax. Must be set before CUDA
# initializes (i.e. before torch is imported by the provider below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import steered_hf_provider  # noqa: E402,F401  registers `steered-hf`
from eval import role_confusion  # noqa: E402

from inspect_ai import eval as inspect_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("off", "project"), default="off")
    ap.add_argument("--variant", default="base-injection",
                    choices=("base-injection", "clean", "both"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-id", default=None,
                    help="comma-separated sample ids (e.g. 007__base-injection)")
    ap.add_argument("--layers", default=None, help="comma list; default=suggested")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--message-limit", type=int, default=24)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--hf-model", default="openai/gpt-oss-20b")
    ap.add_argument("--attn-implementation", default=None,
                    help="override HF attn impl (default: flash kernel, then eager)")
    args = ap.parse_args()

    variant = None if args.variant == "both" else args.variant
    model_args: dict = {
        "mode": args.mode,
        "max_new_tokens": args.max_new_tokens,
        "hf_model": args.hf_model,
    }
    if args.layers:
        model_args["layers"] = args.layers
    if args.attn_implementation:
        model_args["attn_implementation"] = args.attn_implementation

    log_dir = args.log_dir or f"logs/{args.variant}_{args.mode}"
    eval_kwargs: dict = dict(
        model=f"steered-hf/{args.hf_model}",
        model_args=model_args,
        max_connections=1,
        log_dir=log_dir,
        sandbox_cleanup=False,
    )
    if args.sample_id:
        eval_kwargs["sample_id"] = [
            x.strip() for x in args.sample_id.split(",") if x.strip()
        ]
    elif args.limit is not None:
        eval_kwargs["limit"] = args.limit

    inspect_eval(
        role_confusion(variant=variant, message_limit=args.message_limit),
        **eval_kwargs,
    )


if __name__ == "__main__":
    main()
