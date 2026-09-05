"""Run the agent-injection eval.

Single cell:
  uv run python src/run.py --backend local --project off --variant clean --limit 24
  uv run python src/run.py --backend local --project userness --layers 12 --variant base-injection

Matched cells (same article idxs):
  uv run python src/run.py --backend openrouter --four-cell --off-only
  uv run python src/run.py --backend local --four-cell --layers 12 --limit 24
  uv run python src/run.py --backend local --four-cell --layer-sweep --limit 48
  # logs/proj/{L12,L12plus,L16,L16plus}/{clean_project,injected_project}

Pairwise quality judge (one Inspect sample per article; sees clean article + two summaries):
  uv run python src/pairwise.py --pair capabilities --log-root logs/four_cell
  uv run python src/pairwise.py --pair dirty --log-root logs/four_cell
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
load_dotenv(_SRC.parent / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CELLS: list[tuple[str, str, str]] = [
    ("clean", "off", "clean_off"),
    ("clean", "on", "clean_project"),
    ("base-injection", "off", "injected_off"),
    ("base-injection", "on", "injected_project"),
]
EVEN_LAYERS = list(range(0, 24, 2))
# Paper TEST_LAYER_IX = mid even layer (12). Probe acc peaks at 16.
KEY_LAYERS = (12, 16)


def layers_plus(start: int) -> str:
    return ",".join(str(L) for L in EVEN_LAYERS if L >= start)


def all_article_idxs(min_idx: int = 0) -> list[int]:
    pages = pathlib.Path(__file__).resolve().parent.parent / "data" / "pages.jsonl"
    seen: set[int] = set()
    for line in pages.read_text().splitlines():
        if line.strip():
            seen.add(int(json.loads(line)["idx"]))
    return [i for i in sorted(seen) if i >= min_idx]


def article_idxs(sample_id: str | None, min_idx: int, limit: int | None) -> list[int]:
    """Article indices shared across clean and injected sample ids."""
    if sample_id:
        seen: set[int] = set()
        out: list[int] = []
        for tok in sample_id.split(","):
            tok = tok.strip()
            if not tok:
                continue
            idx = int(tok.split("__")[0])
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out
    if limit is None:
        return all_article_idxs(min_idx)
    return list(range(min_idx, min_idx + limit))


def sample_ids_for(idxs: list[int], variant: str) -> list[str]:
    return [f"{i:03d}__{variant}" for i in idxs]


def run_one(
    *,
    backend: str,
    project: str,
    variant: str | None,
    layers: str | None,
    probes: str,
    model: str,
    temperature: float,
    log_dir: str,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
    max_connections: int | None = None,
) -> None:
    from eval import role_confusion
    from inspect_ai import eval as inspect_eval

    kw: dict = dict(log_dir=log_dir, sandbox_cleanup=False, temperature=temperature)
    if sample_ids:
        kw["sample_id"] = sample_ids
    elif limit is not None:
        kw["limit"] = limit

    if backend == "openrouter":
        kw.update(model=f"openrouter/{model}", max_connections=max_connections or 2)
    else:
        import local as _local  # noqa: F401
        kw.update(
            model=f"steered-hf/{model}",
            max_connections=1,
            model_args={
                "project": project,
                "layers": layers or "",
                "probes": probes,
                "hf_model": model,
            },
        )
    inspect_eval(role_confusion(variant=variant), **kw)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_four_cell(args: argparse.Namespace) -> None:
    on = args.project if args.project != "off" else "userness"
    idxs = article_idxs(args.sample_id, args.min_idx, args.limit)
    root = args.log_dir or "logs/four_cell"
    cells = list(CELLS)
    if args.off_only:
        cells = [c for c in cells if c[1] == "off"]
    elif getattr(args, "project_only", False):
        cells = [c for c in cells if c[1] == "on"]
    if args.variant and args.variant != "both":
        cells = [c for c in cells if c[0] == args.variant]
    print(f"[four-cell] articles {idxs[0]}..{idxs[-1]} (n={len(idxs)})  "
          f"off_only={args.off_only} project_only={getattr(args, 'project_only', False)} "
          f"project={on} layers={args.layers}  logs={root}",
          flush=True)
    for variant, arm, sub in cells:
        project = "off" if arm == "off" else on
        ids = sample_ids_for(idxs, variant)
        log_dir = str(pathlib.Path(root) / sub)
        print(f"\n=== {sub}: {variant}  project={project}  n={len(ids)} ===", flush=True)
        run_one(
            backend=args.backend,
            project=project,
            variant=variant,
            layers=args.layers,
            probes=args.probes,
            model=args.model,
            temperature=args.temperature,
            log_dir=log_dir,
            sample_ids=ids,
            max_connections=args.max_connections,
        )
        _free_gpu()


def run_layer_sweep(args: argparse.Namespace) -> None:
    """Userness at each key layer, then that layer + every even layer past it."""
    root = args.log_dir or "logs/proj"
    keys = tuple(int(x) for x in args.sweep_layers.split(",") if x.strip())
    args.project_only = True
    if args.project == "off":
        args.project = "userness"
    for L in keys:
        for name, layers in ((f"L{L}", str(L)), (f"L{L}plus", layers_plus(L))):
            ns = argparse.Namespace(**vars(args))
            ns.layers = layers
            ns.log_dir = str(pathlib.Path(root) / name)
            ns.off_only = False
            ns.project_only = True
            print(f"\n######## sweep {name}  layers={layers}  ########", flush=True)
            run_four_cell(ns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("openrouter", "local"), required=True)
    ap.add_argument("--project", choices=("off", "userness", "userness-toolness"), default="off")
    ap.add_argument("--layers", default=None, help="required when projecting; unused with --off-only")
    ap.add_argument("--probes", default="data/role_probes.pkl")
    ap.add_argument("--variant", default="base-injection", choices=("base-injection", "clean", "both"))
    ap.add_argument(
        "--limit", type=int, default=None,
        help="sample count, or article count with --four-cell (default: all articles)",
    )
    ap.add_argument("--min-idx", type=int, default=0, help="first article idx for --four-cell")
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-connections", type=int, default=None,
                    help="Inspect max_connections (default: 2 OpenRouter / 1 local)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument(
        "--four-cell", action="store_true",
        help="matched cells on the same article idxs",
    )
    ap.add_argument(
        "--off-only", action="store_true",
        help="with --four-cell: only clean_off + injected_off (no probes)",
    )
    ap.add_argument(
        "--project-only", action="store_true",
        help="with --four-cell: only clean_project + injected_project",
    )
    ap.add_argument(
        "--layer-sweep", action="store_true",
        help="project-only userness at key layers and each + later even layers",
    )
    ap.add_argument(
        "--sweep-layers", default="12,16",
        help="key layers for --layer-sweep (default: paper mid-layer 12, peak 16)",
    )
    args = ap.parse_args()

    if args.off_only and not args.four_cell:
        ap.error("--off-only requires --four-cell")
    if args.project_only and not args.four_cell:
        ap.error("--project-only requires --four-cell")
    if args.off_only and args.project_only:
        ap.error("pick one of --off-only / --project-only")
    if args.layer_sweep:
        if args.backend != "local":
            ap.error("--layer-sweep needs --backend local")
        run_layer_sweep(args)
        return
    if args.backend == "openrouter" and (args.project != "off" or (args.four_cell and not args.off_only)):
        ap.error("projection needs --backend local")
    if args.four_cell:
        if not args.off_only and not args.layers:
            ap.error("--layers is required unless --off-only")
        run_four_cell(args)
        return
    if args.project != "off" and not args.layers:
        ap.error("--layers is required when projecting")

    variant = None if args.variant == "both" else args.variant
    sample_ids = [x.strip() for x in args.sample_id.split(",") if x.strip()] if args.sample_id else None
    run_one(
        backend=args.backend,
        project=args.project,
        variant=variant,
        layers=args.layers,
        probes=args.probes,
        model=args.model,
        temperature=args.temperature,
        log_dir=args.log_dir or _log_dir(args),
        sample_ids=sample_ids,
        limit=None if sample_ids else args.limit,
        max_connections=args.max_connections,
    )


def _log_dir(args: argparse.Namespace) -> str:
    if args.backend == "openrouter":
        return "logs/openrouter"
    if args.project == "off":
        return f"logs/local_off_{args.variant}"
    return f"logs/local_{args.project.replace('-', '_')}"


if __name__ == "__main__":
    main()
