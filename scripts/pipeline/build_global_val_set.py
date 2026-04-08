#!/usr/bin/env python3
"""
Build a global validation set by merging per-layer val files proportionally.

Each layer contributes queries in proportion to its training set size (the weights
you set, e.g. 5000 2500 2500 1500). The per-layer val split already holds out 10%
of each layer's queries — this script samples from those holdouts proportionally
to produce a fixed global benchmark.

Usage — full set once all 4 layers are ready:
    python scripts/pipeline/build_global_val_set.py \
        --layer1-val data/layer1_val_queries.json \
        --layer2-val data/layer2_val_queries.json \
        --layer3-val data/layer3_val_queries.json \
        --layer4-val data/layer4_val_queries.json \
        --layer-weights 5000 2500 2500 1500 \
        --global-val-size 500 \
        --output data/global_val_queries.json

Usage — incremental (missing layers skipped gracefully, weights auto-adjusted):
    python scripts/pipeline/build_global_val_set.py \
        --layer1-val data/layer1_val_queries.json \
        --layer2-val data/layer2_val_queries.json \
        --layer-weights 5000 2500 2500 1500 \
        --global-val-size 500 \
        --output data/global_val_queries.json

If --global-val-size 0 (default): use ALL available val queries from each layer,
weighted proportionally (i.e. cap larger layers so ratios match weights).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ALL_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
DEFAULT_WEIGHTS = [5000, 2500, 2500, 1500]


def load_val_file(path: Path, layer_name: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    for q in queries:
        if "layer" not in q:
            q["layer"] = layer_name
    return queries


def group_queries(queries: list) -> dict:
    """Group queries by their generation group (the 'date' field is a GROUP-* id)."""
    groups = defaultdict(list)
    for q in queries:
        groups[q.get("date", "unknown")].append(q)
    return dict(groups)


def sample_group_aware(queries: list, target: int, seed: int) -> list:
    """
    Sample up to `target` queries from `queries`, keeping generation groups intact.
    If target <= 0 or target >= len(queries), returns all queries.
    """
    if target <= 0 or target >= len(queries):
        return queries

    groups = group_queries(queries)
    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    selected = []
    for key in group_keys:
        if len(selected) >= target:
            break
        selected.extend(groups[key])

    return selected


def compute_quotas(
    weights: list,
    available_counts: dict,
    layer_names: list,
    global_val_size: int,
    seed: int,
) -> dict:
    """
    Compute how many queries to draw from each layer.

    If global_val_size > 0: distribute proportionally by weight, capped by available.
    If global_val_size == 0: use all available from each layer, but cap larger layers
    so the ratios across layers match the weights as closely as possible.
    """
    # Only consider layers that have data
    active = [(name, w) for name, w in zip(layer_names, weights)
              if name in available_counts and available_counts[name] > 0]
    if not active:
        return {}

    active_names, active_weights = zip(*active)
    total_weight = sum(active_weights)

    if global_val_size > 0:
        # Proportional allocation, capped by available
        raw_quotas = {
            name: max(1, round(global_val_size * w / total_weight))
            for name, w in zip(active_names, active_weights)
        }
        # Cap by available
        quotas = {
            name: min(raw_quotas[name], available_counts[name])
            for name in active_names
        }
    else:
        # Use all available, but scale down layers that are over-represented
        # Find the layer whose (available / weight) ratio is smallest — that's the binding constraint
        ratios = {
            name: available_counts[name] / w
            for name, w in zip(active_names, active_weights)
        }
        min_ratio = min(ratios.values())
        quotas = {
            name: min(available_counts[name], max(1, round(min_ratio * w)))
            for name, w in zip(active_names, active_weights)
        }

    return quotas


def main():
    parser = argparse.ArgumentParser(
        description="Build a proportionally-weighted global val set from per-layer val files."
    )
    parser.add_argument("--layer1-val", default="data/layer1_val_queries.json")
    parser.add_argument("--layer2-val", default="data/layer2_val_queries.json")
    parser.add_argument("--layer3-val", default="data/layer3_val_queries.json")
    parser.add_argument("--layer4-val", default="data/layer4_val_queries.json")
    parser.add_argument(
        "--layer-weights",
        type=int,
        nargs=4,
        default=DEFAULT_WEIGHTS,
        metavar=("W1", "W2", "W3", "W4"),
        help=(
            "Training set size for each layer — controls sampling proportions. "
            f"Default: {DEFAULT_WEIGHTS} (matches CLAUDE.md targets)"
        ),
    )
    parser.add_argument(
        "--global-val-size",
        type=int,
        default=0,
        help=(
            "Total target size for the global val set. "
            "0 (default) = use all available val queries, weighted proportionally."
        ),
    )
    parser.add_argument("--output", default="data/global_val_queries.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layer_paths = {
        "layer1": Path(args.layer1_val),
        "layer2": Path(args.layer2_val),
        "layer3": Path(args.layer3_val),
        "layer4": Path(args.layer4_val),
    }
    weights_by_layer = dict(zip(ALL_LAYERS, args.layer_weights))

    # ── 1. Load available layers ──────────────────────────────────────────────
    per_layer_queries = {}
    layers_included = []
    layers_missing = []

    for layer in ALL_LAYERS:
        path = layer_paths[layer]
        if path.exists():
            queries = load_val_file(path, layer)
            per_layer_queries[layer] = queries
            layers_included.append(layer)
            print(f"  Loaded {len(queries):>4} queries  {layer}  ({path})")
        else:
            layers_missing.append(layer)
            print(f"  MISSING  {layer}  ({path}) — skipped")

    if not layers_included:
        print("ERROR: no layer val files found.")
        raise SystemExit(1)

    # ── 2. Compute per-layer quotas ───────────────────────────────────────────
    available_counts = {name: len(qs) for name, qs in per_layer_queries.items()}
    active_weights = [weights_by_layer[name] for name in layers_included]

    quotas = compute_quotas(
        weights=active_weights,
        available_counts=available_counts,
        layer_names=layers_included,
        global_val_size=args.global_val_size,
        seed=args.seed,
    )

    # ── 3. Sample and merge ───────────────────────────────────────────────────
    merged = []
    per_layer_counts = {}

    for layer in layers_included:
        quota = quotas.get(layer, 0)
        sampled = sample_group_aware(per_layer_queries[layer], quota, args.seed)
        per_layer_counts[layer] = len(sampled)
        merged.extend(sampled)

    # ── 4. Write output ───────────────────────────────────────────────────────
    complete = len(layers_missing) == 0
    out = {
        "metadata": {
            "layers_included": layers_included,
            "layers_missing": layers_missing,
            "layer_weights": weights_by_layer,
            "global_val_size_target": args.global_val_size,
            "total_queries": len(merged),
            "per_layer_counts": per_layer_counts,
            "complete": complete,
            "seed": args.seed,
            "note": (
                "Global val set — proportionally sampled from per-layer val splits. "
                "Weights match training set targets. Use as fixed benchmark across all training stages."
            ),
        },
        "queries": merged,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print()
    print("=== Global val set ===")
    print(f"  Layers included : {layers_included}")
    if layers_missing:
        print(f"  Layers missing  : {layers_missing}  ← re-run when ready")
    print(f"  Weights used    : {weights_by_layer}")
    print(f"  Per-layer counts: {per_layer_counts}")
    print(f"  Total queries   : {len(merged)}")
    print(f"  Complete        : {complete}")
    print(f"  Output          : {out_path}")


if __name__ == "__main__":
    main()
