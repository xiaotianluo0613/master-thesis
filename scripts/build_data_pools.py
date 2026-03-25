#!/usr/bin/env python3
"""
Build layered train pools + test pool from comprehensive volume fingerprints.

User rules implemented:
- Base filtering:
  - Exclude document types: Register/Registers, Other, Unknown
  - Exclude low quality: avg_pc_score < 0.95, blank_page_ratio > 0.1
- Layered train pools (time-restricted to train period):
  1) Court_Book, Court_Records, Reports
  2) Protocols, District
  3) Legal
  4) City
- Test pool (time-restricted to test period): all remaining eligible types,
  but only "seen types" (types present in train period) to satisfy
  seen-type / unseen-time evaluation.

Outputs:
- output/data_pools/pool_summary.json
- output/data_pools/{pool_name}.txt (one volume_id per line)
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def safe_int(v: str) -> Optional[int]:
    try:
        if v in (None, ""):
            return None
        return int(float(v))
    except Exception:
        return None


def normalize_type(doc_type: str) -> str:
    t = (doc_type or "").strip()
    # Keep canonical labels as in fingerprints where possible
    mapping = {
        "court book": "Court_Book",
        "court_book": "Court_Book",
        "court records": "Court_Records",
        "court_records": "Court_Records",
        "police report": "Reports",
        "police reports": "Reports",
        "police_report": "Reports",
        "police_reports": "Reports",
        "report": "Reports",
        "reports": "Reports",
        "protocol": "Protocols",
        "protocols": "Protocols",
        "district": "District",
        "legal": "Legal",
        "city": "City",
        "register": "Registers",
        "registers": "Registers",
        "unknown": "Unknown",
        "other": "Other",
    }
    key = t.lower().replace('-', ' ').replace('/', ' ').strip()
    return mapping.get(key, t)


def choose_temporal_cutoff(years: List[int], min_test_ratio: float = 0.15, max_test_ratio: float = 0.35) -> int:
    """Pick a cutoff year so that test is a reasonable tail period."""
    uniq = sorted(set(years))
    n = len(years)

    best_cut = uniq[len(uniq) // 2]
    best_score = 10**9

    for c in uniq:
        test_n = sum(1 for y in years if y > c)
        ratio = test_n / n if n else 0
        # Score: prefer inside ratio band and close to 0.2
        penalty = 0
        if ratio < min_test_ratio:
            penalty += (min_test_ratio - ratio) * 10
        if ratio > max_test_ratio:
            penalty += (ratio - max_test_ratio) * 10
        score = penalty + abs(ratio - 0.2)
        if test_n > 0 and score < best_score:
            best_score = score
            best_cut = c

    return best_cut


def main():
    parser = argparse.ArgumentParser(description="Build layered data pools with temporal separation")
    parser.add_argument("--fingerprints", default="output/comprehensive_volume_fingerprints.csv")
    parser.add_argument("--outdir", default="output/data_pools")
    parser.add_argument("--pc-min", type=float, default=0.95)
    parser.add_argument("--blank-max", type=float, default=0.1)
    parser.add_argument("--year-min", type=int, default=1600)
    parser.add_argument("--year-max", type=int, default=1910)
    parser.add_argument("--cutoff-year", type=int, default=0,
                        help="Optional manual train/test cutoff year. If 0, auto-select cutoff.")
    args = parser.parse_args()

    fp = Path(args.fingerprints)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    with fp.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Normalize types and parse quality/year
    for r in rows:
        r["document_type"] = normalize_type(r.get("document_type", ""))
        r["avg_pc_score_f"] = safe_float(r.get("avg_pc_score", "0"), 0.0)
        r["blank_page_ratio_f"] = safe_float(r.get("blank_page_ratio", "0"), 0.0)
        r["year_i"] = safe_int(r.get("year", ""))

    excluded_types = {"Registers", "Other", "Unknown"}

    # Base filter
    eligible = [
        r for r in rows
        if r["document_type"] not in excluded_types
        and r["avg_pc_score_f"] >= args.pc_min
        and r["blank_page_ratio_f"] <= args.blank_max
        and r["year_i"] is not None  # needed for strict temporal split
        and args.year_min <= r["year_i"] <= args.year_max
    ]

    if not eligible:
        raise RuntimeError("No eligible rows after filtering. Relax filters or inspect input data.")

    years = [r["year_i"] for r in eligible]
    cutoff = args.cutoff_year if args.cutoff_year > 0 else choose_temporal_cutoff(years)

    train_rows = [r for r in eligible if r["year_i"] <= cutoff]
    test_rows_all = [r for r in eligible if r["year_i"] > cutoff]

    # Seen-type constraint: test should contain types seen in train
    seen_train_types: Set[str] = {r["document_type"] for r in train_rows}
    test_rows = [r for r in test_rows_all if r["document_type"] in seen_train_types]

    # Layer definitions
    layer1_types = {"Court_Book", "Court_Records", "Reports", "Police_Report", "Police_Reports", "Report"}
    layer2_types = {"Protocols", "District"}
    layer3_types = {"Legal"}
    layer4_types = {"City"}

    def ids_for(rows_subset: List[Dict], types: Set[str]) -> List[str]:
        return sorted({r["volume_id"] for r in rows_subset if r["document_type"] in types})

    layer_pools = {
        "train_layer1_pool": ids_for(train_rows, layer1_types),
        "train_layer2_pool": ids_for(train_rows, layer2_types),
        "train_layer3_pool": ids_for(train_rows, layer3_types),
        "train_layer4_pool": ids_for(train_rows, layer4_types),
    }

    # Pick 2 few-shot source volumes per layer from TRAIN pools only.
    # (These are explicitly excluded from test pool for safety.)
    fewshot_by_layer = {
        "layer1": layer_pools["train_layer1_pool"][:2],
        "layer2": layer_pools["train_layer2_pool"][:2],
        "layer3": layer_pools["train_layer3_pool"][:2],
        "layer4": layer_pools["train_layer4_pool"][:2],
    }
    fewshot_ids = sorted({v for lst in fewshot_by_layer.values() for v in lst})

    test_ids = sorted({r["volume_id"] for r in test_rows})
    test_ids = [v for v in test_ids if v not in set(fewshot_ids)]

    pools = {
        **layer_pools,
        "test_pool_all_seen_types": test_ids,
    }

    # Save pool txt files
    for name, vids in pools.items():
        with (outdir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for v in vids:
                f.write(v + "\n")

    # Save few-shot source files (requested: separate files)
    with (outdir / "fewshot_source_volume_ids.txt").open("w", encoding="utf-8") as f:
        for v in fewshot_ids:
            f.write(v + "\n")

    with (outdir / "fewshot_source_by_layer.json").open("w", encoding="utf-8") as f:
        json.dump(fewshot_by_layer, f, ensure_ascii=False, indent=2)

    # Additional diagnostics
    train_type_counts: Dict[str, int] = {}
    test_type_counts: Dict[str, int] = {}
    for r in train_rows:
        train_type_counts[r["document_type"]] = train_type_counts.get(r["document_type"], 0) + 1
    for r in test_rows:
        test_type_counts[r["document_type"]] = test_type_counts.get(r["document_type"], 0) + 1

    summary = {
        "filters": {
            "excluded_types": sorted(excluded_types),
            "avg_pc_score_min": args.pc_min,
            "blank_page_ratio_max": args.blank_max,
            "year_required": True,
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
        "temporal_split": {
            "train_year_max_inclusive": cutoff,
            "test_year_min_exclusive": cutoff,
            "train_year_range": [min(r["year_i"] for r in train_rows), max(r["year_i"] for r in train_rows)] if train_rows else None,
            "test_year_range": [min(r["year_i"] for r in test_rows), max(r["year_i"] for r in test_rows)] if test_rows else None,
            "eligible_total": len(eligible),
            "train_total": len(train_rows),
            "test_total_before_seen_filter": len(test_rows_all),
            "test_total_after_seen_filter": len(test_rows),
        },
        "counts_by_pool": {k: len(v) for k, v in pools.items()},
        "fewshot_source_counts": {k: len(v) for k, v in fewshot_by_layer.items()},
        "fewshot_source_total": len(fewshot_ids),
        "train_type_counts": dict(sorted(train_type_counts.items())),
        "test_type_counts": dict(sorted(test_type_counts.items())),
        "pool_files": {k: str((outdir / f"{k}.txt")) for k in pools.keys()},
        "fewshot_files": {
            "fewshot_source_volume_ids": str(outdir / "fewshot_source_volume_ids.txt"),
            "fewshot_source_by_layer": str(outdir / "fewshot_source_by_layer.json"),
        },
    }

    with (outdir / "pool_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print concise report
    print("=" * 80)
    print("DATA POOL SUMMARY")
    print("=" * 80)
    print(f"Eligible volumes after base filters: {len(eligible)}")
    print(f"Train period: <= {cutoff} | Test period: > {cutoff}")
    print(f"Train volumes: {len(train_rows)} | Test volumes (seen types): {len(test_rows)}")
    print("\nPool counts:")
    for k, v in summary["counts_by_pool"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved summary: {outdir / 'pool_summary.json'}")


if __name__ == "__main__":
    main()
