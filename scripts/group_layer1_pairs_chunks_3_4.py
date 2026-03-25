#!/usr/bin/env python3
"""Group layer1 pilot chunks within each volume into groups of size 3-4.

Input: data/layer1_pilot_pairs_550.json (with `pairs` list)
Output: chunk-style JSON compatible with N-to-N generator, where each chunk has:
- date: GROUP-<volume_id>-<index>
- group_id
- sub_chunk_index (position inside group)

Grouping preserves page order within each volume.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def partition_3_4(n: int) -> List[int]:
    """Return a list of group sizes (3/4) summing to n.

    Strategy: use as many 4s as possible while keeping remainder divisible by 3.
    """
    if n < 3:
        raise ValueError(f"Cannot partition n={n} into groups of 3-4")

    for num4 in range(n // 4, -1, -1):
        rem = n - 4 * num4
        if rem % 3 == 0:
            num3 = rem // 3
            sizes = [4] * num4 + [3] * num3
            # make sizes smoother by interleaving 4 and 3 when both exist
            if num4 > 0 and num3 > 0:
                out: List[int] = []
                a = [4] * num4
                b = [3] * num3
                while a or b:
                    if a:
                        out.append(a.pop())
                    if b:
                        out.append(b.pop())
                return out
            return sizes
    raise ValueError(f"No 3/4 partition found for n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group layer1 pilot chunks into 3-4 chunk groups per volume")
    parser.add_argument("--input", default="data/layer1_pilot_pairs_550.json")
    parser.add_argument("--output", default="data/layer1_pilot_pairs_550_grouped_3_4.json")
    parser.add_argument("--summary", default="data/layer1_pilot_pairs_550_grouped_3_4_summary.txt")
    args = parser.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])

    by_volume = {}
    for p in pairs:
        by_volume.setdefault(p["volume_id"], []).append(p)

    # sort each volume by page order
    for vid in by_volume:
        by_volume[vid].sort(key=lambda x: (x.get("page_index", -1), x.get("chunk_id", "")))

    grouped_chunks = []
    lines = []
    total_groups = 0

    for vid in sorted(by_volume.keys()):
        vol = by_volume[vid]
        sizes = partition_3_4(len(vol))
        lines.append(f"Volume {vid}: {len(vol)} chunks -> groups {sizes}")

        start = 0
        for gi, gsz in enumerate(sizes, 1):
            grp = vol[start:start + gsz]
            group_id = f"GROUP-{vid}-{gi:04d}"
            for si, item in enumerate(grp):
                row = {
                    "chunk_id": item["chunk_id"],
                    "volume_id": item["volume_id"],
                    "document_type": item.get("document_type"),
                    "source_file": item.get("source_file"),
                    "page_index": item.get("page_index"),
                    "text": item.get("text", ""),
                    "text_without_prefix": item.get("text_without_prefix", item.get("text", "")),
                    "text_prefix": item.get("text_prefix", ""),
                    "text_with_prefix": item.get("text_with_prefix", item.get("text", "")),
                    "pair_id": item.get("pair_id"),
                    "pair_group": item.get("pair_group"),
                    "group_id": group_id,
                    "date": group_id,
                    "sub_chunk_index": si,
                    "group_size": gsz,
                }
                grouped_chunks.append(row)
            start += gsz
            total_groups += 1

    out = {
        "metadata": {
            "source": str(src),
            "method": "group-by-volume-3-4",
            "total_chunks": len(grouped_chunks),
            "total_groups": total_groups,
        },
        "chunks": grouped_chunks,
    }

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved grouped chunks: {args.output}")
    print(f"Saved summary: {args.summary}")
    print(f"Total chunks: {len(grouped_chunks)} | Total groups: {total_groups}")


if __name__ == "__main__":
    main()
