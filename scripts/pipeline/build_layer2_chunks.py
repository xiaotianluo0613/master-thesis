#!/usr/bin/env python3
"""
Build Layer 2 chunk dataset for full-scale training.

Chunk unit: one XML page with metadata prefix.
Sampling: proportional to corpus distribution.
  - District: use ALL available (scarce, only 39 volumes)
  - Protocols: sampled proportionally to fill the target

Usage:
    python scripts/pipeline/build_layer2_chunks.py \
        --target-chunks 2500 \
        --output data/layer2_chunks.json
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import xml.etree.ElementTree as ET

ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v4#}"

TARGET_TYPES = ["Protocols", "District"]
SCARCE_TYPE = "District"


def normalize_type(doc_type: str) -> str:
    t = (doc_type or "").strip()
    mapping = {
        "protocols": "Protocols",
        "protocol": "Protocols",
        "district": "District",
        "district court": "District",
        "district_court": "District",
    }
    key = t.lower().replace('-', ' ').replace('/', ' ').strip()
    return mapping.get(key, t)


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_volume_meta(fingerprints_csv: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with fingerprints_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = r.get("volume_id", "")
            out[vid] = {
                "document_type": normalize_type(r.get("document_type", "")),
                "year": str(r.get("year", "") or "").strip(),
                "volume_title": str(r.get("volume_title", "") or "").strip(),
            }
    return out


def extract_text(xml_path: Path) -> str:
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        lines = []
        for tl in root.findall(f".//{ALTO_NS}TextLine"):
            toks = []
            for s in tl.findall(f"./{ALTO_NS}String"):
                content = (s.attrib.get("CONTENT") or "").strip()
                if content:
                    toks.append(content)
            line = ' '.join(toks)
            if line:
                lines.append(line)
        return '\n'.join(lines)
    except Exception:
        return ""


def parse_page_index(xf: Path) -> int:
    m = re.search(r"(\d+)$", xf.stem)
    return int(m.group(1)) if m else -1


def infer_city(volume_title: str) -> str:
    if not volume_title:
        return "okänd ort"
    candidates = [
        "Stockholm", "Göteborg", "Malmö", "Uppsala", "Lund", "Linköping", "Västerås",
        "Örebro", "Norrköping", "Helsingborg", "Gävle", "Sundsvall", "Umeå", "Luleå",
        "Råneå", "Kalmar", "Laholm", "Kronoberg", "Skövde", "Sköfde"
    ]
    low = volume_title.lower()
    for c in candidates:
        if c.lower() in low:
            return c
    m = re.search(r"[A-Za-zÅÄÖåäö]{4,}", volume_title)
    if m:
        return m.group(0)
    return "okänd ort"


def build_text_prefix(document_type: str, volume_id: str, source_file: str, page_index: int,
                      year: str = "", volume_title: str = "") -> str:
    _ = document_type
    _ = page_index
    y = year if year else "okänt år"
    city = infer_city(volume_title)
    title = volume_title if volume_title else "okänd volymtitel"
    src = Path(source_file).name
    return (
        f"Detta är ett juridiskt dokument från cirka {y} i {city}, Sverige. "
        f"Det kommer från volymtiteln [{title}]. "
        f"Källa: {src} (volym {volume_id})."
    )


def make_proportional_quotas(total: int, available: Dict[str, int]) -> Dict[str, int]:
    """
    Proportional sampling: use all District (scarce), fill the rest from Protocols.
    """
    district_count = min(available.get("District", 0), total)
    remaining = total - district_count

    protocols_avail = available.get("Protocols", 0)
    protocols_quota = min(remaining, protocols_avail)

    return {
        "District": district_count,
        "Protocols": protocols_quota,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer 2 chunks for full-scale training")
    parser.add_argument("--pool", default="output/data_pools/train_layer2_pool.txt")
    parser.add_argument("--fingerprints", default="output/comprehensive_volume_fingerprints.csv")
    parser.add_argument("--transcriptions-root",
                        default="Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions")
    parser.add_argument("--target-chunks", type=int, default=2500)
    parser.add_argument("--min-text-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/layer2_chunks.json")
    args = parser.parse_args()

    random.seed(args.seed)

    pool_ids = read_ids(Path(args.pool))
    vmeta = load_volume_meta(Path(args.fingerprints))
    root = Path(args.transcriptions_root)

    # group volumes by type
    vols_by_type: Dict[str, List[str]] = defaultdict(list)
    for vid in pool_ids:
        t = vmeta.get(vid, {}).get("document_type", "")
        if t in TARGET_TYPES:
            vols_by_type[t].append(vid)

    # Gather candidate chunks per type with early stopping
    cap_per_type = args.target_chunks
    candidates: Dict[str, List[Dict]] = defaultdict(list)
    for t in TARGET_TYPES:
        vols = vols_by_type.get(t, [])
        random.shuffle(vols)
        for vid in vols:
            if len(candidates[t]) >= cap_per_type:
                break
            vdir = root / vid
            if not vdir.exists():
                continue
            for xf in sorted(vdir.glob("*.xml")):
                if len(candidates[t]) >= cap_per_type:
                    break
                txt = extract_text(xf)
                if len(txt) < args.min_text_chars:
                    continue
                page_idx = parse_page_index(xf)
                vm = vmeta.get(vid, {})
                prefix = build_text_prefix(
                    t, vid, str(xf), page_idx,
                    year=vm.get("year", ""),
                    volume_title=vm.get("volume_title", ""),
                )
                candidates[t].append({
                    "document_type": t,
                    "volume_id": vid,
                    "chunk_id": f"{vid}_{xf.stem}",
                    "page_index": page_idx,
                    "source_file": str(xf),
                    "text": txt,
                    "text_without_prefix": txt,
                    "text_prefix": prefix,
                    "text_with_prefix": f"{prefix}\n\n{txt}",
                })

    available = {t: len(candidates[t]) for t in TARGET_TYPES}
    quotas = make_proportional_quotas(args.target_chunks, available)

    print("Available candidates:", available)
    print("Quotas:", quotas)

    chunks = []
    stats = {"available_candidates": available, "selected": {}, "volumes_by_type": {}}

    for t in TARGET_TYPES:
        stats["volumes_by_type"][t] = len(vols_by_type.get(t, []))
        q = quotas[t]
        pool = candidates[t]
        if not pool or q == 0:
            stats["selected"][t] = 0
            continue
        selected = random.sample(pool, min(q, len(pool)))
        for i, item in enumerate(selected):
            item = item.copy()
            item["pair_id"] = f"{t}_pair_{i+1:04d}"
            item["pair_group"] = "layer2"
            chunks.append(item)
        stats["selected"][t] = len(selected)

    # sort by volume then page for deterministic grouping downstream
    chunks.sort(key=lambda x: (x["document_type"], x["volume_id"], x.get("page_index", -1)))

    out = {
        "metadata": {
            "dataset": "layer2_chunks",
            "target_chunks": args.target_chunks,
            "actual_chunks": len(chunks),
            "target_types": TARGET_TYPES,
            "scarce_type": SCARCE_TYPE,
            "quotas": quotas,
            "sampling": "proportional — all District used, Protocols sampled to fill target",
            "min_text_chars": args.min_text_chars,
            "seed": args.seed,
        },
        "stats": stats,
        "chunks": chunks,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Built layer2 chunks")
    print(f"Output: {out_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Selected by type: {stats['selected']}")


if __name__ == "__main__":
    main()
