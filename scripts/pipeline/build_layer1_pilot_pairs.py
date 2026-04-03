#!/usr/bin/env python3
"""
Build a fair layer1 pilot pair dataset (query-target candidate pairs).

Pair unit: one text chunk candidate (typically one XML page) with metadata.
Fairness: near-equal allocation across layer1 target types.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import xml.etree.ElementTree as ET

ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v4#}"


TARGET_TYPES = ["Court_Book", "Court_Records", "Reports"]


def normalize_type(doc_type: str) -> str:
    t = (doc_type or "").strip()
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


def ordinal(n: int) -> str:
    if n <= 0:
        return "unknown"
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def human_doc_type(t: str) -> str:
    mapping = {
        "Court_Book": "court book",
        "Court_Records": "court record",
        "Reports": "police report",
    }
    return mapping.get(t, t.replace("_", " ").lower())


def infer_city(volume_title: str) -> str:
    """Heuristic city/place extraction from title metadata."""
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

    # fallback: first alphabetic token-like word in title
    m = re.search(r"[A-Za-zÅÄÖåäö]{4,}", volume_title)
    if m:
        return m.group(0)

    return "okänd ort"


def build_text_prefix(document_type: str, volume_id: str, source_file: str, page_index: int,
                      year: str = "", volume_title: str = "") -> str:
    _ = document_type  # intentionally ignored in prefix per user request
    _ = page_index  # intentionally ignored in prefix per user request

    y = year if year else "okänt år"
    city = infer_city(volume_title)
    title = volume_title if volume_title else "okänd volymtitel"
    src = Path(source_file).name

    return (
        f"Detta är ett juridiskt dokument från cirka {y} i {city}, Sverige. "
        f"Det kommer från volymtiteln [{title}]. "
        f"Källa: {src} (volym {volume_id})."
    )


def make_quotas(total: int, types: List[str]) -> Dict[str, int]:
    base = total // len(types)
    rem = total % len(types)
    q = {t: base for t in types}
    for i in range(rem):
        q[types[i]] += 1
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fair layer1 pilot pairs")
    parser.add_argument("--pool", default="output/data_pools/train_layer1_pool.txt")
    parser.add_argument("--fingerprints", default="output/comprehensive_volume_fingerprints.csv")
    parser.add_argument("--transcriptions-root", default="Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions")
    parser.add_argument("--target-pairs", type=int, default=550)
    parser.add_argument("--min-text-chars", type=int, default=220)
    parser.add_argument("--output", default="data/layer1_pilot_pairs_550.json")
    args = parser.parse_args()

    pool_ids = read_ids(Path(args.pool))
    vmeta = load_volume_meta(Path(args.fingerprints))
    root = Path(args.transcriptions_root)

    # volumes by target type
    vols_by_type: Dict[str, List[str]] = defaultdict(list)
    for vid in pool_ids:
        t = vmeta.get(vid, {}).get("document_type", "")
        if t in TARGET_TYPES:
            vols_by_type[t].append(vid)

    quotas = make_quotas(args.target_pairs, TARGET_TYPES)

    # Gather all candidate chunks by type
    candidates: Dict[str, List[Dict]] = defaultdict(list)
    for t in TARGET_TYPES:
        for vid in vols_by_type.get(t, []):
            vdir = root / vid
            if not vdir.exists():
                continue
            for xf in sorted(vdir.glob("*.xml")):
                txt = extract_text(xf)
                if len(txt) < args.min_text_chars:
                    continue
                page_idx = parse_page_index(xf)
                vm = vmeta.get(vid, {})
                prefix = build_text_prefix(
                    t,
                    vid,
                    str(xf),
                    page_idx,
                    year=vm.get("year", ""),
                    volume_title=vm.get("volume_title", ""),
                )
                candidates[t].append(
                    {
                        "document_type": t,
                        "volume_id": vid,
                        "chunk_id": f"{vid}_{xf.stem}",
                        "page_index": page_idx,
                        "source_file": str(xf),
                        "text": txt,
                        "text_without_prefix": txt,
                        "text_prefix": prefix,
                        "text_with_prefix": f"{prefix}\n\n{txt}",
                    }
                )

    pairs = []
    stats = {"available_candidates": {}, "selected_pairs": {}, "volumes_by_type": {}}

    for t in TARGET_TYPES:
        stats["available_candidates"][t] = len(candidates[t])
        stats["volumes_by_type"][t] = len(vols_by_type.get(t, []))

        if len(candidates[t]) == 0:
            continue

        # sample with cycling to guarantee quota even for small type pools
        cyc = cycle(candidates[t])
        for i in range(quotas[t]):
            item = next(cyc).copy()
            item["pair_id"] = f"{t}_pair_{i+1:04d}"
            item["pair_group"] = "layer1_pilot"
            pairs.append(item)

        stats["selected_pairs"][t] = quotas[t]

    # deterministic ordering
    pairs.sort(key=lambda x: (x["document_type"], x["volume_id"], x.get("page_index", -1), x["pair_id"]))

    out = {
        "metadata": {
            "dataset": "layer1_pilot_pairs",
            "target_pairs": args.target_pairs,
            "target_types": TARGET_TYPES,
            "quotas": quotas,
            "min_text_chars": args.min_text_chars,
        },
        "stats": stats,
        "pairs": pairs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Built layer1 pilot pairs")
    print(f"Output: {out_path}")
    print(f"Total pairs: {len(pairs)}")
    print("Quotas:", quotas)
    print("Available candidates by type:", stats["available_candidates"])
    print("Volumes by type:", stats["volumes_by_type"])


if __name__ == "__main__":
    main()
