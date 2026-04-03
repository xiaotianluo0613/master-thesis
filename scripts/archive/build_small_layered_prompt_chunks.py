#!/usr/bin/env python3
"""
Build a tiny multi-layer chunk dataset for prompt testing only.

Output format is compatible with generate_n_to_n_queries_layered.py.
Each selected volume becomes one grouped "date" key so multiple pages map
into one N-to-N query target.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from lxml import etree

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def extract_text(xml_path: Path) -> str:
    try:
        tree = etree.parse(str(xml_path))
        lines = []
        for tl in tree.xpath('//alto:TextLine', namespaces=ALTO_NS):
            toks = tl.xpath('./alto:String/@CONTENT', namespaces=ALTO_NS)
            line = ' '.join(t.strip() for t in toks if t and t.strip())
            if line:
                lines.append(line)
        return '\n'.join(lines)
    except Exception:
        return ""


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Build tiny layered chunks for prompt tests')
    parser.add_argument('--pool-dir', default='output/data_pools')
    parser.add_argument('--transcriptions-root', default='Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions')
    parser.add_argument('--per-layer-volumes', type=int, default=1)
    parser.add_argument('--pages-per-volume', type=int, default=3)
    parser.add_argument('--min-text-chars', type=int, default=200)
    parser.add_argument('--output', default='data/layered_prompt_test_chunks.json')
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    root = Path(args.transcriptions_root)

    layer_files = {
        'layer1': pool_dir / 'train_layer1_pool.txt',
        'layer2': pool_dir / 'train_layer2_pool.txt',
        'layer3': pool_dir / 'train_layer3_pool.txt',
        'layer4': pool_dir / 'train_layer4_pool.txt',
    }

    chosen = {}
    for layer, f in layer_files.items():
        ids = read_ids(f)
        chosen[layer] = ids[:args.per_layer_volumes]

    chunks = []
    for layer, vol_ids in chosen.items():
        for vid in vol_ids:
            vdir = root / vid
            if not vdir.exists():
                continue

            xmls = sorted(vdir.glob('*.xml'))
            kept = 0
            for idx, xf in enumerate(xmls, 1):
                txt = extract_text(xf)
                if len(txt) < args.min_text_chars:
                    continue

                chunk_id = f"{vid}_{xf.stem}"
                chunks.append({
                    'chunk_id': chunk_id,
                    'volume_id': vid,
                    'layer': layer,
                    # group all pages of one volume together for N-to-N prompt tests
                    'date': f"VOLUME-{vid}",
                    'sub_chunk_index': kept,
                    'text_without_prefix': txt,
                    'text': txt,
                })
                kept += 1
                if kept >= args.pages_per_volume:
                    break

    out = {
        'metadata': {
            'purpose': 'small layered prompt test',
            'per_layer_volumes': args.per_layer_volumes,
            'pages_per_volume': args.pages_per_volume,
            'layers': list(chosen.keys()),
            'selected_volumes': chosen,
            'total_chunks': len(chunks),
        },
        'chunks': chunks,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    print('✅ Built small layered test chunks')
    print(f'Output: {out_path}')
    print(f'Total chunks: {len(chunks)}')
    print('Selected volumes by layer:')
    for k, v in chosen.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
