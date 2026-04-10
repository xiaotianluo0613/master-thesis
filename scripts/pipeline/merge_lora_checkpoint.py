#!/usr/bin/env python3
"""
Merge a LoRA adapter checkpoint into a full BGE-M3 model.

After each layer's LoRA fine-tune, the saved checkpoint contains only the
adapter weights (adapter_config.json + adapter_model.safetensors) plus the
BGE-M3 specific heads (colbert_linear.pt, sparse_linear.pt).

FlagEmbedding's runner cannot load this as a starting point for the next
layer because AutoConfig.from_pretrained() expects a full config.json.
This script merges the LoRA adapter into the base model and saves a clean
full checkpoint ready for continued training.

Usage:
    python scripts/pipeline/merge_lora_checkpoint.py \
        --adapter-dir output/models/layer1-bge-m3-lora-dense-b4 \
        --output-dir  output/models/layer1-bge-m3-lora-dense-b4-merged \
        --base-model  BAAI/bge-m3
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


BGE_M3_EXTRA_FILES = ["colbert_linear.pt", "sparse_linear.pt"]


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into full BGE-M3 model")
    parser.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output-dir", required=True, help="Path for merged model output")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Base model name or path")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {adapter_dir}")
    print(f"Output     : {output_dir}")

    # Load base model + LoRA adapter, then merge
    print("\nLoading base model...")
    base = AutoModel.from_pretrained(args.base_model)

    print("Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    print("Saving merged model...")
    merged.save_pretrained(str(output_dir))

    # Save tokenizer
    print("Saving tokenizer...")
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(str(output_dir))

    # Copy BGE-M3 specific heads
    for fname in BGE_M3_EXTRA_FILES:
        src = adapter_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"Copied {fname}")
        else:
            print(f"WARNING: {fname} not found in adapter dir — skipping")

    print(f"\nDone: {output_dir}")
    print("Contents:", sorted(p.name for p in output_dir.iterdir()))


if __name__ == "__main__":
    main()
