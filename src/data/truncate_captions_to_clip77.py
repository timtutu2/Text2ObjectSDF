#!/usr/bin/env python3
"""
Preprocess captions.json so every caption fits within CLIP's 77-token limit.
Preserves the start of each caption (main description). Output can be used
as the training captions file to avoid losing the tail at tokenizer time.

Usage:
  python scripts/truncate_captions_to_clip77.py [--input src/data/captions.json] [--output src/data/captions_clip77.json]
  Then in training, use --captions-file src/data/captions_clip77.json (or set captions_file in dataset to this path).
"""

import argparse
import json
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser(
        description="Truncate captions to fit CLIP max 77 tokens; write new JSON."
    )
    p.add_argument(
        "--input", "-i",
        default=PROJECT_ROOT / "src" / "data" / "captions.json",
        type=Path,
        help="Input captions JSON (model_id -> list of caption strings)",
    )
    p.add_argument(
        "--output", "-o",
        default=PROJECT_ROOT / "src" / "data" / "captions_clip77.json",
        type=Path,
        help="Output JSON with truncated captions",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=77,
        help="CLIP max position embeddings (default 77)",
    )
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    out_dict = {}
    total_before = 0
    total_after = 0
    truncated_count = 0

    for model_id, captions in captions_dict.items():
        if isinstance(captions, str):
            captions = [captions]
        shortened = []
        for cap in captions:
            total_before += 1
            ids = tokenizer.encode(cap, add_special_tokens=True, truncation=False)
            if len(ids) <= args.max_tokens:
                shortened.append(cap)
                total_after += 1
                continue
            # Keep first max_tokens tokens and decode back to text (preserves start = main description)
            truncated = tokenizer.decode(
                ids[: args.max_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            shortened.append(truncated)
            total_after += 1
            truncated_count += 1
        out_dict[model_id] = shortened

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=0, ensure_ascii=False)

    print(f"Read {len(captions_dict)} models from {args.input}")
    print(f"Total captions: {total_before} -> {total_after} (truncated: {truncated_count})")
    print(f"Wrote {args.output}")
    print("Use this file as captions_file in your dataset (e.g. captions_clip77.json).")


if __name__ == "__main__":
    main()
