#!/usr/bin/env python3
"""
One-time conversion: CSV captions -> JSON {modelId: [caption, ...]}.
Run when the CSV changes. Training then loads the JSON (single parse, no row loop).

Usage:
  python scripts/csv_captions_to_json.py [--input path/to/captions.csv] [--output path/to/captions.json]
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description="Convert captions CSV to JSON for fast dataset loading.")
    p.add_argument("--input", "-i", default="src/data/captions.tablechair.csv", help="Input CSV path")
    p.add_argument("--output", "-o", default="src/data/captions.json", help="Output JSON path")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    captions_dict = defaultdict(list)
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["modelId"].strip()
            desc = (row.get("description") or "").strip()
            category = (row.get("category") or "").strip()
            caption = f"{desc}" if category else desc
            if caption:
                captions_dict[model_id].append(caption)

    captions_dict = dict(captions_dict)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=0, ensure_ascii=False)

    print(f"Wrote {len(captions_dict)} model IDs -> {out_path}")


if __name__ == "__main__":
    main()
