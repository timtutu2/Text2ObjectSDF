#!/usr/bin/env python3
"""
Clean low-quality captions with hard rules + semantic rules.

Input format:
  {
    "model_id_1": ["caption a", "caption b", ...],
    ...
  }

Output:
  1) Cleaned captions JSON (same structure)
  2) JSON report with removed captions and reasons

Usage:
  python src/data/clean_bad_captions.py \
    --input src/data/captions_clip77.json \
    --output src/data/captions_clip77_clean.json \
    --report src/data/captions_clip77_clean_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


OBJECT_WORDS = {
    "chair",
    "table",
    "desk",
    "sofa",
    "armchair",
    "stool",
    "bench",
    "seat",
}

COLOR_WORDS = {
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "pink",
    "purple",
    "violet",
    "silver",
    "gold",
    "beige",
    "cream",
    "ash",
    "dark",
    "light",
}

MATERIAL_WORDS = {
    "wood",
    "wooden",
    "metal",
    "steel",
    "iron",
    "plastic",
    "glass",
    "fabric",
    "leather",
    "cotton",
    "wicker",
    "resin",
}

SHAPE_WORDS = {
    "round",
    "rectangular",
    "rectangle",
    "square",
    "oval",
    "circular",
    "curved",
    "triangular",
    "hexagon",
    "hexagonal",
    "l",
    "u",
    "x",
    "s",
}

PART_WORDS = {
    "leg",
    "legs",
    "back",
    "backrest",
    "seat",
    "arm",
    "arms",
    "armrest",
    "cushion",
    "cushioned",
    "top",
    "base",
    "frame",
    "shelf",
}

NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "single",
    "double",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "it",
    "its",
    "this",
    "that",
    "to",
    "for",
    "of",
    "and",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "as",
    "be",
    "has",
    "have",
    "had",
    "made",
    "looks",
    "look",
    "very",
    "use",
    "used",
    "type",
    "model",
    "style",
}

GENERIC_NOISE_WORDS = {
    "material",
    "appearance",
    "physical",
    "model",
    "type",
    "use",
    "used",
}

CONNECTOR_WORDS = {"and", "of", "with", "in", "on", "to", "for", "by"}

ATTRIBUTE_WORDS = COLOR_WORDS | MATERIAL_WORDS | SHAPE_WORDS | PART_WORDS | NUMBER_WORDS
SPLIT_VOCAB = OBJECT_WORDS | ATTRIBUTE_WORDS | STOPWORDS | GENERIC_NOISE_WORDS | {
    "colored",
    "colour",
    "coloured",
    "dining",
    "office",
    "coffee",
    "room",
    "rest",
    "high",
    "low",
    "small",
    "large",
    "thin",
    "thick",
    "modern",
    "classic",
    "simple",
    "comfortable",
    "short",
    "long",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
NON_ALLOWED_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s,.'\-_/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean low-quality captions with hard + semantic rules.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("src/data/captions_clip77.json"),
        help="Input captions JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("src/data/captions_clip77_clean.json"),
        help="Output cleaned captions JSON.",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=Path("src/data/captions_clip77_clean_report.json"),
        help="Output report JSON for removed/kept stats.",
    )
    parser.add_argument("--min-tokens", type=int, default=4, help="Minimum token count for a valid caption.")
    parser.add_argument("--max-tokens", type=int, default=20, help="Maximum token count for a valid caption.")
    parser.add_argument(
        "--min-unique-ratio",
        type=float,
        default=0.5,
        help="Min unique-token ratio to prevent repetitive captions.",
    )
    parser.add_argument(
        "--max-stopword-ratio",
        type=float,
        default=0.6,
        help="Max ratio of stopwords in a caption.",
    )
    parser.add_argument(
        "--max-connector-ratio",
        type=float,
        default=0.35,
        help="Max ratio of connector words such as 'and/with/of'.",
    )
    parser.add_argument(
        "--max-generic-ratio",
        type=float,
        default=0.4,
        help="Max ratio of generic words such as 'material/model/type'.",
    )
    parser.add_argument(
        "--enforce-model-object-match",
        action="store_true",
        help="If set, caption object word must match model's dominant object word.",
    )
    parser.add_argument(
        "--keep-one-per-model",
        action="store_true",
        default=True,
        help="Keep at least one best-effort caption if all fail (default: True).",
    )
    parser.add_argument(
        "--no-keep-one-per-model",
        action="store_false",
        dest="keep_one_per_model",
        help="Disable fallback keep-one behavior.",
    )
    return parser.parse_args()


def maybe_split_glued_token(token: str) -> list[str]:
    if token in SPLIT_VOCAB or not token.isalpha() or len(token) < 6:
        return [token]

    n = len(token)
    best: list[str] | None = None
    dp: list[list[str] | None] = [None] * (n + 1)
    dp[0] = []

    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 2, min(n, i + 14) + 1):
            part = token[i:j]
            if part in SPLIT_VOCAB:
                candidate = dp[i] + [part]
                if dp[j] is None or len(candidate) < len(dp[j]):
                    dp[j] = candidate

    best = dp[n]
    if best and len(best) >= 2:
        return best
    return [token]


def collapse_repetition(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens

    dedup = []
    for tok in tokens:
        if dedup and dedup[-1] == tok:
            continue
        dedup.append(tok)

    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(dedup):
            if (
                i + 3 < len(dedup)
                and dedup[i] == "and"
                and dedup[i + 2] == "and"
                and dedup[i + 1] == dedup[i + 3]
            ):
                out.extend(["and", dedup[i + 1]])
                i += 4
                changed = True
            else:
                out.append(dedup[i])
                i += 1
        dedup = out

    out = []
    for tok in dedup:
        if out and tok in CONNECTOR_WORDS and out[-1] == tok:
            continue
        out.append(tok)

    while out and out[0] in CONNECTOR_WORDS:
        out = out[1:]
    while out and out[-1] in CONNECTOR_WORDS:
        out = out[:-1]

    return out


def normalize_caption(text: str) -> tuple[str, list[str]]:
    s = text.strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", " ", s)
    s = s.replace("_", " ").replace("/", " ")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()

    raw_tokens = TOKEN_RE.findall(s)
    expanded: list[str] = []
    for tok in raw_tokens:
        expanded.extend(maybe_split_glued_token(tok))

    collapsed = collapse_repetition(expanded)
    norm = " ".join(collapsed)
    return norm, collapsed


def infer_expected_object(captions: Iterable[str]) -> str | None:
    counter: Counter[str] = Counter()
    for caption in captions:
        _, tokens = normalize_caption(caption)
        for tok in tokens:
            if tok in OBJECT_WORDS:
                counter[tok] += 1
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def hard_rule_reasons(
    original: str,
    tokens: list[str],
    args: argparse.Namespace,
) -> list[str]:
    reasons = []
    n_tokens = len(tokens)
    if n_tokens < args.min_tokens:
        reasons.append("hard_too_short")
    if n_tokens > args.max_tokens:
        reasons.append("hard_too_long")

    if n_tokens > 0:
        unique_ratio = len(set(tokens)) / n_tokens
        if unique_ratio < args.min_unique_ratio:
            reasons.append("hard_low_unique_ratio")

        stop_ratio = sum(tok in STOPWORDS for tok in tokens) / n_tokens
        if stop_ratio > args.max_stopword_ratio:
            reasons.append("hard_high_stopword_ratio")

        conn_ratio = sum(tok in CONNECTOR_WORDS for tok in tokens) / n_tokens
        if conn_ratio > args.max_connector_ratio:
            reasons.append("hard_high_connector_ratio")

        generic_ratio = sum(tok in GENERIC_NOISE_WORDS for tok in tokens) / n_tokens
        if generic_ratio > args.max_generic_ratio:
            reasons.append("hard_high_generic_ratio")

    if NON_ALLOWED_CHAR_RE.search(original):
        reasons.append("hard_non_allowed_chars")
    if REPEATED_CHAR_RE.search(original.lower()):
        reasons.append("hard_repeated_chars")
    if any(len(tok) > 24 for tok in tokens):
        reasons.append("hard_extreme_token_length")
    return reasons


def semantic_rule_reasons(
    tokens: list[str],
    expected_object: str | None,
    enforce_object_match: bool,
) -> list[str]:
    reasons = []
    objects = {tok for tok in tokens if tok in OBJECT_WORDS}
    attrs = {tok for tok in tokens if tok in ATTRIBUTE_WORDS}

    if not objects:
        reasons.append("semantic_missing_object")
    if not attrs:
        reasons.append("semantic_missing_attribute")
    if "material" in tokens and not any(tok in MATERIAL_WORDS for tok in tokens):
        reasons.append("semantic_unspecific_material_word")
    if enforce_object_match and expected_object and objects and expected_object not in objects:
        reasons.append("semantic_object_mismatch")
    return reasons


def pick_fallback(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    # Prefer fewer reasons, then higher lexical diversity, then medium length.
    scored = []
    for c in candidates:
        tokens = c["tokens"]
        n = len(tokens)
        unique_ratio = (len(set(tokens)) / n) if n else 0.0
        reason_count = len(c["reasons"])
        length_penalty = abs(10 - n)
        scored.append((reason_count, -unique_ratio, length_penalty, c))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned: dict[str, list[str]] = {}
    removed_by_model: dict[str, list[dict]] = {}
    kept_by_model: dict[str, list[dict]] = {}
    reason_counter: Counter[str] = Counter()

    total_before = 0
    total_after = 0
    fallback_count = 0

    for model_id, captions in data.items():
        caption_list = captions if isinstance(captions, list) else [captions]
        expected_object = infer_expected_object(caption_list)
        seen = set()
        kept_clean = []
        kept_detail = []
        removed_detail = []
        candidates = []

        for caption in caption_list:
            total_before += 1
            norm, tokens = normalize_caption(caption)
            reasons = []

            if not norm:
                reasons.append("hard_empty_after_normalize")
            else:
                reasons.extend(hard_rule_reasons(caption, tokens, args))
                reasons.extend(
                    semantic_rule_reasons(
                        tokens=tokens,
                        expected_object=expected_object,
                        enforce_object_match=args.enforce_model_object_match,
                    )
                )
                if norm in seen:
                    reasons.append("hard_duplicate_after_normalize")

            if reasons:
                for r in reasons:
                    reason_counter[r] += 1
                removed_detail.append(
                    {
                        "original": caption,
                        "normalized": norm,
                        "reasons": reasons,
                    }
                )
                candidates.append(
                    {
                        "original": caption,
                        "normalized": norm,
                        "tokens": tokens,
                        "reasons": reasons,
                    }
                )
                continue

            seen.add(norm)
            kept_clean.append(norm)
            kept_detail.append({"original": caption, "normalized": norm})

        if not kept_clean and args.keep_one_per_model:
            fallback = pick_fallback(candidates)
            if fallback and fallback["normalized"]:
                kept_clean = [fallback["normalized"]]
                kept_detail.append(
                    {
                        "original": fallback["original"],
                        "normalized": fallback["normalized"],
                        "fallback_kept": True,
                        "original_reasons": fallback["reasons"],
                    }
                )
                reason_counter["fallback_kept_best_candidate"] += 1
                fallback_count += 1

        if kept_clean:
            cleaned[model_id] = kept_clean
            total_after += len(kept_clean)
            kept_by_model[model_id] = kept_detail
        else:
            cleaned[model_id] = []
            kept_by_model[model_id] = []

        if removed_detail:
            removed_by_model[model_id] = removed_detail

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "total_models": len(data),
        "models_with_nonempty_output": sum(1 for v in cleaned.values() if v),
        "total_captions_before": total_before,
        "total_captions_after": total_after,
        "removed_count": total_before - total_after,
        "fallback_kept_model_count": fallback_count,
        "reason_counts": dict(reason_counter.most_common()),
        "removed_by_model": removed_by_model,
        "kept_by_model": kept_by_model,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Input models: {len(data)}")
    print(f"Captions before/after: {total_before} -> {total_after}")
    print(f"Fallback kept models: {fallback_count}")
    print(f"Wrote cleaned captions: {args.output}")
    print(f"Wrote cleaning report: {args.report}")
    if reason_counter:
        print("Top reasons:")
        for reason, count in reason_counter.most_common(10):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
