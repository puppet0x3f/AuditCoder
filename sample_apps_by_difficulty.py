"""Randomly sample APPS problems by difficulty.

Default behavior:
  - introductory: 100 problems
  - interview: 200 problems
  - competition: 100 problems

Usage:
  python sample_apps_by_difficulty.py
  python sample_apps_by_difficulty.py --seed 42 --overwrite
  python sample_apps_by_difficulty.py --output data/apps_sampled_400.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "data" / "apps.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "apps_sampled_400.jsonl"
DEFAULT_IDS_OUTPUT = REPO_ROOT / "data" / "apps_sampled_400_ids.json"

DEFAULT_COUNTS = {
    "introductory": 100,
    "interview": 200,
    "competition": 100,
}


def load_apps_by_difficulty(path: Path) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc

            difficulty = item.get("difficulty")
            if difficulty:
                buckets[difficulty].append(item)

    return dict(buckets)


def sample_items(
    buckets: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    sampled: dict[str, list[dict[str, Any]]] = {}

    for difficulty, count in counts.items():
        candidates = buckets.get(difficulty, [])
        if len(candidates) < count:
            raise ValueError(
                f"Not enough {difficulty} problems: need {count}, found {len(candidates)}"
            )
        sampled[difficulty] = rng.sample(candidates, count)

    return sampled


def write_jsonl(path: Path, items: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_ids(path: Path, sampled: dict[str, list[dict[str, Any]]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    ids_by_difficulty = {
        difficulty: [item.get("id") for item in items]
        for difficulty, items in sampled.items()
    }
    payload = {
        "counts": {difficulty: len(ids) for difficulty, ids in ids_by_difficulty.items()},
        "ids_by_difficulty": ids_by_difficulty,
        "ids": [
            problem_id
            for difficulty in DEFAULT_COUNTS
            for problem_id in ids_by_difficulty.get(difficulty, [])
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample APPS problems by difficulty."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="APPS jsonl path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="sampled jsonl path")
    parser.add_argument(
        "--ids-output",
        type=Path,
        default=DEFAULT_IDS_OUTPUT,
        help="JSON file for sampled problem ids",
    )
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--introductory", type=int, default=DEFAULT_COUNTS["introductory"])
    parser.add_argument("--interview", type=int, default=DEFAULT_COUNTS["interview"])
    parser.add_argument("--competition", type=int, default=DEFAULT_COUNTS["competition"])
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="shuffle all selected problems together instead of grouping by difficulty",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "introductory": args.introductory,
        "interview": args.interview,
        "competition": args.competition,
    }

    rng = random.Random(args.seed)
    buckets = load_apps_by_difficulty(args.input)
    sampled = sample_items(buckets, counts, rng)

    output_items = [
        item
        for difficulty in DEFAULT_COUNTS
        for item in sampled.get(difficulty, [])
    ]
    if args.shuffle_output:
        rng.shuffle(output_items)

    write_jsonl(args.output, output_items, args.overwrite)
    write_ids(args.ids_output, sampled, args.overwrite)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"IDs: {args.ids_output}")
    print(f"Seed: {args.seed}")
    for difficulty in DEFAULT_COUNTS:
        print(f"{difficulty}: {len(sampled[difficulty])}")
    print(f"total: {len(output_items)}")


if __name__ == "__main__":
    main()
