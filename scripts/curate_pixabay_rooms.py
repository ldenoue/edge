#!/usr/bin/env python3
"""Curate the downloaded Pixabay room set for synthetic-person compositing.

The rejected files are moved to a sibling directory instead of deleted, so the
selection is reversible. The active and rejected manifests are rewritten to
match the files in each directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


# Manual review of the 100-image contact sheet. Keep empty or lightly furnished
# real interiors; reject prominent people/animals, illustrations, and close-ups
# that do not provide a useful scene for compositing a synthetic person.
REJECTIONS = {
    27: "prominent person",
    28: "illustration",
    30: "prominent person",
    32: "prominent animal",
    35: "illustration",
    38: "prominent people",
    39: "prominent person",
    41: "prominent person",
    43: "prominent people",
    48: "prominent people",
    50: "prominent person",
    53: "prominent person",
    54: "prominent person",
    60: "prominent person",
    61: "prominent person",
    63: "prominent people",
    66: "object close-up, not a room scene",
    68: "prominent people",
    69: "prominent person",
    70: "prominent people",
    76: "object close-up, not a room scene",
    83: "prominent people",
    86: "prominent person",
    88: "prominent person",
    90: "object close-up, not a room scene",
    91: "prominent person",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/backgrounds/pixabay_rooms"),
        help="Downloaded image directory containing manifest.json",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/backgrounds/pixabay_rooms_rejected"),
        help="Recoverable destination for rejected images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.input / "manifest.json"
    records = json.loads(manifest_path.read_text())
    args.rejected.mkdir(parents=True, exist_ok=True)

    kept = []
    rejected_manifest_path = args.rejected / "manifest.json"
    previously_rejected = (
        json.loads(rejected_manifest_path.read_text())
        if rejected_manifest_path.exists()
        else []
    )
    rejected_by_id = {record["pixabay_id"]: record for record in previously_rejected}
    for record in records:
        sequence = int(record["file"].split("-", 1)[0])
        reason = REJECTIONS.get(sequence)
        if reason is None:
            kept.append(record)
            continue

        source = args.input / record["file"]
        destination = args.rejected / record["file"]
        if source.exists():
            shutil.move(source, destination)
        rejected_by_id[record["pixabay_id"]] = {**record, "rejection_reason": reason}

    rejected = sorted(
        rejected_by_id.values(), key=lambda record: int(record["file"].split("-", 1)[0])
    )
    manifest_path.write_text(json.dumps(kept, indent=2) + "\n")
    rejected_manifest_path.write_text(json.dumps(rejected, indent=2) + "\n")
    print(f"kept {len(kept)} useful backgrounds in {args.input}")
    print(f"moved {len(rejected)} rejects to {args.rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
