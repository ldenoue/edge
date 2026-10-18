#!/usr/bin/env python3
"""Download a small, attributed room-background set via Pixabay's official API.

Create a free Pixabay API key, then keep it out of shell history:

    export PIXABAY_API_KEY="..."
    python3 scripts/download_pixabay_rooms.py \
      --output data/backgrounds/pixabay_rooms --count 100

The API key is never written to disk. A manifest records the source page,
contributor, tags, and Pixabay content-license URL for every downloaded image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://pixabay.com/api/"
LICENSE_URL = "https://pixabay.com/service/license-summary/"
USER_AGENT = "edge-synthetic-training-background-downloader/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/backgrounds/pixabay_rooms"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--query", default="room")
    parser.add_argument("--latest", action="store_true", help="Order by latest instead of popularity")
    args = parser.parse_args()
    if not 1 <= args.count <= 200:
        parser.error("--count must be between 1 and 200")
    return args


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not api_key:
        print(
            "PIXABAY_API_KEY is not set. Create a free key at "
            "https://pixabay.com/api/docs/ and export it before running this script.",
            file=sys.stderr,
        )
        return 2

    params = {
        "key": api_key,
        "q": args.query,
        "image_type": "photo",
        "orientation": "horizontal",
        "min_width": 960,
        "min_height": 540,
        "safesearch": "true",
        "order": "latest" if args.latest else "popular",
        "per_page": args.count,
        "page": 1,
    }
    payload = request_json(f"{API_URL}?{urllib.parse.urlencode(params)}")
    hits = payload.get("hits", [])[: args.count]
    if not hits:
        print("Pixabay returned no matching photos.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, hit in enumerate(hits, start=1):
        image_id = int(hit["id"])
        filename = f"{index:03d}-pixabay-{image_id}.jpg"
        destination = args.output / filename
        image_url = hit.get("largeImageURL") or hit["webformatURL"]
        if not destination.exists():
            download(image_url, destination)
        manifest.append(
            {
                "file": filename,
                "pixabay_id": image_id,
                "source_page": hit["pageURL"],
                "contributor": hit.get("user"),
                "contributor_id": hit.get("user_id"),
                "tags": hit.get("tags", ""),
                "source_width": hit.get("imageWidth"),
                "source_height": hit.get("imageHeight"),
                "download_width": hit.get("webformatWidth"),
                "download_height": hit.get("webformatHeight"),
                "license": LICENSE_URL,
            }
        )
        print(f"downloaded {index}/{len(hits)}: {filename}", flush=True)

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} photos and attribution metadata to {args.output.resolve()}")
    if len(manifest) < args.count:
        print(f"warning: requested {args.count}, but Pixabay returned {len(manifest)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
