#!/usr/bin/env python3
"""Batch face registration script.

Registers all images from a directory. Each filename (without extension)
is used as the person's name.

Usage:
    python batch_register.py ./photos
    python batch_register.py ./photos --url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

import requests

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_face(url: str, name: str, image_path: Path) -> dict:
    """Register a single face via the /register endpoint."""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{url}/register",
            data={"name": name},
            files={"image": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch register faces from a directory.")
    parser.add_argument("directory", type=Path, help="Directory containing face images.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the face recognition service (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    images = sorted(
        p for p in args.directory.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No images found in {args.directory}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images. Registering...")
    success = 0
    failed = 0

    for img_path in images:
        name = img_path.stem.replace("_", " ").replace("-", " ").title()
        try:
            result = register_face(args.url, name, img_path)
            print(f"  OK   {name} -> {result['id']}")
            success += 1
        except Exception as exc:
            print(f"  FAIL {name} -> {exc}")
            failed += 1

    print(f"\nDone. {success} registered, {failed} failed.")


if __name__ == "__main__":
    main()
