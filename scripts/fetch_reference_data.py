from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        response = client.get(url)
        response.raise_for_status()
        destination.write_bytes(response.content)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-if-present", action="store_true")
    args = parser.parse_args()

    settings = get_settings()

    if not (args.skip_if_present and settings.raw_catalog_path.exists()):
        print(f"Downloading catalog to {settings.raw_catalog_path}")
        download(settings.catalog_url, settings.raw_catalog_path)

    if not (args.skip_if_present and settings.raw_traces_zip_path.exists()):
        print(f"Downloading sample traces to {settings.raw_traces_zip_path}")
        download(settings.sample_traces_url, settings.raw_traces_zip_path)

    traces_root = settings.public_traces_dir.parent
    if traces_root.exists() and not args.skip_if_present:
        shutil.rmtree(traces_root)
    extract_zip(settings.raw_traces_zip_path, traces_root)
    print(f"Extracted sample traces to {traces_root}")


if __name__ == "__main__":
    main()
