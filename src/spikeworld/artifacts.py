"""Download the public representative checkpoint with integrity verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/Oooorca/SpikeWorld/releases/download/v0.1.0/"
    "spikeworld_seed42401.pt"
)
CHECKPOINT_SHA256 = "6513bc37d9cd3a5a91dedb478ccd54ce87850e6cd06cb1f3fb9437767dece1ff"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(destination: str | Path, force: bool = False) -> Path:
    target = Path(destination)
    if target.exists() and not force:
        if sha256(target) == CHECKPOINT_SHA256:
            return target
        raise ValueError(f"existing file has the wrong SHA-256: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="spikeworld-", suffix=".pt")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
        observed = sha256(temporary)
        if observed != CHECKPOINT_SHA256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {observed}"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="artifacts/spikeworld_seed42401.pt", help="output path"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(fetch(args.output, args.force))


if __name__ == "__main__":
    main()
