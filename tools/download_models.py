"""Download starter models from the GitHub Releases asset URL, verify checksums."""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(os.environ.get("THUMBNAIL_EXTRACTOR_MODEL_DIR", REPO_ROOT / "models"))
CHECKSUMS_PATH = REPO_ROOT / "models" / "checksums.json"
RELEASE_BASE_URL = os.environ.get(
    "THUMBNAIL_EXTRACTOR_RELEASE_URL",
    "https://github.com/REMvisual/podcast-thumbnail-extractor/releases/download/v0.1.0",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not CHECKSUMS_PATH.exists():
        print(f"ERROR: checksums file missing at {CHECKSUMS_PATH}")
        return 1
    checksums = json.loads(CHECKSUMS_PATH.read_text())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not checksums:
        print("  [skip] no models listed in checksums.json (empty manifest)")
        return 0

    failures = 0
    for name, expected_sha in checksums.items():
        target = MODEL_DIR / name
        if target.exists() and _sha256(target) == expected_sha:
            print(f"  [skip] {name} - already present, checksum OK")
            continue
        url = f"{RELEASE_BASE_URL}/{name}"
        print(f"  [fetch] {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as e:
            print(f"  [error] {name}: {e}")
            failures += 1
            continue
        actual = _sha256(target)
        if actual != expected_sha:
            print(f"  [error] {name}: checksum mismatch (expected {expected_sha}, got {actual})")
            target.unlink(missing_ok=True)
            failures += 1
            continue
        print(f"  [ok]    {name}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
