"""Strip pickle-metadata leaks from a .pth by round-tripping through torch.load/save.

Usage: python tools/scrub_pth.py models/x.pth [models/y.pth ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: torch required. `pip install torch`")
    sys.exit(1)


def scrub(path: Path) -> None:
    print(f"[{path}] loading...")
    state = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[{path}] re-saving (new zipfile format, no path metadata)...")
    tmp = path.with_suffix(".pth.tmp")
    torch.save(state, tmp, _use_new_zipfile_serialization=True)
    tmp.replace(path)
    print(f"[{path}] done.")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: scrub_pth.py <file> [<file> ...]")
        return 2
    for arg in argv[1:]:
        scrub(Path(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
