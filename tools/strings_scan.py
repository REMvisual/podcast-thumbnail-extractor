"""Scan binary files (.pth, .onnx, .bin) for leaked personal strings or absolute paths."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Patterns we consider leaks. Add more as needed.
# Each entry: (pattern_name, compiled_regex, case_insensitive)
LEAK_PATTERNS: list[tuple[str, re.Pattern[bytes], bool]] = [
    ("windows_path", re.compile(rb"[A-Za-z]:[\\/][\w\\/. -]+"), False),
    ("unix_home", re.compile(rb"/(?:Users|home)/\w+"), False),
    ("personal_name_simango", re.compile(rb"simango", re.IGNORECASE), True),
    ("personal_name_yuval", re.compile(rb"yuval", re.IGNORECASE), True),
    ("automationstation", re.compile(rb"automationstation", re.IGNORECASE), True),
    ("standalone_path", re.compile(rb"/Standalone/", re.IGNORECASE), True),
    ("email", re.compile(rb"[\w.+-]+@[\w.-]+\.\w+"), False),
    ("api_key_hint", re.compile(rb"(?:api[_-]?key|secret|token)[\"':= ]+[\w]{8,}", re.IGNORECASE), True),
]


@dataclass
class Leak:
    pattern: str
    match: str
    offset: int
    context: str  # 40 bytes either side, printable-ascii-ized


def _extract_strings(data: bytes, min_len: int = 4) -> list[tuple[int, bytes]]:
    """Yield (offset, run) for each run of printable ASCII at least min_len long."""
    out: list[tuple[int, bytes]] = []
    start = None
    for i, b in enumerate(data):
        if 0x20 <= b < 0x7F:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_len:
                out.append((start, data[start:i]))
            start = None
    if start is not None and (len(data) - start) >= min_len:
        out.append((start, data[start:]))
    return out


def scan_file(path: Path) -> list[Leak]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    runs = _extract_strings(data, min_len=4)
    leaks: list[Leak] = []
    for offset, run in runs:
        for name, rx, _ in LEAK_PATTERNS:
            for m in rx.finditer(run):
                ctx_start = max(0, offset + m.start() - 20)
                ctx_end = min(len(data), offset + m.end() + 20)
                ctx = data[ctx_start:ctx_end].decode("latin-1", errors="replace")
                ctx_clean = "".join(c if 0x20 <= ord(c) < 0x7F else "." for c in ctx)
                leaks.append(Leak(
                    pattern=name,
                    match=m.group(0).decode("latin-1", errors="replace"),
                    offset=offset + m.start(),
                    context=ctx_clean,
                ))
    return leaks


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: strings_scan.py <file> [<file> ...]")
        return 2
    any_leak = False
    for arg in argv[1:]:
        leaks = scan_file(Path(arg))
        if leaks:
            any_leak = True
            print(f"[{arg}] {len(leaks)} leak(s):")
            for leak in leaks:
                print(f"  - {leak.pattern} @ 0x{leak.offset:x}: {leak.match!r}")
                print(f"    ...{leak.context}...")
        else:
            print(f"[{arg}] CLEAN")
    return 1 if any_leak else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
