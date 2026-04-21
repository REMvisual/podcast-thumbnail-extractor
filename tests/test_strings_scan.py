"""Tests for strings_scan — binary-file leak detection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from strings_scan import scan_file, Leak


@pytest.fixture
def clean_binary(tmp_path):
    p = tmp_path / "clean.bin"
    p.write_bytes(b"\x00\x01\x02\x03PyTorchModel\x00\x00sha256:abc\x00")
    return p


@pytest.fixture
def dirty_binary(tmp_path):
    p = tmp_path / "dirty.bin"
    p.write_bytes(b"\x00\x00prefix\x00C:\\Users\\simango\\secret.pth\x00suffix\x00")
    return p


def test_scan_file_clean(clean_binary):
    leaks = scan_file(clean_binary)
    assert leaks == []


def test_scan_file_detects_personal_name(dirty_binary):
    leaks = scan_file(dirty_binary)
    assert any("simango" in leak.match for leak in leaks)


def test_scan_file_detects_absolute_path(dirty_binary):
    leaks = scan_file(dirty_binary)
    assert any(leak.pattern == "windows_path" for leak in leaks)


def test_leak_is_dataclass():
    leak = Leak(pattern="test", match="xyz", offset=0, context="ctx")
    assert leak.pattern == "test"
    assert leak.match == "xyz"


def test_scan_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_file(tmp_path / "nonexistent.bin")
