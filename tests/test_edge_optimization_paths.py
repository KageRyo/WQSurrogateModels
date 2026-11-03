"""Protect inputs and unrelated files from mistaken CLI output paths."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_optimization_paths import artifact_path, atomic_json


@pytest.mark.parametrize("path", ["/etc/edge-test.json", "/home/edge-test.json"])
def test_unrelated_destination_is_rejected(path):
    with pytest.raises(ValueError, match="Artifacts must"):
        artifact_path(path)


def test_traversal_and_input_directory_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        artifact_path(tmp_path / "run" / ".." / "original.json")
    with pytest.raises(ValueError, match="input/model"):
        atomic_json(tmp_path / "models" / "manifest.json", {"overwrite": True})


def test_symlink_destination_preserves_original_file(tmp_path):
    original = tmp_path / "original.json"
    original.write_text("original")
    destination = tmp_path / "output.json"
    destination.symlink_to(original)
    with pytest.raises(ValueError, match="symbolic"):
        atomic_json(destination, {"overwrite": True})
    assert original.read_text() == "original"


def test_predictable_temporary_link_cannot_redirect_metadata_write(tmp_path):
    original = tmp_path / "original.json"
    original.write_text("original")
    destination = tmp_path / "execution.json"
    destination.with_name(destination.name + ".tmp").symlink_to(original)
    atomic_json(destination, {"phase": "running"})
    assert original.read_text() == "original"
    assert json.loads(destination.read_text()) == {"phase": "running"}
