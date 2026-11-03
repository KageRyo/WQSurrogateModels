"""Confine research writes to result trees, container outputs or private scratch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def artifact_path(path):
    """Reject traversal, links and destinations outside designated artifact areas.

    Input/model directories are never writable destinations. CLI entry points
    additionally refuse existing run/evidence paths, or validate frozen resume
    identities before updating their own metadata.
    """
    raw = Path(path).absolute()
    if ".." in raw.parts:
        raise ValueError("Artifact paths must not contain traversal components")
    if any(p.is_symlink() for p in [raw, *raw.parents]):
        raise ValueError("Artifact paths must not traverse symbolic links")
    resolved = raw.resolve()
    if any(name in {"data", "models", ".git"} for name in resolved.parts):
        raise ValueError("Refuse to write into an input/model/Git directory")
    scratch = Path(tempfile.gettempdir()).resolve()
    permitted = scratch in resolved.parents or Path("/output") in resolved.parents
    result_roots = [p for p in resolved.parents if p.name == "results"]
    if not permitted and not result_roots:
        raise ValueError(
            "Artifacts must be inside results/, /output/, or private temporary scratch"
        )
    return resolved


def atomic_json(path, value):
    target = artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive random temporaries prevent an existing predictable .tmp link
    # from redirecting a metadata update into a model or other unrelated file.
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        artifact_path(target)  # Reject a link introduced before publication.
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
