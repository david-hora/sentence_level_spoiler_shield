"""Publish immutable model files through one atomically replaced manifest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def create_export_stage(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stage = output / (".export-" + uuid4().hex)
    stage.mkdir()
    return stage


def publish_export(output: Path, stage: Path, metadata: dict) -> Path:
    """Commit the manifest only after its immutable artifact is fully available.

    Readers load metadata.json once and use the named artifact. Older artifacts
    stay in place, so interruption or concurrent publication cannot invalidate a
    reader's snapshot. A failed publication leaves the previous manifest intact.
    """
    if stage.resolve().parent != output.resolve() or not stage.name.startswith(".export-"):
        raise ValueError("Export staging directory must be inside the output directory")
    staged_artifact = stage / "model.joblib"
    digest = hashlib.sha256(staged_artifact.read_bytes()).hexdigest()
    if metadata["artifact_sha256"] != digest or metadata["model_id"] != "tfidf-a-" + digest[:12]:
        raise ValueError("Export metadata does not identify its staged artifact")
    filename = f"model-{digest[:16]}.joblib"
    destination = output / filename
    manifest = {**metadata, "artifact_file": filename}
    staged_manifest = stage / "metadata.json"
    with staged_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("An artifact filename collision was detected")
    else:
        staged_artifact.replace(destination)
    # This single atomic replacement publishes both the artifact reference and
    # its matching metadata. Concurrent writers can only select a complete pair.
    staged_manifest.replace(output / "metadata.json")
    if staged_artifact.exists():
        staged_artifact.unlink()
    stage.rmdir()
    return destination
