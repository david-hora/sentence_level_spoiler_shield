"""Download and verify the immutable, data-only inference package once per host."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import Request, urlopen
import uuid
import zipfile

from spoiler_shield.model_adapter import ModelLoadError

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "models" / "distilbert" / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> None:
    if manifest.get("format_version") != 1:
        raise ModelLoadError("Unsupported DistilBERT inference package")
    digest = manifest.get("archive_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ModelLoadError("Invalid model package checksum")
    if manifest.get("model_id") != "distilbert-synopsis-" + digest[:12]:
        raise ModelLoadError("Model ID and inference package checksum differ")
    if not isinstance(manifest.get("archive_size"), int) or not 0 < manifest["archive_size"] < 1024**3:
        raise ModelLoadError("Invalid model package size")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelLoadError("Missing model package file checksums")
    for name, record in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or path.parts[0] not in ("classifier", "retriever"):
            raise ModelLoadError("Invalid model package path")
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise ModelLoadError("Invalid model file size")
        file_hash = record.get("sha256", "")
        if len(file_hash) != 64 or any(c not in "0123456789abcdef" for c in file_hash):
            raise ModelLoadError("Invalid model file checksum")


def verify_directory(directory: Path, manifest: dict) -> bool:
    for name, record in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            return False
        if sha256_file(path) != record["sha256"]:
            return False
    return True


def extract_verified_archive(archive: Path, destination: Path, manifest: dict) -> None:
    """Only allow the manifest's regular files, never archive-provided paths/code."""
    if archive.stat().st_size != manifest["archive_size"] or sha256_file(archive) != manifest["archive_sha256"]:
        raise ModelLoadError("The downloaded inference package failed its checksum")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(manifest["files"]) or {m.filename for m in members} != set(manifest["files"]):
            raise ModelLoadError("Unexpected files in the inference package")
        for member in members:
            if member.file_size != manifest["files"][member.filename]["size"] or member.is_dir():
                raise ModelLoadError("Invalid inference package member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ModelLoadError("Model package symlinks are not supported")
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if not verify_directory(destination, manifest):
        raise ModelLoadError("The extracted model files failed their checksums")


def ensure_model_package(cache_root: Path | None = None) -> tuple[Path, dict]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        # A writable cache outside the checkout also works on Streamlit Cloud.
        cache_root = cache_root or Path(tempfile.gettempdir()) / "spoiler-shield-models"
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / manifest["model_id"]
        if target.exists():
            if not verify_directory(target, manifest):
                raise ModelLoadError("The cached inference package is corrupt; clear its cache and retry")
            return target, manifest
        staging = cache_root / (".download-" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            archive = staging / "runtime.zip"
            request = Request(manifest["archive_url"], headers={"User-Agent": "SpoilerShield/1.0"})
            received = 0
            with urlopen(request, timeout=60) as response, archive.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > manifest["archive_size"]:
                        raise ModelLoadError("Model download exceeds its expected size")
                    output.write(chunk)
            extracted = staging / "extracted"
            extracted.mkdir()
            extract_verified_archive(archive, extracted, manifest)
            try:
                os.rename(extracted, target)
            except OSError:
                # Another process may have published this same immutable export.
                if not target.is_dir() or not verify_directory(target, manifest):
                    raise
        finally:
            shutil.rmtree(staging)
        return target, manifest
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError("The DistilBERT inference package could not be downloaded or verified") from exc
