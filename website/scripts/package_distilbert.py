"""Package the supplied classifier with a pinned, local MiniLM snapshot.

This never trains or modifies model weights. The retriever directory must contain
the official snapshot named below (safetensors only). See models/distilbert/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

CLASSIFIER_FILES = {
    "config.json", "model.safetensors", "spoiler_shield_config.json",
    "tokenizer.json", "tokenizer_config.json",
}
RETRIEVER_FILES = {
    "1_Pooling/config.json", "README.md", "config.json", "config_sentence_transformers.json",
    "model.safetensors", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt",
}
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_SHA256 = "ef24ce49eab8d89c2afd851e8a4fc3c58717964d03dd65f60da27b612425c04d"
URL = "https://github.com/david-hora/sentence_level_spoiler_shield/releases/download/distilbert-synopsis-2026-11-25/spoiler-shield-runtime.zip"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-zip", type=Path, required=True)
    parser.add_argument("--retriever-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if checksum(args.classifier_zip) != SOURCE_SHA256:
        raise ValueError("This packager is pinned to the supplied 25 Nov 2026 export")
    records = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.classifier_zip) as source, zipfile.ZipFile(
            args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        if set(source.namelist()) != CLASSIFIER_FILES:
            raise ValueError("Unexpected classifier export files")
        for folder, names in (("classifier", CLASSIFIER_FILES), ("retriever", RETRIEVER_FILES)):
            for name in sorted(names):
                data = source.read(name) if folder == "classifier" else (args.retriever_directory / name).read_bytes()
                member = f"{folder}/{name}"
                entry = zipfile.ZipInfo(member, date_time=(2026, 11, 25, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                output.writestr(entry, data)
                records[member] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    archive_hash = checksum(args.output)
    manifest = {
        "format_version": 1,
        "model_id": "distilbert-synopsis-" + archive_hash[:12],
        "archive_url": URL,
        "archive_sha256": archive_hash,
        "archive_size": args.output.stat().st_size,
        "supplied_classifier_zip_sha256": SOURCE_SHA256,
        "retriever": {"repository": "sentence-transformers/all-MiniLM-L6-v2", "revision": REVISION},
        "training_provenance": "Not supplied. Same architecture/input design as reported model E; exact training run and dataset version unverified.",
        "threshold": 0.2114982306957245,
        "files": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("model_id", "archive_size", "archive_sha256")}, indent=2))


if __name__ == "__main__":
    main()
