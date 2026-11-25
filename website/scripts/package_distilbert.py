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
RETRIEVER_SHA256 = {
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
RETRIEVER_FILES = set(RETRIEVER_SHA256)
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_SHA256 = "ef24ce49eab8d89c2afd851e8a4fc3c58717964d03dd65f60da27b612425c04d"
URL = "https://github.com/david-hora/sentence_level_spoiler_shield/releases/download/distilbert-synopsis-2026-11-25/spoiler-shield-runtime.zip"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_retriever(directory: Path) -> None:
    for name, expected in RETRIEVER_SHA256.items():
        if checksum(directory / name) != expected:
            raise ValueError(f"Retriever file does not match pinned revision {REVISION}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-zip", type=Path, required=True)
    parser.add_argument("--retriever-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if checksum(args.classifier_zip) != SOURCE_SHA256:
        raise ValueError("This packager is pinned to the supplied 25 Nov 2026 export")
    validate_retriever(args.retriever_directory)
    records = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.classifier_zip) as source, zipfile.ZipFile(
            args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        if set(source.namelist()) != CLASSIFIER_FILES:
            raise ValueError("Unexpected classifier export files")
        for folder, names in (("classifier", CLASSIFIER_FILES), ("retriever", RETRIEVER_FILES)):
            for name in sorted(names):
                data = source.read(name) if folder == "classifier" else (args.retriever_directory / name).read_bytes()
                if folder == "retriever" and hashlib.sha256(data).hexdigest() != RETRIEVER_SHA256[name]:
                    raise ValueError(f"Retriever file changed after validation: {name}")
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
