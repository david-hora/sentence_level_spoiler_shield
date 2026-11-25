import hashlib
import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from spoiler_shield.model_adapter import TfidfSpoilerPredictor
from spoiler_shield.model_export import create_export_stage, publish_export


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.gettempdir()).resolve()
        self.output = self.temp_root / ("spoiler-export-test-" + uuid4().hex)
        self.output.mkdir()
        self.addCleanup(self.remove_test_output)

    def remove_test_output(self) -> None:
        resolved = self.output.resolve()
        if resolved.parent != self.temp_root or not resolved.name.startswith("spoiler-export-test-"):
            raise RuntimeError("Refusing to clean up an unexpected test directory")
        shutil.rmtree(resolved)

    def stage(self, data: bytes):
        stage = create_export_stage(self.output)
        (stage / "model.joblib").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        metadata = {"artifact_sha256": digest, "model_id": "tfidf-a-" + digest[:12]}
        return stage, metadata

    def assert_manifest_matches_artifact(self) -> dict:
        metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        artifact = self.output / metadata["artifact_file"]
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), metadata["artifact_sha256"])
        self.assertEqual(metadata["model_id"], "tfidf-a-" + metadata["artifact_sha256"][:12])
        return metadata

    def test_interruption_before_manifest_preserves_previous_export(self) -> None:
        old_stage, old_metadata = self.stage(b"old model")
        old_artifact = publish_export(self.output, old_stage, old_metadata)
        old_manifest = (self.output / "metadata.json").read_bytes()
        new_stage, new_metadata = self.stage(b"new model")
        original_replace = Path.replace

        def interrupt_manifest(path, target):
            if path.name == "metadata.json":
                # The old reader still has its matching artifact even after
                # the new immutable file has been published.
                self.assert_manifest_matches_artifact()
                raise OSError("Simulated interruption before manifest commit")
            return original_replace(path, target)

        with patch.object(Path, "replace", interrupt_manifest):
            with self.assertRaisesRegex(OSError, "Simulated interruption"):
                publish_export(self.output, new_stage, new_metadata)
        self.assertEqual((self.output / "metadata.json").read_bytes(), old_manifest)
        self.assertEqual(old_artifact.read_bytes(), b"old model")
        self.assert_manifest_matches_artifact()

    def test_concurrent_exports_always_publish_a_complete_pair(self) -> None:
        inputs = [self.stage(f"model {index}".encode()) for index in range(6)]
        self.assertEqual(len({stage for stage, _ in inputs}), 6)
        with ThreadPoolExecutor(max_workers=3) as workers:
            artifacts = list(workers.map(lambda pair: publish_export(self.output, *pair), inputs))
        selected = self.assert_manifest_matches_artifact()
        self.assertIn(selected["artifact_file"], {path.name for path in artifacts})
        for artifact in artifacts:
            self.assertTrue(artifact.is_file())

    def test_invalid_metadata_does_not_replace_published_pair(self) -> None:
        stage, metadata = self.stage(b"valid")
        publish_export(self.output, stage, metadata)
        stage, metadata = self.stage(b"invalid")
        metadata["model_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "metadata"):
            publish_export(self.output, stage, metadata)
        self.assertEqual(self.assert_manifest_matches_artifact()["artifact_sha256"],
                         hashlib.sha256(b"valid").hexdigest())

    def test_loader_accepts_atomically_published_export(self) -> None:
        source = Path(__file__).resolve().parents[1] / "models" / "tfidf"
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        data = (source / metadata.get("artifact_file", "model.joblib")).read_bytes()
        stage, _ = self.stage(data)
        publish_export(self.output, stage, metadata)
        model = TfidfSpoilerPredictor(self.output)
        self.assertEqual(model.model_id, metadata["model_id"])
        self.assertEqual(model.threshold, metadata["threshold"])
