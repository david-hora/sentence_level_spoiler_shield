import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
import uuid
import zipfile
from urllib.request import urlopen

from scripts import package_distilbert

from spoiler_shield.catalog import Movie, load_catalog
from spoiler_shield.distilbert_adapter import (
    DistilbertSpoilerPredictor, MAX_REVIEW_SENTENCES, position_tag, select_context, synopsis_sentences, validate_settings,
)
from spoiler_shield.model_adapter import ModelLoadError
from spoiler_shield.model_assets import (
    MANIFEST_PATH, ensure_model_package, extract_verified_archive, sha256_file, validate_manifest,
)


class SynopsisTests(unittest.TestCase):
    def test_too_many_sentences_rejected_before_loading_or_using_models(self):
        predictor = object.__new__(DistilbertSpoilerPredictor)
        with self.assertRaisesRegex(ValueError, "at most 100 sentences"):
            predictor.predict(Movie("x", "Test", None, "", "", ()), ["Short."] * (MAX_REVIEW_SENTENCES + 1))

    def test_short_synopsis_uses_summary_and_removes_author(self):
        movie = Movie("x", "Test", None, "Short synopsis.",
                      "First event. Second event. Written by a contributor", ())
        self.assertEqual(synopsis_sentences(movie), [("First event.", 0.0), ("Second event.", 1.0)])
        self.assertEqual(synopsis_sentences(Movie("x", "Empty", None, "", "", ())), [])

    def test_retrieval_restores_plot_order_and_position_boundaries(self):
        sentences = [("Start.", 0.0), ("Early.", 0.2), ("Middle.", 0.5), ("Ending.", 0.75)]
        self.assertEqual(select_context(sentences, [0.1, 0.8, 0.7, 0.9]),
                         "(beginning) Early. (middle) Middle. (end) Ending.")
        self.assertEqual(position_tag(0.2), "(beginning)")
        self.assertEqual(position_tag(0.75), "(end)")

    def test_export_settings_reject_unsupported_input_changes(self):
        # The export's supported inference settings.
        settings = {"threshold": 0.2114982306957245, "max_length": 256, "truncation": "longest_first",
                    "top_k_synopsis_sentences": 3, "retriever": "sentence-transformers/all-MiniLM-L6-v2",
                    "position_tags": {"beginning": "pos <= 0.2", "end": "pos >= 0.75", "middle": "otherwise"}}
        validate_settings(settings)
        for changes in ({"threshold": float("nan")}, {"max_length": 512}, {"top_k_synopsis_sentences": 1}):
            with self.assertRaises(ModelLoadError):
                validate_settings(settings | changes)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / ("spoiler-package-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.archive = self.root / "test.zip"
        self.content = b"model test bytes"
        with zipfile.ZipFile(self.archive, "w") as bundle:
            bundle.writestr("classifier/config.json", self.content)
        digest = sha256_file(self.archive)
        self.manifest = {"format_version": 1, "model_id": "distilbert-synopsis-" + digest[:12],
                         "archive_sha256": digest, "archive_size": self.archive.stat().st_size,
                         "archive_url": self.archive.as_uri(),
                         "files": {"classifier/config.json": {"size": len(self.content),
                                   "sha256": hashlib.sha256(self.content).hexdigest()}}}
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_download_publish_reuse_and_corruption_detection(self):
        cache = self.root / "cache"
        with patch("spoiler_shield.model_assets.MANIFEST_PATH", self.manifest_path):
            target, _ = ensure_model_package(cache)
            self.assertEqual((target / "classifier/config.json").read_bytes(), self.content)
            with patch("spoiler_shield.model_assets.urlopen", side_effect=AssertionError("Must use cache")):
                self.assertEqual(ensure_model_package(cache)[0], target)
            (target / "classifier/config.json").write_bytes(b"corrupt")
            with self.assertRaises(ModelLoadError):
                ensure_model_package(cache)

    def test_checksum_failure_does_not_publish_and_cleans_staging(self):
        self.manifest["files"]["classifier/config.json"]["sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        cache = self.root / "cache"
        with patch("spoiler_shield.model_assets.MANIFEST_PATH", self.manifest_path):
            with self.assertRaises(ModelLoadError):
                ensure_model_package(cache)
        # Native backends may retain the small lock file after releasing it.
        self.assertLessEqual({path.name for path in cache.iterdir()}, {".download.lock"})

    def test_abandoned_staging_removed_even_when_model_already_cached(self):
        cache = self.root / "cache"
        with patch("spoiler_shield.model_assets.MANIFEST_PATH", self.manifest_path):
            target, _ = ensure_model_package(cache)
            abandoned = cache / (".download-" + "a" * 32)
            abandoned.mkdir()
            (abandoned / "runtime.zip").write_bytes(b"interrupted download")
            unrelated = cache / ".download-unrelated"
            unrelated.mkdir()
            self.assertEqual(ensure_model_package(cache)[0], target)
            self.assertFalse(abandoned.exists())
            self.assertTrue(unrelated.exists())

    def test_concurrent_loaders_do_not_remove_active_download_or_download_twice(self):
        cache = self.root / "cache"
        started, release = Event(), Event()

        def delayed_download(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError("Test download was not released")
            return urlopen(*args, **kwargs)

        with patch("spoiler_shield.model_assets.MANIFEST_PATH", self.manifest_path), \
                patch("spoiler_shield.model_assets.urlopen", side_effect=delayed_download) as download, \
                ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(ensure_model_package, cache)
            try:
                self.assertTrue(started.wait(5))
                second = executor.submit(ensure_model_package, cache)
                self.assertFalse(second.done())
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5)[0], second.result(timeout=5)[0])
            self.assertEqual(download.call_count, 1)

    def test_packager_rejects_wrong_retriever_bytes(self):
        retriever = self.root / "retriever"
        retriever.mkdir()
        file = retriever / "config.json"
        file.write_bytes(b"pinned snapshot")
        hashes = {"config.json": hashlib.sha256(file.read_bytes()).hexdigest()}
        with patch.object(package_distilbert, "RETRIEVER_SHA256", hashes):
            package_distilbert.validate_retriever(retriever)
            file.write_bytes(b"different snapshot")
            with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
                package_distilbert.validate_retriever(retriever)

    def test_rejects_traversal_and_changed_archive(self):
        with self.assertRaises(ModelLoadError):
            validate_manifest(self.manifest | {"files": {"classifier/../../outside": {"size": 0, "sha256": "0"*64}}})
        with self.archive.open("ab") as archive:
            archive.write(b"changed")
        with self.assertRaises(ModelLoadError):
            extract_verified_archive(self.archive, self.root / "unpublished", self.manifest)


@unittest.skipUnless(os.environ.get("SPOILER_SHIELD_TEST_MODEL_DIR"), "Set SPOILER_SHIELD_TEST_MODEL_DIR for real local weights")
class SuppliedWeightsTests(unittest.TestCase):
    def test_real_scores_threshold_formatting_and_empty_context(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        predictor = DistilbertSpoilerPredictor(Path(os.environ["SPOILER_SHIELD_TEST_MODEL_DIR"]), manifest)
        root = Path(__file__).resolve().parents[1]
        movie = next(m for m in load_catalog(root.parent / "data/plot_synopses.json", root / "data/movie_titles.json")
                     if m.movie_id == "tt0803096")
        texts = ["  The acting is excellent.\n\n", "At the end, Medivh dies and the portal is destroyed.", "\n"]
        predictions = predictor.predict(movie, texts)
        self.assertEqual([p.text for p in predictions], texts)
        self.assertEqual([p.is_spoiler for p in predictions], [False, True, False])
        self.assertAlmostEqual(predictions[0].score, 0.002994347, places=4)
        self.assertAlmostEqual(predictions[1].score, 0.968612432, places=4)
        self.assertEqual(predictor.threshold, manifest["threshold"])
        self.assertEqual(predictor.predict(movie, []), [])
        self.assertEqual(len(predictor.predict(Movie("x", "No context", None, "", "", ()), ["Good acting."])), 1)


if __name__ == "__main__":
    unittest.main()
