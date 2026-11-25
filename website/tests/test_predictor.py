import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from spoiler_shield.catalog import Movie
from spoiler_shield.model_adapter import ModelLoadError, TfidfSpoilerPredictor


ROOT = Path(__file__).resolve().parents[1]
MOVIE = Movie(movie_id="example", title="Example", year=2020, synopsis="A detective investigates.",
              plot_summary="A mystery.", genres=("Mystery",))


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = TfidfSpoilerPredictor()

    def test_export_is_real_deterministic_and_preserves_formatting(self) -> None:
        sentences = ["Beautiful cinematography. ", "\n\nAt the end he dies and the killer is revealed.", "\n\n"]
        first = self.predictor.predict(MOVIE, sentences)
        self.assertEqual(first, self.predictor.predict(MOVIE, sentences))
        self.assertEqual([item.text for item in first], sentences)
        self.assertFalse(self.predictor.is_mock)
        self.assertTrue(all(0 <= item.score <= 1 for item in first))
        self.assertEqual([item.is_spoiler for item in first],
                         [bool(text.strip()) and item.score >= self.predictor.threshold
                          for item, text in zip(first, sentences)])
        self.assertFalse(first[-1].is_spoiler)
        self.assertEqual(self.predictor.predict(MOVIE, []), [])

    def test_export_matches_original_held_out_scores(self) -> None:
        with (ROOT.parent / "results" / "predictions_test.csv").open(encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        # Independent scores produced by the original notebook, spanning both
        # low- and high-scoring test sentences across multiple movies.
        rows = sorted(all_rows, key=lambda row: float(row["score_A"]))
        sample = rows[::max(1, len(rows) // 40)] + rows[-5:]
        predicted = self.predictor.predict(MOVIE, [row["text"] for row in sample])
        for row, prediction in zip(sample, predicted):
            # The original CSV stores scores rounded to four decimal places.
            self.assertAlmostEqual(prediction.score, float(row["score_A"]), delta=0.000050001)
        self.assertTrue(any(item.is_spoiler for item in predicted))
        self.assertTrue(any(not item.is_spoiler for item in predicted))
        for item in predicted:
            if item.is_spoiler:
                self.assertIn("trained text classifier", item.explanation)
                self.assertNotIn("Mock", item.explanation)

    def test_partitions_are_disjoint_and_test_was_not_used_for_threshold(self) -> None:
        metadata = self.predictor.metadata
        groups = metadata["partitions"]
        self.assertEqual([groups[key]["sentences"] for key in ["train", "validation", "test"]],
                         [34756, 9489, 15944])
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            self.assertFalse(set(groups[left]["movie_ids"]) & set(groups[right]["movie_ids"]))
        self.assertEqual(metadata["export_verification"]["max_score_difference"], 0)
        self.assertEqual(metadata["threshold_selection"], "Maximum F1 on validation movies only")

    def test_missing_artifact_fails_without_mock_fallback(self) -> None:
        with self.assertRaises(ModelLoadError):
            TfidfSpoilerPredictor(ROOT / "models" / "missing-model")

    def test_corrupt_artifact_is_rejected_before_deserializing(self) -> None:
        with patch("pathlib.Path.read_bytes", return_value=b"corrupt"):
            with patch("joblib.load") as deserialize:
                with self.assertRaisesRegex(ModelLoadError, "checksum"):
                    TfidfSpoilerPredictor()
                deserialize.assert_not_called()

    def test_incompatible_runtime_is_rejected(self) -> None:
        metadata = json.loads((ROOT / "models" / "tfidf" / "metadata.json").read_text(encoding="utf-8"))
        metadata["versions"]["scikit-learn"] = "incompatible"
        with patch("pathlib.Path.read_text", return_value=json.dumps(metadata)):
            with self.assertRaisesRegex(ModelLoadError, "dependencies"):
                TfidfSpoilerPredictor()

    def test_mismatched_model_id_is_rejected(self) -> None:
        metadata = dict(self.predictor.metadata)
        metadata["model_id"] = "tfidf-a-stale-export"
        with patch("pathlib.Path.read_text", return_value=json.dumps(metadata)):
            with patch("joblib.load") as deserialize:
                with self.assertRaisesRegex(ModelLoadError, "model ID"):
                    TfidfSpoilerPredictor()
                deserialize.assert_not_called()
