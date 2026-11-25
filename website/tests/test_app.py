import unittest
from html import escape
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import streamlit as st
from spoiler_shield.model_adapter import Prediction
from spoiler_shield.reviews import load_reviews


ROOT = Path(__file__).resolve().parents[1]
MOVIE_ID = "tt0050083"


class FakePredictor:
    """Keep UI tests offline; the supplied weights have a separate integration test."""
    name = "DistilBERT + movie synopsis"
    is_mock = False
    model_id = "distilbert-synopsis-ui-provenance-sentinel-49281"

    def predict(self, movie, sentences):
        return [Prediction(text, False, 0.1, "UI test prediction") for text in sentences]


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()
        mock = patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor", FakePredictor)
        mock.start()
        self.addCleanup(mock.stop)
        self.addCleanup(st.cache_resource.clear)

    def open_movie(self) -> AppTest:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        app.button(key=f"open_{MOVIE_ID}").click().run()
        self.assertFalse(app.exception)
        return app

    def open_composer(self, app: AppTest) -> None:
        app.get_by_key("detail_view").set_value("Add a review").run()
        self.assertFalse(app.exception)

    def submit(self, app: AppTest, headline="A new review", text="Great acting. The culprit is revealed.") -> None:
        app.text_input(key=f"review_headline_{MOVIE_ID}").set_value(headline)
        app.text_area(key=f"review_text_{MOVIE_ID}").set_value(text)
        app.button(key="submit_review").click().run()
        self.assertFalse(app.exception)

    def html(self, app: AppTest) -> str:
        return "\n".join(element.value for element in app.get("html"))

    def test_library_and_movie_review_views_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "app.py"), default_timeout=15).run()

        self.assertFalse(app.exception)
        self.assertIsNotNone(app.text_input(key="movie_search"))
        self.assertEqual(sum(str(button.key).startswith("open_") for button in app.button), 12)
        self.assertEqual(app.button(key="hero_add_review").label, "Add a Review")

        app.button(key="open_tt0050083").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(title.value == "12 Angry Men" for title in app.title))
        dataset = load_reviews(ROOT.parent / "data" / "spoiler_detection_dataset_1.3.1.json")[MOVIE_ID]
        self.assertEqual(self.html(app).count('<header class="review-heading">'), 5)
        for review in dataset[:5]:
            self.assertIn(f"<h3>{escape(review.summary)}</h3>", self.html(app))
        self.assertNotIn(f"<h3>{escape(dataset[5].summary)}</h3>", self.html(app))

    def test_opening_another_movie_resets_to_reviews(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        app.button(key="back_to_library").click().run()
        # Streamlit can clean up an unmounted widget; seed the stale value to
        # ensure the navigation callback itself also guarantees the default.
        app.session_state["detail_view"] = "Add a review"
        other_movie = next(button for button in app.button
                           if str(button.key).startswith("open_") and button.key != f"open_{MOVIE_ID}")
        other_movie_id = other_movie.key.removeprefix("open_")
        other_movie.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_movie_id"], other_movie_id)
        self.assertEqual(app.session_state["detail_view"], "Reviews")
        self.assertEqual(self.html(app).count('<header class="review-heading">'), 5)

    def test_submission_updates_totals_page_and_is_idempotent(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        app.session_state[f"review_page_{MOVIE_ID}"] = 4
        predictions = [Prediction("Great acting.", False, 0.1, "Safe"),
                       Prediction(" The culprit is revealed.", True, 0.9, "Classifier evidence")]
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict", return_value=predictions) as predict:
            self.submit(app)
            self.assertEqual(len(app.session_state["added_reviews"][MOVIE_ID]), 1)
            self.assertEqual(app.session_state["added_review_counter"], 1)
            self.assertEqual(app.session_state[f"review_page_{MOVIE_ID}"], 1)
            self.assertIn("<dt>Reviews</dt><dd>41</dd>", self.html(app))
            self.assertIn("<dt>Hidden sentences</dt><dd>65</dd>", self.html(app))
            self.assertTrue(app.success)
            self.submit(app)
            self.assertEqual(len(app.session_state["added_reviews"][MOVIE_ID]), 1)
            self.assertEqual(app.session_state["added_review_counter"], 1)
            self.assertEqual(predict.call_count, 1)
            self.assertTrue(app.info)
            # A different submission is allowed; retrying the first later is
            # still idempotent, rather than merely checking the last click.
            self.submit(app, headline="Another review")
            self.submit(app)
            self.assertEqual(len(app.session_state["added_reviews"][MOVIE_ID]), 2)
            self.assertEqual(predict.call_count, 2)
        app.get_by_key("detail_view").set_value("Reviews").run()
        self.assertFalse(app.exception)
        self.assertEqual(self.html(app).count('<header class="review-heading">'), 5)
        self.assertIn("<h3>A new review</h3>", self.html(app))

    def test_legacy_session_review_without_fingerprint_is_not_added_again(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        legacy = {
            "id": 1, "summary": "A new review", "rating": 8,
            "predictions": [
                Prediction("Great acting.", False, 0.1, "Safe"),
                Prediction(" The culprit is revealed.", True, 0.9, "Mock explanation"),
            ],
        }
        app.session_state["added_reviews"] = {MOVIE_ID: [legacy]}
        app.session_state["added_review_counter"] = 1
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict") as predict:
            self.submit(app)
            predict.assert_not_called()
        self.assertTrue(app.info)
        self.assertEqual(app.session_state["added_review_counter"], 1)
        self.assertEqual(app.session_state["added_reviews"], {MOVIE_ID: [legacy]})
        self.assertIn("<dt>Reviews</dt><dd>41</dd>", self.html(app))

    def test_submission_success_is_not_replayed_for_a_new_draft(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        self.submit(app)
        self.assertTrue(app.success)
        app.get_by_key("detail_view").set_value("Reviews").run()
        self.open_composer(app)
        self.assertFalse(app.success)
        self.assertFalse(app.info)
        self.assertNotIn('<header class="review-heading">', self.html(app))
        self.submit(app)
        self.assertTrue(app.info)
        app.run()
        self.assertFalse(app.info)
        self.assertFalse(app.success)
        self.assertEqual(len(app.session_state["added_reviews"][MOVIE_ID]), 1)

    def test_session_limit_applies_across_movies_and_preserves_draft(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        stored = [{"id": index, "summary": "Saved review", "rating": 8, "predictions": []}
                  for index in range(1, 101)]
        app.session_state["added_reviews"] = {"tt0468569": stored}
        app.session_state["added_review_counter"] = 100
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict") as predict:
            self.submit(app, text="Keep this draft.")
            predict.assert_not_called()
        self.assertTrue(app.error)
        self.assertIn("limit of 100", app.error[0].value)
        self.assertEqual(app.session_state["added_review_counter"], 100)
        self.assertEqual(app.session_state["added_reviews"], {"tt0468569": stored})
        self.assertEqual(app.text_area(key=f"review_text_{MOVIE_ID}").value, "Keep this draft.")

    def test_last_available_slot_and_duplicate_at_limit(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        app.session_state["added_reviews"] = {"tt0468569": [
            {"id": index, "summary": "Saved review", "rating": 8, "predictions": []}
            for index in range(1, 100)
        ]}
        app.session_state["added_review_counter"] = 99
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict",
                   side_effect=lambda movie, chunks: [Prediction(text, False, 0.1, "Safe") for text in chunks]) as predict:
            self.submit(app)
            self.assertTrue(app.success)
            self.submit(app)
            self.assertTrue(app.info)
            self.submit(app, headline="Review beyond the limit")
            self.assertTrue(app.error)
            self.assertEqual(predict.call_count, 1)
        self.assertEqual(app.session_state["added_review_counter"], 100)
        self.assertEqual(sum(map(len, app.session_state["added_reviews"].values())), 100)

    def test_failed_analysis_is_logged_and_can_be_retried(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict", side_effect=RuntimeError("Predictor offline")):
            with self.assertLogs("spoiler_shield.analysis", level="ERROR") as logs:
                self.submit(app, text="Keep my review.")
        self.assertIn("Predictor offline", "\n".join(logs.output))
        self.assertIn(MOVIE_ID, "\n".join(logs.output))
        self.assertTrue(app.error)
        self.assertEqual(app.session_state["added_reviews"], {})
        self.assertEqual(app.session_state["added_review_counter"], 0)
        self.assertEqual(app.text_area(key=f"review_text_{MOVIE_ID}").value, "Keep my review.")
        self.submit(app, text="Keep my review.")
        self.assertTrue(app.success)
        self.assertEqual(len(app.session_state["added_reviews"][MOVIE_ID]), 1)

    def test_submission_records_model_provenance(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        self.assertIn("Trained spoiler detection", self.html(app))
        self.assertNotIn("mock predictor", self.html(app))
        self.submit(app, text="Lovely acting.\n\nAt the end he dies and the killer is revealed.")
        saved = app.session_state["added_reviews"][MOVIE_ID][0]
        self.assertEqual(saved["model_name"], "DistilBERT + movie synopsis")
        self.assertEqual(saved["model_id"], FakePredictor.model_id)
        self.assertEqual("".join(item.text for item in saved["predictions"]),
                         "Lovely acting.\n\nAt the end he dies and the killer is revealed.")
        self.assertIn("Your review (DistilBERT + movie synopsis)", self.html(app))

    def test_invalid_prediction_contract_keeps_draft(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict", return_value=[]):
            with self.assertLogs("spoiler_shield.analysis", level="ERROR"):
                self.submit(app)
        self.assertTrue(app.error)
        self.assertEqual(app.session_state["added_reviews"], {})
        self.assertEqual(app.text_area(key=f"review_text_{MOVIE_ID}").value,
                         "Great acting. The culprit is revealed.")

    def test_safe_review_shows_its_percentage(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor.predict",
                   return_value=[Prediction("Lovely acting.", False, 0.123, "Below threshold")]):
            self.submit(app, text="Lovely acting.")
        self.assertTrue(app.success)
        self.assertIn("No sentences hidden", self.html(app))
        self.assertIn('>12.3%</span>', self.html(app))
        self.assertNotIn('<details class="spoiler-mask">', self.html(app))

    def test_sentence_limit_preserves_draft_without_loading_predictor(self) -> None:
        app = self.open_movie()
        self.open_composer(app)
        text = "Short. " * 101
        with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor") as factory:
            self.submit(app, text=text)
            factory.assert_not_called()
        self.assertTrue(app.error)
        self.assertIn("100 sentences", app.error[0].value)
        self.assertEqual(app.text_area(key=f"review_text_{MOVIE_ID}").value, text)
        self.assertEqual(app.session_state["added_reviews"], {})
        self.assertEqual(app.session_state["added_review_counter"], 0)

    def test_incomplete_adapter_keeps_draft_without_incrementing_counter(self) -> None:
        class IncompleteAdapter:
            name = "Incomplete adapter"
            is_mock = False

            def predict(self, movie, sentences):
                return [Prediction(text, False, 0.1, "Safe") for text in sentences]

        app = self.open_movie()
        self.open_composer(app)
        st.cache_resource.clear()
        try:
            with patch("spoiler_shield.distilbert_adapter.DistilbertSpoilerPredictor", return_value=IncompleteAdapter()):
                with self.assertLogs("spoiler_shield.analysis", level="ERROR"):
                    self.submit(app)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state["added_review_counter"], 0)
            self.assertEqual(app.session_state["added_reviews"], {})
            self.assertEqual(app.text_area(key=f"review_text_{MOVIE_ID}").value,
                             "Great acting. The culprit is revealed.")
        finally:
            st.cache_resource.clear()
