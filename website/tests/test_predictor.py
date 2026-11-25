import unittest

from spoiler_shield.catalog import Movie
from spoiler_shield.model_adapter import MockSpoilerPredictor


MOVIE = Movie(
    movie_id="tt0000001",
    title="Example",
    year=2020,
    synopsis="A detective learns that Morgan is responsible for the crime.",
    plot_summary="A detective investigates a mystery.",
    genres=("Mystery",),
)


class PredictorTests(unittest.TestCase):
    def test_mock_is_deterministic_and_keeps_order(self) -> None:
        sentences = ["Beautiful cinematography. ", "Morgan turns out to be the killer!"]
        predictor = MockSpoilerPredictor()
        first = predictor.predict(MOVIE, sentences)
        second = predictor.predict(MOVIE, sentences)
        self.assertEqual(first, second)
        self.assertEqual([item.text for item in first], sentences)
        self.assertFalse(first[0].is_spoiler)
        self.assertTrue(first[1].is_spoiler)
