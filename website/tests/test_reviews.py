import unittest
import json
from pathlib import Path
from unittest.mock import mock_open, patch

from spoiler_shield.reviews import load_reviews


class ReviewDatasetTests(unittest.TestCase):
    def test_full_dataset_loads_and_groups_reviews_by_movie(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reviews = load_reviews(root.parent / "data" / "spoiler_detection_dataset_1.3.1.json")

        self.assertEqual(len(reviews), 80)
        self.assertEqual(sum(len(items) for items in reviews.values()), 3200)
        self.assertTrue(all(len(items) == 40 for items in reviews.values()))
        self.assertEqual(
            sum(len(review.sentences) for items in reviews.values() for review in items),
            60189,
        )

    def test_reviews_expose_sentence_level_spoiler_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reviews = load_reviews(root.parent / "data" / "spoiler_detection_dataset_1.3.1.json")
        dark_knight = reviews["tt0468569"]

        self.assertEqual(len(dark_knight), 40)
        all_reviews = [review for items in reviews.values() for review in items]
        self.assertEqual(sum(review.rating is not None for review in all_reviews), 3200)
        for review in all_reviews:
            self.assertIsNotNone(review.rating)
            self.assertTrue(1 <= review.rating <= 10)
        self.assertEqual(dark_knight[0].rating, 10)
        for movie_id, expected in {"tt0468569": 67, "tt0050083": 64, "tt0111161": 77}.items():
            with self.subTest(movie=movie_id):
                self.assertEqual(sum(review.spoiler_count for review in reviews[movie_id]), expected)
        self.assertEqual(sum(review.spoiler_count for review in all_reviews), 4677)

    def test_loader_retains_boundary_whitespace(self) -> None:
        texts = ["  First paragraph.\n\n", "\tA spoiler.\r\n\r\n", " Last paragraph.  "]
        payload = {"data": [{
            "movie_id": "example", "rating": "8",
            "sentences": [{"text": text, "is_spoiler": index == 1}
                          for index, text in enumerate(texts)],
        }]}
        with patch("pathlib.Path.open", mock_open(read_data=json.dumps(payload))):
            review = load_reviews(Path("fixture.json"))["example"][0]
        self.assertEqual([sentence.text for sentence in review.sentences], texts)
        self.assertEqual([sentence.is_spoiler for sentence in review.sentences], [False, True, False])
