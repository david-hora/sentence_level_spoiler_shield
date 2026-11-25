import unittest
from pathlib import Path

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
        self.assertTrue(any(review.spoiler_count > 0 for review in dark_knight))
        self.assertTrue(all(1 <= review.rating <= 10 for review in dark_knight if review.rating))
