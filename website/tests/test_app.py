import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
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
        self.assertGreaterEqual(len(app.get("html")), 6)
