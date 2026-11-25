import unittest
from pathlib import Path
from unittest.mock import patch

from spoiler_shield.catalog import CatalogError, load_catalog


class CatalogTests(unittest.TestCase):
    def test_real_catalog_loads_all_annotated_movies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        movies = load_catalog(root.parent / "data" / "plot_synopses.json", root / "data" / "movie_titles.json")
        self.assertEqual(len(movies), 80)
        self.assertEqual(movies, sorted(movies, key=lambda movie: (movie.title.casefold(), movie.year or 0)))
        self.assertIn("The Dark Knight", {movie.title for movie in movies})

    def test_catalog_rejects_missing_title(self) -> None:
        with patch("spoiler_shield.catalog._read_json", side_effect=[{"tt1": {}}, {}]):
            with self.assertRaisesRegex(CatalogError, "missing"):
                load_catalog(Path("synopses.json"), Path("titles.json"))
