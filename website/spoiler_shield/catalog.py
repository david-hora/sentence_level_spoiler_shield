from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CatalogError(RuntimeError):
    """Raised when the movie catalog cannot be validated."""


@dataclass(frozen=True)
class Movie:
    movie_id: str
    title: str
    year: int | None
    synopsis: str
    plot_summary: str
    genres: tuple[str, ...]
    rating: str = ""
    duration: str = ""
    release_date: str = ""


def _read_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise CatalogError(f"missing file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"invalid JSON in {path.name} (line {exc.lineno})") from exc


def load_catalog(synopses_path: Path, titles_path: Path) -> list[Movie]:
    synopses = _read_json(synopses_path)
    titles = _read_json(titles_path)
    if not isinstance(synopses, dict) or not isinstance(titles, dict):
        raise CatalogError("catalog files must contain JSON objects")

    missing_titles = sorted(set(synopses) - set(titles))
    if missing_titles:
        raise CatalogError(f"{len(missing_titles)} movie title(s) are missing from movie_titles.json")

    movies: list[Movie] = []
    for movie_id, record in synopses.items():
        title_record = titles[movie_id]
        if not isinstance(record, dict) or not isinstance(title_record, dict):
            raise CatalogError(f"invalid catalog entry for {movie_id}")
        title = str(title_record.get("title", "")).strip()
        if not title:
            raise CatalogError(f"empty title for {movie_id}")
        genres = record.get("genre") or []
        movies.append(
            Movie(
                movie_id=movie_id,
                title=title,
                year=title_record.get("year"),
                synopsis=str(record.get("plot_synopsis") or ""),
                plot_summary=str(record.get("plot_summary") or ""),
                genres=tuple(str(item) for item in genres),
                rating=str(record.get("rating") or ""),
                duration=str(record.get("duration") or ""),
                release_date=str(record.get("release_date") or ""),
            )
        )
    return sorted(movies, key=lambda movie: (movie.title.casefold(), movie.year or 0))
