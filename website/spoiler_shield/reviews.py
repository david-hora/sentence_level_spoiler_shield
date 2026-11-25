from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ReviewDataError(RuntimeError):
    """Raised when the annotated review dataset cannot be loaded."""


@dataclass(frozen=True)
class AnnotatedSentence:
    text: str
    is_spoiler: bool


@dataclass(frozen=True)
class Review:
    review_id: str
    movie_id: str
    user_id: str
    is_spoiler_review: bool
    rating: int | None
    summary: str
    split: str
    sentences: tuple[AnnotatedSentence, ...]

    @property
    def spoiler_count(self) -> int:
        return sum(sentence.is_spoiler for sentence in self.sentences)


def _parse_rating(value: object) -> int | None:
    try:
        rating = int(str(value))
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 10 else None


def load_reviews(path: Path) -> dict[str, tuple[Review, ...]]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ReviewDataError(f"Missing review dataset: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewDataError(f"Invalid JSON in {path.name} (line {exc.lineno})") from exc

    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ReviewDataError("The annotated review dataset has no valid data array")

    grouped: dict[str, list[Review]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ReviewDataError(f"Invalid review record at position {index}")
        movie_id = str(record.get("movie_id") or "").strip()
        raw_sentences = record.get("sentences")
        if not movie_id or not isinstance(raw_sentences, list):
            raise ReviewDataError(f"Review {index + 1} is missing a movie ID or sentences")

        sentences: list[AnnotatedSentence] = []
        for sentence_index, sentence in enumerate(raw_sentences):
            if not isinstance(sentence, dict) or not isinstance(sentence.get("text"), str):
                raise ReviewDataError(
                    f"Review {index + 1}, sentence {sentence_index + 1} has an invalid format"
                )
            sentences.append(
                AnnotatedSentence(
                    text=sentence["text"],
                    is_spoiler=bool(sentence.get("is_spoiler", False)),
                )
            )

        review = Review(
            review_id=f"dataset-{index}",
            movie_id=movie_id,
            user_id=str(record.get("user_id") or "Unknown user"),
            is_spoiler_review=bool(record.get("is_spoiler_review", False)),
            rating=_parse_rating(record.get("rating")),
            summary=str(record.get("review_summary") or "Untitled review").strip(),
            split=str(record.get("split") or "dataset"),
            sentences=tuple(sentences),
        )
        grouped.setdefault(movie_id, []).append(review)

    return {movie_id: tuple(reviews) for movie_id, reviews in grouped.items()}
