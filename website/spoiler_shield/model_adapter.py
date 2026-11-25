from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from spoiler_shield.catalog import Movie


_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
_STOP_WORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between",
    "could", "does", "from", "have", "into", "just", "more", "movie", "really",
    "some", "than", "that", "their", "there", "these", "they", "this", "through",
    "very", "what", "when", "where", "which", "while", "with", "would", "your",
}
_CUES = {
    "turns out": 0.30,
    "is the killer": 0.34,
    "was the killer": 0.34,
    "final scene": 0.24,
    "at the end": 0.24,
    "the ending": 0.22,
    "reveals": 0.20,
    "revealed": 0.20,
    "actually": 0.10,
    "betrays": 0.24,
    "sacrifices": 0.23,
    "dies": 0.29,
    "killed": 0.27,
    "killer": 0.25,
    "murderer": 0.27,
    "survives": 0.22,
    "defeats": 0.20,
    "identity": 0.18,
}


@dataclass(frozen=True)
class Prediction:
    text: str
    is_spoiler: bool
    score: float
    explanation: str


class SpoilerPredictor(Protocol):
    """Stable boundary between the website and any prediction implementation."""

    name: str
    is_mock: bool

    def predict(self, movie: Movie, sentences: Sequence[str]) -> list[Prediction]:
        """Return exactly one prediction for each input sentence, in the same order."""


class MockSpoilerPredictor:
    """Deterministic UI placeholder. This is not a trained classifier."""

    name = "Rule-based mock predictor"
    is_mock = True
    threshold = 0.56

    def predict(self, movie: Movie, sentences: Sequence[str]) -> list[Prediction]:
        plot_words = self._meaningful_words(f"{movie.plot_summary} {movie.synopsis}")
        results = [self._predict_one(movie, sentence, plot_words) for sentence in sentences]
        if len(results) != len(sentences):
            raise RuntimeError("Predictor returned a different number of results than sentences")
        return results

    def _predict_one(self, movie: Movie, sentence: str, plot_words: set[str]) -> Prediction:
        clean = sentence.strip()
        lower = clean.casefold()
        sentence_words = self._meaningful_words(clean)
        overlap = len(sentence_words & plot_words) / max(1, len(sentence_words))
        cue_matches = [(cue, weight) for cue, weight in _CUES.items() if cue in lower]
        cue_score = min(0.48, sum(weight for _, weight in cue_matches))
        overlap_score = min(0.24, overlap * 0.55)
        # A stable, small variation makes the demo less uniform without randomness.
        digest = hashlib.sha256(f"{movie.movie_id}\0{clean}".encode("utf-8")).digest()
        jitter = (digest[0] / 255) * 0.08
        score = min(0.97, 0.08 + cue_score + overlap_score + jitter)
        is_spoiler = bool(clean) and score >= self.threshold

        if is_spoiler:
            reasons: list[str] = []
            if cue_matches:
                cues = ", ".join(f'“{cue}”' for cue, _ in cue_matches[:2])
                reasons.append(f"contains plot-reveal wording such as {cues}")
            if overlap >= 0.18:
                reasons.append("shares several specific terms with the selected movie's synopsis")
            explanation = "Mock flag: " + " and ".join(reasons or ["matched the demonstration heuristic"] ) + "."
        else:
            explanation = "Mock result: no strong plot-reveal pattern was found."
        return Prediction(text=sentence, is_spoiler=is_spoiler, score=round(score, 3), explanation=explanation)

    @staticmethod
    def _meaningful_words(text: str) -> set[str]:
        return {
            word.casefold()
            for word in _WORD.findall(text)
            if len(word) >= 4 and word.casefold() not in _STOP_WORDS
        }
