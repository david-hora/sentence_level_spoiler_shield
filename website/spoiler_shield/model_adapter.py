from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from spoiler_shield.catalog import Movie


@dataclass(frozen=True)
class Prediction:
    text: str
    is_spoiler: bool
    score: float
    explanation: str


class SpoilerPredictor(Protocol):
    """Boundary for sentence-only and future synopsis-aware predictors."""

    name: str
    is_mock: bool

    def predict(self, movie: Movie, sentences: Sequence[str]) -> list[Prediction]:
        """Return one prediction per sentence, preserving text and order."""


class ModelLoadError(RuntimeError):
    """The bundled trained model is missing, incompatible or corrupt."""


class TfidfSpoilerPredictor:
    """Fitted word/character TF-IDF and logistic regression; no synopsis input."""

    name = "TF-IDF + logistic regression"
    is_mock = False

    def __init__(self, directory: Path | None = None) -> None:
        import joblib
        import numpy as np
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        directory = directory or Path(__file__).resolve().parents[1] / "models" / "tfidf"
        try:
            self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            artifact = directory / "model.joblib"
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != self.metadata["artifact_sha256"]:
                raise ModelLoadError("The trained model checksum does not match its metadata")
            if self.metadata["versions"]["scikit-learn"] != sklearn.__version__:
                raise ModelLoadError("Install the model dependencies from website/requirements.txt")
            # Load only this repository's verified artifact, never a user upload.
            bundle = joblib.load(artifact)
            if bundle["format_version"] != 1 or self.metadata["format_version"] != 1:
                raise ModelLoadError("Unsupported trained model format")
            self.features = bundle["features"]
            self.classifier = bundle["classifier"]
            self.threshold = float(bundle["threshold"])
            if not isinstance(self.features, FeatureUnion) or not isinstance(self.classifier, LogisticRegression):
                raise ModelLoadError("Unexpected trained model components")
            if self.classifier.classes_.tolist() != [0, 1]:
                raise ModelLoadError("Unexpected spoiler class labels")
            if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ModelLoadError("Invalid spoiler threshold")
            if self.threshold != self.metadata["threshold"]:
                raise ModelLoadError("Model and metadata thresholds differ")
            self.word_names = self.features.transformer_list[0][1].get_feature_names_out()
            self.word_count = len(self.word_names)
            self.model_id = self.metadata["model_id"]
            self._numpy = np
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("The trained TF-IDF model could not be loaded") from exc

    def predict(self, movie: Movie, sentences: Sequence[str]) -> list[Prediction]:
        if not sentences:
            return []
        matrix = self.features.transform([text.strip() for text in sentences]).tocsr()
        scores = self.classifier.predict_proba(matrix)[:, 1]
        if len(scores) != len(sentences) or not self._numpy.isfinite(scores).all():
            raise RuntimeError("The classifier returned invalid prediction scores")
        predictions = []
        for index, (text, raw_score) in enumerate(zip(sentences, scores)):
            score = float(raw_score) if text.strip() else 0.0
            is_spoiler = bool(text.strip()) and score >= self.threshold
            explanation = "The trained text classifier scored this sentence below its spoiler threshold."
            if is_spoiler:
                contributions = matrix[index, :self.word_count].multiply(
                    self.classifier.coef_[0, :self.word_count],
                ).tocsr()
                ranked = sorted(zip(contributions.indices, contributions.data), key=lambda pair: -pair[1])
                words = [str(self.word_names[column]) for column, contribution in ranked if contribution > 0][:3]
                explanation = (
                    f"The trained text classifier exceeded its spoiler threshold ({self.threshold:.1%}). "
                    "It recognises wording associated with spoiler-labelled sentences."
                )
                if words:
                    explanation += " Positive word features: " + ", ".join(f'“{word}”' for word in words) + "."
                explanation += " This describes the model's evidence; it does not verify the movie's plot."
            predictions.append(Prediction(text, is_spoiler, score, explanation))
        return predictions
