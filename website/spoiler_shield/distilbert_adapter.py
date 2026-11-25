"""Inference matching the supplied export and the team's synopsis preprocessing."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
from typing import Sequence

from spoiler_shield.catalog import Movie
from spoiler_shield.model_adapter import ModelLoadError, Prediction
from spoiler_shield.model_assets import ensure_model_package, validate_manifest, verify_directory

SYNOPSIS_SPLIT = re.compile(r'''(?<=[.!?])\s+(?=["'(\[]?[A-Z0-9])''')


def synopsis_sentences(movie: Movie) -> list[tuple[str, float]]:
    text = movie.synopsis
    if len(text.split()) < 50:
        text = re.sub(r"\s*Written by.*$", "", movie.plot_summary, flags=re.S)
    sentences = [s.strip() for s in SYNOPSIS_SPLIT.split(re.sub(r"\s+", " ", text)) if len(s.strip()) > 3]
    return [(text, i / max(len(sentences) - 1, 1)) for i, text in enumerate(sentences)]


def position_tag(position: float) -> str:
    return "(beginning)" if position <= 0.2 else "(end)" if position >= 0.75 else "(middle)"


def select_context(sentences: list[tuple[str, float]], similarities, top_k: int = 3) -> str:
    import numpy as np

    # Retrieve by similarity, then restore plot order, exactly as in the notebook.
    top = np.argsort(-np.asarray(similarities))[:top_k]
    return " ".join(f"{position_tag(sentences[i][1])} {sentences[i][0]}" for i in sorted(top))


def validate_settings(settings: dict) -> None:
    threshold = settings.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ModelLoadError("Invalid exported spoiler threshold")
    if (settings.get("max_length") != 256 or settings.get("truncation") != "longest_first"
            or settings.get("top_k_synopsis_sentences") != 3
            or settings.get("retriever") != "sentence-transformers/all-MiniLM-L6-v2"
            or settings.get("position_tags") != {
                "beginning": "pos <= 0.2", "end": "pos >= 0.75", "middle": "otherwise"}):
        raise ModelLoadError("Unsupported exported synopsis preprocessing settings")


class DistilbertSpoilerPredictor:
    name = "DistilBERT + movie synopsis"
    is_mock = False

    def __init__(self, directory: Path | None = None, manifest: dict | None = None) -> None:
        try:
            if directory is None:
                directory, manifest = ensure_model_package()
            elif manifest is None:
                raise ModelLoadError("A local model package requires its checksum manifest")
            assert manifest is not None
            validate_manifest(manifest)
            if not verify_directory(directory, manifest):
                raise ModelLoadError("The inference package failed verification")
            self.settings = json.loads((directory / "classifier/spoiler_shield_config.json").read_text(encoding="utf-8"))
            validate_settings(self.settings)
            if self.settings["threshold"] != manifest["threshold"]:
                raise ModelLoadError("The supplied threshold differs from the package manifest")
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Small batches and two CPU threads keep concurrent Cloud sessions usable.
            torch.set_num_threads(2)
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                directory / "classifier", local_files_only=True, trust_remote_code=False)
            if self.tokenizer is None:
                raise ModelLoadError("The supplied classifier tokenizer is missing")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                directory / "classifier", local_files_only=True, trust_remote_code=False,
                use_safetensors=True).to("cpu").eval()
            if self.model.config.num_labels != 2 or self.model.config.model_type != "distilbert":
                raise ModelLoadError("Unexpected spoiler classifier architecture")
            self.encoder = SentenceTransformer(
                str(directory / "retriever"), device="cpu", local_files_only=True, trust_remote_code=False)
            self.encoder.eval()
            self.threshold = float(self.settings["threshold"])
            self.model_id = manifest["model_id"]
            self._lock = Lock()
            self._synopsis_cache: OrderedDict[str, object] = OrderedDict()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("The supplied DistilBERT model could not be loaded") from exc

    def predict(self, movie: Movie, sentences: Sequence[str]) -> list[Prediction]:
        if not sentences:
            return []
        import numpy as np

        active = [i for i, text in enumerate(sentences) if text.strip()]
        scores = [0.0] * len(sentences)
        plot = synopsis_sentences(movie)
        # This object is shared by st.cache_resource; serialize mutable tokenizer/cache access.
        with self._lock, self._torch.inference_mode():
            plot_embeddings = None
            if plot and active:
                cache_key = hashlib.sha256(json.dumps(plot).encode("utf-8")).hexdigest()
                if cache_key not in self._synopsis_cache:
                    self._synopsis_cache[cache_key] = self.encoder.encode(
                        [text for text, _ in plot], normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False)
                self._synopsis_cache.move_to_end(cache_key)
                plot_embeddings = self._synopsis_cache[cache_key]
                if len(self._synopsis_cache) > 80:
                    self._synopsis_cache.popitem(last=False)
            for offset in range(0, len(active), 8):
                indices = active[offset:offset + 8]
                texts = [sentences[i].strip() for i in indices]
                contexts = [""] * len(texts)
                if plot_embeddings is not None:
                    queries = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    contexts = [select_context(plot, plot_embeddings @ query) for query in queries]
                assert self.tokenizer is not None
                tokens = self.tokenizer(
                    texts, contexts, truncation="longest_first", max_length=256,
                    padding=True, return_tensors="pt")
                batch_scores = self._torch.softmax(self.model(**tokens).logits.float(), dim=-1)[:, 1].tolist()
                if len(batch_scores) != len(indices) or not np.isfinite(batch_scores).all():
                    raise RuntimeError("DistilBERT returned invalid prediction scores")
                for i, score in zip(indices, batch_scores):
                    scores[i] = float(score)
        predictions = []
        for text, score in zip(sentences, scores):
            flagged = bool(text.strip()) and score >= self.threshold
            relation = "reached" if flagged else "stayed below"
            explanation = (
                f"The trained DistilBERT classifier {relation} its exported spoiler threshold ({self.threshold:.1%}). "
                + ("It used this sentence and up to three relevant synopsis sentences with plot-position tags. "
                   if plot else "No synopsis context was available; it used an empty context, as during preprocessing. ")
                + "This is a model decision, not a verified explanation of the plot."
            )
            predictions.append(Prediction(text, flagged, score, explanation))
        return predictions
