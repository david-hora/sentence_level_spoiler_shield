"""Recreate model A without modifying the team's notebook or results directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import warnings

WEBSITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBSITE))
from spoiler_shield.tfidf_features import make_features
from spoiler_shield.model_export import create_export_stage, publish_export


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "sentences": len(labels), "spoilers": int(labels.sum()),
    }


def train(dataset: Path, output: Path) -> dict:
    started = time.monotonic()
    dataset_bytes = dataset.read_bytes()
    records = json.loads(dataset_bytes)["data"]
    rows = [(sentence["text"], int(sentence["is_spoiler"]), record["movie_id"], record["split"])
            for record in records for sentence in record["sentences"]]
    texts = np.array([row[0] for row in rows], dtype=object)
    labels = np.array([row[1] for row in rows], dtype=np.int64)
    movies = np.array([row[2] for row in rows], dtype=object)
    splits = np.array([row[3] for row in rows], dtype=object)
    if set(splits) != {"train", "test"} or set(labels) != {0, 1}:
        raise ValueError("Expected the sentence labels and train/test splits from dataset 1.3.1")
    train_all = np.flatnonzero(splits == "train")
    test = np.flatnonzero(splits == "test")
    splitter = GroupShuffleSplit(n_splits=1, test_size=12 / 59, random_state=42)
    train_relative, val_relative = next(splitter.split(train_all, groups=movies[train_all]))
    train_indices, validation = train_all[train_relative], train_all[val_relative]
    partitions = {"train": train_indices, "validation": validation, "test": test}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(movies[partitions[left]]) & set(movies[partitions[right]]):
            raise ValueError("Movie leakage between partitions")
    partition_info = {name: {
        "sentences": len(indices), "spoilers": int(labels[indices].sum()),
        "movie_ids": sorted(set(movies[indices])),
    } for name, indices in partitions.items()}
    print({name: {"sentences": len(indices), "movies": len(set(movies[indices]))}
           for name, indices in partitions.items()}, flush=True)
    features = make_features()
    print("Fitting word and character TF-IDF on training sentences only...", flush=True)
    x_train = features.fit_transform(texts[train_indices]).tocsr()
    x_val = features.transform(texts[validation]).tocsr()
    candidates = []
    best = None
    for c_value in (0.1, 0.3, 1, 3, 10):
        classifier = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(x_train, labels[train_indices])
        scores = classifier.predict_proba(x_val)[:, 1]
        ap = float(average_precision_score(labels[validation], scores))
        candidates.append({"C": c_value, "validation_pr_auc": ap, "iterations": int(classifier.n_iter_[0])})
        print(f"C={c_value}: validation PR-AUC={ap:.6f}", flush=True)
        if best is None or ap > best[0]:
            best = (ap, classifier, scores)
    _, classifier, val_scores = best
    threshold = best_f1_threshold(labels[validation], val_scores)
    print(f"Selected C={classifier.C}; validation F1 threshold={threshold:.8f}", flush=True)
    # The test partition is first scored after both model selection and threshold tuning.
    test_scores = classifier.predict_proba(features.transform(texts[test]).tocsr())[:, 1]
    stage = create_export_stage(output)
    pending = stage / "model.joblib"
    joblib.dump({"format_version": 1, "features": features, "classifier": classifier,
                 "threshold": threshold}, pending, compress=3)
    restored = joblib.load(pending)
    restored_scores = restored["classifier"].predict_proba(
        restored["features"].transform(texts[test]).tocsr(),
    )[:, 1]
    difference = float(np.max(np.abs(restored_scores - test_scores)))
    if difference > 1e-12:
        raise RuntimeError("Exported model predictions differ from the training model")
    model_hash = hashlib.sha256(pending.read_bytes()).hexdigest()
    metadata = {
        "format_version": 1, "model_id": "tfidf-a-" + model_hash[:12],
        "name": "TF-IDF + logistic regression", "language": "English",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_notebook": "notebooks/spoiler_models.ipynb", "source_model": "A",
        "dataset": dataset.name, "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "artifact_sha256": model_hash, "seed": 42,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "features": {"word": len(features.transformer_list[0][1].vocabulary_),
                     "char": len(features.transformer_list[1][1].vocabulary_)},
        "partitions": partition_info, "candidates": candidates, "selected_C": classifier.C,
        "threshold": threshold, "threshold_selection": "Maximum F1 on validation movies only",
        "validation": metrics(labels[validation], val_scores, threshold),
        "test": metrics(labels[test], test_scores, threshold),
        "export_verification": {"sentences": len(test), "max_score_difference": difference},
        "training_seconds": round(time.monotonic() - started, 2),
        "limitations": ["Sentence wording only; no synopsis or movie-specific context.",
                        "Class-weighted scores are not calibrated confidence estimates.",
                        "Training annotations contain LLM-produced labels and uncertain cases.",
                        "This is a new training run; use its own evaluation metrics."],
    }
    artifact = publish_export(output, stage, metadata)
    print(json.dumps({"artifact": str(artifact), "bytes": artifact.stat().st_size,
                      "test": metadata["test"], "seconds": metadata["training_seconds"]}, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=WEBSITE.parent / "data" / "spoiler_detection_dataset_1.3.1.json")
    parser.add_argument("--output", type=Path, default=WEBSITE / "models" / "tfidf")
    arguments = parser.parse_args()
    with threadpool_limits(limits=2):
        train(arguments.dataset, arguments.output)
