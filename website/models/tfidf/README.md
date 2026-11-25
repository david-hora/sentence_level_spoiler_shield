# Trained TF-IDF spoiler classifier

This is a newly trained recreation of **model A** from the team's
`notebooks/spoiler_models.ipynb`, exported on 25 November 2026. It is a fitted
classifier, not a lookup table of old predictions or a rule-based mock.

Files:

- `model.joblib`: fitted word and character TF-IDF `FeatureUnion`, binary
  `LogisticRegression`, format version and selected decision threshold.
- `metadata.json`: exact dependency versions, data/artifact checksums, training
  parameters, movie IDs per split, candidate validation results and held-out metrics.

Dataset: `data/spoiler_detection_dataset_1.3.1.json`. The 34,756 training sentences
come from 47 movies, 9,489 validation sentences from 12 other movies, and 15,944
test sentences from 21 further movies. These partitions do not overlap by movie.
Word/character vocabularies are fitted on training data only. Regularization is
chosen from `0.1, 0.3, 1, 3, 10` using validation average precision. The threshold
maximizes validation F1. No training, parameter selection or threshold tuning
uses the test partition.

The selected `C=1` and exported threshold `0.5002023450008355` reproduce model A's
reported results. This export's test PR-AUC is **0.359141**, F1 **0.400763**, precision
**0.323475** and recall **0.526580**. Confusion matrix, with rows as true labels
and columns as predicted labels, ordered `[safe, spoiler]`:

| | Predicted safe | Predicted spoiler |
|---|---:|---:|
| True safe | 13,849 | 1,098 |
| True spoiler | 472 | 525 |

All 15,944 test scores were identical before and after serializing/reloading the
artifact (maximum absolute score difference: zero). The original team's
`results/` files and training notebook are preserved separately.

The classifier analyses English wording in each sentence. It does not read a
synopsis or check plot facts. Training labels include LLM annotations and
uncertain cases. False positives and missed spoilers are expected; the displayed
score is not a calibrated probability of correctness. Feature contributions in
explanations describe how the classifier reached its decision, not plot truth.

Run `python scripts/train_tfidf.py` from `website/` to recreate both files using
the pinned requirements. New runs use unique staging directories and publish
immutable `model-<hash>.joblib` files through one atomic `metadata.json` update.
Older artifacts remain available for readers that already read their manifest.
The original bundled flat `model.joblib` remains supported. The website loads
the selected trusted artifact, verifies its SHA-256, derived export ID and
scikit-learn version, and fails visibly on analysis errors rather
than silently substituting a mock. Never replace it with an untrusted joblib
file; joblib can execute code during loading.
