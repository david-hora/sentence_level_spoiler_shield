# Spoiler Shield — movie review database MVP

Spoiler Shield is a Streamlit movie-review database built around the team's sentence-level IMDb spoiler annotations. Users can browse 80 movies, read 3,200 reviews with spoiler sentences hidden, reveal individual spoilers, and add a new review analysed by a trained TF-IDF and logistic regression model.

## Run it in VS Code

Open the shared repository in VS Code. Enter the website folder first so Streamlit
loads its theme, then run:

```powershell
cd website
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically. Keep the terminal running while using the website; stop it with `Ctrl+C`.

For an existing virtual environment, install the updated requirements before
starting the app. The trained model and its metadata are already included in
`models/tfidf/`; training is not needed to run the website.

## Current experience

- Search and paginate through all 80 movies in the annotated dataset.
- Open a movie to see its release year, runtime, genres, IMDb rating, and review counts.
- Read all 40 dataset reviews for each movie, five at a time.
- Reveal human-labelled spoiler sentences independently while leaving safe text visible.
- Add a review with a headline and 1–10 rating.
- Analyse new reviews with the exported, trained TF-IDF classifier and add the protected result to the current browser session.
- Use a light theme with a cinema photo fading into white, clickable movie tiles, and centred review pages.

Existing dataset reviews use the `is_spoiler` sentence labels in the repository's
`../data/spoiler_detection_dataset_1.3.1.json`. Reviews entered through the website
use the trained classifier; existing dataset annotations are not replaced.

## Data and persistence

The repository contains:

- 80 movies;
- 3,200 reviews, with 40 reviews per movie;
- 60,189 annotated sentences;
- movie titles, metadata, summaries, and available full synopses.

Reviews added through the website are intentionally stored only in Streamlit Session State. They disappear when the browser session or server ends and never modify the team's dataset. Permanent review storage can be added later if it becomes part of the assignment.

Each session can store up to 100 new reviews, each at most 12,000 characters.
Submitting the same movie, headline, rating and review text again returns the
saved result without adding another copy or running analysis again. At the limit,
new submissions are refused without removing existing reviews or clearing drafts.
Successful submissions refresh the totals and return review pagination to page 1.
Analysis failures keep the draft and log a traceback with the movie ID to the
server console.

Dataset sentence text retains any supplied whitespace, including paragraph
breaks. Where adjacent records have no boundary whitespace, the renderer inserts
one space. Paragraphs already removed from the source dataset cannot be restored.

## Project structure

```text
app.py                           Streamlit database and interaction flow
spoiler_shield/catalog.py       Movie metadata loading and validation
spoiler_shield/reviews.py       Annotated review loading and grouping
spoiler_shield/text.py          Formatting-preserving sentence splitting
spoiler_shield/model_adapter.py Trained predictor loading and sentence inference
spoiler_shield/tfidf_features.py Original model A feature settings
spoiler_shield/rendering.py     Safe spoiler redaction and reveal rendering
spoiler_shield/site.css         Website styling
assets/cinema-hero.png          Hero photograph
data/movie_titles.json          Website's movie title index
models/tfidf/model.joblib       Fitted word/character vectorizers and classifier
models/tfidf/metadata.json      Provenance, versions, threshold, held-out metrics
scripts/train_tfidf.py          Reproducible training and export command
../data/                       Existing shared team data read by the website
tests/                           Logic, dataset, rendering, and Streamlit tests
```

## Trained model and limitations

`TfidfSpoilerPredictor` loads the fitted model A pipeline recreated from
`../notebooks/spoiler_models.ipynb`. It uses word unigrams/bigrams and character
3–5-grams with a class-balanced logistic regression classifier. New review text
is split into sentence-like chunks, scored, and redacted using the exported
validation threshold. The model is cached once per server process and never
trained while the app is running. Missing or incompatible artifacts produce an
analysis error, preserve the draft, and are logged; there is no mock fallback.

This recreation uses dataset 1.3.1 with 47 training movies, 12 validation movies,
and 21 held-out test movies. Vocabulary fitting uses training sentences only;
regularization is selected by validation PR-AUC, then the threshold by validation
F1. Test sentences are scored only after those choices. The model is not refit
on validation or test data. Original team results and notebooks are unchanged.

- Held-out PR-AUC: **0.3591**; F1: **0.4008**.
- Spoiler precision: **0.3235**; recall: **0.5266**.
- Threshold: **0.5002023450**; selected regularization `C=1`.
- These are this new run's own metrics and reproduce the original model A results.
- It was trained on English reviews. It uses sentence wording, not the selected
  movie's synopsis, and does not verify whether a plot event actually occurs.
- It misses spoilers and flags safe text; training annotations include LLM labels
  and uncertain cases. Its class-weighted model scores are not calibrated confidence.
- Reveal explanations describe the decision threshold and positive word-feature
  contributions. They are statistical evidence, not verified plot explanations.

See `models/tfidf/README.md` and `metadata.json` for the full export details.

## Recreate and export the model

From `website/`, after installing requirements:

```powershell
.venv\Scripts\python scripts/train_tfidf.py
```

This writes `models/tfidf/model.joblib` and `metadata.json`, including data and
artifact SHA-256 hashes, movie splits, dependency versions, candidate scores and
evaluation metrics. The export is reloaded and checked against all 15,944
held-out test scores before completion. Run training outside a live deployment
and commit both artifact and metadata together. Keep the pinned dependencies
when loading the trusted repository artifact; do not load user-supplied joblib files.

## Connect another model later

The website retains one predictor contract in `spoiler_shield/model_adapter.py`:

```python
predict(movie: Movie, sentences: Sequence[str]) -> list[Prediction]
```

An adapter returns one prediction per input chunk, in the same order, preserving
the text exactly. The `Movie` object supplies the movie metadata and synopsis,
so a later synopsis-aware model can use the same interface. The current TF-IDF
adapter intentionally ignores movie context, matching model A.

To add another model, implement `SpoilerPredictor`, update `get_predictor()` in
`app.py`, and update the visible model description. New reviews store the model
name and export ID along with the predictions for provenance.

The movie library, dataset-review redactions, review cards, pagination, and add-review interface do not need to be rebuilt.

## Run tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The tests cover dataset integrity, original held-out score parity, export loading,
invalid model handling, text preservation, safe rendering, submission provenance,
duplicate prevention, limits, failure retry and the main Streamlit navigation flow.

## Shared repository layout

The website's source, theme, assets, dependencies and tests all live inside
`website/`. It reads the existing plot synopses and annotated reviews from the
repository's root `data/` directory. It does not write to the shared dataset.

This website uses dataset 1.3.1. A later dataset version should be checked for
sentence and annotation changes before switching the website and training to it.

The public deployment reads `david-hora/sentence_level_spoiler_shield`, branch
`main`, entrypoint `website/app.py`. Updates to the team's original repository
must be synced into that fork to reach the hosted app. No secrets or external
model API calls are needed. Newly added reviews remain session-only.

The local `.gitignore` excludes virtual environments, caches and
`.streamlit/secrets.toml`.
