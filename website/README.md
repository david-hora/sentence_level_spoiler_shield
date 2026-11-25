# Spoiler Shield — movie review database MVP

Spoiler Shield is a Streamlit movie-review database built around the team's sentence-level IMDb spoiler annotations. Users can browse 80 movies, read 3,200 reviews with spoiler sentences hidden, reveal individual spoilers, and add a new review for mock analysis.

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

If the virtual environment already exists, only the last command is needed.

## Current experience

- Search and paginate through all 80 movies in the annotated dataset.
- Open a movie to see its release year, runtime, genres, IMDb rating, and review counts.
- Read all 40 dataset reviews for each movie, five at a time.
- Reveal human-labelled spoiler sentences independently while leaving safe text visible.
- Add a review with a headline and 1–10 rating.
- Analyse new reviews with the clearly labelled mock predictor and add the protected result to the current browser session.
- Use a light theme with a cinema photo fading into white, clickable movie tiles, and centred review pages.

Existing dataset reviews use the `is_spoiler` sentence labels in the repository's
`../data/spoiler_detection_dataset_1.3.1.json`. Only reviews entered through the website use the mock predictor.

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
spoiler_shield/model_adapter.py Predictor contract and temporary mock
spoiler_shield/rendering.py     Safe spoiler redaction and reveal rendering
spoiler_shield/site.css         Website styling
assets/cinema-hero.png          Hero photograph
data/movie_titles.json          Website's movie title index
../data/                       Existing shared team data read by the website
tests/                           Logic, dataset, rendering, and Streamlit tests
```

## Mock predictor limitations

`MockSpoilerPredictor` is deterministic and uses plot-reveal phrases, synopsis overlap, and a stable score variation. It demonstrates the add-review experience, but it:

- has not learned from the annotated dataset;
- can miss genuine spoilers and hide harmless sentences;
- does not understand context, negation, character relationships, or plot chronology;
- produces illustrative scores and explanations that must not be reported as model results.

## Connect the trained model later

The website depends on one stable contract in `spoiler_shield/model_adapter.py`:

```python
predict(movie: Movie, sentences: Sequence[str]) -> list[Prediction]
```

A real adapter returns one prediction per input sentence, in the same order. The `Movie` object supplies the selected movie ID, title, summary, and full synopsis, so both synopsis-free and synopsis-aware models can use the same website.

When the trained model arrives:

1. Add a class such as `TrainedModelPredictor` that follows `SpoilerPredictor`.
2. Load the exported model and its tokenizer or vectorizer inside that adapter.
3. Replace `MockSpoilerPredictor()` in `app.py` with the real adapter.
4. Update the prototype notice only after genuine model predictions are being used.

The movie library, dataset-review redactions, review cards, pagination, and add-review interface do not need to be rebuilt.

## Run tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The tests cover the complete 3,200-review dataset, catalog integrity, exact text preservation, safe rendering, deterministic mock output, and the main Streamlit navigation flow.

## Shared repository layout

The website's source, theme, assets, dependencies and tests all live inside
`website/`. It reads the existing plot synopses and annotated reviews from the
repository's root `data/` directory. It does not write to the shared dataset.

This website uses dataset 1.3.1. A later dataset version should be checked for
sentence and annotation changes before switching the website to it. The trained
model is still pending and new-review analysis uses the labelled mock predictor.

The local `.gitignore` excludes virtual environments, caches and
`.streamlit/secrets.toml`.
