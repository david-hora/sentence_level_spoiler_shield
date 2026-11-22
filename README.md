# Sentence-level Spoiler Shield

Detect and mask spoiler sentences in IMDB movie reviews.

## Data (`data/`)
| File | What it is |
|---|---|
| `spoiler_detection_dataset_1.3.1.json` | 80 movies, 3,200 reviews, 60,189 sentences with a sentence-level `is_spoiler` label and a fixed train/test split by movie |
| `plot_synopses.json` | Plot summary + full synopsis (incl. ending) for each of the 80 movies, keyed by `movie_id` (subset of the file below, plain JSON) |
| `IMDB_movie_details.json` | Original Kaggle file, all 1,572 movies, one JSON object per line (JSON Lines) |
| `annotation_guidelines.md` | What counts as a spoiler (the rules the labels follow) |
| `notes.md` | How the dataset was made, statistics, JSON structure |

## Load it
```python
import json
import pandas as pd

d = json.load(open("data/spoiler_detection_dataset_1.3.1.json", encoding="utf-8"))["data"]
df = pd.DataFrame([
    {"movie_id": r["movie_id"], "split": r["split"], "review_idx": i, "sent_idx": j,
     "text": s["text"], "is_spoiler": s["is_spoiler"], "unsure": s.get("unsure", False)}
    for i, r in enumerate(d) for j, s in enumerate(r["sentences"])
])
train, test = df[df.split == "train"], df[df.split == "test"]

plots = json.load(open("data/plot_synopses.json", encoding="utf-8"))
synopsis = plots["tt0111161"]["plot_synopsis"]  # empty for tt0369610 -> use plot_summary
```

## Rules of thumb for modelling
- **Label:** `is_spoiler` on each sentence. `is_spoiler_review` (review level) is the reviewer's own IMDB
  flag and is often wrong — a feature at most, never the target.
- **Validation:** take a few *train movies* as a validation set (split by `movie_id`, never by random
  sentence or review). Use the 21 test movies only for the final evaluation.
- **Imbalance:** only ~7–8% of sentences are spoilers. Use class weights / threshold tuning and report
  precision, recall and F1 for the spoiler class — accuracy alone is misleading (~93% by predicting "no").
- **Context helps:** a sentence's neighbours and the movie's plot synopsis carry a lot of the signal
  (e.g. "he dies at the end" only matters if it is true for this film).

## Web app: split new reviews the same way
The model is trained on the dataset's sentences, so split user reviews with `scripts/sentence_split.py`:
```python
from scripts.sentence_split import split_sentences
sentences = split_sentences(review_text)   # -> list of sentences to classify and mask
```
