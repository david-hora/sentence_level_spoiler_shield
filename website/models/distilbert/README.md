# Supplied DistilBERT + synopsis export

The live predictor uses the classifier supplied on 25 November 2026, packaged
without changing its weights, tokenizer or inference settings. `manifest.json`
records the original ZIP checksum, each runtime file checksum, the archive URL,
and the exact MiniLM retriever revision. Large safetensors files live in the
hosting fork's public GitHub Release, not Git history.

The runtime ZIP contains two local model directories:

- `classifier/`: the five files from the supplied ZIP, including 267,832,560-byte
  `model.safetensors`, plus the supplied tokenizer and settings.
- `retriever/`: the safetensors/tokenizer/configuration/model-card files from
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache 2.0).

## Inference

Both models run on the Streamlit server's CPU. They load only verified local
safetensors with remote code disabled. There are no external prediction API
calls and reviews are not sent to GitHub or Hugging Face. The first analysis
downloads the 330 MB runtime package and caches it under the operating system's
temporary directory, `spoiler-shield-models/<model-id>/`. Cold starts or a cleared
host cache need another download. Subsequent analyses reuse cached models.
An incomplete download is never published as a usable package. Download,
checksum or loading failures preserve the draft and produce an analysis error;
there is no silent TF-IDF or mock fallback.

Preprocessing matches the supplied export and the team's notebook:

1. Use the full synopsis when it has at least 50 words; otherwise use the plot
   summary, removing its trailing `Written by` attribution.
2. Normalize synopsis whitespace and split with the notebook's sentence regex.
3. Encode synopsis and review sentences using normalized MiniLM embeddings.
4. Retrieve the three most similar synopsis sentences, restore their plot order,
   and prefix each with `(beginning)`, `(middle)` or `(end)` according to its
   relative position (beginning <= 0.2; end >= 0.75).
5. Tokenize the review sentence and synopsis context as a text pair with
   `max_length=256`, `truncation="longest_first"`. Missing context is an empty
   string, matching the notebook.
6. Take softmax class 1 and hide scores >= **0.2114982306957245**, the supplied
   export's threshold. Review whitespace is preserved in the returned predictions.

Inference uses evaluation mode, inference mode, batches of at most eight and two
CPU threads. Shared model access is serialized. Synopsis embeddings are cached
for at most 80 distinct plots. Scores are not calibrated confidence values.
Explanations state how the decision was made; they do not invent plot evidence.

## Provenance and evaluation limits

This export has the architecture and synopsis input design of reported model E,
which had test average precision 0.493 and used threshold 0.1145169735. The supplied
archive instead specifies 0.2114982307. No training log, dataset version, label
mapping document or run identifier accompanied the archive. The classifier's two
classes use the notebook's convention (0 safe, 1 spoiler), checked with inference
smoke tests. **Do not attribute the old report's metrics to this exact export**
until its training provenance and held-out evaluation are confirmed.

Local smoke checks on Warcraft: “The acting is excellent.” scores about 0.0030;
“At the end, Medivh dies and the portal is destroyed.” about 0.9686. The previously
reported clay-monster sentence scores about 0.0835 and is still missed. These
examples verify wiring, not general accuracy, and were not used to change the threshold.

## Repackage / verify

Download only the retriever files listed in `scripts/package_distilbert.py` from
the pinned revision above. Keep the original supplied classifier ZIP unchanged.
Then, from `website/`:

```powershell
python scripts/package_distilbert.py --classifier-zip "path/to/supplied.zip" --retriever-directory "path/to/pinned-minilm" --output "path/to/spoiler-shield-runtime.zip" --manifest models/distilbert/manifest.json
```

The packager fixes archive timestamps and stores SHA-256 checksums. Compare the
generated retriever checksums with the reviewed manifest before publishing.
Upload the runtime ZIP to the manifest's release URL before deploying the source.
If publishing a new export, use a new release tag, update the manifest and settings
validation deliberately, and run tests again. Never replace assets under an old ID.

To run the real-weights integration test, first download the package by running
one website analysis, then set the model directory (use your OS's temp path):

```powershell
$env:SPOILER_SHIELD_TEST_MODEL_DIR = "$env:TEMP/spoiler-shield-models/distilbert-synopsis-8141afd2ce53"
python -m unittest discover -s tests -p test_distilbert.py -v
```

Default tests stay offline and skip the real-weights test unless this variable is
set. TF-IDF remains available as a reproducible baseline in `models/tfidf/` and is
tested separately; the website selects DistilBERT explicitly in `get_predictor()`.
