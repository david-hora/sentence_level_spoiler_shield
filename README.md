# Sentence-Level Spoiler Shield

## Project Overview

Sentence-Level Spoiler Shield is a Natural Language Processing (NLP) project focused on detecting spoilers in movie reviews at the sentence level.

Unlike traditional spoiler detection approaches that classify entire reviews, this project aims to identify individual sentences containing spoilers. This allows potentially sensitive plot information to be hidden while preserving the remaining spoiler-free content.

The project also investigates whether additional movie plot information improves spoiler detection performance.

## Project Objectives

- Detect spoilers at the sentence level.
- Build and validate a labelled movie review dataset.
- Compare traditional machine learning and transformer-based models.
- Evaluate the impact of movie plot synopses on classification performance.
- Minimize unnecessary masking of spoiler-free content.
- Analyse model limitations and common classification errors.

## Dataset

The dataset is based on IMDb movie reviews and contains sentence-level spoiler annotations.

| Attribute | Value |
|---|---:|
| Movies | 80 |
| Reviews | 3,200 |
| Sentences | 60,189 |
| Spoiler sentences | 4,677 |
| Non-spoiler sentences | 55,512 |
| Training movies | 59 |
| Testing movies | 21 |

The dataset was processed using AI-assisted annotation, a local Qwen model, and manual verification to improve sentence segmentation and annotation quality.

The training and testing datasets are split by movie to prevent data leakage between the two sets.

## Methodology

The project explores several approaches to spoiler detection:

- TF-IDF with Logistic Regression
- Synopsis similarity-based classification
- Logistic Regression using synopsis features
- Combined TF-IDF and synopsis features
- DistilBERT using review text
- DistilBERT using review text and movie synopses

These approaches allow us to compare traditional text classification methods with contextual language models and investigate the contribution of additional plot information.

## Evaluation

Model performance is evaluated using the following metrics:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC

Additional evaluation focuses on practical spoiler protection, including the percentage of missed spoilers, unnecessarily masked text, and successfully protected reviews.

Error analysis and stress testing are performed to identify challenging cases and understand model behaviour.

## Project Structure

    sentence_level_spoiler_shield/
    ├── data/
    │   ├── IMDB_movie_details.json
    │   ├── plot_synopses.json
    │   ├── spoiler_detection_dataset_1.3.1.json
    │   ├── annotation_guidelines.md
    │   └── notes.md
    ├── notebooks/
    │   └── spoiler_models.ipynb
    └── results/
        ├── metrics_test.csv
        ├── predictions_test.csv
        ├── pr_curves.png
        ├── per_movie_ap.csv
        ├── stress_tests.csv
        ├── run_config.json
        └── README.md

- **data/** — datasets, movie information, plot synopses and annotation guidelines.
- **notebooks/** — data processing, model training and evaluation.
- **results/** — evaluation metrics, predictions, visualizations and error analysis.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- DistilBERT
- Qwen
- Jupyter Notebook
- Google Colab

## Results

The experimental results show that transformer-based models generally outperform traditional machine learning approaches in sentence-level spoiler detection.

Movie plot synopses can provide useful additional context, although their contribution depends on the model and the characteristics of individual reviews.

Detailed experimental results, performance comparisons and error analysis are available in the [results directory](results/README.md).

## Getting Started

1. Clone the repository.
2. Open `notebooks/spoiler_models.ipynb` in Jupyter Notebook or Google Colab.
3. Load the datasets from the `data/` directory.
4. Run the notebook to reproduce the model training and evaluation.
5. Explore the generated results and performance metrics.

## Potential Applications

- Spoiler-free movie review platforms
- Automatic spoiler masking
- Content moderation systems
- Personalized spoiler protection
- Context-aware text classification

## Conclusion

This project investigates how NLP and machine learning can improve the experience of reading movie reviews by automatically identifying potentially unwanted plot information without hiding entire reviews.

The main challenge is finding a balance between effective spoiler detection and preserving as much useful, spoiler-free content as possible.
