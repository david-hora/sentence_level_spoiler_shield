We decided to use a local Qwen model, which we left running overnight.

After Qwen "cleaned" the dataset, we still had to go through it ourselves and fix the mistakes it made. 55,660 sentences before -> 60,189 sentences after fixing Qwen's errors. We verified these errors manually.

Result -> A training/validation dataset with 80 movies, 3,200 reviews, and 60,189 sentences.



## Overview
- 80 movies, 40 reviews each (20 marked spoiler and 20 marked non-spoiler by the reviewer on IMDB) = 3,200 reviews
- Sentence-level spoiler labels for every review
- Fixed train/test split by movie (no movie appears in both)
- File: `spoiler_detection_dataset_1.3.1.json`

| Split | Movies | Reviews | Sentences | Spoiler sentences |
|---|---|---|---|---|
| train | 59 | 2,360 | 44,245 | 3,680 (8.3%) |
| test  | 21 | 840 | 15,944 | 997 (6.3%) |
| total | 80 | 3,200 | 60,189 | 4,677 (7.8%) |


## Structure
```
{
  "description", "version": "1.3.0", "total_reviews": 3200, "movies": 80, "reviews_per_movie": 40,
  "annotation_format": "...",
  "split_movies": {"test": [movie_ids], "train": [movie_ids]},
  "data": [
    {
      "movie_id": "ttXXXXXXX",
      "user_id": "urXXXXX",
      "is_spoiler_review": true/false,   // reviewer's own IMDB flag — noisy, not ground truth
      "rating": "1"-"10",
      "review_summary": "...",
      "movie_plot_preview": "...",       // first 100 chars only, as in v1.2.0
      "split": "train" | "test",
      "annotation_source": "...",
      "sentences": [
        {
          "text": "...",
          "is_spoiler": true/false,      // THE label to train/evaluate on
          "orig_sentence_idx": 3,        // index of the v1.2.0 sentence it came from
          "unsure": true                 // train only, optional: Sonnet was unsure (label = its best decision)
        }
      ]
    }
  ]
}
```
