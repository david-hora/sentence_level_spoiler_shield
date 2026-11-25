"""Feature settings copied from model A in the team's training notebook."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=100_000,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=150_000, sublinear_tf=True,
        )),
    ])
