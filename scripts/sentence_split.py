"""Sentence splitter matching the dataset (v1.3.1).

The dataset's sentences = the original v1.2.0 sentence split, plus a fix that splits sentences
glued together without a space ("...real.Bruce Wayne..."). Use `split_sentences` on new reviews
(e.g. in the web app) so the model sees input shaped like its training data.

    from scripts.sentence_split import split_sentences
    split_sentences("Great film.The ending though... Bruce dies!")
    # ['Great film.', 'The ending though...', 'Bruce dies!']

Uses NLTK's punkt tokenizer if installed (pip install nltk; nltk.download('punkt_tab')),
otherwise a simple regex. Both are followed by the glued-sentence fix used to build the dataset.
"""
import re

# glued boundary: lowercase/digit/closing char + .!? (1-3x) directly followed by Capital+lowercase
_GLUE = re.compile(
    r'(?<=[a-z0-9\)\]"\'”’][.!?])(?=["“]?[A-Z][a-z])'
    r'|(?<=[a-z0-9\)][.!?]{2})(?=[A-Z][a-z])'
    r'|(?<=[a-z0-9\)][.!?]{3})(?=[A-Z][a-z])')
_ABBR = re.compile(r'(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Lt|Cpt|Capt|Col|Gen|Sgt|Prof|vs|etc|No|Vol|U\.S|U\.K)[.])$')
_SIMPLE = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”’)]))\s+(?=["“(]?[A-Z0-9])')


def fix_glued(sentence):
    """Split one sentence at glued boundaries (the exact rule used for the dataset)."""
    parts, start = [], 0
    for m in _GLUE.finditer(sentence):
        left = sentence[start:m.start()]
        if _ABBR.search(left):
            continue
        if left.strip():
            parts.append(left.strip())
        start = m.start()
    rest = sentence[start:].strip()
    if rest:
        parts.append(rest)
    return parts or [sentence]


def _base_split(text):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:  # nltk or its punkt data missing
        parts, start = [], 0
        for m in _SIMPLE.finditer(text):
            left = text[start:m.start()].strip()
            if _ABBR.search(left) or re.search(r'\b[A-Z]\.$', left):  # "Mr.", "J." -> not a boundary
                continue
            if left:
                parts.append(left)
            start = m.end()
        rest = text[start:].strip()
        if rest:
            parts.append(rest)
        return parts


def split_sentences(text):
    out = []
    for s in _base_split(text.strip()):
        out.extend(fix_glued(s))
    return out


if __name__ == '__main__':
    import sys
    for s in split_sentences(sys.stdin.read()):
        print(s)
