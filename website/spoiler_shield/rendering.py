from __future__ import annotations

import html
from collections.abc import Sequence

from spoiler_shield.model_adapter import Prediction
from spoiler_shield.reviews import AnnotatedSentence


REVIEW_STYLES = """
<style>
.protected-review {
    color: #3f322e;
    max-width: 78ch;
    font-size: 1rem;
    line-height: 1.75;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}
.spoiler-mask {
    display: inline-block;
    margin: 0.12rem 0.16rem;
    vertical-align: baseline;
    white-space: normal;
}
.spoiler-mask summary {
    background: #4a312d;
    border: 2px solid #4a312d;
    border-radius: 0.35rem;
    color: #ffffff;
    cursor: pointer;
    display: inline-block;
    font-size: 0.86rem;
    font-weight: 650;
    list-style: none;
    letter-spacing: 0.015em;
    padding: 0.18rem 0.62rem;
    user-select: none;
}
.spoiler-mask summary::-webkit-details-marker { display: none; }
.spoiler-mask summary:focus-visible { outline: 3px solid #ad402e; outline-offset: 3px; }
.spoiler-mask .hide-label, .spoiler-mask[open] .show-label { display: none; }
.spoiler-mask[open] .hide-label { display: inline; }
.spoiler-mask[open] {
    background: #fff7ea;
    border: 1px solid #e54a3f;
    border-radius: 0.5rem;
    display: block;
    margin: 0.55rem 0;
    padding: 0.7rem 0.8rem;
}
.spoiler-mask[open] summary {
    background: transparent;
    border-color: transparent;
    color: #b52e27;
    padding: 0 0 0.45rem;
}
.reveal-copy { color: #3f322e; white-space: pre-wrap; }
.reveal-why {
    border-top: 1px solid #e8c8af;
    color: #755d55;
    font-size: 0.82rem;
    margin-top: 0.55rem;
    padding-top: 0.5rem;
}
</style>
"""


def _render_sentence(text: str, is_spoiler: bool, explanation: str, spoiler_number: int) -> str:
    escaped_text = html.escape(text)
    if not is_spoiler:
        return f'<span class="safe-sentence">{escaped_text}</span>'
    return (
        '<details class="spoiler-mask">'
        f'<summary><span class="show-label">Reveal spoiler {spoiler_number}</span>'
        f'<span class="hide-label">Hide spoiler {spoiler_number}</span></summary>'
        f'<div class="reveal-copy">{escaped_text}</div>'
        f'<div class="reveal-why">{html.escape(explanation)}</div>'
        '</details>'
    )


def render_annotated_review(sentences: Sequence[AnnotatedSentence]) -> str:
    """Render a dataset review using its human-approved sentence labels."""

    parts = ['<div class="protected-review" aria-label="Spoiler-protected review">']
    spoiler_number = 0
    for index, sentence in enumerate(sentences):
        if index:
            parts.append(" ")
        if sentence.is_spoiler:
            spoiler_number += 1
        parts.append(
            _render_sentence(
                sentence.text,
                sentence.is_spoiler,
                "Hidden because this sentence is labelled as a spoiler in the annotated dataset.",
                spoiler_number,
            )
        )
    parts.append("</div>")
    return "".join(parts)


def render_protected_review(predictions: Sequence[Prediction]) -> str:
    """Render escaped mock predictions with independently revealable spoiler regions."""

    parts = ['<div class="protected-review" aria-label="Spoiler-protected review">']
    spoiler_number = 0
    for prediction in predictions:
        if prediction.is_spoiler:
            spoiler_number += 1
        explanation = prediction.explanation
        if prediction.is_spoiler:
            explanation = f"{explanation} Mock score: {prediction.score:.0%}."
        parts.append(
            _render_sentence(
                prediction.text,
                prediction.is_spoiler,
                explanation,
                spoiler_number,
            )
        )
    parts.append("</div>")
    return "".join(parts)
