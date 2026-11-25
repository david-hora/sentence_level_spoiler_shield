from __future__ import annotations

import re


_SENTENCE = re.compile(r".*?(?:[.!?]+[\"'”’\)\]]*(?=\s|$)|$)", re.DOTALL)


def split_preserving_formatting(text: str) -> list[str]:
    """Split text into sentence-like chunks without consuming whitespace.

    Leading spaces and line breaks remain attached to the following chunk, so
    joining the returned list always reproduces the original text exactly.
    """

    if not text:
        return []
    chunks = [match.group(0) for match in _SENTENCE.finditer(text) if match.group(0)]
    if "".join(chunks) != text:
        raise ValueError("Review text could not be split without changing its formatting")
    return chunks
