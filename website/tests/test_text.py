import unittest

from spoiler_shield.text import split_preserving_formatting


class TextSplittingTests(unittest.TestCase):
    def test_split_preserves_original_text_exactly(self) -> None:
        review = 'Great acting.\n\n"The ending surprised me!"  Final thought without punctuation'
        chunks = split_preserving_formatting(review)
        self.assertEqual("".join(chunks), review)
        self.assertEqual(len(chunks), 3)

    def test_empty_review_has_no_chunks(self) -> None:
        self.assertEqual(split_preserving_formatting(""), [])
