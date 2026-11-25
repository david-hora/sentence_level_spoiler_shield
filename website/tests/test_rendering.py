import unittest

from spoiler_shield.model_adapter import Prediction
from spoiler_shield.rendering import render_annotated_review, render_protected_review
from spoiler_shield.reviews import AnnotatedSentence


class RenderingTests(unittest.TestCase):
    def test_renderer_escapes_user_text_and_creates_independent_reveals(self) -> None:
        predictions = [
            Prediction("<script>alert(1)</script>. ", False, 0.1, "Safe"),
            Prediction("A dies.", True, 0.8, "Mock <reason>"),
            Prediction(" B survives.", True, 0.9, "Another reason"),
        ]
        rendered = render_protected_review(predictions)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertEqual(rendered.count('<details class="spoiler-mask">'), 2)
        self.assertIn("Mock &lt;reason&gt;", rendered)

    def test_annotated_renderer_hides_only_labelled_sentences(self) -> None:
        rendered = render_annotated_review(
            [
                AnnotatedSentence("Great acting.", False),
                AnnotatedSentence("The culprit is revealed.", True),
            ]
        )
        self.assertEqual(rendered.count('<details class="spoiler-mask">'), 1)
        self.assertIn("Great acting.", rendered)
        self.assertIn("labelled as a spoiler", rendered)
