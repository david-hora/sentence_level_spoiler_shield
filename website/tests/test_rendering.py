import unittest
from html.parser import HTMLParser

from spoiler_shield.model_adapter import Prediction
from spoiler_shield.rendering import render_annotated_review, render_protected_review
from spoiler_shield.reviews import AnnotatedSentence


class DisclosureParser(HTMLParser):
    """Separate visible text from each disclosure's hidden sentence body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.disclosures: list[dict[str, object]] = []
        self.visible: list[str] = []
        self.all_text: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        attributes = dict(attrs)
        self.stack.append((tag, attributes))
        if tag == "details":
            self.disclosures.append({"attrs": attributes, "copy": []})

    def handle_endtag(self, tag) -> None:
        self.assert_matching_end(tag)

    def assert_matching_end(self, tag) -> None:
        if not self.stack or self.stack[-1][0] != tag:
            raise AssertionError(f"Unbalanced rendered HTML: {tag}")
        self.stack.pop()

    def handle_data(self, data) -> None:
        self.all_text.append(data)
        inside_details = any(tag == "details" for tag, _ in self.stack)
        if not inside_details or any(tag == "summary" for tag, _ in self.stack):
            self.visible.append(data)
        if any(attrs.get("class") == "reveal-copy" for _, attrs in self.stack):
            self.disclosures[-1]["copy"].append(data)


class RenderingTests(unittest.TestCase):
    def assert_spoilers_are_closed(self, rendered: str, spoilers: list[str]) -> None:
        parsed = DisclosureParser()
        parsed.feed(rendered)
        self.assertFalse(parsed.stack)
        self.assertEqual(len(parsed.disclosures), len(spoilers))
        for disclosure, text in zip(parsed.disclosures, spoilers):
            self.assertNotIn("open", disclosure["attrs"])
            self.assertEqual("".join(disclosure["copy"]), text)
            self.assertNotIn(text, "".join(parsed.visible))
            self.assertEqual("".join(parsed.all_text).count(text), 1)

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
        self.assert_spoilers_are_closed(rendered, ["A dies.", "B survives."])

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
        self.assert_spoilers_are_closed(rendered, ["The culprit is revealed."])

    def test_annotated_paragraph_boundaries_stay_outside_masks(self) -> None:
        rendered = render_annotated_review([
            AnnotatedSentence("  Great acting.\n\n", False),
            AnnotatedSentence("\tThe culprit is revealed.\r\n\r\n", True),
            AnnotatedSentence("  A memorable score.  ", False),
        ])
        self.assertIn("Great acting.\n\n</span>\t<details", rendered)
        self.assertIn('</details>\r\n\r\n<span class="safe-sentence">  A memorable score.  </span>', rendered)
        self.assert_spoilers_are_closed(rendered, ["The culprit is revealed."])

    def test_unseparated_records_get_one_readable_space(self) -> None:
        rendered = render_annotated_review([
            AnnotatedSentence("First.", False), AnnotatedSentence("Second.", False),
        ])
        self.assertIn('</span> <span class="safe-sentence">Second.', rendered)
