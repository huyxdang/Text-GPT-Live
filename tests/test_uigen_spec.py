"""Validator coverage for the generated-UI vocabulary.

The renderer is a strict allowlist: the delegated model may choose content,
component order, span, emphasis and accent, but anything outside those closed
vocabularies — or any value that would render as a placeholder — must be
rejected before it reaches the browser.
"""
from __future__ import annotations

import unittest

from app.uigen import (
    UI_ACCENTS,
    _DEMO_SPEC,
    _validate_build_plan,
    validate_ui_spec,
)


def spec(*components, **extra) -> dict:
    return {"title": "Panel", "components": list(components), **extra}


class UISpecVocabularyTests(unittest.TestCase):
    def test_every_new_component_type_validates(self) -> None:
        problems = validate_ui_spec(
            spec(
                {"type": "hero", "name": "Elon Musk", "tagline": "Engineer and investor"},
                {"type": "stat_card", "label": "Revenue", "value": "$14.0B"},
                {
                    "type": "chart",
                    "chart": "bar",
                    "label": "Growth",
                    "series": [
                        {"label": "Revenue", "value": 20},
                        {"label": "Bookings", "value": 22},
                        {"label": "EBITDA", "value": 35},
                    ],
                },
                {
                    "type": "timeline",
                    "events": [
                        {"date": "1971", "text": "Born in Pretoria."},
                        {"date": "2002", "text": "Founded SpaceX."},
                    ],
                },
                {
                    "type": "fact",
                    "facts": [
                        {"label": "Role", "value": "CEO"},
                        {"label": "Founded", "value": "SpaceX"},
                    ],
                },
                {"type": "callout", "text": "Momentum across every segment."},
                {
                    "type": "comparison",
                    "columns": [
                        {"title": "Mobility", "items": ["Ride hailing"]},
                        {"title": "Delivery", "items": ["Uber Eats"]},
                    ],
                },
                {"type": "list", "label": "Highlights", "items": ["Exports led growth."]},
                {"type": "sources", "links": [{"title": "IR", "url": "https://example.com"}]},
                accent="violet",
            )
        )
        self.assertEqual(problems, [])

    def test_demo_spec_stays_renderable(self) -> None:
        self.assertEqual(validate_ui_spec(_DEMO_SPEC), [])
        self.assertIn(_DEMO_SPEC["accent"], UI_ACCENTS)

    def test_presentation_vocabularies_are_closed(self) -> None:
        for field, value in (("span", "third"), ("tone", "hot"), ("emphasis", "huge")):
            component = {"type": "callout", "text": "Fine text", field: value}
            with self.subTest(field=field):
                self.assertTrue(validate_ui_spec(spec(component)))

    def test_accent_must_come_from_the_palette(self) -> None:
        self.assertTrue(validate_ui_spec(spec({"type": "callout", "text": "Hi"}, accent="chartreuse")))
        self.assertEqual(
            validate_ui_spec(spec({"type": "callout", "text": "Hi"}, accent="amber")), []
        )

    def test_only_one_component_may_lead(self) -> None:
        problems = validate_ui_spec(
            spec(
                {"type": "callout", "text": "First", "emphasis": "lead"},
                {"type": "callout", "text": "Second", "emphasis": "lead"},
            )
        )
        self.assertTrue(any("lead" in problem for problem in problems))

    def test_per_type_caps_are_enforced(self) -> None:
        problems = validate_ui_spec(
            spec(
                {"type": "hero", "name": "A", "tagline": "one"},
                {"type": "hero", "name": "B", "tagline": "two"},
            )
        )
        self.assertTrue(any("hero" in problem for problem in problems))


class UISpecHonestyTests(unittest.TestCase):
    """Guards that keep unusable content off the stage."""

    def test_placeholder_stat_card_is_rejected(self) -> None:
        for value in ("$X.XB", "TBD", "N/A", "unknown", "—"):
            with self.subTest(value=value):
                problems = validate_ui_spec(
                    spec({"type": "stat_card", "label": "Revenue", "value": value})
                )
                self.assertTrue(problems, f"{value!r} should not render")

    def test_placeholder_fact_row_is_rejected(self) -> None:
        problems = validate_ui_spec(
            spec(
                {
                    "type": "fact",
                    "facts": [
                        {"label": "HQ", "value": "TBD"},
                        {"label": "CEO", "value": "Dara Khosrowshahi"},
                    ],
                }
            )
        )
        self.assertTrue(problems)

    def test_chart_rejects_calendar_years_as_values(self) -> None:
        problems = validate_ui_spec(
            spec(
                {
                    "type": "chart",
                    "chart": "bar",
                    "label": "Milestones",
                    "series": [
                        {"label": "Born", "value": 1971},
                        {"label": "Zip2", "value": 1995},
                        {"label": "SpaceX", "value": 2002},
                    ],
                }
            )
        )
        self.assertTrue(any("calendar years" in problem for problem in problems))

    def test_chart_rejects_all_zero_and_flat_series(self) -> None:
        zeros = validate_ui_spec(
            spec(
                {
                    "type": "chart",
                    "chart": "bar",
                    "label": "Empty",
                    "series": [
                        {"label": "a", "value": 0},
                        {"label": "b", "value": 0},
                        {"label": "c", "value": 0},
                    ],
                }
            )
        )
        self.assertTrue(any("zeros" in problem for problem in zeros))

        flat = validate_ui_spec(
            spec(
                {
                    "type": "chart",
                    "chart": "bar",
                    "label": "Flat",
                    "series": [
                        {"label": "a", "value": 100},
                        {"label": "b", "value": 101},
                        {"label": "c", "value": 102},
                    ],
                }
            )
        )
        self.assertTrue(any("nearly identical" in problem for problem in flat))

    def test_a_real_metric_chart_still_passes(self) -> None:
        self.assertEqual(
            validate_ui_spec(
                spec(
                    {
                        "type": "chart",
                        "chart": "bar",
                        "label": "GDP growth (%)",
                        "series": [
                            {"label": "2021", "value": 2.6},
                            {"label": "2022", "value": 8.0},
                            {"label": "2023", "value": 5.1},
                        ],
                    }
                )
            ),
            [],
        )


class UIBuildPlanTests(unittest.TestCase):
    def test_plan_returns_title_accent_and_types(self) -> None:
        title, accent, types = _validate_build_plan(
            {"title": "  Uber  ", "accent": "green", "component_types": ["hero", "stat_card"]}
        )
        self.assertEqual((title, accent), ("Uber", "green"))
        self.assertEqual(types, ["hero", "stat_card"])

    def test_plan_defaults_to_blue_and_rejects_bad_input(self) -> None:
        _, accent, _ = _validate_build_plan({"title": "T", "component_types": ["callout"]})
        self.assertEqual(accent, "blue")
        with self.assertRaises(ValueError):
            _validate_build_plan({"title": "T", "component_types": ["hero", "hero"]})
        with self.assertRaises(ValueError):
            _validate_build_plan({"title": "T", "component_types": ["nope"]})


if __name__ == "__main__":
    unittest.main()
