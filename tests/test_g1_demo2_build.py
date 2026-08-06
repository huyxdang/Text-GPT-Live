from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.g1_authored_demo2 import validate_demo2_batches as _real_validate_demo2_batches
from datagen.g1_demo2 import Demo2Targets
from scripts.g1_demo2_build import (
    Demo2BuildError,
    build_demo2_artifacts,
    validate_demo2_rows,
    visible_text_from_prompt,
)

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "g1_demo2_sample"
FIXTURE_TARGETS = Demo2Targets(
    errors=17,
    matches=8,
    episodes=12,
    cards=140,
    empty_per_kind=2,
    hard_idle_cards=24,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


SINGLE_AUTHOR_TARGETS = Demo2Targets(
    errors=7, matches=2, episodes=4, cards=45, empty_per_kind=1, hard_idle_cards=6
)


def _single_author_source(root: Path) -> None:
    """One real author's own tranche (4 records) as its only authored file.

    This is exactly the situation the fix addresses: a lone author's slice of
    the corpus structurally cannot satisfy the cross-author distribution
    gates (3-4 authors, 5+ personas, 5+ domains, 4+ registers), so it can
    never verify its own work compiles without ``--allow-small-corpus``.
    """

    demo2_dir = root / "demo2"
    demo2_dir.mkdir(parents=True)
    data = (SAMPLE_ROOT / "demo2" / "d2_sample_sol_a.json").read_text(encoding="utf-8")
    (demo2_dir / "only.json").write_text(data, encoding="utf-8")


def _build(root: Path, **overrides) -> tuple[dict, Path, Path, Path]:
    train_base_path = root / "data" / "train_g1_demo2.jsonl"
    dev_path = root / "data" / "dev_g1_demo2.jsonl"
    artifact_dir = root / "artifacts"
    manifest = build_demo2_artifacts(
        authored_root=overrides.pop("authored_root", SAMPLE_ROOT),
        train_base_path=train_base_path,
        dev_path=dev_path,
        artifact_dir=artifact_dir,
        targets=overrides.pop("targets", FIXTURE_TARGETS),
        dev_fraction=overrides.pop("dev_fraction", 1 / 12),
        **overrides,
    )
    return manifest, train_base_path, dev_path, artifact_dir


def _rows_from(manifest: dict, dev_path: Path) -> list[dict]:
    rows: list[dict] = []
    for entry in manifest["files"]["train"]["shards"]:
        rows.extend(_load_jsonl(Path(entry["path"])))
    rows.extend(_load_jsonl(dev_path))
    return rows


class Demo2FullBuildTests(unittest.TestCase):
    def test_the_build_is_deterministic_auditable_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_base_path = root / "data" / "train_g1_demo2.jsonl"
            train_base_path.parent.mkdir(parents=True)
            train_base_path.write_text("obsolete single-file build\n", encoding="utf-8")
            stale = train_base_path.with_name("train_g1_demo2-00001-of-00002.jsonl")
            stale.write_text("obsolete shard\n", encoding="utf-8")

            first, _, dev_path, artifact_dir = _build(root)
            shard_paths = [Path(e["path"]) for e in first["files"]["train"]["shards"]]
            outputs = [
                *shard_paths,
                dev_path,
                artifact_dir / "manifest.json",
                artifact_dir / "coverage.json",
                artifact_dir / "inspection_samples.md",
            ]
            first_bytes = {path: path.read_bytes() for path in outputs}
            second, *_ = _build(root)

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, {path: path.read_bytes() for path in outputs})
            self.assertFalse(train_base_path.exists())
            self.assertFalse(stale.exists())
            self.assertEqual(len(shard_paths), 3)

            rows = _rows_from(first, dev_path)
            self.assertEqual(len(rows), 140)
            self.assertTrue(all(row["schema_version"] == "g1" for row in rows))
            self.assertTrue(all(row["demo"] == "demo-2" for row in rows))
            self.assertTrue(
                all(row["prompt"].endswith("<PREDICT_THIS_ACTION>") for row in rows)
            )
            self.assertTrue(
                all(result["exact"] for result in first["exact_target_evidence"].values())
            )

            aggregate = hashlib.sha256()
            for path, entry in zip(shard_paths, first["files"]["train"]["shards"], strict=True):
                payload = path.read_bytes()
                aggregate.update(payload)
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(first["files"]["train"]["aggregate_sha256"], aggregate.hexdigest())

            classes = {row["expected_class"] for row in rows}
            self.assertEqual(classes, {"idle", "respond", "suggest_edit", "highlight"})
            self.assertEqual(first["row_validation"]["completion_suggest_edit"], 17)
            self.assertEqual(first["row_validation"]["completion_highlight"], 8)
            self.assertEqual(first["row_validation"]["completion_respond"], 12)

            coverage = json.loads((artifact_dir / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(
                coverage["selected_empty_kinds"], {"cleared": 2, "initial": 2, "unchanged": 2}
            )
            self.assertEqual(
                sorted(coverage["selected_traps"]),
                ["clean_text", "instruction_mention", "non_literal_match", "self_correction"],
            )
            self.assertEqual(coverage["typing_delta_chars"], {"min": 4, "max": 7})
            self.assertGreater(coverage["time_gap_ms"]["max"], 10_000)
            self.assertGreater(len(coverage["source_skeletons"]), 1)

            inspection = (artifact_dir / "inspection_samples.md").read_text(encoding="utf-8")
            for role in coverage["selected_roles"]:
                self.assertIn(f"### role: {role}", inspection)
            for trap in coverage["selected_traps"]:
                self.assertIn(f"### trap: {trap}", inspection)
            self.assertIn("mode dominant", inspection)
            self.assertIn("longest selected prompt", inspection)

    def test_every_episode_stays_on_one_side_of_the_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, dev_path, _ = _build(Path(temporary))
            rows = _rows_from(manifest, dev_path)
            by_episode: dict[str, set[str]] = {}
            for row in rows:
                by_episode.setdefault(row["episode"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_episode.values()))
            self.assertEqual(len(by_episode), 12)

    def test_a_target_that_does_not_match_the_source_blocks_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                _build(root, targets=Demo2Targets(errors=99, matches=8, episodes=12))
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "artifacts").exists())

    def test_a_missing_source_tree_is_a_clear_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Demo2BuildError, "No Demo 2 authored JSON files"):
                _build(root, authored_root=root / "empty")

    def test_a_single_author_tranche_fails_the_distribution_gate_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            out = Path(temporary) / "out"
            _single_author_source(source)
            with self.assertRaises(Demo2BuildError) as caught:
                build_demo2_artifacts(
                    authored_root=source,
                    train_base_path=out / "data" / "train_g1_demo2.jsonl",
                    dev_path=out / "data" / "dev_g1_demo2.jsonl",
                    artifact_dir=out / "artifacts",
                    targets=SINGLE_AUTHOR_TARGETS,
                    dev_fraction=0.25,
                )
            self.assertIn("3-4 authors", str(caught.exception))

    def test_allow_small_corpus_lets_a_single_author_slot_build_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            out = Path(temporary) / "out"
            _single_author_source(source)
            manifest = build_demo2_artifacts(
                authored_root=source,
                train_base_path=out / "data" / "train_g1_demo2.jsonl",
                dev_path=out / "data" / "dev_g1_demo2.jsonl",
                artifact_dir=out / "artifacts",
                targets=SINGLE_AUTHOR_TARGETS,
                dev_fraction=0.25,
                allow_small_corpus=True,
            )
            self.assertFalse(manifest["distribution_gates_enforced"])
            self.assertEqual(manifest["row_counts"]["total"], 45)
            self.assertEqual(
                set(manifest["source"]["distributions"]["agent"]), {"d2-sample-sol-a"}
            )


def _inject_warning(*args, **kwargs):
    result = dict(_real_validate_demo2_batches(*args, **kwargs))
    result["warnings"] = [*result["warnings"], "synthetic-test-warning: canary only"]
    return result


class WarningsAreNonFatalTests(unittest.TestCase):
    def test_warnings_are_non_fatal_by_default_and_land_in_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "scripts.g1_demo2_build.validate_demo2_batches",
                side_effect=_inject_warning,
            ):
                manifest, *_ = _build(root)
            self.assertIn(
                "synthetic-test-warning: canary only", manifest["source_warnings"]
            )

    def test_fail_on_warnings_turns_the_same_warning_into_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "scripts.g1_demo2_build.validate_demo2_batches",
                side_effect=_inject_warning,
            ):
                with self.assertRaises(Demo2BuildError) as caught:
                    _build(root, fail_on_warnings=True)
            self.assertIn("synthetic-test-warning", str(caught.exception))
            self.assertIn("--fail-on-warnings", str(caught.exception))


class Demo2RowValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        manifest, _, dev_path, _ = _build(Path(self._temporary.name))
        self.rows = _rows_from(manifest, dev_path)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _row(self, expected_class: str) -> dict:
        return dict(next(row for row in self.rows if row["expected_class"] == expected_class))

    def test_the_unmodified_rows_validate(self) -> None:
        self.assertEqual(validate_demo2_rows(self.rows)["rows"], 140)

    def test_the_visible_textbox_is_recovered_from_the_prompt(self) -> None:
        row = self._row("suggest_edit")
        visible = visible_text_from_prompt(row["prompt"])
        quote = json.loads(row["completion"][len("<action>suggest_edit(") : -len(")</action>")])
        self.assertEqual(visible.count(quote["quote"]), 1)
        self.assertTrue(visible.endswith(quote["quote"]))

    def test_a_non_unique_edit_quote_is_rejected(self) -> None:
        row = self._row("suggest_edit")
        row["completion"] = '<action>suggest_edit({"quote":"the","replacement":"thee"})</action>'
        with self.assertRaisesRegex(Demo2BuildError, "occurs .* times in the visible textbox"):
            validate_demo2_rows([row])

    def test_an_edit_that_quotes_the_instruction_line_is_rejected(self) -> None:
        row = self._row("suggest_edit")
        instruction = visible_text_from_prompt(row["prompt"]).split("\n", 1)[0]
        payload = json.dumps(
            {"quote": instruction, "replacement": instruction + " Thanks."},
            separators=(",", ":"),
            sort_keys=True,
        )
        row["completion"] = f"<action>suggest_edit({payload})</action>"
        with self.assertRaisesRegex(Demo2BuildError, "standing instruction line"):
            validate_demo2_rows([row])

    def test_a_highlight_occurrence_past_the_end_is_rejected(self) -> None:
        row = self._row("highlight")
        payload = json.loads(row["completion"][len("<action>highlight(") : -len(")</action>")])
        payload["occurrence"] = 99
        row["completion"] = (
            "<action>highlight(" + json.dumps(payload, separators=(",", ":"), sort_keys=True) + ")</action>"
        )
        with self.assertRaisesRegex(Demo2BuildError, "does not exist"):
            validate_demo2_rows([row])

    def test_a_mislabelled_expected_class_is_rejected(self) -> None:
        row = self._row("idle")
        row["expected_class"] = "respond"
        with self.assertRaisesRegex(Demo2BuildError, "does not match"):
            validate_demo2_rows([row])

    def test_unpaired_neighbours_are_rejected(self) -> None:
        rows = [row for row in self.rows if row["candidate_role"] != "error-after"]
        with self.assertRaisesRegex(Demo2BuildError, "neighbours ship in pairs"):
            validate_demo2_rows(rows)

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        row = self._row("idle")
        with self.assertRaisesRegex(Demo2BuildError, "duplicate candidate_id"):
            validate_demo2_rows([row, dict(row)])

    def test_a_missing_key_is_rejected(self) -> None:
        row = self._row("idle")
        del row["trigger_kind"]
        with self.assertRaisesRegex(Demo2BuildError, "missing required keys"):
            validate_demo2_rows([row])


if __name__ == "__main__":
    unittest.main()
