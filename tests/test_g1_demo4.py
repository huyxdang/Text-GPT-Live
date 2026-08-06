from __future__ import annotations

import json
import unittest

from app.domain import ActionKind, EventSource
from app.stream import g1_action_completion, parse_g1_action
from datagen.g1_demo4 import (
    Demo4Targets,
    closing_shape,
    compile_demo4_dataset,
    compile_demo4_record,
    compile_demo4_records,
    demo4_banks_from_batches,
    expand_demo4_episodes,
    opening_shape,
    render_demo4_card,
    select_demo4_candidates,
)


def _requests(n: int = 6) -> list[dict]:
    out = []
    for index in range(n):
        out.append(
            {
                "id": f"req-{index:03d}",
                "text": f"Can you build a little panel about topic {index} please?",
                "task": f"generate a UI to visualize topic {index} statistics",
                "persona": f"persona-{index % 6}",
                "domain": f"domain-{index % 5}",
                "register": f"register-{index % 4}",
                "author_slot": f"slot-{index % 3}",
                "author_model": "test-model",
                "author_tranche": "1",
            }
        )
    return out


def _progress(n: int = 6) -> list[dict]:
    out = []
    for index in range(n):
        kind = "check" if index % 2 == 0 else "nudge"
        item = {
            "id": f"prog-{index:03d}",
            "kind": kind,
            "question": f"how's it going {index % 9}" if kind == "check" else f"got that {index % 9}?",
            "persona": f"persona-{index % 6}",
            "domain": f"domain-{index % 5}",
            "register": f"register-{index % 4}",
            "author_slot": f"slot-{index % 3}",
            "author_model": "test-model",
            "author_tranche": "1",
        }
        if kind == "check":
            item["reply"] = f"Still working on it — should be up soon ({index})."
        out.append(item)
    return out


class ExpansionTests(unittest.TestCase):
    def test_expansion_cross_products_requests_and_variants(self) -> None:
        episodes = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        self.assertEqual(len(episodes), 15)
        ids = {episode["id"] for episode in episodes}
        self.assertEqual(len(ids), 15)

    def test_expansion_is_deterministic(self) -> None:
        first = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        second = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        self.assertEqual(first, second)

    def test_expansion_requires_both_progress_kinds(self) -> None:
        only_checks = [item for item in _progress(6) if item["kind"] == "check"]
        with self.assertRaises(ValueError) as caught:
            expand_demo4_episodes(_requests(6), only_checks, count=6)
        self.assertIn("nudge", str(caught.exception))

    def test_content_kind_and_outcome_cycle_deterministically(self) -> None:
        episodes = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        kinds = [episode["content_kind"] for episode in episodes]
        self.assertEqual(kinds, ["check", "nudge", "narration"] * 5)
        outcomes = [episode["outcome"] for episode in episodes]
        self.assertEqual(outcomes.count("failure"), 3)


class CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        self.compiled = {record.record_id: record for record in compile_demo4_records(self.episodes)}

    def test_every_episode_compiles(self) -> None:
        self.assertEqual(len(self.compiled), 15)

    def test_delegate_fires_exactly_once_on_the_completion_tick(self) -> None:
        for record in self.compiled.values():
            fires = [
                turn
                for turn in record.turns
                if turn.action.kind is ActionKind.TOOL and turn.action.tool_name == "delegate"
            ]
            self.assertEqual(len(fires), 1, record.record_id)

    def test_tool_events_are_always_graded_idle(self) -> None:
        for record in self.compiled.values():
            for turn in record.turns:
                if turn.event.source is EventSource.TOOL:
                    self.assertIs(turn.action.kind, ActionKind.IDLE, record.record_id)

    def test_accepted_payload_is_byte_exact(self) -> None:
        record = next(iter(self.compiled.values()))
        accepted = next(
            turn for turn in record.turns if turn.event.source is EventSource.TOOL
        )
        self.assertEqual(accepted.event.content, '{"status":"accepted"}')
        self.assertEqual(accepted.event.tool_name, "delegate")
        self.assertIsNotNone(accepted.event.job_id)

    def test_completed_payload_is_byte_exact(self) -> None:
        record = next(r for r in self.compiled.values() if r.outcome == "success")
        completed = next(
            turn
            for turn in record.turns
            if turn.event.source is EventSource.TOOL
            and json.loads(turn.event.content).get("status") == "completed"
        )
        self.assertEqual(
            completed.event.content, f'{{"status":"completed","task":"{record.request_task}"}}'
        )

    def test_failed_payload_carries_task_and_error_in_runtime_key_order(self) -> None:
        record = next(r for r in self.compiled.values() if r.outcome == "failure")
        failed = next(
            turn
            for turn in record.turns
            if turn.event.source is EventSource.TOOL
            and json.loads(turn.event.content).get("status") == "failed"
        )
        # Runtime key order is insertion order (status, task, error), not
        # alphabetical -- this is what "train equals serve" means for a tool
        # payload, since app/runtime.py does not sort_keys.
        prefix = f'{{"status":"failed","task":"{record.request_task}","error":"'
        self.assertTrue(failed.event.content.startswith(prefix), failed.event.content)

    def test_job_id_never_repeats_inside_the_payload(self) -> None:
        for record in self.compiled.values():
            for turn in record.turns:
                if turn.event.source is EventSource.TOOL:
                    self.assertNotIn("job_id", turn.event.content)

    def test_pendency_is_within_the_sampled_range(self) -> None:
        for record in self.compiled.values():
            accepted_index = next(
                turn.event.index
                for turn in record.turns
                if turn.event.source is EventSource.TOOL
                and json.loads(turn.event.content).get("status") == "accepted"
            )
            terminal_index = next(
                turn.event.index
                for turn in record.turns
                if turn.event.source is EventSource.TOOL
                and json.loads(turn.event.content).get("status") in {"completed", "failed"}
            )
            pendency = terminal_index - accepted_index - 1
            if record.outcome == "success":
                self.assertGreaterEqual(pendency, 2, record.record_id)
                self.assertLessEqual(pendency, 8, record.record_id)
            else:
                self.assertGreaterEqual(pendency, 5, record.record_id)
                self.assertLessEqual(pendency, 10, record.record_id)

    def test_completion_never_lands_instantly(self) -> None:
        for record in self.compiled.values():
            accepted_index = next(
                turn.event.index
                for turn in record.turns
                if turn.event.source is EventSource.TOOL
                and json.loads(turn.event.content).get("status") == "accepted"
            )
            terminal_index = next(
                turn.event.index
                for turn in record.turns
                if turn.event.source is EventSource.TOOL
                and json.loads(turn.event.content).get("status") in {"completed", "failed"}
            )
            self.assertGreater(terminal_index, accepted_index + 1)

    def test_nudge_never_causes_a_second_delegate(self) -> None:
        record = next(r for r in self.compiled.values() if r.content_kind == "nudge")
        fires = [
            turn
            for turn in record.turns
            if turn.action.kind is ActionKind.TOOL and turn.action.tool_name == "delegate"
        ]
        self.assertEqual(len(fires), 1)
        nudge_candidate = next(c for c in record.candidates if c.role == "nudge-idle")
        self.assertIs(record.turns[nudge_candidate.turn_offset].action.kind, ActionKind.IDLE)

    def test_check_positive_is_a_respond_before_the_job_resolves(self) -> None:
        record = next(r for r in self.compiled.values() if r.content_kind == "check")
        candidate = next(c for c in record.candidates if c.role == "check-positive")
        turn = record.turns[candidate.turn_offset]
        self.assertIs(turn.action.kind, ActionKind.RESPOND)
        self.assertEqual(turn.action.target, turn.event.index)
        terminal_indices = [
            t.event.index
            for t in record.turns
            if t.event.source is EventSource.TOOL
            and json.loads(t.event.content).get("status") in {"completed", "failed"}
        ]
        self.assertTrue(all(turn.event.index < idx for idx in terminal_indices))

    def test_failure_check_never_claims_completion(self) -> None:
        record = next(r for r in self.compiled.values() if r.outcome == "failure")
        candidate = next(c for c in record.candidates if c.role == "failure-check-positive")
        turn = record.turns[candidate.turn_offset]
        self.assertIs(turn.action.kind, ActionKind.RESPOND)
        message = turn.action.message or ""
        for banned in ("here it is", "it's ready", "here's your", "all done"):
            self.assertNotIn(banned, message.lower())

    def test_human_cadence_and_non_uniform_gaps(self) -> None:
        record = next(iter(self.compiled.values()))
        deltas: list[int] = []
        gaps: list[int] = []
        prior = ""
        prior_time = None
        for turn in record.turns:
            content = turn.event.content
            if content and content.startswith(prior) and len(content) > len(prior):
                deltas.append(len(content) - len(prior))
            if prior_time is not None:
                gaps.append(turn.event.elapsed_ms - prior_time)
            prior = content
            prior_time = turn.event.elapsed_ms
        self.assertTrue(deltas)
        self.assertGreaterEqual(min(deltas), 4)
        self.assertLessEqual(max(deltas), 7)
        self.assertGreater(len(set(gaps)), 3)

    def test_event_indices_are_contiguous(self) -> None:
        for record in self.compiled.values():
            self.assertEqual(
                [turn.event.index for turn in record.turns],
                list(range(1, len(record.turns) + 1)),
            )

    def test_exactly_one_candidate_per_turn(self) -> None:
        for record in self.compiled.values():
            offsets = [c.turn_offset for c in record.candidates]
            self.assertEqual(len(offsets), len(set(offsets)), record.record_id)


class SkeletonTests(unittest.TestCase):
    def test_opening_and_closing_shapes_vary_across_episodes(self) -> None:
        openings = {opening_shape(f"demo4-shape-{i:03d}") for i in range(200)}
        closings = {closing_shape(f"demo4-shape-{i:03d}") for i in range(200)}
        self.assertEqual(len(openings), 4)
        self.assertEqual(len(closings), 4)

    def test_shape_selection_is_deterministic(self) -> None:
        self.assertEqual(opening_shape("demo4-x"), opening_shape("demo4-x"))
        self.assertEqual(closing_shape("demo4-x"), closing_shape("demo4-x"))


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        episodes = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        self.compiled = {record.record_id: record for record in compile_demo4_records(episodes)}

    def test_every_card_round_trips_through_the_serving_parser(self) -> None:
        for record in self.compiled.values():
            for candidate in record.candidates:
                row = render_demo4_card(record, candidate)
                parsed = parse_g1_action(row["completion"])
                self.assertTrue(parsed.valid, row["candidate_id"])
                self.assertTrue(row["prompt"].endswith("<PREDICT_THIS_ACTION>"))

    def test_delegate_completion_matches_g1_action_completion(self) -> None:
        record = next(iter(self.compiled.values()))
        candidate = next(c for c in record.candidates if c.role == "request-positive")
        row = render_demo4_card(record, candidate)
        parsed = parse_g1_action(row["completion"])
        self.assertEqual(row["completion"], g1_action_completion(parsed))
        self.assertEqual(parsed.arguments, {"task": record.request_task})

    def test_respond_target_is_the_current_event(self) -> None:
        for record in self.compiled.values():
            for candidate in record.candidates:
                row = render_demo4_card(record, candidate)
                parsed = parse_g1_action(row["completion"])
                if parsed.kind is ActionKind.RESPOND:
                    self.assertEqual(parsed.target, row["current_event_index"])

    def test_job_id_only_present_on_tool_ticks(self) -> None:
        record = next(iter(self.compiled.values()))
        for candidate in record.candidates:
            row = render_demo4_card(record, candidate)
            if candidate.role in {"accepted-idle", "completed-idle", "failed-idle"}:
                self.assertIsNotNone(row["job_id"])
            else:
                self.assertIsNone(row["job_id"])


class DeterminismTests(unittest.TestCase):
    def test_same_source_compiles_byte_identically(self) -> None:
        episodes = expand_demo4_episodes(_requests(6), _progress(6), count=15)
        first = compile_demo4_records(episodes)
        second = compile_demo4_records(episodes)
        first_by_id = {record.record_id: record for record in first}
        second_by_id = {record.record_id: record for record in second}
        for record_id in first_by_id:
            record_a = first_by_id[record_id]
            record_b = second_by_id[record_id]
            rows_a = [render_demo4_card(record_a, c) for c in record_a.candidates]
            rows_b = [render_demo4_card(record_b, c) for c in record_b.candidates]
            self.assertEqual(rows_a, rows_b)

    def test_utf8_hashing_is_stable_for_non_ascii_ids(self) -> None:
        episode = expand_demo4_episodes(_requests(6), _progress(6), count=1)[0]
        episode = dict(episode)
        episode["id"] = "demo4-记录-01"
        first = compile_demo4_record(episode)
        second = compile_demo4_record(episode)
        self.assertEqual(first.opening_shape, second.opening_shape)
        self.assertEqual(
            [turn.event.elapsed_ms for turn in first.turns],
            [turn.event.elapsed_ms for turn in second.turns],
        )


def _batch(slot: str, requests: list[dict], progress: list[dict]) -> dict:
    return {
        "schema_version": "g1-authored-demo4-1",
        "demo": "demo-4",
        "author": {"model": "test-model", "slot": slot, "tranche": 1},
        "requests": requests,
        "progress_pairs": progress,
    }


class BankFlatteningTests(unittest.TestCase):
    def test_flattening_attaches_author_provenance(self) -> None:
        raw_requests = [
            {"id": f"r{i}", "text": f"Can you make a quick chart {i} please?", "task": f"visualize {i}",
             "persona": "p", "domain": "d", "register": "reg"}
            for i in range(3)
        ]
        raw_progress = [
            {"id": "c0", "kind": "check", "question": "how's it going?", "reply": "Almost there.",
             "persona": "p", "domain": "d", "register": "reg"},
            {"id": "n0", "kind": "nudge", "question": "got that?", "persona": "p", "domain": "d", "register": "reg"},
        ]
        batches = [_batch("slot-a", raw_requests, raw_progress)]
        requests, progress = demo4_banks_from_batches(batches)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(progress), 2)
        self.assertEqual(requests[0]["author_slot"], "slot-a")
        self.assertEqual(progress[0]["author_slot"], "slot-a")


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = _requests(6)
        self.progress = _progress(6)
        self.episodes = expand_demo4_episodes(self.requests, self.progress, count=15)
        self.compiled = compile_demo4_records(self.episodes)
        self.targets = Demo4Targets(
            requests=6,
            progress_pairs=6,
            episodes=15,
            cards=128,
            empty_per_kind=10,
            min_check_positive=5,
            min_nudge_idle=5,
            min_narration_idle=5,
            min_failure_check=3,
        )

    def test_selection_hits_the_exact_card_target(self) -> None:
        selected = select_demo4_candidates(self.compiled, targets=self.targets)
        self.assertEqual(len(selected), 128)
        roles = {c.role for c in selected}
        self.assertIn("request-positive", roles)
        self.assertEqual(sum(1 for c in selected if c.role == "request-positive"), 15)
        self.assertEqual(sum(1 for c in selected if c.role == "accepted-idle"), 15)
        self.assertEqual(sum(1 for c in selected if c.role == "ack-positive"), 15)
        for kind in ("initial", "unchanged", "cleared"):
            self.assertEqual(sum(1 for c in selected if c.empty_kind == kind), 10)

    def test_wrong_source_size_is_a_hard_failure(self) -> None:
        with self.assertRaises(ValueError) as caught:
            select_demo4_candidates(
                self.compiled, targets=Demo4Targets(episodes=99, cards=100)
            )
        self.assertIn("99 episodes", str(caught.exception))

    def test_unreachable_card_target_is_diagnosed(self) -> None:
        with self.assertRaises(ValueError):
            select_demo4_candidates(
                self.compiled,
                targets=Demo4Targets(
                    episodes=15,
                    cards=10_000,
                    empty_per_kind=10,
                    min_check_positive=5,
                    min_nudge_idle=5,
                    min_narration_idle=5,
                    min_failure_check=3,
                ),
            )

    def test_dataset_build_is_deterministic_and_parseable(self) -> None:
        first = compile_demo4_dataset(self.episodes, targets=self.targets)
        second = compile_demo4_dataset(self.episodes, targets=self.targets)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(len(first.rows), 128)
        for row in first.rows:
            self.assertTrue(parse_g1_action(row["completion"]).valid)
        self.assertEqual(
            set(first.coverage["selected_empty_kinds"]), {"initial", "unchanged", "cleared"}
        )

    def test_default_targets_reach_exactly_870_cards_at_production_scale(self) -> None:
        """The headline number the brief asks for must actually be reachable."""

        requests = _requests(50)
        progress = _progress(50)
        targets = Demo4Targets()
        episodes = expand_demo4_episodes(requests, progress, count=targets.episodes)
        compiled = compile_demo4_records(episodes)
        selected = select_demo4_candidates(compiled, targets=targets)
        self.assertEqual(len(selected), 870)
        self.assertEqual(targets.episodes, 130)


if __name__ == "__main__":
    unittest.main()
