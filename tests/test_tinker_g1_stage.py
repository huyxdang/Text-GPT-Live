from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from train import tinker_run


def _report(
    *,
    strict: float = 0.8,
    fire: float = 0.7,
    wait: float = 0.9,
    silence: float = 1.0,
    clause: float = 0.95,
    gates: bool = False,
) -> dict:
    return {
        "summary": {
            "strict_row_accuracy": strict,
            "should_fire_recall": fire,
            "reminder_wait_accuracy": wait,
            "ordinary_silence_idle_accuracy": silence,
            "clause_boundary_accuracy": clause,
            "canonical_exact_rate": 1.0,
            "format_validity": 1.0,
        },
        "hard_gates": {"passed": gates, "gates": {}},
        "rows": [],
    }


class _Future:
    def __init__(self, value=None):
        self.value = value

    def result(self, timeout=None):
        return self.value


class _TrainingClient:
    model_id = "fake-model"

    def __init__(self):
        self.adam_params = []

    def forward_backward(self, batch, loss):
        return _Future(
            SimpleNamespace(
                metrics={"loss:sum": float(len(batch)), "loss_fn_output_weight:sum": len(batch)}
            )
        )

    def optim_step(self, params):
        self.adam_params.append(params)
        return _Future()

    def save_state(self, name):
        return _Future(SimpleNamespace(path=f"tinker://state/{name}"))

    def save_weights_and_get_sampling_client(self):
        return object()

    def save_weights_for_sampler(self, name):
        return _Future(SimpleNamespace(path=f"tinker://sampler/{name}"))


class _LengthTokenizer:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def apply_chat_template(self, *args, **kwargs):
        return list(range(self.prompt_tokens))

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": list(range(self.completion_tokens))}

    def convert_tokens_to_ids(self, token):
        return 999


class TinkerG1SelectionTests(unittest.TestCase):
    def test_behavior_floor_beats_misleading_aggregate_accuracy(self) -> None:
        always_idle_like = _report(strict=0.94, fire=0.0, wait=1.0, silence=1.0, clause=1.0)
        balanced = _report(strict=0.86, fire=0.8, wait=0.9, silence=0.95, clause=0.9)
        self.assertGreater(
            tinker_run.g1_checkpoint_selection_key(balanced),
            tinker_run.g1_checkpoint_selection_key(always_idle_like),
        )

    def test_missing_slice_metric_fails_before_checkpoint_selection(self) -> None:
        report = _report()
        del report["summary"]["should_fire_recall"]
        with self.assertRaisesRegex(ValueError, "should_fire_recall"):
            tinker_run.g1_checkpoint_selection_key(report)

    def test_target_count_matches_datum_targets_and_enforces_exact_boundary(self) -> None:
        pair = {
            "schema_version": "g1",
            "candidate_id": "boundary",
            "prompt": "prompt",
            "completion": "<action>idle()</action>",
        }
        tokenizer = _LengthTokenizer(
            prompt_tokens=tinker_run.TINKER_MAX_TARGET_TOKENS - 6,
            completion_tokens=6,
        )
        fake_tinker = SimpleNamespace(
            ModelInput=SimpleNamespace(from_ints=lambda values: values),
            Datum=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        target_count = tinker_run.g1_target_token_count(tokenizer, pair)
        datum = tinker_run.build_datum(fake_tinker, tokenizer, pair)
        self.assertEqual(target_count, tinker_run.TINKER_MAX_TARGET_TOKENS)
        self.assertEqual(len(datum.loss_fn_inputs["target_tokens"]), target_count)
        tinker_run._validate_g1_context_limit(tokenizer, [pair], [])

        tokenizer.prompt_tokens += 1
        with self.assertRaisesRegex(SystemExit, "boundary"):
            tinker_run._validate_g1_context_limit(tokenizer, [pair], [])


class TinkerG1StageTests(unittest.TestCase):
    def test_cli_exposes_g1_and_removes_v7(self) -> None:
        parser = tinker_run.build_parser()
        g1_args = parser.parse_args(["--stage", "g1"])
        self.assertEqual(g1_args.stage, "g1")
        self.assertEqual(g1_args.base_model, tinker_run.G1_BASE_MODEL)
        self.assertEqual(g1_args.seed, tinker_run.G1_SEED)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--stage", "v7"])

    def test_tag_must_stay_inside_g1_namespace(self) -> None:
        with self.assertRaisesRegex(SystemExit, "outside the g1 namespace"):
            tinker_run.stage_g1(tag="experiment")

    def test_legacy_mutating_stages_reject_reserved_g1_tag(self) -> None:
        parser = tinker_run.build_parser()
        for stage in (
            "train", "resume", "adapt", "v5", "v6", "v6-evals", "v6-promote", "loop"
        ):
            arguments = parser.parse_args(["--stage", stage, "--tag", "g1-bad"])
            with self.assertRaisesRegex(SystemExit, "reserved g1 tag"):
                tinker_run.validate_cli_namespace(arguments)

    def test_legacy_stages_reject_reserved_g1_source_tag(self) -> None:
        parser = tinker_run.build_parser()
        for stage in ("adapt", "v5", "v6", "v6-evals"):
            arguments = parser.parse_args(
                ["--stage", stage, "--source-tag", "g1-source"]
            )
            with self.assertRaisesRegex(SystemExit, "reserved g1 source tag"):
                tinker_run.validate_cli_namespace(arguments)

    def test_namespace_guard_runs_before_credentials_or_network(self) -> None:
        with (
            mock.patch.object(
                sys, "argv", ["tinker_run.py", "--stage", "train", "--tag", "g1"]
            ),
            mock.patch.object(tinker_run, "load_env_key") as load_env_key,
        ):
            with self.assertRaisesRegex(SystemExit, "reserved g1 tag"):
                tinker_run.main()
        load_env_key.assert_not_called()

    def test_cli_does_not_replace_explicit_zero_hyperparameters_with_defaults(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "tinker_run.py",
                    "--stage",
                    "g1",
                    "--epochs",
                    "0",
                    "--learning-rate",
                    "0",
                ],
            ),
            mock.patch.object(tinker_run, "load_env_key"),
            mock.patch.object(tinker_run, "stage_g1") as stage_g1,
        ):
            tinker_run.main()
        self.assertEqual(stage_g1.call_args.kwargs["epochs"], 0)
        self.assertEqual(stage_g1.call_args.kwargs["learning_rate"], 0.0)

    def test_one_mocked_epoch_selects_without_legacy_overall_score(self) -> None:
        client = _TrainingClient()
        service = SimpleNamespace(create_lora_training_client=lambda **kwargs: client)
        fake_tinker = SimpleNamespace(
            ServiceClient=lambda **kwargs: service,
            AdamParams=lambda **kwargs: kwargs,
        )
        train_pairs = [
            {"schema_version": "g1", "episode": "train-1"},
            {"schema_version": "g1", "episode": "train-2"},
        ]
        dev_pairs = [{"schema_version": "g1", "episode": "dev-1"}]

        def load_pairs(name: str):
            return train_pairs if name == "train_g1" else dev_pairs

        state_updates: list[dict] = []
        with (
            mock.patch.dict(sys.modules, {"tinker": fake_tinker}),
            mock.patch.object(
                tinker_run,
                "g1_train_dev_dataset_names",
                return_value=("train_g1", "dev_g1"),
            ),
            mock.patch.object(tinker_run, "load_pairs", side_effect=load_pairs),
            mock.patch.object(tinker_run, "get_tokenizer", return_value=object()),
            mock.patch.object(tinker_run, "_validate_g1_context_limit"),
            mock.patch.object(tinker_run, "build_datum", side_effect=lambda *_: object()),
            mock.patch.object(tinker_run, "_validate_g1_dev_contract"),
            mock.patch.object(tinker_run, "run_eval", return_value=_report()),
            mock.patch.object(
                tinker_run, "save_state_json", side_effect=lambda value: state_updates.append(value)
            ),
        ):
            result = tinker_run.stage_g1(tag="g1-test", epochs=1)

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["best_score"], 0.8)
        self.assertTrue(result["sampler_path"].startswith("tinker://sampler/"))
        self.assertTrue(
            any("g1-test:best_dev_selection_key" in update for update in state_updates)
        )
        self.assertEqual(client.adam_params[0]["learning_rate"], tinker_run.G1_LEARNING_RATE)

    def test_invalid_paid_run_inputs_fail_before_service_creation(self) -> None:
        service_client = mock.Mock()
        fake_tinker = SimpleNamespace(ServiceClient=service_client)
        invalid_cases = (
            {"epochs": 0},
            {"batch_size": 0},
            {"learning_rate": -1.0},
            {"base_model": ""},
            {"seed": -1},
        )
        with mock.patch.dict(sys.modules, {"tinker": fake_tinker}):
            for overrides in invalid_cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaises(SystemExit):
                        tinker_run.stage_g1(**overrides)
        service_client.assert_not_called()

    def test_tokenization_failure_happens_before_service_creation(self) -> None:
        service_client = mock.Mock()
        fake_tinker = SimpleNamespace(ServiceClient=service_client)
        pairs = [{"schema_version": "g1", "completion": "<action>idle()</action>"}]
        with (
            mock.patch.dict(sys.modules, {"tinker": fake_tinker}),
            mock.patch.object(
                tinker_run,
                "g1_train_dev_dataset_names",
                return_value=("train_g1", "dev_g1"),
            ),
            mock.patch.object(tinker_run, "load_pairs", return_value=pairs),
            mock.patch.object(tinker_run, "_validate_g1_dev_contract"),
            mock.patch.object(
                tinker_run, "get_tokenizer", side_effect=ValueError("bad tokenizer")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "bad tokenizer"):
                tinker_run.stage_g1()
        service_client.assert_not_called()

    def test_datum_failure_happens_before_service_creation(self) -> None:
        service_client = mock.Mock()
        fake_tinker = SimpleNamespace(ServiceClient=service_client)
        pairs = [{"schema_version": "g1", "completion": "<action>idle()</action>"}]
        with (
            mock.patch.dict(sys.modules, {"tinker": fake_tinker}),
            mock.patch.object(
                tinker_run,
                "g1_train_dev_dataset_names",
                return_value=("train_g1", "dev_g1"),
            ),
            mock.patch.object(tinker_run, "load_pairs", return_value=pairs),
            mock.patch.object(tinker_run, "_validate_g1_dev_contract"),
            mock.patch.object(tinker_run, "get_tokenizer", return_value=object()),
            mock.patch.object(tinker_run, "_validate_g1_context_limit"),
            mock.patch.object(
                tinker_run, "build_datum", side_effect=ValueError("malformed row")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "malformed row"):
                tinker_run.stage_g1()
        service_client.assert_not_called()

    def test_context_overflow_fails_before_service_creation(self) -> None:
        service_client = mock.Mock()
        fake_tinker = SimpleNamespace(ServiceClient=service_client)
        pairs = [
            {
                "schema_version": "g1",
                "candidate_id": "too-long",
                "completion": "<action>idle()</action>",
            }
        ]
        with (
            mock.patch.dict(sys.modules, {"tinker": fake_tinker}),
            mock.patch.object(
                tinker_run,
                "g1_train_dev_dataset_names",
                return_value=("train_g1", "dev_g1"),
            ),
            mock.patch.object(tinker_run, "load_pairs", return_value=pairs),
            mock.patch.object(tinker_run, "_validate_g1_dev_contract"),
            mock.patch.object(tinker_run, "get_tokenizer", return_value=object()),
            mock.patch.object(
                tinker_run,
                "g1_target_token_count",
                return_value=tinker_run.TINKER_MAX_TARGET_TOKENS + 1,
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "too-long"):
                tinker_run.stage_g1()
        service_client.assert_not_called()

    def test_final_uses_the_selected_g1_model_and_seed(self) -> None:
        sampler = object()
        service = SimpleNamespace(create_sampling_client=lambda **kwargs: sampler)
        fake_tinker = SimpleNamespace(ServiceClient=lambda **kwargs: service)
        tokenizer = object()
        pairs = [{"schema_version": "g1", "completion": "<action>idle()</action>"}]
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "run_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "g1-custom:sampler_path": "tinker://sampler/custom",
                        "g1-custom:base_model": "Qwen/Qwen3.5-9B",
                        "g1-custom:seed": 1234,
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.dict(sys.modules, {"tinker": fake_tinker}),
                mock.patch.object(tinker_run, "STATE_PATH", state_path),
                mock.patch.object(
                    tinker_run, "get_tokenizer", return_value=tokenizer
                ) as get_tokenizer,
                mock.patch.object(tinker_run, "load_pairs", return_value=pairs),
                mock.patch.object(tinker_run, "run_eval", return_value=_report()) as run_eval,
                mock.patch.object(tinker_run, "save_state_json"),
            ):
                tinker_run.stage_final("g1-custom", "dev_g1")

        get_tokenizer.assert_called_once_with("Qwen/Qwen3.5-9B")
        self.assertEqual(run_eval.call_args.kwargs["base_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(run_eval.call_args.kwargs["seed"], 1234)


if __name__ == "__main__":
    unittest.main()
