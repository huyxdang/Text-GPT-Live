from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.g1_v2_delivery_repair_build import _source_paths


def test_source_paths_resolve_stage2_candidate_from_run_state(tmp_path: Path) -> None:
    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "g1-v2:selected_state": "tinker://state",
                "g1-v2:selected_sampler_path": "tinker://sampler",
            }
        ),
        encoding="utf-8",
    )

    assert _source_paths(state_path, None, None) == (
        "tinker://state",
        "tinker://sampler",
    )


def test_source_paths_accept_explicit_checkpoint_without_local_state(tmp_path: Path) -> None:
    assert _source_paths(
        tmp_path / "missing.json",
        "tinker://external-state",
        "tinker://external-sampler",
    ) == ("tinker://external-state", "tinker://external-sampler")


def test_source_paths_require_complete_checkpoint_pair(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="together"):
        _source_paths(tmp_path / "missing.json", "tinker://state", None)


def test_source_paths_explain_missing_stage2_state(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="promoted stage-2"):
        _source_paths(tmp_path / "missing.json", None, None)
