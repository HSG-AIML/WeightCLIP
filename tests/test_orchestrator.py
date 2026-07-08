from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sane.orchestrator import Orchestrator, _build_runner, _config_hash, _resolve_trainer_class
from sane.runner.base import RunResult


def _make_config(tmp_path: Path) -> dict:
    return {
        "experiment_name": "test_exp",
        "root_dir": str(tmp_path),
        "trainer": "SANETrainer",
        "runner": "local",
        "epochs": 5,
        "checkpoint_frequency": 2,
    }


def _mock_build_runner(tmp_path: Path, cfg: dict) -> MagicMock:
    runner = MagicMock()
    runner.run.return_value = RunResult(
        checkpoint_path=tmp_path / "ckpt", config=cfg, metrics={"loss": 0.1}
    )
    return runner


def test_orchestrator_creates_storage_dir(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    with patch("sane.orchestrator._build_runner", return_value=_mock_build_runner(tmp_path, cfg)):
        Orchestrator(config=cfg).run()

    storage_dir = tmp_path / cfg["experiment_name"]
    assert storage_dir.is_dir()


def test_orchestrator_saves_config_yaml(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    cfg = _make_config(tmp_path)

    with patch("sane.orchestrator._build_runner", return_value=_mock_build_runner(tmp_path, cfg)):
        Orchestrator(config=cfg).run()

    config_path = tmp_path / cfg["experiment_name"] / f"config_{_config_hash(cfg)}.yaml"
    assert config_path.exists()
    saved = OmegaConf.to_container(OmegaConf.load(config_path))
    assert saved["experiment_name"] == "test_exp"
    assert saved["trainer"] == "SANETrainer"
    assert saved["epochs"] == 5


def test_orchestrator_delegates_to_runner(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    expected_result = RunResult(
        checkpoint_path=tmp_path / "ckpt", config=cfg, metrics={"loss": 0.1}
    )
    runner = MagicMock()
    runner.run.return_value = expected_result

    with patch("sane.orchestrator._build_runner", return_value=runner):
        result = Orchestrator(config=cfg).run()

    runner.run.assert_called_once()
    call_kwargs = runner.run.call_args[1]
    assert call_kwargs["epochs"] == 5
    assert call_kwargs["checkpoint_frequency"] == 2
    assert call_kwargs["storage_dir"] == tmp_path / cfg["experiment_name"]
    assert result is expected_result


def test_orchestrator_force_local_passed_to_build_runner(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg["runner"] = "ray"

    with patch("sane.orchestrator._build_runner", return_value=_mock_build_runner(tmp_path, cfg)) as mock_build:
        Orchestrator(config=cfg, force_local=True).run()

    _, kwargs = mock_build.call_args
    assert kwargs["force_local"] is True




def test_build_runner_force_local_overrides_ray(tmp_path: Path) -> None:
    from sane.runner.local_runner import LocalRunner

    runner = _build_runner({"runner": "ray"}, force_local=True)
    assert isinstance(runner, LocalRunner)


def test_build_runner_local(tmp_path: Path) -> None:
    from sane.runner.local_runner import LocalRunner

    runner = _build_runner({"runner": "local"})
    assert isinstance(runner, LocalRunner)


def test_build_runner_ray_backend(tmp_path: Path) -> None:
    from sane.runner.ray_runner import RayRunner

    runner = _build_runner({
        "runner": "ray",
        "num_cpus": 4,
        "num_gpus": 1,
        "cpu_per_trial": 2,
        "gpu_per_trial": 0.5,
        "num_samples": 1,
        "num_checkpoints_to_keep": 3,
        "ray_verbose": 1,
        "wandb_project": None,
        "wandb_group": None,
        "wandb_api_key_file": None,
    })
    assert isinstance(runner, RayRunner)
    assert runner._num_cpus == 4
    assert runner._num_gpus == 1


def test_resolve_trainer_class_known() -> None:
    from sane.trainer import SANETrainer

    cls = _resolve_trainer_class("SANETrainer")
    assert cls is SANETrainer


def test_resolve_trainer_class_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown trainer class"):
        _resolve_trainer_class("Bogus")