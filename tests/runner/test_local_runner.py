from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from sane.runner.local_runner import LocalRunner


class FakeTrainer:
    def __init__(self) -> None:
        self._epoch = 0

    def setup(self, config):
        self._config = config

    def step(self):
        self._epoch += 1
        return {"loss": 1.0 / self._epoch, "epoch": self._epoch}

    def save_checkpoint(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / "checkpoint.pt").write_text("fake")

    def load_checkpoint(self, checkpoint_dir):
        pass


def test_run_single_trial(tmp_path: Path) -> None:
    runner = LocalRunner()
    result = runner.run(
        trainer_class=FakeTrainer,
        config={"lr": 0.01},
        epochs=3,
        checkpoint_frequency=1,
        storage_dir=tmp_path,
    )

    assert result.checkpoint_path.exists()
    assert (result.checkpoint_path / "checkpoint.pt").exists()
    assert "loss" in result.metrics
    assert "epoch" in result.metrics
    assert result.metrics["epoch"] == 3


def test_trial_saves_config_yaml_and_json(tmp_path: Path) -> None:
    runner = LocalRunner()
    runner.run(
        trainer_class=FakeTrainer,
        config={"lr": 0.01},
        epochs=1,
        checkpoint_frequency=1,
        storage_dir=tmp_path,
    )

    trial_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    config_yaml = trial_dir / "config.yaml"
    config_json = trial_dir / "config.json"
    assert config_yaml.exists()
    assert config_json.exists()

    loaded_yaml = yaml.safe_load(config_yaml.read_text())
    loaded_json = json.loads(config_json.read_text())
    assert loaded_yaml["lr"] == 0.01
    assert loaded_json["lr"] == 0.01


def test_checkpoint_frequency(tmp_path: Path) -> None:
    runner = LocalRunner()
    runner.run(
        trainer_class=FakeTrainer,
        config={"lr": 0.01},
        epochs=5,
        checkpoint_frequency=2,
        storage_dir=tmp_path,
    )

    trial_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]

    checkpoint_dirs = sorted(p.name for p in trial_dir.iterdir() if p.name.startswith("checkpoint_"))
    assert "checkpoint_000002" in checkpoint_dirs
    assert "checkpoint_000004" in checkpoint_dirs
    assert "checkpoint_000005" in checkpoint_dirs
    assert "checkpoint_000001" not in checkpoint_dirs
    assert "checkpoint_000003" not in checkpoint_dirs


def test_trial_dir_injected_into_config(tmp_path: Path) -> None:
    runner = LocalRunner()
    result = runner.run(
        trainer_class=FakeTrainer,
        config={"lr": 0.01},
        epochs=1,
        checkpoint_frequency=1,
        storage_dir=tmp_path,
    )

    assert "trial_dir" in result.config
    assert Path(result.config["trial_dir"]).is_dir()


def test_logger_called(tmp_path: Path) -> None:
    mock_logger = MagicMock()
    runner = LocalRunner(logger=mock_logger)
    epochs = 4

    runner.run(
        trainer_class=FakeTrainer,
        config={"lr": 0.01},
        epochs=epochs,
        checkpoint_frequency=1,
        storage_dir=tmp_path,
    )

    assert mock_logger.log.call_count == epochs
    for i, call in enumerate(mock_logger.log.call_args_list, start=1):
        _, kwargs = call
        assert kwargs["step"] == i