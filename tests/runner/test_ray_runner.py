from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sane.runner.ray_runner import (
    RayRunner,
    _inject_ray_grids,
    _make_trainable,
    _resolve_best_checkpoint_path,
    _should_use_ray_train_ddp,
    _strip_distributed_only_config,
)


# ---------------------------------------------------------------------------
# _inject_ray_grids
# ---------------------------------------------------------------------------


def test_inject_ray_grids_no_grids():
    cfg = {"a": 1, "b": {"c": 2}}
    result = _inject_ray_grids(cfg)
    assert result == cfg


def test_inject_ray_grids_top_level():
    from ray import tune

    cfg = {"lr": {"grid": [0.1, 0.01]}}
    result = _inject_ray_grids(cfg)
    expected = tune.grid_search([0.1, 0.01])
    assert result["lr"] == expected


def test_inject_ray_grids_nested():
    from ray import tune

    cfg = {"optimizer": {"lr": {"grid": [1e-3, 1e-4]}, "weight_decay": 0.0}}
    result = _inject_ray_grids(cfg)
    assert result["optimizer"]["lr"] == tune.grid_search([1e-3, 1e-4])
    assert result["optimizer"]["weight_decay"] == 0.0


def test_inject_ray_grids_none_strings():
    from ray import tune

    cfg = {"scheduler": {"grid": ["cosine", "none", "None"]}}
    result = _inject_ray_grids(cfg)
    assert result["scheduler"] == tune.grid_search(["cosine", None, None])


def test_inject_ray_grids_in_list():
    from ray import tune

    cfg = {"items": [{"grid": [1, 2]}, "keep"]}
    result = _inject_ray_grids(cfg)
    assert result["items"][0] == tune.grid_search([1, 2])
    assert result["items"][1] == "keep"


# ---------------------------------------------------------------------------
# _StubTrainer for _make_trainable and RayRunner tests
# ---------------------------------------------------------------------------


class _StubTrainer:
    """Minimal trainer for unit tests."""

    def __init__(self) -> None:
        self.epoch = 0
        self._config: dict[str, Any] = {}

    def setup(self, config: dict[str, Any]) -> None:
        self._config = config

    def step(self) -> dict[str, Any]:
        self.epoch += 1
        return {"loss": 1.0 / self.epoch, "epoch": self.epoch}

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / "checkpoint.pt").write_text("stub")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        pass


# ---------------------------------------------------------------------------
# _make_trainable
# ---------------------------------------------------------------------------


def test_make_trainable_returns_callable():
    fn = _make_trainable(_StubTrainer, epochs=3, checkpoint_frequency=1)
    assert callable(fn)


def test_make_trainable_reports_every_epoch(tmp_path: Path):
    fn = _make_trainable(_StubTrainer, epochs=3, checkpoint_frequency=1)

    mock_checkpoint = MagicMock()
    mock_checkpoint.from_directory.return_value = "fake_ckpt"

    mock_tune = MagicMock()
    mock_tune.get_checkpoint.return_value = None
    mock_tune.get_context.return_value.get_storage.return_value.trial_fs_path = str(tmp_path)
    mock_tune.Checkpoint = mock_checkpoint

    with patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner.Checkpoint", mock_checkpoint):
        fn({"some_key": "value"})

    assert mock_tune.report.call_count == 3
    for c in mock_tune.report.call_args_list:
        assert "checkpoint" in c.kwargs
        assert "metrics" in c.kwargs
    assert (tmp_path / "trial.yaml").exists()


def test_make_trainable_checkpoints_only_at_frequency(tmp_path: Path):
    fn = _make_trainable(_StubTrainer, epochs=3, checkpoint_frequency=3)

    mock_checkpoint = MagicMock()
    mock_checkpoint.from_directory.return_value = "fake_ckpt"

    mock_tune = MagicMock()
    mock_tune.get_checkpoint.return_value = None
    mock_tune.get_context.return_value.get_storage.return_value.trial_fs_path = str(tmp_path)
    mock_tune.Checkpoint = mock_checkpoint

    with patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner.Checkpoint", mock_checkpoint):
        fn({})

    assert mock_tune.report.call_count == 3
    calls = mock_tune.report.call_args_list
    # Epochs 1 and 2: no checkpoint; epoch 3: checkpoint (last + freq match)
    assert "checkpoint" not in calls[0].kwargs
    assert "checkpoint" not in calls[1].kwargs
    assert "checkpoint" in calls[2].kwargs


def test_make_trainable_resumes_from_checkpoint(tmp_path: Path):
    """Verify that checkpoint resume reads trainer.epoch, not the file."""
    resumed_epoch = 5

    class _ResumableTrainer(_StubTrainer):
        def load_checkpoint(self, checkpoint_dir: str) -> None:
            self.epoch = resumed_epoch

    fn = _make_trainable(_ResumableTrainer, epochs=7, checkpoint_frequency=7)

    fake_checkpoint = MagicMock()
    fake_checkpoint.as_directory.return_value.__enter__ = MagicMock(
        return_value="/fake/ckpt"
    )
    fake_checkpoint.as_directory.return_value.__exit__ = MagicMock(
        return_value=False
    )

    mock_checkpoint_cls = MagicMock()
    mock_checkpoint_cls.from_directory.return_value = "fake_ckpt"

    mock_tune = MagicMock()
    mock_tune.get_checkpoint.return_value = fake_checkpoint
    mock_tune.get_context.return_value.get_storage.return_value.trial_fs_path = str(tmp_path)

    with patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner.Checkpoint", mock_checkpoint_cls):
        fn({})

    # Should run epochs 6 and 7 (resumed_epoch + 1 through epochs)
    assert mock_tune.report.call_count == 2


def test_should_use_ray_train_ddp_only_for_integer_multi_gpu() -> None:
    assert _should_use_ray_train_ddp(2) is True
    assert _should_use_ray_train_ddp(4.0) is True
    assert _should_use_ray_train_ddp(0) is False
    assert _should_use_ray_train_ddp(1) is False
    assert _should_use_ray_train_ddp(1.5) is False


def test_strip_distributed_only_config_clears_downstream_tasks() -> None:
    original = {"downstream_tasks": [{"class": "SomeTask"}], "seed": 7}

    stripped = _strip_distributed_only_config(original)

    assert stripped["downstream_tasks"] == []
    assert stripped["seed"] == 7
    assert original["downstream_tasks"] == [{"class": "SomeTask"}]


def test_resolve_best_checkpoint_path_uses_metric_path_when_needed(tmp_path: Path) -> None:
    best = MagicMock()
    best.checkpoint = None
    best.metrics = {
        "loss": 0.1,
        "checkpoint_path": str(tmp_path / "inner_train_ckpt"),
    }

    resolved = _resolve_best_checkpoint_path(best, tmp_path / "fallback")

    assert resolved == tmp_path / "inner_train_ckpt"


# ---------------------------------------------------------------------------
# RayRunner
# ---------------------------------------------------------------------------


def test_ray_runner_satisfies_protocol():
    from sane.runner.base import Runner

    runner = RayRunner(
        num_cpus=1, num_gpus=0, cpus_per_trial=1,
        gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
    )
    assert isinstance(runner, Runner)


def test_ray_runner_init_defaults():
    runner = RayRunner(
        num_cpus=1, num_gpus=0, cpus_per_trial=1,
        gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
    )
    assert runner._cpus_per_trial == 1
    assert runner._gpus_per_trial == 0
    assert runner._num_samples == 1
    assert runner._num_to_keep == 1
    assert runner._verbose == 3
    assert runner._wandb_project is None
    assert runner._wandb_group is None
    assert runner._wandb_api_key_file is None


def test_ray_runner_init_custom():
    runner = RayRunner(
        num_cpus=8,
        num_gpus=2,
        cpus_per_trial=4,
        gpus_per_trial=1,
        num_samples=5,
        num_to_keep=3,
        verbose=1,
        wandb_project="my_project",
        wandb_group="my_group",
        wandb_api_key_file="/tmp/wandb.key",
    )
    assert runner._num_cpus == 8
    assert runner._num_gpus == 2
    assert runner._cpus_per_trial == 4
    assert runner._gpus_per_trial == 1
    assert runner._num_samples == 5
    assert runner._num_to_keep == 3
    assert runner._verbose == 1
    assert runner._wandb_project == "my_project"
    assert runner._wandb_group == "my_group"
    assert runner._wandb_api_key_file == "/tmp/wandb.key"


def test_ray_runner_run_delegates_to_tuner(tmp_path):
    from sane.runner.base import RunResult

    fake_best = MagicMock()
    fake_best.checkpoint = MagicMock()
    fake_best.checkpoint.path = str(tmp_path / "best_ckpt")
    fake_best.config = {"lr": 0.01}
    fake_best.metrics = {"loss": 0.5}

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    mock_tune = MagicMock()
    mock_tuner_instance = MagicMock()
    mock_tuner_instance.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner_instance
    mock_tune.with_resources.return_value = "wrapped_trainable"

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune):
        runner = RayRunner(
            num_cpus=1, num_gpus=0, cpus_per_trial=1,
            gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
        )
        config = {
            "experiment_name": "test_exp",
            "trainer": "SANETrainer",
            "epochs": 2,
            "checkpoint_frequency": 1,
        }

        result = runner.run(
            trainer_class=_StubTrainer,
            config=config,
            epochs=2,
            checkpoint_frequency=1,
            storage_dir=tmp_path / "storage",
        )

        mock_tune.Tuner.assert_called_once()
        mock_tuner_instance.fit.assert_called_once()
        assert isinstance(result, RunResult)
        assert result.config == {"lr": 0.01}
        assert result.metrics == {"loss": 0.5}


def test_ray_runner_run_inits_ray_if_needed(tmp_path):
    fake_best = MagicMock()
    fake_best.checkpoint = None
    fake_best.config = {}
    fake_best.metrics = {}

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = False
    mock_ray.cluster_resources.return_value = {"CPU": 2}

    mock_tune = MagicMock()
    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner
    mock_tune.with_resources.return_value = "wrapped"

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune):
        runner = RayRunner(
            num_cpus=2, num_gpus=1, cpus_per_trial=1,
            gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
        )
        config = {"experiment_name": "exp", "epochs": 1, "checkpoint_frequency": 1}

        runner.run(
            trainer_class=_StubTrainer,
            config=config,
            epochs=1,
            checkpoint_frequency=1,
            storage_dir=tmp_path,
        )

        mock_ray.init.assert_called_once_with(
            num_cpus=2, num_gpus=1, ignore_reinit_error=True
        )


def test_ray_runner_adds_wandb_callback_when_configured(tmp_path):
    fake_best = MagicMock()
    fake_best.checkpoint = MagicMock()
    fake_best.checkpoint.path = str(tmp_path / "ckpt")
    fake_best.config = {}
    fake_best.metrics = {}

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    mock_tune = MagicMock()
    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner
    mock_tune.with_resources.return_value = "wrapped"

    mock_wandb_cls = MagicMock()

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner.WandbLoggerCallback", mock_wandb_cls):
        runner = RayRunner(
            num_cpus=1, num_gpus=0, cpus_per_trial=1,
            gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
            wandb_project="test_proj",
            wandb_group="test_grp",
            wandb_api_key_file="/tmp/key",
        )
        runner.run(
            trainer_class=_StubTrainer,
            config={"experiment_name": "exp", "epochs": 1, "checkpoint_frequency": 1},
            epochs=1,
            checkpoint_frequency=1,
            storage_dir=tmp_path,
        )

    mock_wandb_cls.assert_called_once_with(
        project="test_proj",
        group="test_grp",
        api_key_file="/tmp/key",
    )
    run_config_kwargs = mock_tune.RunConfig.call_args
    assert run_config_kwargs.kwargs["callbacks"] == [mock_wandb_cls.return_value]


def test_ray_runner_no_wandb_callback_when_not_configured(tmp_path):
    fake_best = MagicMock()
    fake_best.checkpoint = MagicMock()
    fake_best.checkpoint.path = str(tmp_path / "ckpt")
    fake_best.config = {}
    fake_best.metrics = {}

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    mock_tune = MagicMock()
    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner
    mock_tune.with_resources.return_value = "wrapped"

    mock_wandb_cls = MagicMock()

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner.WandbLoggerCallback", mock_wandb_cls):
        runner = RayRunner(
            num_cpus=1, num_gpus=0, cpus_per_trial=1,
            gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
        )
        runner.run(
            trainer_class=_StubTrainer,
            config={"experiment_name": "exp", "epochs": 1, "checkpoint_frequency": 1},
            epochs=1,
            checkpoint_frequency=1,
            storage_dir=tmp_path,
        )

    mock_wandb_cls.assert_not_called()
    run_config_kwargs = mock_tune.RunConfig.call_args
    assert run_config_kwargs.kwargs["callbacks"] is None


def test_ray_runner_fallback_checkpoint_path_when_none(tmp_path):
    fake_best = MagicMock()
    fake_best.checkpoint = None
    fake_best.config = None
    fake_best.metrics = None

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    mock_tune = MagicMock()
    mock_tuner = MagicMock()
    mock_tuner.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner
    mock_tune.with_resources.return_value = "wrapped"

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune):
        runner = RayRunner(
            num_cpus=1, num_gpus=0, cpus_per_trial=1,
            gpus_per_trial=0, num_samples=1, num_to_keep=1, verbose=3,
        )
        config = {"experiment_name": "fallback_exp", "epochs": 1, "checkpoint_frequency": 1}
        storage = tmp_path / "store"

        result = runner.run(
            trainer_class=_StubTrainer,
            config=config,
            epochs=1,
            checkpoint_frequency=1,
            storage_dir=storage,
        )

        assert result.checkpoint_path == storage
        assert result.config == config
        assert result.metrics == {}


def test_ray_runner_uses_distributed_driver_for_integer_multi_gpu(tmp_path: Path) -> None:
    from sane.runner.base import RunResult

    fake_best = MagicMock()
    fake_best.checkpoint = None
    fake_best.config = {"lr": 0.01}
    fake_best.metrics = {
        "loss": 0.25,
        "checkpoint_path": str(tmp_path / "distributed_ckpt"),
    }

    fake_result_grid = MagicMock()
    fake_result_grid.get_best_result.return_value = fake_best

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True

    mock_tune = MagicMock()
    mock_tuner_instance = MagicMock()
    mock_tuner_instance.fit.return_value = fake_result_grid
    mock_tune.Tuner.return_value = mock_tuner_instance
    mock_tune.with_resources.return_value = "wrapped_distributed_trainable"

    with patch("sane.runner.ray_runner.ray", mock_ray), \
         patch("sane.runner.ray_runner.tune", mock_tune), \
         patch("sane.runner.ray_runner._make_distributed_trainable", return_value="distributed_trainable") as mock_make_distributed, \
         patch("sane.runner.ray_runner._make_trainable") as mock_make_single:
        runner = RayRunner(
            num_cpus=8,
            num_gpus=4,
            cpus_per_trial=4,
            gpus_per_trial=2,
            num_samples=1,
            num_to_keep=2,
            verbose=3,
        )
        result = runner.run(
            trainer_class=_StubTrainer,
            config={"experiment_name": "exp", "epochs": 1, "checkpoint_frequency": 1},
            epochs=1,
            checkpoint_frequency=1,
            storage_dir=tmp_path / "storage",
        )

    mock_make_distributed.assert_called_once()
    mock_make_single.assert_not_called()
    mock_tune.with_resources.assert_called_once_with(
        "distributed_trainable",
        resources={"cpu": 1},
    )
    assert isinstance(result, RunResult)
    assert result.checkpoint_path == tmp_path / "distributed_ckpt"
    assert result.metrics["loss"] == 0.25