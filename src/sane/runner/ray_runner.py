import copy
import json
import logging
from math import isclose
import numbers
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import ray
import ray.train.torch as ray_train_torch
import torch
from ray import train as ray_train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import Checkpoint as TuneCheckpoint
from torch.nn.parallel import DistributedDataParallel

from sane.runner.base import RunResult

logger = logging.getLogger(__name__)
Checkpoint = TuneCheckpoint

_OFAT_KEY = "_ofat"
_OFAT_TRIAL_KEY = "__ofat_overrides__"


class RayRunner:
    """Runner that delegates training to Ray Tune."""

    def __init__(
        self,
        num_cpus: int,
        num_gpus: int,
        cpus_per_trial: int,
        gpus_per_trial: float,
        num_samples: int,
        num_to_keep: int,
        verbose: int,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        wandb_api_key_file: str | None = None,
    ) -> None:
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._cpus_per_trial = cpus_per_trial
        self._gpus_per_trial = gpus_per_trial
        self._num_samples = num_samples
        self._num_to_keep = num_to_keep
        self._verbose = verbose
        self._wandb_project = wandb_project
        self._wandb_group = wandb_group
        self._wandb_api_key_file = wandb_api_key_file

    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> RunResult:
        we_initialized = not ray.is_initialized()
        ray.init(num_cpus=self._num_cpus, num_gpus=self._num_gpus, ignore_reinit_error=True)
        if we_initialized:
            logger.info("Ray initialized: %s", ray.cluster_resources())

        try:
            return self._run_tuner(
                trainer_class,
                config,
                epochs,
                checkpoint_frequency,
                storage_dir,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        finally:
            if we_initialized:
                ray.shutdown()

    def _run_tuner(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> RunResult:
        ofat_overrides = _extract_ofat_overrides(config)
        param_space = _inject_ray_grids(config)
        if ofat_overrides is not None:
            param_space[_OFAT_TRIAL_KEY] = tune.grid_search(ofat_overrides)
        use_ray_train = _should_use_ray_train_ddp(self._gpus_per_trial)

        if resume_from_checkpoint is not None:
            logger.info("Will resume training from checkpoint: %s", resume_from_checkpoint)

        if use_ray_train:
            train_func = _make_distributed_trainable(
                trainer_class=trainer_class,
                epochs=epochs,
                checkpoint_frequency=checkpoint_frequency,
                num_workers=int(self._gpus_per_trial),
                cpus_per_worker=max(1, self._cpus_per_trial // int(self._gpus_per_trial)),
                num_to_keep=self._num_to_keep,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            trainable = tune.with_resources(train_func, resources={"cpu": 1})
        else:
            train_func = _make_trainable(trainer_class, epochs, checkpoint_frequency, resume_from_checkpoint=resume_from_checkpoint)
            trainable = tune.with_resources(train_func, resources={"cpu": self._cpus_per_trial, "gpu": self._gpus_per_trial})

        experiment_name = config["experiment_name"]

        callbacks = []
        if self._wandb_project:
            callbacks.append(WandbLoggerCallback(
                project=self._wandb_project,
                group=self._wandb_group,
                api_key_file=self._wandb_api_key_file,
            ))

        resolved = storage_dir.resolve()

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=self._num_samples,
                reuse_actors=False,
            ),
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=str(resolved.parent),
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=self._num_to_keep,
                ),
                verbose=self._verbose,
                callbacks=callbacks or None,
            ),
        )

        result_grid = tuner.fit()
        best = result_grid.get_best_result(metric="loss", mode="min")

        checkpoint_path = _resolve_best_checkpoint_path(best, storage_dir)

        best_config = best.config or config
        best_metrics = dict(best.metrics) if best.metrics else {}

        return RunResult(checkpoint_path=checkpoint_path, config=best_config, metrics=best_metrics)


def _make_trainable(trainer_class: type, epochs: int, checkpoint_frequency: int, resume_from_checkpoint: Path | None = None):
    """Build a Ray Tune function trainable that wraps a plain trainer."""

    resume_path = str(resume_from_checkpoint) if resume_from_checkpoint is not None else None

    def train_func(config: dict) -> None:

        config = _apply_ofat_overrides(config)

        # Get trial directory and add to config (used in trainer setup)
        trial_dir = Path(tune.get_context().get_storage().trial_fs_path)
        config["trial_dir"] = str(trial_dir)
        _save_trial_artifacts(config, trial_dir)

        start_epoch = 1
        trainer = trainer_class()
        trainer.setup(config)

        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as ckpt_dir:
                trainer.load_checkpoint(ckpt_dir)
                start_epoch = trainer.epoch + 1
        elif resume_path is not None:
            trainer.load_checkpoint(resume_path)
            start_epoch = int(getattr(trainer, "epoch", 0)) + 1
            logger.info("Resumed trainer from %s at epoch %d", resume_path, start_epoch - 1)

        for epoch in range(start_epoch, epochs + 1):
            metrics = trainer.step()

            should_checkpoint = (checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0) or epoch == epochs

            if should_checkpoint:
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer.save_checkpoint(tmpdir)
                    tune.report(metrics=metrics, checkpoint=TuneCheckpoint.from_directory(tmpdir))
            else:
                tune.report(metrics=metrics)

    return train_func


def _make_distributed_trainable(
    trainer_class: type,
    epochs: int,
    checkpoint_frequency: int,
    num_workers: int,
    cpus_per_worker: int,
    num_to_keep: int,
    resume_from_checkpoint: Path | None = None,
):
    """Build a Ray Tune driver that launches a Ray Train TorchTrainer."""

    worker_loop = _make_distributed_worker_loop(trainer_class=trainer_class, epochs=epochs, checkpoint_frequency=checkpoint_frequency)

    resume_path_str = str(resume_from_checkpoint) if resume_from_checkpoint is not None else None

    def train_driver(config: dict) -> None:
        config = _apply_ofat_overrides(config)
        trial_dir = Path(tune.get_context().get_storage().trial_fs_path)
        driver_config = copy.deepcopy(config)
        driver_config["trial_dir"] = str(trial_dir)
        driver_config = _strip_distributed_only_config(driver_config)
        _save_trial_artifacts(driver_config, trial_dir)

        resume_checkpoint = None
        if resume_path_str is not None:
            resume_checkpoint = ray_train.Checkpoint.from_directory(resume_path_str)
            logger.info("Wrapping TorchTrainer with resume_from_checkpoint=%s", resume_path_str)

        trainer = ray_train_torch.TorchTrainer(
            train_loop_per_worker=worker_loop,
            train_loop_config=driver_config,
            scaling_config=ray_train.ScalingConfig(
                num_workers=num_workers,
                use_gpu=True,
                resources_per_worker={"CPU": cpus_per_worker, "GPU": 1},
            ),
            run_config=ray_train.RunConfig(
                name=f"train_{trial_dir.name}",
                storage_path=str(trial_dir),
                checkpoint_config=ray_train.CheckpointConfig(num_to_keep=num_to_keep),
            ),
            resume_from_checkpoint=resume_checkpoint,
        )

        result = trainer.fit()
        metrics = dict(result.metrics) if result.metrics else {}
        if result.checkpoint is not None:
            metrics["checkpoint_path"] = result.checkpoint.path
        tune.report(metrics=metrics)

    return train_driver


def _make_distributed_worker_loop(trainer_class: type, epochs: int, checkpoint_frequency: int):
    """Build the per-worker Ray Train loop for AE-only DDP training."""

    class DistributedTrainer(trainer_class):
        def setup(self, config: dict[str, Any]) -> None:
            worker_config = _configure_distributed_worker_config(config)
            super().setup(worker_config)

            find_unused_parameters = _requires_ddp_unused_parameter_detection(worker_config)
            if find_unused_parameters:
                logger.warning(
                    "Enabling DDP find_unused_parameters because the configured "
                    "loss disables the contrastive projector branch."
                )

            ddp_kwargs = ({"find_unused_parameters": True} if find_unused_parameters else None)
            self.sane_ae = ray_train_torch.prepare_model(self.sane_ae, parallel_strategy_kwargs=ddp_kwargs)
            self.trainloader = ray_train_torch.prepare_data_loader(self.trainloader)
            if self.valloader is not None:
                self.valloader = ray_train_torch.prepare_data_loader(self.valloader)
            if self.testloader is not None:
                self.testloader = ray_train_torch.prepare_data_loader(self.testloader)


        def save_checkpoint(self, checkpoint_dir: str) -> None:
            wrapped_model = self.sane_ae
            self.sane_ae = _unwrap_ddp_model(self.sane_ae)
            try:
                super().save_checkpoint(checkpoint_dir)
            finally:
                self.sane_ae = wrapped_model

        def load_checkpoint(self, checkpoint_dir: str) -> None:
            wrapped_model = self.sane_ae
            self.sane_ae = _unwrap_ddp_model(self.sane_ae)
            try:
                super().load_checkpoint(checkpoint_dir)
            finally:
                self.sane_ae = wrapped_model

    def train_loop(config: dict) -> None:
        trainer = DistributedTrainer()
        trainer.setup(config)

        start_epoch = 1
        checkpoint = ray_train.get_checkpoint()
        if checkpoint is not None:
            with checkpoint.as_directory() as checkpoint_dir:
                trainer.load_checkpoint(checkpoint_dir)
                start_epoch = trainer.epoch + 1

        for epoch in range(start_epoch, epochs + 1):
            _set_sampler_epoch_if_supported(trainer.trainloader, epoch)
            metrics = trainer.step()
            report_metrics = _synchronize_report_metrics(metrics)

            should_checkpoint = (checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0) or epoch == epochs

            if should_checkpoint and _distributed_rank() == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer.save_checkpoint(tmpdir)
                    ray_train.report(metrics=report_metrics, checkpoint=ray_train.Checkpoint.from_directory(tmpdir))
            else:
                ray_train.report(metrics=report_metrics)

    return train_loop


def _save_trial_artifacts(config: dict[str, Any], trial_dir: Path) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), str(trial_dir / "trial.yaml"))
    OmegaConf.save(OmegaConf.create(config), str(trial_dir / "config.yaml"))
    (trial_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))


def _should_use_ray_train_ddp(gpus_per_trial: float) -> bool:
    """Use Ray Train DDP only for integer multi-GPU trials."""
    gpus = float(gpus_per_trial)
    return gpus >= 2 and gpus.is_integer()


def _strip_distributed_only_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop unsupported features from the distributed config."""
    stripped = copy.deepcopy(config)
    if stripped.get("downstream_tasks"):
        logger.warning("Distributed Ray Train runs ignore downstream_tasks; clearing them.")
    stripped["downstream_tasks"] = []
    return stripped


def _configure_distributed_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve device, seed, and downstream-task behavior for one worker."""
    worker_config = _strip_distributed_only_config(config)
    worker_config["device"] = str(ray_train_torch.get_device())

    seed = worker_config.get("seed")
    if seed is not None:
        worker_config["seed"] = int(seed) + _distributed_rank()

    return worker_config


def _requires_ddp_unused_parameter_detection(config: Mapping[str, Any]) -> bool:
    """Detect configs that intentionally leave part of the SANE graph inactive.

    The joint alignment ResNet config uses ``model: sane`` (which builds a
    contrastive projector) together with ``GammaContrastReconLoss`` at
    ``gamma=0``. In that case the projector branch is executed in ``forward``
    but never participates in the loss, so DDP must track unused parameters.
    """

    model_cfg = config.get("model")
    loss_cfg = config.get("loss")
    if not isinstance(model_cfg, Mapping) or not isinstance(loss_cfg, Mapping):
        return False

    projector_cfg = model_cfg.get("projector")
    if projector_cfg in (None, "null"):
        return False

    if loss_cfg.get("class") != "GammaContrastReconLoss":
        return False

    gamma = loss_cfg.get("gamma")
    try:
        return isclose(float(gamma), 0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _unwrap_ddp_model(model: Any) -> Any:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _set_sampler_epoch_if_supported(dataloader: Any, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _synchronize_report_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Average scalar metrics across ranks and broadcast rank-0 metadata."""
    scalar_metrics: dict[str, float] = {}
    rank_zero_payload: dict[str, Any] = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar_metrics[key] = float(value.detach().item())
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            scalar_metrics[key] = float(value)
        elif _distributed_rank() == 0:
            rank_zero_payload[key] = value

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        report_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key, value in scalar_metrics.items():
            tensor = torch.tensor(value, device=report_device, dtype=torch.float64)
            torch.distributed.all_reduce(tensor)
            rank_zero_payload[key] = tensor.item() / world_size

        payload = [rank_zero_payload if _distributed_rank() == 0 else None]
        torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    rank_zero_payload.update(scalar_metrics)
    return rank_zero_payload


def _resolve_best_checkpoint_path(best: Any, storage_dir: Path) -> Path:
    best_checkpoint = getattr(best, "checkpoint", None)
    if best_checkpoint is not None:
        return Path(best_checkpoint.path)

    metrics = dict(best.metrics) if best.metrics else {}
    checkpoint_path = metrics.get("checkpoint_path")
    if checkpoint_path:
        return Path(checkpoint_path)

    return storage_dir


def _extract_ofat_overrides(config: MutableMapping[str, Any]) -> list[dict] | None:
    spec = config.pop(_OFAT_KEY, None)
    return None if spec is None else [dict(e) for e in spec["grid"]]


def _apply_ofat_overrides(config: dict) -> dict:
    overrides = config.pop(_OFAT_TRIAL_KEY, None)
    if not overrides:
        return config
    cfg = OmegaConf.create(config)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return OmegaConf.to_container(cfg, resolve=False)


def _inject_ray_grids(node: Any) -> Any:
    """Convert ``{"grid": [v1, v2, ...]}`` specs to ``tune.grid_search``."""
    if isinstance(node, Mapping) and len(node) == 1 and next(iter(node)) == "grid":
        values = list(node.values())[0]
        values = [None if v in ("none", "None") else v for v in values]
        return tune.grid_search(values)
    if isinstance(node, MutableMapping):
        return {k: _inject_ray_grids(v) for k, v in node.items()}
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        t = type(node)
        try:
            return t(_inject_ray_grids(v) for v in node)
        except TypeError:
            return [_inject_ray_grids(v) for v in node]
    return node
