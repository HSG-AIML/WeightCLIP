import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sane.trainer.dummy_trainer import SANEDummyTrainer


@pytest.fixture()
def dummy_config() -> dict:
    config_dir = str(Path(__file__).resolve().parents[2] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="dummy_tiny")
        return OmegaConf.to_container(cfg, resolve=True)


def _setup_trained_trainer(config: dict) -> SANEDummyTrainer:
    trainer = SANEDummyTrainer()
    trainer.setup(config)
    trainer.step()
    return trainer


def _save_legacy_checkpoint(trainer: SANEDummyTrainer, checkpoint_dir: Path) -> None:
    """Save a checkpoint without embedded config to simulate old format."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": trainer.epoch,
        "model": trainer.sane_ae.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
        "scaler": trainer.scaler.state_dict() if trainer.scaler else None,
    }
    torch.save(state, checkpoint_dir / "checkpoint.pt")


def _assert_weights_match(original: SANEDummyTrainer, loaded: SANEDummyTrainer) -> None:
    for (name_a, p_a), (name_b, p_b) in zip(
        original.sane_ae.state_dict().items(),
        loaded.sane_ae.state_dict().items(),
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), f"Mismatch in parameter {name_a}"


def _assert_trainers_match(original: SANEDummyTrainer, loaded: SANEDummyTrainer) -> None:
    assert original.config.get("seed") == loaded.config.get("seed")
    _assert_weights_match(original, loaded)


# -- Tests ------------------------------------------------------------------ #


@pytest.mark.slow
def test_roundtrip_with_embedded_config(dummy_config: dict, tmp_path: Path) -> None:
    """New checkpoints embed config; from_checkpoint should use it directly."""
    trainer = _setup_trained_trainer(dummy_config)

    ckpt_dir = tmp_path / "checkpoint_000001"
    trainer.save_checkpoint(str(ckpt_dir))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    _assert_trainers_match(trainer, loaded)


@pytest.mark.slow
def test_roundtrip_falls_back_to_trial_yaml(
    dummy_config: dict, tmp_path: Path
) -> None:
    """Legacy checkpoints (no embedded config) should fall back to trial.yaml."""
    trainer = _setup_trained_trainer(dummy_config)

    trial_dir = tmp_path / "trial"
    ckpt_dir = trial_dir / "checkpoint_000001"
    _save_legacy_checkpoint(trainer, ckpt_dir)
    OmegaConf.save(OmegaConf.create(dummy_config), str(trial_dir / "trial.yaml"))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    _assert_trainers_match(trainer, loaded)


@pytest.mark.slow
def test_roundtrip_falls_back_to_params_json(
    dummy_config: dict, tmp_path: Path
) -> None:
    """Legacy checkpoints should fall back to params.json (Ray Tune format)."""
    trainer = _setup_trained_trainer(dummy_config)

    trial_dir = tmp_path / "trial"
    ckpt_dir = trial_dir / "checkpoint_000001"
    _save_legacy_checkpoint(trainer, ckpt_dir)
    (trial_dir / "params.json").write_text(json.dumps(dummy_config))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    _assert_trainers_match(trainer, loaded)


@pytest.mark.slow
def test_roundtrip_unwraps_ray_train_loop_config(
    dummy_config: dict, tmp_path: Path
) -> None:
    """params.json written by Ray wraps config under 'train_loop_config'."""
    trainer = _setup_trained_trainer(dummy_config)

    trial_dir = tmp_path / "trial"
    ckpt_dir = trial_dir / "checkpoint_000001"
    _save_legacy_checkpoint(trainer, ckpt_dir)
    wrapped = {"train_loop_config": dummy_config}
    (trial_dir / "params.json").write_text(json.dumps(wrapped))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    _assert_trainers_match(trainer, loaded)


@pytest.mark.slow
def test_embedded_config_takes_precedence_over_sidecar(
    dummy_config: dict, tmp_path: Path
) -> None:
    """When both embedded config and trial.yaml exist, embedded wins."""
    trainer = _setup_trained_trainer(dummy_config)

    trial_dir = tmp_path / "trial"
    ckpt_dir = trial_dir / "checkpoint_000001"
    trainer.save_checkpoint(str(ckpt_dir))

    # Write a trial.yaml with a different seed — it should be ignored
    bad_config = {**dummy_config, "seed": 999999}
    OmegaConf.save(OmegaConf.create(bad_config), str(trial_dir / "trial.yaml"))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    assert loaded.config.get("seed") == dummy_config["seed"]


@pytest.mark.slow
def test_from_checkpoint_missing_config_raises(
    dummy_config: dict, tmp_path: Path
) -> None:
    """from_checkpoint should raise FileNotFoundError when no config is available."""
    trainer = _setup_trained_trainer(dummy_config)

    trial_dir = tmp_path / "trial"
    ckpt_dir = trial_dir / "checkpoint_000001"
    _save_legacy_checkpoint(trainer, ckpt_dir)
    # No trial.yaml, no params.json, no embedded config

    with pytest.raises(FileNotFoundError, match="No config found"):
        SANEDummyTrainer.from_checkpoint(str(ckpt_dir))


@pytest.mark.slow
def test_from_checkpoint_does_not_restore_epoch(
    dummy_config: dict, tmp_path: Path
) -> None:
    """from_checkpoint is a cold load; epoch resets to 0."""
    trainer = _setup_trained_trainer(dummy_config)
    assert trainer.epoch >= 1

    ckpt_dir = tmp_path / "checkpoint_000001"
    trainer.save_checkpoint(str(ckpt_dir))

    loaded = SANEDummyTrainer.from_checkpoint(str(ckpt_dir))
    assert loaded.epoch == 0


@pytest.mark.slow
def test_load_checkpoint_restores_training_state(
    dummy_config: dict, tmp_path: Path
) -> None:
    """load_checkpoint (used for mid-training resume) restores state correctly."""
    trainer = _setup_trained_trainer(dummy_config)
    epoch_after_step = trainer.epoch

    ckpt_dir = tmp_path / "checkpoint_000001"
    trainer.save_checkpoint(str(ckpt_dir))

    fresh = SANEDummyTrainer()
    fresh.setup(dummy_config)
    assert fresh.epoch == 0

    fresh.load_checkpoint(str(ckpt_dir))
    assert fresh.epoch == epoch_after_step
    _assert_weights_match(trainer, fresh)