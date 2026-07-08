import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sane.trainer.trainer import _load_trial_config


SAMPLE_CONFIG = {"seed": 42, "model": {"n_tokens": 128}, "optimizer": {"lr": 1e-3}}


def test_loads_trial_yaml(tmp_path: Path) -> None:
    OmegaConf.save(OmegaConf.create(SAMPLE_CONFIG), str(tmp_path / "trial.yaml"))
    assert _load_trial_config(tmp_path) == SAMPLE_CONFIG


def test_loads_params_json(tmp_path: Path) -> None:
    (tmp_path / "params.json").write_text(json.dumps(SAMPLE_CONFIG))
    assert _load_trial_config(tmp_path) == SAMPLE_CONFIG


def test_unwraps_ray_train_loop_config(tmp_path: Path) -> None:
    wrapped = {"train_loop_config": SAMPLE_CONFIG}
    (tmp_path / "params.json").write_text(json.dumps(wrapped))
    assert _load_trial_config(tmp_path) == SAMPLE_CONFIG


def test_trial_yaml_takes_precedence_over_params_json(tmp_path: Path) -> None:
    yaml_config = {**SAMPLE_CONFIG, "seed": 99}
    json_config = {**SAMPLE_CONFIG, "seed": 1}
    OmegaConf.save(OmegaConf.create(yaml_config), str(tmp_path / "trial.yaml"))
    (tmp_path / "params.json").write_text(json.dumps(json_config))
    loaded = _load_trial_config(tmp_path)
    assert loaded["seed"] == 99


def test_raises_when_no_config_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No config found"):
        _load_trial_config(tmp_path)