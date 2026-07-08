from copy import deepcopy

import pytest

from sane.runner.local_runner import expand_grid


def test_no_grid_returns_original():
    config = {"model": "resnet", "lr": 0.01}
    result = expand_grid(config)
    assert result == [config]


def test_single_grid_top_level():
    config = {"lr": {"grid": [0.01, 0.001]}, "epochs": 10}
    result = expand_grid(config)
    assert len(result) == 2
    assert result[0] == {"lr": 0.01, "epochs": 10}
    assert result[1] == {"lr": 0.001, "epochs": 10}


def test_nested_grid():
    config = {"optimizer": {"type": "adam", "lr": {"grid": [0.01, 0.1]}}}
    result = expand_grid(config)
    assert len(result) == 2
    assert result[0] == {"optimizer": {"type": "adam", "lr": 0.01}}
    assert result[1] == {"optimizer": {"type": "adam", "lr": 0.1}}


def test_multiple_grids_cartesian_product():
    config = {
        "lr": {"grid": [0.01, 0.1]},
        "batch_size": {"grid": [16, 32]},
    }
    result = expand_grid(config)
    assert len(result) == 4
    combos = [(r["lr"], r["batch_size"]) for r in result]
    assert (0.01, 16) in combos
    assert (0.01, 32) in combos
    assert (0.1, 16) in combos
    assert (0.1, 32) in combos


@pytest.mark.parametrize("none_str", ["none", "None"])
def test_none_string_converted_to_none(none_str: str):
    config = {"value": {"grid": [none_str, 1]}}
    result = expand_grid(config)
    assert len(result) == 2
    assert result[0]["value"] is None
    assert result[1]["value"] == 1


def test_original_config_not_mutated():
    config = {"optimizer": {"lr": {"grid": [0.01, 0.1]}}, "seed": 42}
    original = deepcopy(config)
    expand_grid(config)
    assert config == original