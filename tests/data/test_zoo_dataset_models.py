import torch

from sane.data.datasets.zoo_dataset_models import get_model


def test_get_model_cnn3_uses_configured_output_dim():
    config = {
        "model::type": "CNN3",
        "model::channels_in": 3,
        "model::o_dim": 8,
        "model::nlin": "gelu",
        "model::dropout": 0.0,
        "model::init_type": "kaiming_uniform",
    }

    model = get_model(config)
    state_dict = model.state_dict()

    assert tuple(state_dict["module_list.16.weight"].shape) == (8, 20)
    assert tuple(state_dict["module_list.16.bias"].shape) == (8,)

    loaded = model.load_state_dict(state_dict)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys


def test_get_model_resnet18slim_uses_configured_output_dim_and_width_mult():
    config = {
        "model::type": "ResNet18Slim",
        "model::channels_in": 3,
        "model::o_dim": 8,
        "model::nlin": "relu",
        "model::dropout": 0.0,
        "model::init_type": "kaiming_uniform",
        "model::width_mult": 0.5,
    }

    model = get_model(config)
    state_dict = model.state_dict()

    assert tuple(state_dict["conv1.weight"].shape) == (32, 3, 3, 3)
    assert tuple(state_dict["fc.weight"].shape) == (8, 256)
    assert tuple(state_dict["fc.bias"].shape) == (8,)

    loaded = model.load_state_dict(state_dict)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys