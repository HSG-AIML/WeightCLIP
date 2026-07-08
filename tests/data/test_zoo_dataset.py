import json
import os
from pathlib import Path

import pytest
import torch

from sane.data.datasets.zoo_dataset import ZooDataset
from sane.data.datasets.zoo_dataset_models import get_model



IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.slow
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test runs only locally, not in GitHub Actions.")
def test_dataset_init_and_getitem():
    """ 
    Note: This test requires a model zoo dataset. Therefore, it can only be tested
    locally where access to the model zoo datasets is granted.  
    """

    root_dir = "/ds2/model_zoos/zoos_resnet/zoos/CIFAR100/resnet18/kaiming_uniform/tune_zoo_cifar100_resnet18_kaiming_uniform"
    if not os.path.exists(root_dir):
        return
    
    epoch_idx = [-1,-2, -3]
    ds = ZooDataset(root_dir, epoch_idx)

    # Checks that the dataset len is equals the number of model directories 
    # multiplied by the number of checkpoints per model.
    file_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    model_dir_paths = [p for p in file_paths if os.path.isdir(p)]
    assert(len(model_dir_paths) * len(epoch_idx) == len(ds))

    # Gets the number of checkpoints.
    model_dir_path = model_dir_paths[0]
    ckpt_paths = [os.path.join(model_dir_path, f) for f in os.listdir(model_dir_path)] 
    ckpt_no = len([p for p in ckpt_paths if os.path.isdir(p)])

    for i in [0, 1, 2]:
        m = ds[i]
        # Checks that an item has been properly created.
        assert(m.model)
        assert(m.metadata)
        
        # The checkpoints of the same model are stored consecutively.
        # Checks that the correct epoch's checkpoint is stored.
        # This model zoo has ckpt_no checkpoints per model (index: 0-(ckpt_no-1)).
        assert(m.metadata['eval_results']['training_iteration'] == (ckpt_no-1)-i)


def test_dataset_getitem_reconciles_legacy_output_dim_metadata(tmp_path: Path):
    root_dir = tmp_path / "toy_zoo"
    model_dir = root_dir / "model_000"
    checkpoint_dir = model_dir / "checkpoint_0"
    checkpoint_dir.mkdir(parents=True)

    torch.save(
        get_model(
            {
                "model::type": "CNN",
                "model::channels_in": 1,
                "model::o_dim": 10,
                "model::nlin": "relu",
                "model::dropout": 0.0,
                "model::init_type": "kaiming_uniform",
            }
        ).state_dict(),
        checkpoint_dir / "checkpoints",
    )

    (model_dir / "params.json").write_text(
        json.dumps(
            {
                "model::type": "CNN",
                "model::channels_in": 1,
                "model::o_dim": 4,
                "model::nlin": "relu",
                "model::dropout": 0.0,
                "model::init_type": "kaiming_uniform",
            }
        )
    )
    (model_dir / "result.json").write_text(json.dumps({"training_iteration": 0}) + "\n")

    dataset = ZooDataset(str(root_dir), epoch_idx=[-1])
    checkpoint = dataset[0]

    assert checkpoint.metadata["model_config"]["model::o_dim"] == 10
    assert tuple(checkpoint.model.state_dict()["module_list.11.weight"].shape) == (10, 20)
    assert tuple(checkpoint.model.state_dict()["module_list.11.bias"].shape) == (10,)