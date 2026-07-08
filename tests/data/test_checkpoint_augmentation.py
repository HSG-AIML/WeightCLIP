import torch
from copy import deepcopy
from sane.data.checkpoint import Checkpoint
from sane.data.checkpoint.checkpoint_augmentation import CheckpointAugmentation, CheckpointAugmentationPipeline, PermutationAugmentation, CheckpointAligner
from sane.utils.git_re_basin import mlp_permutation_spec

class MLP(torch.nn.Module):
    """A simple MLP model for testing purposes."""
    def __init__(self):
        super(MLP, self).__init__()
        self.layer0 = torch.nn.Linear(16, 8)
        self.relu = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer0(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x

def generate_mlp_checkpoint():
    """Generates a simple checkpoint for testing."""
    model = MLP()
    checkpoint = Checkpoint(model=model)

    perm_spec = mlp_permutation_spec(num_hidden_layers=1)

    return checkpoint, perm_spec

def test_permutation_augmentation():
    """Test the PermutationAugmentation class."""
    # Generate a simple checkpoint and its permutation specification
    checkpoint, perm_spec = generate_mlp_checkpoint()

    # Initialize the PermutationAugmentation with the generated permutation spec
    augmentation = PermutationAugmentation(permutation_spec=perm_spec, num_permutations=2)

    # Apply the augmentation to the checkpoint
    augmented_checkpoints = augmentation(checkpoint)

    # Check that we have the expected number of augmented checkpoints
    assert len(augmented_checkpoints) == 3  # Original + 2 permutations

    # Check that the first checkpoint is the original one
    for key in checkpoint.model.state_dict():
        assert torch.equal(augmented_checkpoints[0].model.state_dict()[key], checkpoint.model.state_dict()[key]), f"Checkpoint {key} does not match original"

    # Check that all checkpoints have the same behaviour
    x = torch.randn(1, 16)
    y = None
    for i in range(len(augmented_checkpoints)):
        model = MLP()
        model.load_state_dict(augmented_checkpoints[i].model.state_dict())
        output = model(x)

        if y is None:
            y = output
        else:
            assert torch.allclose(output, y), f"Output of checkpoint {i} does not match the first checkpoint output"
        

def test_preprocessing_augmentation_pipeline():
    """Test the CheckpointAugmentationPipeline class."""
    # Generate a simple checkpoint and its permutation specification
    checkpoint, perm_spec = generate_mlp_checkpoint()

    # Initialize the PermutationAugmentation with the generated permutation spec
    identity = CheckpointAugmentation()
    permutation = PermutationAugmentation(permutation_spec=perm_spec, num_permutations=2)

    # Check a pipeline of identity augmentations
    pipeline = CheckpointAugmentationPipeline([identity, identity, identity])
    augmented_checkpoints = pipeline(checkpoint)
    assert len(augmented_checkpoints) == 1  # Only the original checkpoint should be returned
    for key in checkpoint.model.state_dict():
        assert torch.equal(augmented_checkpoints[0].model.state_dict()[key], checkpoint.model.state_dict()[key]), f"Checkpoint {key} does not match original in identity pipeline"

    # Check a pipeline of permutation augmentations
    pipeline = CheckpointAugmentationPipeline([permutation, permutation])
    augmented_checkpoints = pipeline(checkpoint)
    assert len(augmented_checkpoints) == 9  # 1 => 3*1 => 3*3

    # Check that the first checkpoint is the original one
    for key in checkpoint.model.state_dict():
        assert torch.equal(augmented_checkpoints[0].model.state_dict()[key], checkpoint.model.state_dict()[key]), f"Checkpoint {key} does not match original in permutation pipeline"

    # Check that all checkpoints have the same behaviour
    x = torch.randn(1, 16)
    y = None
    for i in range(len(augmented_checkpoints)):
        model = augmented_checkpoints[i].model
        output = model(x)

        if y is None:
            y = output
        else:
            assert torch.allclose(output, y, rtol=1e-4), f"Output of checkpoint {i} does not match the first checkpoint output"

    # Check a pipeline of identity and permutation augmentations
    pipeline = CheckpointAugmentationPipeline([identity, permutation, identity])
    augmented_checkpoints = pipeline(checkpoint)
    assert len(augmented_checkpoints) == 3  # 1 => 3

    # Check that the first checkpoint is the original one
    for key in checkpoint.model.state_dict():
        assert torch.equal(augmented_checkpoints[0].model.state_dict()[key], checkpoint.model.state_dict()[key]), f"Checkpoint {key} does not match original in permutation pipeline"

    # Check that all checkpoints have the same behaviour
    x = torch.randn(1, 16)
    y = None
    for i in range(len(augmented_checkpoints)):
        model = augmented_checkpoints[i].model
        output = model(x)

        if y is None:
            y = output
        else:
            assert torch.allclose(output, y, rtol=1e-4), f"Output of checkpoint {i} does not match the first checkpoint output"


def test_checkpoint_aligner():
    """Test the CheckpointAligner class."""
    # Generate a simple checkpoint and its permutation specification
    checkpoint, perm_spec = generate_mlp_checkpoint()

    # Initialize the PermutationAugmentation with the generated permutation spec
    permutation = PermutationAugmentation(permutation_spec=perm_spec, num_permutations=2)

    # Apply the augmentation to the checkpoint
    augmented_checkpoints = permutation(checkpoint)

    # Check that we have the expected number of augmented checkpoints
    assert len(augmented_checkpoints) == 3  # Original + 2 permutations

    # Initialize the CheckpointAligner
    aligner = CheckpointAligner(permutation_spec=perm_spec, reference_checkpoint=checkpoint)

    # Align the augmented checkpoints
    aligned_checkpoint = aligner(augmented_checkpoints[1])[0]

    # Check that the first checkpoint is the original one
    for key in checkpoint.model.state_dict():
        assert torch.equal(aligned_checkpoint.model.state_dict()[key], checkpoint.model.state_dict()[key]), f"Checkpoint {key} does not match original in alignment"