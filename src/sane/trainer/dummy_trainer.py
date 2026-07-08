import torch.nn as nn
import numpy as np
from typing import Dict, Any

from .trainer import SANETrainer
from sane.data.datasets import CheckpointsDataset
from sane.data.checkpoint import Checkpoint
from sane.data.dataset_factory import DatasetFactory
from sane.data.config_schema import (
    CachedWindowedDatasetConfig,
    DataLoaderConfig,
    SplitConfig,
    TokenizerConfig,
    from_dict,
)


class SANEDummyTrainer(SANETrainer):
    """
    A dummy trainer that generates synthetic data instead of loading real checkpoints.
    This is useful for testing, debugging, and development purposes.
    
    The dummy trainer creates fake neural networks with random weights and generates
    synthetic token sequences that mimic real checkpoint data.
    """

    def _load_data(self):
        if 'data' not in self.config:
            raise ValueError("Configuration must contain a 'data' key with the data configuration.")
        data_config = self.config['data']

        checkpoints_dataset = self._create_fake_checkpoints_dataset(data_config)

        factory = DatasetFactory()
        split_config = from_dict(SplitConfig, data_config.get('split', {}))
        splits = factory._split_checkpoints(split_config, checkpoints_dataset)

        tokenizer_config = from_dict(TokenizerConfig, data_config.get('tokenizer', {}))
        tokens_dataset_config = from_dict(CachedWindowedDatasetConfig, data_config.get('tokens_dataset', {}))

        tokens_datasets = [
            factory._build_tokens_dataset(
                tokens_dataset_config,
                tokenizer_config,
                subset,
                split_name,
                "dummy",
            )
            for split_name, subset in splits
            if subset is not None
        ]

        dataloader_config = from_dict(DataLoaderConfig, data_config)
        self.trainloader, self.valloader, self.testloader = factory._make_dataloaders(
            dataloader_config, tokens_datasets
        )
        self.trainset = self.trainloader.dataset
        self.valset = self.valloader.dataset if self.valloader is not None else None
        self.testset = self.testloader.dataset if self.testloader is not None else None

    def _create_fake_checkpoints_dataset(self, data_config: Dict[str, Any]) -> CheckpointsDataset:
        """
        Creates a dataset of fake checkpoints with random neural network weights.
        
        Args:
            data_config: Data configuration dictionary
            
        Returns:
            CheckpointsDataset containing fake checkpoints
        """
        # Get configuration for fake data generation
        num_checkpoints = data_config.get('num_fake_checkpoints', 100)
        input_dim = data_config.get('fake_model_input_dim', 32)
        hidden_dim = data_config.get('fake_model_hidden_dim', 64)
        output_dim = data_config.get('fake_model_output_dim', 10)
        num_layers = data_config.get('fake_model_num_layers', 3)
        
        checkpoints = []
        
        for i in range(num_checkpoints):
            # Create a simple MLP as our fake model
            fake_model = self._create_fake_model(input_dim, hidden_dim, output_dim, num_layers)
            
            # Create metadata for the checkpoint
            metadata = {
                'checkpoint_id': i,
                'epoch': np.random.randint(1, 100),
                'accuracy': np.random.uniform(0.1, 0.9),
                'loss': np.random.uniform(0.1, 2.0)
            }
            
            checkpoint = Checkpoint(model=fake_model, metadata=metadata)
            checkpoints.append(checkpoint)
        
        return CheckpointsDataset(checkpoints=checkpoints)

    def _create_fake_model(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
        """
        Creates a fake neural network model with random weights.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            
        Returns:
            A PyTorch model with random weights
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        model = nn.Sequential(*layers)
        
        # Initialize with random weights (they're already random, but we can be explicit)
        for param in model.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        
        return model