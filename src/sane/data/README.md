# SANE Data Module

This module provides the data handling infrastructure for the SANE (Sequential Autoencoder for Neural Embeddings) project. It includes datasets for loading neural network checkpoints, tokenization functionality, and windowing strategies for processing model weights.

## Overview

The data module is organized into several key components:

1. **SANETokenizer**: Handle different types of model checkpoint sources
2. **Dataset Classes**: Converts model weights into tokenized sequences  
3. **Windowing System**: Extracts subsequences from tokenized models
4. **Preprocessing**: Data augmentation and transformation utilities

## SANETokenizer

**Location**: `tokenizer.py`

The core tokenization component that converts neural network state dictionaries into sequences of tokens suitable for processing by transformers.

### Tokenization Modes

#### 1. Layer-wise Mode (Default)
```python
tokenizer = SANETokenizer(tokensize=256, mode="layer_wise")
result = tokenizer(checkpoint)
# Returns: List[Tuple[tokens, mask, position]] - one tuple per layer
```

**Structure**: Each layer becomes a separate sequence of tokens with its own mask and positional information.

**Benefits**:
- Preserves layer boundaries
- Enables per-layer windowing strategies
- Natural for architectures where layer information is important

#### 2. Full Model Mode  
```python
tokenizer = SANETokenizer(tokensize=256, mode="full_model")
result = tokenizer(checkpoint)
# Returns: Tuple[tokens, mask, position] - single concatenated sequence
```

**Structure**: All layers are concatenated into one long sequence.

**Benefits**:
- Simpler data format
- Global view of model weights
- Easier to apply global transformations

**Cross-Layer Windowing**:
- Compatible with windowing functionality (windows can span multiple layers)
- Can be used with `TokensDataset` when `enable_windowing=True`
- Can be used with `SingleWindowTokensDataset`

### Key Parameters

- **`tokensize`**: Size of each token (default: 256). Determines how model weights are chunked.
- **`device`**: Device for tensor operations (default: 'cpu')  
- **`mode`**: "layer_wise" or "full_model" (default: "layer_wise")
- **`reference_checkpoint`**: Template for reconstructing models during detokenization

### Process Flow

1. **Flatten**: Convert state dict tensors to flat vectors per layer
2. **Slice**: Chunk flattened weights into tokens of specified size
3. **Mask**: Create attention masks for valid tokens (handles padding)
4. **Position**: Assign positional encodings to tokens


## Dataset Classes

### 1. CheckpointsDataset (Base Class)
**Location**: `datasets/checkpoints_dataset.py`

The base class for all checkpoint datasets. Handles a collection of `Checkpoint` objects that contain model state dictionaries and metadata.

```python
from sane.data.datasets import CheckpointsDataset, Checkpoint

# Basic usage
checkpoints = [Checkpoint(state_dict=model.state_dict(), metadata={})]
dataset = CheckpointsDataset(checkpoints)
```

### 2. ZooDataset
**Location**: `datasets/zoo_dataset.py`

Loads checkpoints from model zoos compiled by the AIML lab. Supports loading specific epochs and filtering models.

```python
from sane.data.datasets import ZooDataset

# Load from a model zoo directory
dataset = ZooDataset(zoo_path="/path/to/zoo", epoch_indices=[0, -1])
```

**Use Case**: Working with systematically trained model collections where you need specific training epochs.

### 3. HFDataset (Hugging Face Dataset)
**Location**: `datasets/huggingface_dataset.py`

Downloads and loads models from Hugging Face Hub based on queries.

```python
from sane.data.datasets import HFDataset, HFQuery

# Load models matching specific criteria
query = HFQuery(task="text-classification", library="transformers")
dataset = HFDataset(query, cache_dir="~/.cache/huggingface")
```

**Use Case**: Accessing pretrained models from Hugging Face for analysis or comparison.


### 4. TokensDataset
**Location**: `datasets/tokens_dataset.py`

Applies tokenization to checkpoints from an underlying `CheckpointsDataset`. Returns tokenized model weights with optional windowing.

```python
from sane.data.datasets import TokensDataset
from sane.data.tokenizer import SANETokenizer

# For windowing (requires layer_wise mode)
tokenizer = SANETokenizer(tokensize=256, mode="layer_wise")
dataset = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    num_windows_per_model=4,
    enable_windowing=True,
    pad_windows=False  # Default: variable window sizes
)

# For windowing with consistent sizes (padding enabled)
dataset_padded = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    num_windows_per_model=4,
    enable_windowing=True,
    pad_windows=True  # All windows will be exactly 512 tokens
)

# For full model without windowing
tokenizer_full = SANETokenizer(tokensize=256, mode="full_model") 
dataset_full = TokensDataset(
    checkpoints_dataset,
    tokenizer_full,
    enable_windowing=False
)

# For full model with cross-layer windowing
dataset_full_windowed = TokensDataset(
    checkpoints_dataset,
    tokenizer_full,
    window_size=512,
    num_windows_per_model=2,
    enable_windowing=True  # Now supported! Windows can span layers
)
```

**Returns**: Always returns `List[Tuple[tokens, mask, position]]`
- **Layer_wise mode without windowing**: One tuple per layer (tokenizer returns list directly)
- **Layer_wise mode with windowing**: Multiple windowed tuples (respecting layer boundaries)
- **Full_model mode without windowing**: Single tuple wrapped in list (dataset converts tokenizer's tuple to list)
- **Full_model mode with windowing**: Multiple windowed tuples (can span across layers)

**Use Case**: Training SANE autoencoders with full control over window sampling strategy.

### 5. SingleWindowTokensDataset
**Location**: `datasets/single_window_tokens_dataset.py`

Similar to `TokensDataset` but expands the dataset so each window becomes a separate item. Always returns exactly one window per `__getitem__` call.

```python
# Works with both tokenizer modes
tokenizer = SANETokenizer(tokensize=256, mode="layer_wise")
dataset = SingleWindowTokensDataset(
    checkpoints_dataset,
    tokenizer, 
    window_size=512,
    num_windows_per_model=4,
    pad_windows=False  # Default: variable window sizes
)

# With full_model mode for cross-layer windowing
tokenizer_full = SANETokenizer(tokensize=256, mode="full_model")
dataset_cross_layer = SingleWindowTokensDataset(
    checkpoints_dataset,
    tokenizer_full,
    window_size=512,
    num_windows_per_model=4,
    pad_windows=True  # All windows will be exactly 512 tokens
)

# Dataset size = len(checkpoints_dataset) * num_windows_per_model
# Each index returns a single (tokens, mask, position) tuple
```

**Returns**: Single `Tuple[tokens, mask, position]` per `__getitem__` call

**Features**: 
- Works with both `layer_wise` and `full_model` tokenizer modes
- Always requires windowing (cannot be disabled)
- `full_model` mode enables cross-layer windowing

**Use Case**: When you need individual windows as separate training samples (e.g., for contrastive learning).

## Base Window Dataset

**Location**: `datasets/base_windowed_dataset.py`

The windowing system extracts subsequences (windows) from tokenized models for training. It supports two main strategies:

### Regular Windowing
Samples a fixed number of windows per model, distributed across layers.

```python
dataset = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    num_windows_per_model=4,  # or "auto" for full coverage
    enable_windowing=True
)
```

### Per-Layer Windowing  
Samples a fixed number of windows from each layer independently.

```python
dataset = TokensDataset(
    checkpoints_dataset, 
    tokenizer,
    window_size=512,
    num_windows_per_layer=2,  # 2 windows per layer
    enable_windowing=True
)
```

**Precedence**: When both `num_windows_per_layer` and `num_windows_per_model` are specified, per-layer windowing takes precedence (with a warning).

**Compatibility**: Per-layer windowing requires `layer_wise` tokenizer mode. It cannot be used with `full_model` mode since layers are concatenated into a single sequence.

### Auto Mode with Complete Coverage

When using `num_windows_per_model="auto"`, the system automatically calculates how many complete windows fit in each layer. By default, any remaining tokens that don't fill a complete window are ignored. However, when `pad_windows=True`, these remaining tokens are included in an additional padded window, ensuring complete coverage of all model weights:

```python
# Auto mode without padding - may miss remaining tokens
dataset_partial = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=100,
    num_windows_per_model="auto",
    enable_windowing=True,
    pad_windows=False  # Default: only complete windows included
)

# Auto mode with padding - complete coverage including remaining tokens  
dataset_complete = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=100,
    num_windows_per_model="auto",
    enable_windowing=True,
    pad_windows=True  # Remaining tokens included in padded final window
)
```

**Auto Mode Coverage Behavior**:
- **Without padding**: `layer_length // window_size` windows per layer (ignores remainder)
- **With padding**: `⌈layer_length / window_size⌉` windows per layer (includes all tokens)
- **Benefits**: Ensures no model weights are ignored during training
- **Use case**: Critical for comprehensive model analysis and training

### Cross-Layer Windowing (Full-Model Mode)

When using `full_model` tokenizer mode with windowing enabled, windows can span multiple layers:

```python
# Traditional layer-wise windowing (respects layer boundaries)
tokenizer_layer = SANETokenizer(mode="layer_wise")
dataset_layer = TokensDataset(
    checkpoints_dataset,
    tokenizer_layer,
    window_size=256,
    enable_windowing=True
)
# Windows contain tokens from single layers only

# Cross-layer windowing (can span multiple layers)  
tokenizer_full = SANETokenizer(mode="full_model")
dataset_cross = TokensDataset(
    checkpoints_dataset,
    tokenizer_full,
    window_size=256,
    enable_windowing=True
)
# Windows can contain tokens from multiple layers
```

**Cross-Layer Behavior**:
- **Token mixing**: A single window can contain tokens from different layers
- **Position preservation**: Position information maintains original layer indices
- **Global view**: Enables analysis of inter-layer relationships
- **Use cases**: Cross-layer pattern detection, global model analysis

### Window Padding

By default, windows smaller than `window_size` are returned as-is with variable lengths. You can enable padding to ensure all windows have consistent sizes:

```python
# Without padding (default) - variable window sizes
dataset = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    enable_windowing=True,
    pad_windows=False  # Windows may be < 512 tokens
)

# With padding - consistent window sizes
dataset_padded = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    enable_windowing=True,
    pad_windows=True  # All windows will be exactly 512 tokens
)
```

**Padding Behavior**:
- **Padding tokens**: Filled with zeros (same as tokenizer padding)
- **Padding masks**: Set to `False` (invalid tokens)
- **Padding positions**: Continue indexing from last valid position
- **Benefits**: Consistent batch sizes, easier tensor operations
- **Trade-offs**: Increased memory usage, artificial tokens in data

### Window Overlapping Control

By default, windows do not overlap when multiple windows are sampled from the same sequence. You can control this behavior with the `allow_overlapping_windows` parameter:

```python
# Overlapping windows
dataset_overlap = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=10,
    num_windows_per_model=3,
    enable_windowing=True,
    allow_overlapping_windows=True
)
# Windows might be: [0:10], [5:15], [12:22] (overlapping)

# Non-overlapping windows (default)
dataset_no_overlap = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=10,
    num_windows_per_model=3,
    enable_windowing=True,
    allow_overlapping_windows=False
)
# Windows will be: [0:10], [10:20], [20:30] (evenly spaced)
```

### Window Distribution Strategies

For non-overlapping windows, you can control how windows are distributed across the sequence using the `window_distribution_strategy` parameter:

```python
# Consecutive placement (default) - start from beginning
dataset_consecutive = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=10,
    num_windows_per_model=3,
    enable_windowing=True,
    allow_overlapping_windows=False,
    window_distribution_strategy="consecutive"
)
# Windows: [0:10], [10:20], [20:30] (clustered at start)

# Distributed placement - spread across sequence for better coverage
dataset_distributed = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=10,
    num_windows_per_model=3,
    enable_windowing=True,
    allow_overlapping_windows=False,
    window_distribution_strategy="distributed"
)
# Windows: [0:10], [20:30], [40:50] (spread across full sequence)
```

**Distribution Strategy Benefits**:
- **Consecutive**: Predictable, backward compatible, faster to compute
- **Distributed**: Better sequence coverage, reduced sampling bias, more representative training data
- **Coverage improvement**: Distributed can provide 2x+ better span coverage in longer sequences

**Non-Overlapping Behavior**:
- **Deterministic**: Windows are evenly spaced by `window_size`
- **Layer-wise**: Non-overlapping applies within each layer independently
- **Coverage**: May return fewer windows than requested if sequence is too short
- **Warning**: Issues warning when requested windows don't fit non-overlapping
- **Benefits**: Avoids data leakage, cleaner evaluation, no duplicate tokens
- **Use cases**: Model evaluation, ablation studies, non-redundant sampling

### Window Parameters

- **`window_size`**: Number of tokens per window
- **`num_windows_per_model`**: Windows to sample per model (int or "auto")
- **`num_windows_per_layer`**: Windows to sample per layer  
- **`enable_windowing`**: Whether to enable windowing functionality
- **`pad_windows`**: Whether to pad windows to consistent sizes (default: False)
- **`allow_overlapping_windows`**: Whether to allow overlapping windows (default: False)
- **`window_distribution_strategy`**: Strategy for non-overlapping window placement (default: "consecutive")
  - `"consecutive"`: Place windows sequentially from start
  - `"distributed"`: Spread windows across sequence for better coverage

## How Components Work Together

### Tokenizer Mode Compatibility

Before choosing components, understand the compatibility constraints:

| Dataset | Tokenizer Mode | Windowing | Padding | Supported | Notes |
|---------|----------------|-----------|---------|-----------|-------|
| `TokensDataset` | `layer_wise` | Enabled | Any | ✅ | Respects layer boundaries |
| `TokensDataset` | `layer_wise` | Disabled | N/A | ✅ | Returns list of layer tuples |
| `TokensDataset` | `full_model` | Enabled | Any | ✅ | Cross-layer windowing |
| `TokensDataset` | `full_model` | Disabled | N/A | ✅ | Returns single tuple in list |
| `SingleWindowTokensDataset` | `layer_wise` | Always Enabled | Any | ✅ | Respects layer boundaries |
| `SingleWindowTokensDataset` | `full_model` | Always Enabled | Any | ✅ | Cross-layer windowing |

**Note**: `pad_windows=True` requires `enable_windowing=True`. Padding is not applicable when windowing is disabled.

**Important**: Per-layer windowing (`num_windows_per_layer`) cannot be used with `full_model` tokenizer mode, since the layers are already concatenated into a single sequence. Use regular windowing (`num_windows_per_model`) instead for cross-layer windowing.

### Typical Usage Pipeline

1. **Load Checkpoints**: Use appropriate dataset class (ZooDataset, HFDataset, etc.)
2. **Configure Tokenizer**: Choose mode and parameters based on your use case and compatibility requirements
3. **Apply Windowing**: Select windowing strategy for your training setup (if supported)
4. **Feed to Model**: Use with SANE autoencoder or other transformer models

### Example: Full Pipeline

```python
from sane.data.datasets import ZooDataset, TokensDataset
from sane.data.tokenizer import SANETokenizer

# 1. Load checkpoints from model zoo
checkpoints_dataset = ZooDataset("/path/to/zoo", epoch_indices=[0, 5, 10])

# 2. Configure tokenizer for layer-wise processing  
tokenizer = SANETokenizer(
    tokensize=256,
    mode="layer_wise",
)

# 3. Create windowed token dataset
dataset = TokensDataset(
    checkpoints_dataset,
    tokenizer,
    window_size=512,
    num_windows_per_model="auto",  # Full coverage
    enable_windowing=True
)

# 4. Use with DataLoader for training
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    # batch contains windowed tokenized model weights
    windows = batch  # List of windows per model in batch
    # Process with SANE autoencoder...
```

### Integration with SANE Model

The tokenized data flows into the SANE autoencoder components:

- **Input**: Tokenized sequences (tokens, mask, position)
- **Encoder**: Processes tokens → latent representations  
- **Decoder**: Reconstructs tokens from latent space
- **Output**: Reconstructed model weights

See [model README](../model/README.md) for details on the autoencoder architecture.

## Code Organization

```
sane/data/
├── datasets/
│   ├── __init__.py              # Dataset exports
│   ├── base_windowed_dataset.py # Windowing logic base class  
│   ├── checkpoints_dataset.py   # Base checkpoint dataset
│   ├── huggingface_dataset.py   # Hugging Face integration
│   ├── single_window_tokens_dataset.py  # Single window per item
│   ├── tokens_dataset.py        # Main tokenized dataset
│   ├── zoo_dataset.py          # Model zoo integration  
│   └── zoo_dataset_models.py    # Zoo model definitions
├── preprocessing_augmentations.py  # Data augmentation
├── splitter.py                  # Dataset splitting utilities
└── tokenizer.py                # SANETokenizer implementation
```

## Advanced Features

### Data Augmentation
**Location**: `preprocessing_augmentations.py`

Provides transformations for tokenized model weights to improve training robustness.

### Dataset Splitting
**Location**: `splitter.py`

Utilities for splitting checkpoint datasets into train/validation/test sets while respecting model relationships.

## Performance Considerations

- **Memory**: Layer-wise mode uses less memory per item but creates more items
- **Speed**: Full model mode is faster for global operations
- **Windowing**: Per-layer windowing provides better layer coverage but increases dataset size
- **Padding**: Increases memory usage but enables consistent batch processing
- **Caching**: Tokenization is performed on-demand; consider pre-tokenizing for large datasets

## Common Patterns

### For SANE Training (Variable Window Sizes)
```python
# Use TokensDataset with layer-wise tokenization
dataset = TokensDataset(checkpoints_dataset, tokenizer, 
                       window_size=512, num_windows_per_model=4,
                       pad_windows=False)  # Efficient, variable sizes
```

### For Batch Training (Consistent Window Sizes)
```python
# Use padded windows for easier batching
dataset = TokensDataset(checkpoints_dataset, tokenizer,
                       window_size=512, num_windows_per_model=4,
                       pad_windows=True)  # All windows exactly 512 tokens
```

### For Contrastive Learning  
```python
# Use SingleWindowTokensDataset for individual window samples
dataset = SingleWindowTokensDataset(checkpoints_dataset, tokenizer,
                                  window_size=256, num_windows_per_model=8,
                                  pad_windows=True)  # Consistent sizes help with contrastive learning
```

### For Analysis/Visualization
```python  
# Use full model mode without windowing for complete model view
tokenizer = SANETokenizer(mode="full_model")
dataset = TokensDataset(checkpoints_dataset, tokenizer, enable_windowing=False)
```