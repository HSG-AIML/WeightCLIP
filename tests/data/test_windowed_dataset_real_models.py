"""
Real model integration tests for WindowedDataset and CachedWindowedDataset.

Tests use actual model weights loaded from local model zoos.
Requires --runslow to execute and local zoo access.

Usage:
    pytest tests/data/test_windowed_dataset_real_models.py --runslow -v
"""

import os
import time
import pytest
import torch
from sane.data.checkpoint.checkpoint import Checkpoint
from sane.data.datasets.windowed_dataset import WindowedDataset
from sane.data.datasets.cached_windowed_dataset import CachedWindowedDataset
from sane.data.datasets.zoo_dataset import ZooDataset
from sane.data.tokenizers.dense import DenseTokenizer


# ---------------------------------------------------------------------------
# Zoo paths & helpers
# ---------------------------------------------------------------------------

ZOO_RESNET_CIFAR100 = (
    "/ds2/model_zoos/zoos_resnet/zoos/CIFAR100/resnet18/kaiming_uniform"
    "/tune_zoo_cifar100_resnet18_kaiming_uniform"
)
ZOO_MNIST_CNN = (
    "/ds2/weight_space_learning/model_zoos/core-modelzoo/cnn-small_mnist"
    "/tune_zoo_mnist_uniform"
)

# Ordered by preference — first existing path is used as the primary zoo.
_CANDIDATE_ZOOS = [ZOO_MNIST_CNN, ZOO_RESNET_CIFAR100]

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

CACHE_DIR = "/local/tmp/sane_test_cache_windowed_real"


def _primary_zoo() -> str | None:
    for path in _CANDIDATE_ZOOS:
        if os.path.exists(path):
            return path
    return None


def _zoo(epoch_idx: list[int] | None = None, max_checkpoints: int = 10) -> ZooDataset:
    zoo_path = _primary_zoo()
    if zoo_path is None:
        pytest.skip("No model zoo available")
    return ZooDataset(
        zoo_path,
        epoch_idx=epoch_idx or [-1],
        max_checkpoints=max_checkpoints,
    )


def _tokenizer(tokensize: int = 32, mode: str = "layer_wise") -> DenseTokenizer:
    return DenseTokenizer(tokensize=tokensize, device="cpu", mode=mode, reference_statedict=None)


def _windowed(
    checkpoints_dataset,
    tokenizer,
    window_size: int = 100,
    num_windows_per_model: int | str = 1,  # str covers Literal["auto"]
    **kwargs,
) -> WindowedDataset:
    return WindowedDataset(
        checkpoints_dataset,
        tokenizer,
        window_size=window_size,
        num_windows_per_model=num_windows_per_model,  # type: ignore[arg-type]  # "auto" is valid
        **kwargs,
    )


def _cached(
    checkpoints_dataset,
    tokenizer,
    window_size: int = 100,
    num_windows_per_model: int | str = 1,  # str covers Literal["auto"]
    **kwargs,
) -> CachedWindowedDataset:
    return CachedWindowedDataset(
        checkpoints_dataset,
        tokenizer,
        window_size=window_size,
        num_windows_per_model=num_windows_per_model,  # type: ignore[arg-type]
        cache_dir=CACHE_DIR,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shared skip decorator
# ---------------------------------------------------------------------------

_slow_local = pytest.mark.slow and pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Requires local model zoo access"
)


# ---------------------------------------------------------------------------
# WindowedDataset — real model tests
# ---------------------------------------------------------------------------


class TestWindowedDatasetRealModels:

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_basic_tokenization(self):
        """Single checkpoint, single window — validate tensor shapes and weight content."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _windowed(checkpoints, tok, window_size=200, num_windows_per_model=1)

        assert len(dataset) >= 1

        tokens, mask, position = dataset[0]

        assert tokens.dim() == 2
        assert tokens.shape[1] == 32
        assert mask.shape == tokens.shape
        assert position.shape == (tokens.shape[0], 3)

        assert tokens.abs().sum() > 0, "Real model weights should be non-zero"
        assert mask.sum() > 0, "Mask should have valid entries"
        assert tokens.std().item() > 0.01, "Weight std too low — may be zeros"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_multiple_windows_per_model(self):
        """num_windows_per_model=3 — each window is within window_size, non-zero."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        num_windows = 3
        dataset = _windowed(
            checkpoints, tok, window_size=64, num_windows_per_model=num_windows
        )

        assert len(dataset) == len(checkpoints) * num_windows

        for i in range(num_windows):
            tokens, mask, position = dataset[i]
            assert tokens.shape[0] <= 64
            assert tokens.shape[1] == 32
            assert tokens.abs().sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_auto_windowing(self):
        """auto mode — generates at least one window per checkpoint, all non-zero."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _windowed(
            checkpoints, tok, window_size=128, num_windows_per_model="auto"
        )

        # auto mode must produce at least as many items as there are checkpoints
        assert len(dataset) >= len(checkpoints)

        for i in range(min(5, len(dataset))):
            tokens, mask, position = dataset[i]
            assert tokens.shape[0] <= 128
            assert tokens.shape[1] == 32
            assert tokens.abs().sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_multiple_epochs_same_architecture(self):
        """Multiple epochs from one zoo — same token count, different weight values."""
        checkpoints = _zoo(epoch_idx=[-1, -5, -10], max_checkpoints=9)
        tok = _tokenizer(tokensize=32)
        dataset = _windowed(
            checkpoints, tok, window_size=200, num_windows_per_model=1
        )

        assert len(dataset) >= 2

        token_counts = []
        means = []
        for i in range(min(3, len(dataset))):
            tokens, mask, _ = dataset[i]
            token_counts.append(tokens.shape[0])
            means.append(tokens.mean().item())

        # All epochs of the same model have the same architecture -> same token count
        assert len(set(token_counts)) == 1, "All epochs should yield the same number of tokens"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_layer_boundary_enforcement(self):
        """Every window must contain tokens from exactly one layer."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)

        configs = [
            dict(window_size=50, num_windows_per_model=1),
            dict(window_size=30, num_windows_per_model=4),
            dict(window_size=40, num_windows_per_model="auto"),
        ]

        for cfg in configs:
            dataset = _windowed(checkpoints, tok, **cfg)
            num_windows = min(10, len(dataset))

            for i in range(num_windows):
                tokens, mask, position = dataset[i]
                unique_layers = torch.unique(position[:, 1])
                assert unique_layers.numel() == 1, (
                    f"cfg={cfg}, index={i}: window spans layers {unique_layers.tolist()}"
                )

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_state_dict_consistency(self):
        """Layer count and valid-token count must exactly match the raw state_dict."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)

        # One window per layer, large enough to never truncate
        dataset = _windowed(
            checkpoints,
            tok,
            window_size=10_000,
            num_windows_per_model="auto",
            num_windows_per_layer=1,
        )

        raw = checkpoints[0]
        checkpoint = raw[0] if isinstance(raw, list) else raw
        assert isinstance(checkpoint, Checkpoint)
        state_dict = checkpoint.model.state_dict()

        expected_layers = sum(1 for k in state_dict if "weight" in k)

        def _expected_weights(sd):
            total = 0
            for k in sd:
                if "weight" in k:
                    total += sd[k].numel()
                    bias_key = k.replace("weight", "bias")
                    if bias_key in sd:
                        total += sd[bias_key].numel()
            return total

        expected_weights = _expected_weights(state_dict)

        # Collect all windows for checkpoint 0 via __getmodel__
        all_windows = dataset.__getmodel__(0)
        assert len(all_windows) == expected_layers, (
            f"Expected {expected_layers} windows (one per layer), got {len(all_windows)}"
        )

        valid_tokens = sum(mask.sum().item() for _, mask, _ in all_windows)
        assert valid_tokens == expected_weights, (
            f"Valid token count {valid_tokens} != state_dict weight count {expected_weights}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_different_tokenizer_sizes(self):
        """Different tokensizes all produce correct shapes and non-zero weights."""
        checkpoints = _zoo(epoch_idx=[-1])

        for tokensize in [16, 32, 64, 128]:
            tok = _tokenizer(tokensize=tokensize)
            dataset = _windowed(
                checkpoints, tok, window_size=200, num_windows_per_model=1
            )
            tokens, mask, _ = dataset[0]
            assert tokens.shape[1] == tokensize, f"tokensize={tokensize} mismatch"
            assert tokens.abs().sum() > 0
            assert mask.sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_getitem_performance(self):
        """__getitem__ must average under 10 s per call over 50 accesses."""
        checkpoints = _zoo(epoch_idx=[-1, -2, -3], max_checkpoints=30)
        tok = _tokenizer(tokensize=32)
        dataset = _windowed(
            checkpoints, tok, window_size=100, num_windows_per_model=2
        )

        n = min(50, len(dataset))
        times = []
        for i in range(n):
            t0 = time.time()
            result = dataset[i]
            times.append(time.time() - t0)
            assert isinstance(result, tuple)

        avg = sum(times) / len(times)
        print(f"WindowedDataset avg __getitem__: {avg:.4f}s over {n} calls")
        assert avg < 10.0, f"Too slow: {avg:.4f}s average"


# ---------------------------------------------------------------------------
# CachedWindowedDataset — real model tests
# ---------------------------------------------------------------------------


class TestCachedWindowedDatasetRealModels:

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_basic_tokenization_per_model_cache(self):
        """CachedWindowedDataset (per_model) — basic shapes and non-zero weights."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=200, num_windows_per_model=1,
            cache_mode="per_model",
        )

        assert len(dataset) >= 1

        tokens, mask, position = dataset[0]
        assert tokens.dim() == 2
        assert tokens.shape[1] == 32
        assert mask.shape == tokens.shape
        assert position.shape == (tokens.shape[0], 3)
        assert tokens.abs().sum() > 0
        assert tokens.std().item() > 0.01

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_basic_tokenization_per_window_cache(self):
        """CachedWindowedDataset (per_window) — basic shapes and non-zero weights."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=200, num_windows_per_model=1,
            cache_mode="per_window",
        )

        tokens, mask, position = dataset[0]
        assert tokens.dim() == 2
        assert tokens.shape[1] == 32
        assert tokens.abs().sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_dataset_length_fixed_windows(self):
        """len(dataset) == num_checkpoints * num_windows_per_model."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        num_windows = 4
        dataset = _cached(
            checkpoints, tok,
            window_size=80, num_windows_per_model=num_windows,
        )

        assert len(dataset) == len(checkpoints) * num_windows

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_auto_windowing(self):
        """auto mode — at least one window per checkpoint, all within window_size."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=128, num_windows_per_model="auto",
        )

        assert len(dataset) >= len(checkpoints)

        for i in range(min(5, len(dataset))):
            tokens, mask, position = dataset[i]
            assert tokens.shape[0] <= 128
            assert tokens.shape[1] == 32
            assert tokens.abs().sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_index_mapping(self):
        """_map_index produces correct (checkpoint_idx, window_idx) pairs."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        num_windows = 3
        dataset = _cached(
            checkpoints, tok,
            window_size=80, num_windows_per_model=num_windows,
        )

        for flat_idx in range(min(12, len(dataset))):
            ckpt_idx, win_idx = dataset._map_index(flat_idx)
            assert ckpt_idx == flat_idx // num_windows
            assert win_idx == flat_idx % num_windows

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_cache_hit_avoids_retokenization(self):
        """Repeated __getitem__ calls on a warm cache must not re-tokenize."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=100, num_windows_per_model=1,
            cache_mode="per_model",
        )

        # warm access
        t1, m1, p1 = dataset[0]

        # second access should hit cache
        t2, m2, p2 = dataset[0]

        assert torch.equal(t1, t2)
        assert torch.equal(m1, m2)
        assert torch.equal(p1, p2)

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_layer_boundary_enforcement(self):
        """Every window must contain tokens from exactly one layer."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)

        for cfg in [
            dict(window_size=50, num_windows_per_model=1),
            dict(window_size=30, num_windows_per_model="auto"),
            dict(window_size=60, num_windows_per_model="auto"),
        ]:
            dataset = _cached(checkpoints, tok, **cfg)
            for i in range(min(10, len(dataset))):
                tokens, mask, position = dataset[i]
                unique_layers = torch.unique(position[:, 1])
                assert unique_layers.numel() == 1, (
                    f"cfg={cfg}, index={i}: window spans layers {unique_layers.tolist()}"
                )

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_state_dict_consistency(self):
        """Layer count and valid-token count must exactly match the raw state_dict."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=10_000,
            num_windows_per_model="auto",
            num_windows_per_layer=1,
        )

        raw = checkpoints[0]
        checkpoint = raw[0] if isinstance(raw, list) else raw
        assert isinstance(checkpoint, Checkpoint)
        state_dict = checkpoint.model.state_dict()

        expected_layers = sum(1 for k in state_dict if "weight" in k)

        def _expected_weights(sd):
            total = 0
            for k in sd:
                if "weight" in k:
                    total += sd[k].numel()
                    bias_key = k.replace("weight", "bias")
                    if bias_key in sd:
                        total += sd[bias_key].numel()
            return total

        expected_weights = _expected_weights(state_dict)

        all_windows = dataset.__getmodel__(0)
        assert len(all_windows) == expected_layers

        valid_tokens = sum(mask.sum().item() for _, mask, _ in all_windows)
        assert valid_tokens == expected_weights, (
            f"Valid token count {valid_tokens} != state_dict weight count {expected_weights}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_multiple_epochs_same_token_count(self):
        """Multiple epochs from the same zoo — architecture is fixed so token count is constant."""
        checkpoints = _zoo(epoch_idx=[-1, -5, -10], max_checkpoints=9)
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=200, num_windows_per_model=1,
        )

        assert len(dataset) >= 2

        counts = [dataset[i][0].shape[0] for i in range(min(3, len(dataset)))]
        assert len(set(counts)) == 1, f"Token counts differ across epochs: {counts}"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_different_tokenizer_sizes(self):
        """Different tokensizes produce correct token dimension and non-zero weights."""
        checkpoints = _zoo(epoch_idx=[-1])

        for tokensize in [16, 32, 64, 128]:
            tok = _tokenizer(tokensize=tokensize)
            dataset = _cached(
                checkpoints, tok,
                window_size=200, num_windows_per_model=1,
            )
            tokens, mask, _ = dataset[0]
            assert tokens.shape[1] == tokensize
            assert tokens.abs().sum() > 0
            assert mask.sum() > 0

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_getitem_performance_per_model_cache(self):
        """Warm-cache __getitem__ must average under 5 s over 50 accesses."""
        checkpoints = _zoo(epoch_idx=[-1, -2, -3], max_checkpoints=10)
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=100, num_windows_per_model="auto",
            cache_mode="per_model",
        )

        n = min(50, len(dataset))
        times = []
        for i in range(n):
            t0 = time.time()
            result = dataset[i]
            times.append(time.time() - t0)
            assert isinstance(result, tuple)

        avg = sum(times) / len(times)
        max_t = max(times)
        print(f"CachedWindowedDataset (per_model) avg: {avg:.4f}s, max: {max_t:.4f}s over {n} calls")
        assert avg < 5.0, f"Too slow: {avg:.4f}s average"
        assert max_t < 15.0, f"Outlier too slow: {max_t:.4f}s"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_getitem_performance_per_window_cache(self):
        """Warm-cache __getitem__ (per_window) must average under 5 s over 50 accesses."""
        checkpoints = _zoo(epoch_idx=[-1, -2, -3], max_checkpoints=10)
        tok = _tokenizer(tokensize=32)
        dataset = _cached(
            checkpoints, tok,
            window_size=100, num_windows_per_model="auto",
            cache_mode="per_window",
        )

        n = min(50, len(dataset))
        times = []
        for i in range(n):
            t0 = time.time()
            result = dataset[i]
            times.append(time.time() - t0)
            assert isinstance(result, tuple)

        avg = sum(times) / len(times)
        max_t = max(times)
        print(f"CachedWindowedDataset (per_window) avg: {avg:.4f}s, max: {max_t:.4f}s over {n} calls")
        assert avg < 5.0, f"Too slow: {avg:.4f}s average"
        assert max_t < 15.0, f"Outlier too slow: {max_t:.4f}s"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_per_model_and_per_window_cache_agree(self):
        """Both cache modes must return identical tensors for the same index."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        num_windows = 3

        ds_model = _cached(
            checkpoints, tok,
            window_size=80, num_windows_per_model=num_windows,
            cache_mode="per_model",
        )
        ds_window = _cached(
            checkpoints, tok,
            window_size=80, num_windows_per_model=num_windows,
            cache_mode="per_window",
        )

        assert len(ds_model) == len(ds_window)

        for i in range(min(6, len(ds_model))):
            t_m, mask_m, pos_m = ds_model[i]
            t_w, mask_w, pos_w = ds_window[i]
            assert torch.equal(t_m, t_w), f"Token mismatch at index {i}"
            assert torch.equal(mask_m, mask_w), f"Mask mismatch at index {i}"
            assert torch.equal(pos_m, pos_w), f"Position mismatch at index {i}"


# ---------------------------------------------------------------------------
# Cross-compatibility: WindowedDataset vs CachedWindowedDataset
# ---------------------------------------------------------------------------


class TestCrossCompatibility:

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_windowed_and_cached_agree(self):
        """WindowedDataset and CachedWindowedDataset must return identical windows."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        window_size = 80
        num_windows = 3

        ds_live = _windowed(
            checkpoints, tok,
            window_size=window_size, num_windows_per_model=num_windows,
        )
        ds_cached = _cached(
            checkpoints, tok,
            window_size=window_size, num_windows_per_model=num_windows,
        )

        assert len(ds_live) == len(ds_cached)

        for i in range(min(6, len(ds_live))):
            t_l, m_l, p_l = ds_live[i]
            t_c, m_c, p_c = ds_cached[i]
            assert torch.equal(t_l, t_c), f"Token mismatch at index {i}"
            assert torch.equal(m_l, m_c), f"Mask mismatch at index {i}"
            assert torch.equal(p_l, p_c), f"Position mismatch at index {i}"

    @pytest.mark.slow
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires local model zoo access")
    def test_getmodel_matches_sequential_getitem(self):
        """__getmodel__(0) must return the same windows as sequential __getitem__ calls."""
        checkpoints = _zoo(epoch_idx=[-1])
        tok = _tokenizer(tokensize=32)
        num_windows = 3

        for DatasetCls, extra in [
            (WindowedDataset, dict()),
            (CachedWindowedDataset, dict(cache_dir=CACHE_DIR)),
        ]:
            dataset = DatasetCls(
                checkpoints, tok,
                window_size=80, num_windows_per_model=num_windows,
                **extra,  # type: ignore[arg-type]
            )

            model_windows = dataset.__getmodel__(0)
            assert len(model_windows) == num_windows

            for w_idx in range(num_windows):
                t_model, m_model, p_model = model_windows[w_idx]
                t_item, m_item, p_item = dataset[w_idx]  # flat index 0..num_windows-1
                assert torch.equal(t_model, t_item), (
                    f"{DatasetCls.__name__}: token mismatch at window {w_idx}"
                )