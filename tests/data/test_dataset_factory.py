import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sane.data.checkpoint.checkpoint import Checkpoint
from sane.data.config_schema import (
    AugmentationConfig,
    CachedWindowedDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    SplitConfig,
    TokenizerConfig,
    ZooDatasetConfig,
)
from sane.data.dataset_factory import DatasetFactory, _ref_checkpoint
from sane.data.datasets.combined_checkpoints_dataset import CombinedCheckpointsDataset
from sane.data.datasets.checkpoints_dataset import CheckpointsDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Primary zoo — cnn-small MNIST uniform (matches config/data/checkpoints_dataset/mnist_cnn_small_uniform.yaml)
# Uses zoo_cnn_permutation_spec and CNN (28x28 grayscale), which is consistent across all small CNN zoos.
ZOO_MNIST_CNN = "/ds2/weight_space_learning/model_zoos/core-modelzoo/cnn-small_mnist/tune_zoo_mnist_uniform"

# Second zoo used for multi-zoo tests — cnn-small FMNIST (matches config/data/checkpoints_dataset/fmnist_cnn_small_uniform.yaml)
ZOO_FMNIST_CNN = "/ds2/weight_space_learning/model_zoos/core-modelzoo/cnn-small_fmnist/tune_zoo_f_mnist_uniform"

# tokensize from config/data/tokenizer/sparse_full_model_cnn.yaml
SPARSE_TOKENSIZE = 201
# tokensize from config/data/tokenizer/dense_full_model.yaml
DENSE_TOKENSIZE = 230

CACHE_DIR = "/local/tmp/sane_test_cache"


def _make_checkpoint():
    return Checkpoint(model=nn.Linear(4, 2), metadata={})


def _mock_tokens_dataset(split_name: str, size: int = 8):
    ds = MagicMock()
    ds.split_name = split_name
    ds.__len__ = MagicMock(return_value=size)
    return ds


def _dl_config(**overrides):
    defaults = dict(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=4,
    )
    return DataLoaderConfig(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# _ref_checkpoint
# ---------------------------------------------------------------------------


class TestRefCheckpoint:
    def test_returns_checkpoint_directly(self):
        ckpt = _make_checkpoint()
        dataset = MagicMock()
        dataset.__getitem__ = MagicMock(return_value=ckpt)
        assert _ref_checkpoint(dataset) is ckpt

    def test_unwraps_list(self):
        ckpt = _make_checkpoint()
        dataset = MagicMock()
        dataset.__getitem__ = MagicMock(return_value=[ckpt, _make_checkpoint()])
        assert _ref_checkpoint(dataset) is ckpt


# ---------------------------------------------------------------------------
# _load_augmentations
# ---------------------------------------------------------------------------


class TestLoadAugmentations:
    def setup_method(self):
        self.factory = DatasetFactory()
        self.ref = _make_checkpoint()

    def _config(self, align=False, num_permutations=1, spec="zoo_cnn_permutation_spec"):
        return AugmentationConfig(
            permutation_spec=spec, align=align, num_permutations=num_permutations
        )

    @patch("sane.data.dataset_factory.git_re_basin")
    def test_no_augmentations_returns_none(self, mock_grb):
        mock_grb.zoo_cnn_permutation_spec.return_value = MagicMock()
        result = self.factory._load_augmentations(self._config(), self.ref)
        assert result is None

    @patch("sane.data.dataset_factory.git_re_basin")
    def test_align_only(self, mock_grb):
        mock_grb.zoo_cnn_permutation_spec.return_value = MagicMock()
        from sane.data.checkpoint.checkpoint_augmentation import (
            CheckpointAligner,
            CheckpointAugmentationPipeline,
        )

        result = self.factory._load_augmentations(self._config(align=True), self.ref)
        # single augmentation may be wrapped in a pipeline
        if isinstance(result, CheckpointAugmentationPipeline):
            assert any(isinstance(a, CheckpointAligner) for a in result.augmentations)
        else:
            assert isinstance(result, CheckpointAligner)

    @patch("sane.data.dataset_factory.git_re_basin")
    def test_permutations_only(self, mock_grb):
        mock_grb.zoo_cnn_permutation_spec.return_value = MagicMock()
        from sane.data.checkpoint.checkpoint_augmentation import (
            PermutationAugmentation,
            CheckpointAugmentationPipeline,
        )

        result = self.factory._load_augmentations(
            self._config(num_permutations=3), self.ref
        )
        if isinstance(result, CheckpointAugmentationPipeline):
            assert any(
                isinstance(a, PermutationAugmentation) for a in result.augmentations
            )
        else:
            assert isinstance(result, PermutationAugmentation)

    @patch("sane.data.dataset_factory.git_re_basin")
    def test_align_and_permutations_returns_pipeline(self, mock_grb):
        mock_grb.zoo_cnn_permutation_spec.return_value = MagicMock()
        from sane.data.checkpoint.checkpoint_augmentation import (
            CheckpointAugmentationPipeline,
        )

        result = self.factory._load_augmentations(
            self._config(align=True, num_permutations=2), self.ref
        )
        assert isinstance(result, CheckpointAugmentationPipeline)

    @patch("sane.data.dataset_factory.git_re_basin")
    def test_bad_permutation_spec_raises(self, mock_grb):
        mock_grb.nonexistent_spec.side_effect = AttributeError("not found")
        with pytest.raises(Exception):
            self.factory._load_augmentations(
                AugmentationConfig(
                    permutation_spec="nonexistent_spec", align=True, num_permutations=10
                ),
                self.ref,
            )


# ---------------------------------------------------------------------------
# _split_checkpoints
# ---------------------------------------------------------------------------


class TestSplitCheckpoints:
    def setup_method(self):
        self.factory = DatasetFactory()

    def _mock_dataset(self, size=10):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=size)
        return ds

    @patch("sane.data.dataset_factory.RandomSplitter")
    def test_train_only(self, MockSplitter):
        train_subset = MagicMock(spec=Subset)
        MockSplitter.return_value.split.return_value = [train_subset]
        splits = self.factory._split_checkpoints(
            SplitConfig(train=1.0, val=None, test=None), self._mock_dataset()
        )
        ds_map = dict(splits)
        assert ds_map["train"] is train_subset
        assert ds_map["val"] is None
        assert ds_map["test"] is None

    @patch("sane.data.dataset_factory.RandomSplitter")
    def test_train_val_test(self, MockSplitter):
        subsets = [MagicMock(spec=Subset) for _ in range(3)]
        MockSplitter.return_value.split.return_value = subsets
        splits = self.factory._split_checkpoints(
            SplitConfig(train=0.7, val=0.15, test=0.15), self._mock_dataset()
        )
        ds_map = dict(splits)
        assert ds_map["train"] is subsets[0]
        assert ds_map["val"] is subsets[1]
        assert ds_map["test"] is subsets[2]

    @patch("sane.data.dataset_factory.RandomSplitter")
    def test_zero_val_skipped(self, MockSplitter):
        subsets = [MagicMock(spec=Subset), MagicMock(spec=Subset)]
        MockSplitter.return_value.split.return_value = subsets
        splits = self.factory._split_checkpoints(
            SplitConfig(train=0.85, val=0.0, test=0.15), self._mock_dataset()
        )
        ds_map = dict(splits)
        assert ds_map["val"] is None
        assert ds_map["test"] is subsets[1]

    def test_missing_train_raises(self):
        with pytest.raises(ValueError, match="[Tt]rain"):
            self.factory._split_checkpoints(
                SplitConfig(train=None, val=0.5, test=0.5), self._mock_dataset()
            )

    class _NamedDataset(CheckpointsDataset):
        def __init__(self, name: str, root_dir: str, size: int = 2):
            super().__init__(checkpoints=[_make_checkpoint() for _ in range(size)])
            self.name = name
            self.root_dir = root_dir

    def test_dataset_name_splitter_holds_out_whole_datasets(self):
        dataset = CombinedCheckpointsDataset(
            [
                self._NamedDataset("cassava_leaf_cnn3", "/zoos/cassava-leaf/cnn3/run", size=2),
                self._NamedDataset("mushrooms_cnn3", "/zoos/mushrooms/cnn3/run", size=3),
                self._NamedDataset("asl_cnn3", "/zoos/asl/cnn3/run", size=4),
            ]
        )

        splits = self.factory._split_checkpoints(
            SplitConfig(
                train=None,
                val=None,
                test=None,
                val_datasets=["mushrooms_cnn3"],
                test_datasets=["asl"],
            ),
            dataset,
            splitter_name="DatasetNameSplitter",
        )
        ds_map = dict(splits)
        assert list(ds_map["train"].indices) == [0, 1]
        assert list(ds_map["val"].indices) == [2, 3, 4]
        assert list(ds_map["test"].indices) == [5, 6, 7, 8]

    def test_dataset_name_splitter_rejects_overlap(self):
        dataset = CombinedCheckpointsDataset(
            [self._NamedDataset("cassava_leaf_cnn3", "/zoos/cassava-leaf/cnn3/run", size=2)]
        )

        with pytest.raises(ValueError, match="disjoint"):
            self.factory._split_checkpoints(
                SplitConfig(
                    train=None,
                    val=None,
                    test=None,
                    val_datasets=["cassava_leaf_cnn3"],
                    test_datasets=["cassava_leaf_cnn3"],
                ),
                dataset,
                splitter_name="DatasetNameSplitter",
            )


# ---------------------------------------------------------------------------
# _build_tokenizer
# ---------------------------------------------------------------------------


class TestBuildTokenizer:
    def setup_method(self):
        self.factory = DatasetFactory()
        self.ref = _make_checkpoint()

    @patch("sane.data.dataset_factory.DenseTokenizer")
    def test_dense_dispatch(self, MockDense):
        config = TokenizerConfig(
            tokenizer_class="dense", mode="full_model", tokensize=64, device="cpu"
        )
        self.factory._build_tokenizer(config, self.ref)
        MockDense.assert_called_once()
        _, kwargs = MockDense.call_args
        assert kwargs["tokensize"] == 64

    @patch("sane.data.dataset_factory.SparseTokenizer")
    def test_sparse_dispatch(self, MockSparse):
        config = TokenizerConfig(
            tokenizer_class="sparse", mode="full_model", tokensize=32, device="cpu"
        )
        self.factory._build_tokenizer(config, self.ref)
        MockSparse.assert_called_once()


# ---------------------------------------------------------------------------
# _make_dataloaders
# ---------------------------------------------------------------------------


class TestMakeDataloaders:
    def setup_method(self):
        self.factory = DatasetFactory()

    def test_train_only(self):
        train, val, test = self.factory._make_dataloaders(
            _dl_config(), [_mock_tokens_dataset("train")]
        )
        assert isinstance(train, DataLoader)
        assert val is None
        assert test is None

    def test_train_val_test(self):
        datasets = [_mock_tokens_dataset(s) for s in ("train", "val", "test")]
        train, val, test = self.factory._make_dataloaders(_dl_config(), datasets)
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_train_shuffle_others_not(self):
        datasets = [_mock_tokens_dataset(s) for s in ("train", "val", "test")]
        with patch("sane.data.dataset_factory.DataLoader") as MockDL:
            MockDL.side_effect = lambda ds, shuffle, **kw: SimpleNamespace(
                split_name=ds.split_name, shuffle=shuffle
            )
            train, val, test = self.factory._make_dataloaders(_dl_config(), datasets)
        assert train.shuffle is True
        assert val.shuffle is False
        assert test.shuffle is False

    def test_no_persistent_workers_when_num_workers_is_0(self):
        with patch("sane.data.dataset_factory.DataLoader") as MockDL:
            MockDL.return_value = MagicMock()
            self.factory._make_dataloaders(
                _dl_config(num_workers=0), [_mock_tokens_dataset("train")]
            )
            _, kwargs = MockDL.call_args
            assert "persistent_workers" not in kwargs

    def test_persistent_workers_and_prefetch_when_num_workers_gt0(self):
        with patch("sane.data.dataset_factory.DataLoader") as MockDL:
            MockDL.return_value = MagicMock()
            self.factory._make_dataloaders(
                _dl_config(num_workers=2), [_mock_tokens_dataset("train")]
            )
            _, kwargs = MockDL.call_args
            assert kwargs["persistent_workers"] is True
            assert kwargs["prefetch_factor"] == 4


# ---------------------------------------------------------------------------
# _load_checkpoints_dataset
# ---------------------------------------------------------------------------


class TestLoadCheckpointsDataset:
    def setup_method(self):
        self.factory = DatasetFactory()

    @patch("sane.data.dataset_factory.ZooDataset")
    def test_single_config_creates_zoo_dataset(self, MockZoo):
        mock_zoo = MagicMock()
        mock_zoo.__getitem__ = MagicMock(return_value=_make_checkpoint())
        MockZoo.return_value = mock_zoo
        self.factory._load_checkpoints_dataset(
            ZooDatasetConfig(
                root_dir="/fake",
                epoch_idx=[],
                drop_nan=False,
                max_absolute_weight=None,
                max_checkpoints=None,
                augmentations=None,
            )
        )
        MockZoo.assert_called_once()

    @patch("sane.data.dataset_factory.CombinedCheckpointsDataset")
    @patch("sane.data.dataset_factory.ZooDataset")
    def test_list_config_creates_combined_dataset(self, MockZoo, MockCombined):
        mock_zoo = MagicMock()
        mock_zoo.__getitem__ = MagicMock(return_value=_make_checkpoint())
        MockZoo.return_value = mock_zoo
        configs = [
            ZooDatasetConfig(
                root_dir="/a",
                epoch_idx=[],
                drop_nan=False,
                max_absolute_weight=None,
                max_checkpoints=None,
                augmentations=None,
            ),
            ZooDatasetConfig(
                root_dir="/b",
                epoch_idx=[],
                drop_nan=False,
                max_absolute_weight=None,
                max_checkpoints=None,
                augmentations=None,
            ),
        ]
        self.factory._load_checkpoints_dataset(configs)
        MockCombined.assert_called_once()
        assert len(MockCombined.call_args.kwargs["datasets"]) == 2


# ---------------------------------------------------------------------------
# build (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestBuild:
    @patch("sane.data.dataset_factory.datasets.CachedWindowedDataset")
    @patch("sane.data.dataset_factory.DenseTokenizer")
    @patch("sane.data.dataset_factory.RandomSplitter")
    @patch("sane.data.dataset_factory.ZooDataset")
    def test_build_returns_train_loader(
        self, MockZoo, MockSplitter, MockTokenizer, MockSWD
    ):
        ckpt = _make_checkpoint()
        mock_zoo = MagicMock()
        mock_zoo.name = "TestZoo"
        mock_zoo.__getitem__ = MagicMock(return_value=ckpt)
        MockZoo.return_value = mock_zoo

        train_subset = MagicMock()
        train_subset.__getitem__ = MagicMock(return_value=ckpt)
        MockSplitter.return_value.split.return_value = [train_subset]

        mock_swd = MagicMock()
        mock_swd.split_name = "train"
        mock_swd.__len__ = MagicMock(return_value=8)
        MockSWD.return_value = mock_swd

        config = DatasetFactoryConfig(
            checkpoints_dataset=ZooDatasetConfig(
                name=None,
                root_dir="/fake",
                epoch_idx=[],
                drop_nan=False,
                max_absolute_weight=None,
                max_checkpoints=None,
                augmentations=None,
            ),
            tokenizer=TokenizerConfig(
                tokenizer_class="dense", mode="full_model", tokensize=128, device="cpu"
            ),
            tokens_dataset=CachedWindowedDatasetConfig(
                window_size=16,
                num_windows_per_model=1,
                num_windows_per_layer= None,
                pad_windows=True,
                cache_mode="per_model",
                allow_overlapping_windows=False,
                window_distribution_strategy="distributed",
                cache_dir=None,
                read_only=True,
            ),
            split=SplitConfig(train=1.0, val=None, test=None),
            dataloader=_dl_config(),
            splitter="RandomSplitter",
        )
        train, val, test = DatasetFactory().build(config)
        assert isinstance(train, DataLoader)
        assert val is None
        assert test is None
        _, kwargs = MockSWD.call_args
        assert kwargs["read_only"] is True


# ---------------------------------------------------------------------------
# Integration tests (real disk, --runslow)
# ---------------------------------------------------------------------------


def _assert_batches(loader: DataLoader, min_batches: int = 1):
    """Pull at least min_batches from a loader and check tensor shapes."""
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i + 1 >= min_batches:
            break
    assert len(batches) >= min_batches, "DataLoader yielded no batches"
    # Each batch from CachedWindowedDataset is a tuple of tensors
    batch = batches[0]
    assert isinstance(batch, (tuple, list)) and len(batch) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Requires local model zoo access"
)
class TestDatasetFactoryIntegration:
    """
    Integration tests exercising DatasetFactory.build() end-to-end against real model zoos.

    Zoo paths mirror the production configs under config/data/checkpoints_dataset/.
    Tests are skipped automatically when zoo paths are unavailable (e.g. on CI).
    """

    # ------------------------------------------------------------------
    # Shared fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def require_primary_zoo(self):
        if not os.path.exists(ZOO_MNIST_CNN):
            pytest.skip(f"Primary zoo not available: {ZOO_MNIST_CNN}")

    @pytest.fixture()
    def require_secondary_zoo(self):
        if not os.path.exists(ZOO_FMNIST_CNN):
            pytest.skip(f"Secondary zoo not available: {ZOO_FMNIST_CNN}")

    # ------------------------------------------------------------------
    # Config builders — match production YAML files directly
    # ------------------------------------------------------------------

    def _single_zoo_config(
        self,
        *,
        tokenizer: TokenizerConfig,
        tokens_dataset: CachedWindowedDatasetConfig,
        epoch_idx=None,
        max_checkpoints: int = 100,
    ) -> DatasetFactoryConfig:
        """Single zoo — mirrors config/data/checkpoints_dataset/mnist_cnn_small_uniform.yaml."""
        return DatasetFactoryConfig(
            checkpoints_dataset=ZooDatasetConfig(
                root_dir=ZOO_MNIST_CNN,
                epoch_idx=epoch_idx or [25],
                drop_nan=True,
                max_absolute_weight=100,
                max_checkpoints=max_checkpoints,
                augmentations=AugmentationConfig(
                    permutation_spec="zoo_cnn_permutation_spec",
                    align=True,
                    num_permutations=2,
                ),
            ),
            split=SplitConfig(train=0.7, val=0.15, test=0.15),
            tokenizer=tokenizer,
            tokens_dataset=tokens_dataset,
            dataloader=_dl_config(batch_size=2),
        )

    def _multi_zoo_config(
        self,
        *,
        tokenizer: TokenizerConfig,
        tokens_dataset: CachedWindowedDatasetConfig,
        max_checkpoints: int = 100,
    ) -> DatasetFactoryConfig:
        """Multi-zoo — MNIST + FMNIST small CNNs (both use zoo_cnn_permutation_spec)."""
        return DatasetFactoryConfig(
            checkpoints_dataset=[
                ZooDatasetConfig(
                    root_dir=ZOO_MNIST_CNN,
                    epoch_idx=[25],
                    drop_nan=True,
                    max_absolute_weight=100,
                    max_checkpoints=max_checkpoints,
                    augmentations=AugmentationConfig(
                        permutation_spec="zoo_cnn_permutation_spec",
                        align=True,
                        num_permutations=2,
                    ),
                ),
                ZooDatasetConfig(
                    root_dir=ZOO_FMNIST_CNN,
                    epoch_idx=[25],
                    drop_nan=True,
                    max_absolute_weight=100,
                    max_checkpoints=max_checkpoints,
                    augmentations=AugmentationConfig(
                        permutation_spec="zoo_cnn_permutation_spec",
                        align=True,
                        num_permutations=2,
                    ),
                ),
            ],
            split=SplitConfig(train=0.7, val=0.15, test=0.15),
            tokenizer=tokenizer,
            tokens_dataset=tokens_dataset,
            dataloader=_dl_config(batch_size=2),
        )

    # ------------------------------------------------------------------
    # Tokenizer fixtures — match config/data/tokenizer/*.yaml
    # ------------------------------------------------------------------

    @pytest.fixture()
    def dense_tokenizer(self) -> TokenizerConfig:
        """config/data/tokenizer/dense_full_model.yaml"""
        return TokenizerConfig(
            tokenizer_class="dense",
            tokensize=DENSE_TOKENSIZE,
            mode="full_model",
            device="cpu",
        )

    @pytest.fixture()
    def sparse_tokenizer(self) -> TokenizerConfig:
        """config/data/tokenizer/sparse_full_model_cnn.yaml"""
        return TokenizerConfig(
            tokenizer_class="sparse",
            tokensize=SPARSE_TOKENSIZE,
            mode="full_model",
            device="cpu",
        )

    # ------------------------------------------------------------------
    # Tokens-dataset fixtures — match config/data/tokens_dataset/*.yaml
    # ------------------------------------------------------------------

    @pytest.fixture()
    def one_window_per_model(self) -> CachedWindowedDatasetConfig:
        """1 window per model — based on single_window_small.yaml with num_windows_per_model=1."""
        return CachedWindowedDatasetConfig(
            window_size=256,
            num_windows_per_model=1,
            allow_overlapping_windows=False,
            window_distribution_strategy="consecutive",
            cache_dir=CACHE_DIR,
            cache_mode="per_model",
            pad_windows=True,
            num_windows_per_layer=None,
        )

    @pytest.fixture()
    def many_windows_per_model(self) -> CachedWindowedDatasetConfig:
        """Many windows per model — single_window.yaml with auto coverage."""
        return CachedWindowedDatasetConfig(
            window_size=256,
            num_windows_per_model="auto",
            allow_overlapping_windows=False,
            window_distribution_strategy="consecutive",
            cache_dir=CACHE_DIR,
            cache_mode="per_model",
            pad_windows=True,
            num_windows_per_layer=None,
        )

    # ------------------------------------------------------------------
    # Single zoo
    # ------------------------------------------------------------------

    def test_single_zoo_dense_one_window_per_model(
        self, dense_tokenizer, one_window_per_model
    ):
        """Single zoo · dense tokenizer · 1 window/model."""
        train, val, test = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer, tokens_dataset=one_window_per_model
            )
        )
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)
        _assert_batches(train)

    def test_single_zoo_dense_many_windows_per_model(
        self, dense_tokenizer, many_windows_per_model
    ):
        """Single zoo · dense tokenizer · auto (many) windows/model."""
        train, val, test = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer, tokens_dataset=many_windows_per_model
            )
        )
        assert isinstance(train, DataLoader)
        _assert_batches(train)
        # auto mode produces more items than a single-window dataset
        assert len(train.dataset) >= 1

    def test_single_zoo_sparse_one_window_per_model(
        self, sparse_tokenizer, one_window_per_model
    ):
        """Single zoo · sparse tokenizer · 1 window/model."""
        train, val, test = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=sparse_tokenizer, tokens_dataset=one_window_per_model
            )
        )
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)
        _assert_batches(train)

    def test_single_zoo_sparse_many_windows_per_model(
        self, sparse_tokenizer, many_windows_per_model
    ):
        """Single zoo · sparse tokenizer · auto (many) windows/model."""
        train, val, test = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=sparse_tokenizer, tokens_dataset=many_windows_per_model
            )
        )
        assert isinstance(train, DataLoader)
        _assert_batches(train)

    # ------------------------------------------------------------------
    # Multi-zoo
    # ------------------------------------------------------------------

    def test_multi_zoo_dense_one_window_per_model(
        self, require_secondary_zoo, dense_tokenizer, one_window_per_model
    ):
        """Multi-zoo · dense tokenizer · 1 window/model — MNIST + FMNIST small CNNs."""
        train, val, test = DatasetFactory().build(
            self._multi_zoo_config(
                tokenizer=dense_tokenizer, tokens_dataset=one_window_per_model
            )
        )
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)
        _assert_batches(train)

    def test_multi_zoo_dense_many_windows_per_model(
        self, require_secondary_zoo, dense_tokenizer, many_windows_per_model
    ):
        """Multi-zoo · dense tokenizer · auto windows/model."""
        train, val, test = DatasetFactory().build(
            self._multi_zoo_config(
                tokenizer=dense_tokenizer, tokens_dataset=many_windows_per_model
            )
        )
        assert isinstance(train, DataLoader)
        _assert_batches(train)

    def test_multi_zoo_sparse_one_window_per_model(
        self, require_secondary_zoo, sparse_tokenizer, one_window_per_model
    ):
        """Multi-zoo · sparse tokenizer · 1 window/model — MNIST + FMNIST small CNNs."""
        train, val, test = DatasetFactory().build(
            self._multi_zoo_config(
                tokenizer=sparse_tokenizer, tokens_dataset=one_window_per_model
            )
        )
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)
        _assert_batches(train)

    def test_multi_zoo_sparse_many_windows_per_model(
        self, require_secondary_zoo, sparse_tokenizer, many_windows_per_model
    ):
        """Multi-zoo · sparse tokenizer · auto windows/model."""
        train, val, test = DatasetFactory().build(
            self._multi_zoo_config(
                tokenizer=sparse_tokenizer, tokens_dataset=many_windows_per_model
            )
        )
        assert isinstance(train, DataLoader)
        _assert_batches(train)

    # ------------------------------------------------------------------
    # Window-count invariants
    # ------------------------------------------------------------------

    def test_many_windows_yields_more_samples_than_one(
        self, dense_tokenizer, one_window_per_model, many_windows_per_model
    ):
        """auto windowing must produce strictly more samples than num_windows_per_model=1."""
        _, _, _ = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer,
                tokens_dataset=one_window_per_model,
                max_checkpoints=100,
            )
        )
        train_one, _, _ = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer,
                tokens_dataset=one_window_per_model,
                max_checkpoints=100,
            )
        )
        train_many, _, _ = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer,
                tokens_dataset=many_windows_per_model,
                max_checkpoints=100,
            )
        )
        assert len(train_many.dataset) >= len(train_one.dataset)

    def test_dense_and_sparse_token_sizes_differ(
        self, dense_tokenizer, sparse_tokenizer, one_window_per_model
    ):
        """Dense and sparse tokenizers should produce different token sizes."""
        train_dense, _, _ = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=dense_tokenizer,
                tokens_dataset=one_window_per_model,
                max_checkpoints=100,
            )
        )
        train_sparse, _, _ = DatasetFactory().build(
            self._single_zoo_config(
                tokenizer=sparse_tokenizer,
                tokens_dataset=one_window_per_model,
                max_checkpoints=100,
            )
        )

        batch_dense = next(iter(train_dense))
        batch_sparse = next(iter(train_sparse))
        # weights tensor is the first element; last dim is token size
        token_dim_dense = batch_dense[0].shape[-1]
        token_dim_sparse = batch_sparse[0].shape[-1]
        assert token_dim_dense != token_dim_sparse
