import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict, Tuple
from pathlib import Path
import logging
import __main__

from sane.data.datasets import DatasetPtImageSetSampler
from sane.loss.logical_datasets import infer_logical_dataset_name

logger = logging.getLogger(__name__)


class CachedDataset(torch.utils.data.Dataset):
    """Compatibility shim for unpickling legacy zoo dataset.pt files."""
    def __init__(self, dataset=None, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        if dataset is not None:
            for i in range(len(dataset)):
                img, label = dataset[i]
                self.data.append(img)
                self.targets.append(label)
            self.data = torch.stack(self.data)
            self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.targets) if hasattr(self, 'targets') else 0

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


class GrayscaleToRGB:
    """Compatibility shim for unpickling legacy zoo dataset.pt files."""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


# Inject into __main__ so pickle can find them when unpickling dataset.pt
__main__.CachedDataset = CachedDataset
__main__.GrayscaleToRGB = GrayscaleToRGB


class DatasetAlignmentLoss(nn.Module):
    """Bidirectional token-level contrastive alignment loss.

    Aligns SANE per-token embeddings against DeepSets dataset embeddings:

        L_align = L_{m->d} + L_{d->m}

        L_{m->d}: cross-entropy - each token predicts its dataset.
        L_{d->m}: multi-positive NCE - each dataset predicts its tokens.

        The objective can be configured as either:
                - ``multi_positive``: the default bidirectional token/dataset loss.
                - ``siglip``: a SigLIP-style pairwise sigmoid loss over the same
                    token x dataset similarity matrix.

    Args:
        deepsets_encoder: Dataset encoder with .encode(x) -> [B, embed_dim]
            and .embedding_dim. Input x: [B, set_size, C, H, W].
        temperature: Logit scale applied to cosine similarities. Default 10.0.
        weight: Scalar weight applied to the combined loss. Default 1.0.
        set_size: Images sampled per dataset from dataset.pt. Default 32.
        image_resize: Optional (H, W) to resize dataset images. Accepts int
            (square), (H, W) tuple, or None.
        freeze_sane: Detach SANE embeddings before loss computation.
        freeze_deepsets: Disable gradient flow through the dataset encoder.
        objective: Either ``multi_positive`` or ``siglip``.
        siglip_bias_init: Initial value for the paper-style learnable SigLIP
            scalar logit bias. Only used when ``objective='siglip'``.
    """

    def __init__(
        self,
        deepsets_encoder: nn.Module,
        temperature: float = 10.0,
        weight: float = 1.0,
        set_size: int = 32,
        image_resize: Optional[Any] = None,
        freeze_sane: bool = False,
        freeze_deepsets: bool = False,
        objective: str = "multi_positive",
        siglip_bias_init: float = -10.0,
    ):
        super().__init__()
        self.deepsets_encoder = deepsets_encoder
        self.temperature = temperature
        self.weight = weight
        self.set_size = set_size
        self.image_resize = self._parse_image_resize(image_resize)
        self.freeze_sane = freeze_sane
        self.freeze_deepsets = freeze_deepsets
        self.objective = self._parse_objective(objective)
        self.siglip_bias_init = float(siglip_bias_init)
        self._dataset_cache: Dict[str, Any] = {}
        if self.objective == "siglip":
            encoder_param = next(self.deepsets_encoder.parameters(), None)
            bias_kwargs: Dict[str, Any] = {}
            if encoder_param is not None:
                bias_kwargs["device"] = encoder_param.device
                if encoder_param.is_floating_point():
                    bias_kwargs["dtype"] = encoder_param.dtype
            # SigLIP uses a learnable scalar bias initialized to a large negative value to keep early training stable.
            self.logit_bias = nn.Parameter(torch.tensor(self.siglip_bias_init, **bias_kwargs))
        else:
            self.register_parameter("logit_bias", None)
        target_hw = (32, 32) if self.image_resize is None else self.image_resize
        self.image_sampler = DatasetPtImageSetSampler(
            split="trainset",
            set_size=set_size,
            target_image_size=target_hw,
            target_channels=3,
        )

        if freeze_deepsets:
            self.deepsets_encoder.eval()
            for p in self.deepsets_encoder.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------

    def _get_logical_dataset_name(self, zoo_path: str) -> str:
        """Collapse architecture-specific zoo roots to one logical dataset name."""
        return infer_logical_dataset_name(zoo_path)

    def _load_dataset(self, zoo_path: str):
        """Load and cache dataset.pt from a zoo root."""
        if zoo_path in self._dataset_cache:
            return self._dataset_cache[zoo_path]
        pt = Path(zoo_path) / "dataset.pt"
        if not pt.exists():
            logger.warning("dataset.pt not found in %s", zoo_path)
            self._dataset_cache[zoo_path] = None
            return None
        d = torch.load(str(pt), weights_only=False)
        dataset = d.get('trainset') or d.get('testset')
        if dataset is None:
            logger.warning("No trainset/testset in %s", pt)
        self._dataset_cache[zoo_path] = dataset
        return dataset

    def _parse_image_resize(self, image_resize: Optional[Any]) -> Optional[Tuple[int, int]]:
        if image_resize is None:
            return None
        if isinstance(image_resize, int):
            if image_resize <= 0:
                raise ValueError(f"image_resize must be positive, got {image_resize}")
            return (image_resize, image_resize)
        if isinstance(image_resize, (list, tuple)):
            if len(image_resize) == 1:
                s = int(image_resize[0])
                if s <= 0:
                    raise ValueError(f"image_resize must be positive, got {image_resize}")
                return (s, s)
            if len(image_resize) == 2:
                h, w = int(image_resize[0]), int(image_resize[1])
                if h <= 0 or w <= 0:
                    raise ValueError(f"image_resize values must be positive, got {image_resize}")
                return (h, w)
        raise ValueError(f"image_resize must be None, int, or 1/2-item sequence, got {image_resize!r}")

    def _parse_objective(self, objective: str) -> str:
        if not isinstance(objective, str):
            raise TypeError(f"objective must be a string, got {type(objective).__name__}")
        objective = objective.lower()
        if objective not in {"multi_positive", "siglip"}:
            raise ValueError(
                f"objective must be one of ('multi_positive', 'siglip'), got {objective!r}"
            )
        return objective

    def _normalize_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Standardize to 3 channels and optionally resize."""
        if images.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if self.image_resize is not None and tuple(images.shape[-2:]) != self.image_resize:
            if not torch.is_floating_point(images):
                images = images.float()
            images = F.interpolate(images, size=self.image_resize, mode="bilinear", align_corners=False)
        return images

    def _sample_images(self, dataset, device: torch.device) -> torch.Tensor:
        """Sample set_size random images -> [1, n, C, H, W] on device."""
        total = len(dataset)
        indices = torch.randperm(total)[:min(self.set_size, total)]
        if hasattr(dataset, 'tensors'):
            images = dataset.tensors[0][indices]
        else:
            images = torch.stack([dataset[i.item()][0] for i in indices])
        return self._normalize_image_batch(images).unsqueeze(0).to(device)

    def _encode_datasets(self, zoo_paths: list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return [U, embed_dim] embeddings for logical datasets.

        The dataset encoder is called once on the stacked logical-dataset batch
        so BatchNorm layers see the actual number of datasets in the current
        minibatch rather than a stream of singleton inputs.
        """
        embed_dim = self.deepsets_encoder.embedding_dim
        embeddings = torch.zeros(len(zoo_paths), embed_dim, device=device, dtype=dtype)
        if not zoo_paths:
            return embeddings

        image_sets = []
        valid_positions = []
        for pos, zoo_path in enumerate(zoo_paths):
            try:
                image_sets.append(self.image_sampler.sample_image_set(zoo_path, device=device))
                valid_positions.append(pos)
            except Exception as e:
                logger.error("Failed to sample dataset from %s: %s", zoo_path, e)

        if not image_sets:
            return embeddings.detach() if self.freeze_deepsets else embeddings

        images = torch.stack(image_sets, dim=0)
        was_train = self.deepsets_encoder.training
        force_eval = self.freeze_deepsets or (was_train and images.shape[0] == 1)
        if force_eval:
            self.deepsets_encoder.eval()

        try:
            with torch.set_grad_enabled(not self.freeze_deepsets):
                encoded = self.deepsets_encoder.encode(images)
        except Exception as e:
            logger.error("Failed to encode dataset batch for %d zoo(s): %s", len(valid_positions), e)
            return embeddings.detach() if self.freeze_deepsets else embeddings
        finally:
            if force_eval and was_train:
                self.deepsets_encoder.train()

        encoded = encoded.to(dtype=dtype)
        embeddings[valid_positions] = encoded
        return embeddings.detach() if self.freeze_deepsets else embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sane_embeddings: torch.Tensor,
        dataset_indices: torch.Tensor,
        zoo_paths: list,
    ) -> Dict[str, torch.Tensor]:
        """Compute bidirectional token-level alignment loss.

        Args:
            sane_embeddings: [B, seq_len, D] SANE per-token latent vectors.
            dataset_indices: [B] integer index into zoo_paths for each model.
            zoo_paths: list of zoo root paths, one per unique raw dataset index.

        Returns dict with:
            'alignment_loss'    - weighted scalar (backprop through both encoders).
            'alignment_acc_fwd' - % tokens that pick the correct dataset (no-grad).
            'alignment_acc_bwd' - % datasets that pick >=1 correct token (no-grad).
        """
        B, seq_len, sane_dim = sane_embeddings.shape
        device = sane_embeddings.device
        dtype = sane_embeddings.dtype

        embed_dim = self.deepsets_encoder.embedding_dim
        if sane_dim != embed_dim:
            raise ValueError(
                f"SANE token dim ({sane_dim}) != dataset encoder embedding_dim ({embed_dim}). "
                "Make both encoders use the same embedding size."
            )

        # ------------------------------------------------------------------
        # 1. Collapse zoo paths to logical datasets (architecture-agnostic)
        # ------------------------------------------------------------------
        logical_name_to_pos: Dict[str, int] = {}
        raw_idx_to_pos: Dict[int, int] = {}
        logical_zoo_paths = []
        for raw in dataset_indices.unique():
            ri = raw.item()
            if ri >= len(zoo_paths):
                logger.warning("Dataset index %d out of range", ri)
                continue
            name = self._get_logical_dataset_name(zoo_paths[ri])
            if name not in logical_name_to_pos:
                logical_name_to_pos[name] = len(logical_zoo_paths)
                logical_zoo_paths.append(zoo_paths[ri])
            raw_idx_to_pos[ri] = logical_name_to_pos[name]

        U = len(logical_zoo_paths)
        model_pos = torch.tensor([raw_idx_to_pos[i.item()] for i in dataset_indices], device=device, dtype=torch.long)  # [B]

        # ------------------------------------------------------------------
        # 2. Encode each logical dataset -> [U, embed_dim]
        # ------------------------------------------------------------------
        deepsets_embs = self._encode_datasets(
            logical_zoo_paths,
            device,
            dtype,
        )

        # ------------------------------------------------------------------
        # 3. Per-token bidirectional contrastive loss
        # ------------------------------------------------------------------
        sane_flat = sane_embeddings.reshape(B * seq_len, sane_dim)
        if self.freeze_sane:
            sane_flat = sane_flat.detach()
        token_pos = model_pos.unsqueeze(1).expand(-1, seq_len).reshape(-1)  # [N]

        sane_feat = F.normalize(sane_flat, dim=-1)        # [N, D]
        ds_feat = F.normalize(deepsets_embs, dim=-1)      # [U, D]
        logits = sane_feat @ ds_feat.T * self.temperature  # [N, U]
        if self.logit_bias is not None:
            logits = logits + self.logit_bias.to(device=logits.device, dtype=logits.dtype)

        pos_mask = (token_pos.unsqueeze(0) == torch.arange(U, device=device).unsqueeze(1)).float()  # [U, N]
        if self.objective == "multi_positive":
            # L_{m->d}: each token predicts its dataset
            loss_fwd = F.cross_entropy(logits, token_pos)

            # L_{d->m}: each dataset predicts its tokens (multi-positive NCE)
            log_probs_bwd = F.log_softmax(logits.T, dim=-1)                                          # [U, N]
            n_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
            loss_bwd = (-(pos_mask * log_probs_bwd).sum(dim=1) / n_pos.squeeze(1)).mean()
            total_loss = self.weight * (loss_fwd + loss_bwd)
        else:
            pair_targets = pos_mask.T.mul(2.0).sub(1.0)                                              # [N, U] in {-1, +1}
            total_loss = self.weight * F.softplus(-pair_targets * logits).mean()

        with torch.no_grad():
            acc_fwd = (logits.argmax(dim=1) == token_pos).float().mean() * 100.0
            best_token = logits.T.argmax(dim=1)  # [U]
            acc_bwd = (token_pos[best_token] == torch.arange(U, device=device)).float().mean() * 100.0

        return {
            'alignment_loss': total_loss,
            'alignment_acc_fwd': acc_fwd.item(),
            'alignment_acc_bwd': acc_bwd.item(),
        }
