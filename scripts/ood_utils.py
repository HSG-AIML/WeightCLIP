#!/usr/bin/env python3
from __future__ import annotations

import gc
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
src_dir = repo_root / "src"

for path in (src_dir, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import sane
from sane.data.config_schema import (
    AugmentationConfig,
    CachedWindowedDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    SplitConfig,
    TokenizerConfig,
    ZooDatasetConfig,
    from_dict,
)
from sane.data.dataset_factory import DatasetFactory
from sane.data.datasets import CombinedCheckpointsDataset
from sane.data.datasets.image_set_dataset import CachedDataset, DatasetPtImageSetSampler, resolve_dataset_pt_path
from sane.data.datasets.zoo_dataset_models import CNN3, ResNet18Slim, get_model
from sane.model.autoencoder import SANEAutoEncoder
from sane.model.dataset_encoder import ENCODER_PRESETS, build_dataset_encoder

CNN3_ZOO_BASE = "/local/zoos/metatrain_cnn3"
RESNET18SLIM_ZOO_BASE = "/local/zoos/metatrain_resnet18"

METATEST_DATASETS =  ["colorectal-histology", "covid19", "speed-limit-signs", "honeybee-pollen", "real-or-drawing", "cifar10"]

DATASET_CLASS_COUNTS = {
    "colorectal-histology": 8,
    "covid19": 3,
    "speed-limit-signs": 4,
    "honeybee-pollen": 2,
    "real-or-drawing": 10,
    "cifar10": 10,
}


CNN3_FINETUNE_LR = 1.5e-3
CNN3_FINETUNE_WD = 0.0
RESNET18SLIM_FINETUNE_LR = 1.5e-4
RESNET18SLIM_FINETUNE_WD = 0.0
RESNET18SLIM_MOMENTUM = 0.9

_MODEL_CONFIG_CACHE: dict[str, dict[str, Any] | None] = {}
_IMAGE_SAMPLERS: dict[tuple[int, tuple[int, int]], DatasetPtImageSetSampler] = {}


class GrayscaleToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1) if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[0] == 1 else x


class MemoryBankMixtureTranslator(nn.Module):
    MAX_TRANSFORMER_SEQ_LEN = 1024

    def __init__(self, input_dim=128, seq_len=118, token_dim=128, d_model=256, n_head=4, n_layer=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.register_buffer("memory_bank", None)
        self.num_memory = 0
        self.use_mlp_mode = seq_len > self.MAX_TRANSFORMER_SEQ_LEN

        if self.use_mlp_mode:
            self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
            self.num_prototypes = min(256, seq_len)
            self.position_protos = nn.Parameter(torch.randn(self.num_prototypes, d_model) * 0.02)
            self.output_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)) #old version
        else:
            from sane.model.transformer import SANETransformerEncoder

            self.dataset_proj = nn.Linear(input_dim, d_model)
            self.token_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.transformer = SANETransformerEncoder(input_dim=d_model, output_dim=d_model, d_model=d_model, nhead=n_head, num_layers=n_layer, dropout=dropout, 
                                                      position_embedding="sinusoidal", num_pos_dims=1, transformer_type="gpt", block_size=seq_len + 1)

        self.mixture_head = None

    def set_memory_bank(self, embeddings):
        self.num_memory = int(embeddings.shape[0])
        self.register_buffer("memory_bank", embeddings.clone())
        self.mixture_head = nn.Linear(self.d_model, self.num_memory)

    def forward(self, dataset_emb, return_logits=False):
        if self.memory_bank is None:
            raise ValueError("Memory bank not set! Call set_memory_bank() first.")

        batch_size = dataset_emb.shape[0]
        device = dataset_emb.device

        if self.use_mlp_mode:
            h_global = self.input_proj(dataset_emb)
            proto_positions = torch.linspace(0, 1, self.num_prototypes, device=device)
            target_positions = torch.linspace(0, 1, self.seq_len, device=device)
            pos_weights = 1.0 - (target_positions.unsqueeze(1) - proto_positions.unsqueeze(0)).abs().clamp(0, 1)
            pos_weights = pos_weights / (pos_weights.sum(dim=1, keepdim=True) + 1e-8)

            pos_emb = pos_weights.mm(self.position_protos)
            combined = torch.cat([h_global.unsqueeze(1).expand(-1, self.seq_len, -1), pos_emb.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
            
            token_hidden = self.output_proj(combined)

        else:
            dataset_token = self.dataset_proj(dataset_emb).unsqueeze(1)
            token_queries = self.token_queries.expand(batch_size, -1, -1)
            sequence = torch.cat([dataset_token, token_queries], dim=1)
            positions = torch.arange(sequence.shape[1], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
            token_hidden = self.transformer(sequence, positions)[:, 1:, :]

        logits = self.mixture_head(token_hidden)
        return logits if return_logits else F.softmax(logits, dim=-1)

    def sample(self, dataset_emb, mode="expectation", temperature=1.0, top_k=None):
        with torch.no_grad():
            logits = self(dataset_emb, return_logits=True) / max(float(temperature), 1e-8)

            if top_k is not None and 0 < int(top_k) < self.num_memory:
                topk_vals, topk_idx = logits.topk(int(top_k), dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                logits = filtered.scatter(-1, topk_idx, topk_vals)

            weights = F.softmax(logits, dim=-1)
            memory_bank = self.memory_bank.to(dataset_emb.device)

            if mode in {"expectation", "top_k_expectation"}:
                return torch.einsum("bsm,msd->bsd", weights, memory_bank)

            idx = logits.argmax(dim=-1)
            out = torch.empty(dataset_emb.shape[0], self.seq_len, self.token_dim, device=dataset_emb.device)
            for batch_idx in range(dataset_emb.shape[0]):
                out[batch_idx] = memory_bank[idx[batch_idx], torch.arange(self.seq_len, device=dataset_emb.device)]
            return out


def train_memory_bank_translator(X, Y, epochs=200, lr=0.5e-4, batch_size=320, d_model=512, n_head=8, n_layer=4, device="cuda", leave_one_out=True, use_amp=True, token_subsample=None, patience=300, log_every=25):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(device)
    x_data = X.float().to(device)
    y_data = Y.float().cpu()
    num_samples, seq_len, token_dim = int(y_data.shape[0]), int(y_data.shape[1]), int(y_data.shape[2])
    use_token_subsample = token_subsample is not None and seq_len > int(token_subsample)
    token_subsample = int(token_subsample) if use_token_subsample else seq_len
    effective_leave_one_out = bool(leave_one_out and num_samples > 1)
    soft_target_k = min(16, num_samples - 1 if effective_leave_one_out else num_samples)

    if soft_target_k <= 0:
        effective_leave_one_out = False
        soft_target_k = 1

    model = MemoryBankMixtureTranslator(input_dim=x_data.shape[1], seq_len=seq_len, token_dim=token_dim, d_model=d_model, n_head=n_head, n_layer=n_layer, dropout=0.15).to(device)
    model.set_memory_bank(y_data)
    model = model.to(device)
    model.memory_bank = model.memory_bank.cpu()

    adamw_kwargs = {"lr": float(lr), "weight_decay": 0.05, "betas": (0.9, 0.98)}
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    dataset = TensorDataset(x_data.cpu(), torch.arange(num_samples))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(lr) * 10, epochs=int(epochs), steps_per_epoch=max(len(loader), 1), pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=100)

    with torch.no_grad():
        pooled = y_data.mean(dim=1)
        pooled_dists = torch.cdist(pooled, pooled)
        if effective_leave_one_out:
            pooled_dists.fill_diagonal_(float("inf"))
        topk_dists, topk_idx = pooled_dists.topk(soft_target_k, dim=-1, largest=False)
        topk_weights = F.softmax(-topk_dists / 0.1, dim=-1)
        nearest = pooled_dists.argmin(dim=-1)

    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = {"epoch": [], "total_loss": [], "accuracy": []}

    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_x, batch_idx in loader:
            batch_x = batch_x.to(device)
            target_len = token_subsample if use_token_subsample else seq_len

            with torch.amp.autocast(device_type=device.type, enabled=bool(scaler is not None)):
                logits = model(batch_x, return_logits=True)
                if use_token_subsample:
                    pos_idx = torch.randperm(seq_len, device=device)[:token_subsample]
                    logits = logits[:, pos_idx, :]
                batch_topk_idx = topk_idx[batch_idx].to(device)
                batch_topk_weights = topk_weights[batch_idx].to(device)
                soft_targets = torch.zeros(batch_x.shape[0], model.num_memory, device=device)
                soft_targets.scatter_add_(1, batch_topk_idx, batch_topk_weights)
                soft_targets = soft_targets.unsqueeze(1).expand(-1, target_len, -1)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(soft_targets * log_probs).sum(dim=-1).mean()

            nearest_idx = nearest[batch_idx].unsqueeze(1).expand(-1, target_len).to(device)
            acc = (logits.argmax(dim=-1) == nearest_idx).float().mean()
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += float(loss.item())
            epoch_acc += float(acc.item())

        avg_loss = epoch_loss / max(len(loader), 1)
        avg_acc = epoch_acc / max(len(loader), 1)
        loss_history["epoch"].append(epoch + 1)
        loss_history["total_loss"].append(avg_loss)
        loss_history["accuracy"].append(avg_acc * 100.0)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % int(log_every) == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch + 1}/{int(epochs)}, loss: {avg_loss:.4f}, acc: {avg_acc * 100.0:.1f}%, lr: {current_lr:.2e}")

        if patience_counter >= int(patience):
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, loss_history


def canonicalize_model_type(model_type: Any) -> str:
    if model_type is None:
        return "cnn3"
    if not isinstance(model_type, str):
        model_type = getattr(model_type, "__name__", str(model_type))
    normalized = model_type.lower().replace("-", "_")
    aliases = {"cnn": "cnn3", "cnn3": "cnn3", "resnet": "resnet18slim", "resnet18": "resnet18slim", "resnet18slim": "resnet18slim"}
    return aliases.get(normalized, normalized)


def model_type_display_name(model_type: Any) -> str:
    return {"cnn3": "CNN3", "resnet18slim": "ResNet18Slim"}.get(canonicalize_model_type(model_type), str(model_type))


def mean_std(values: Any) -> tuple[float, float]:
    finite_values = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return float("nan"), float("nan")
    mean = sum(finite_values) / len(finite_values)
    if len(finite_values) == 1:
        return float(mean), 0.0
    std = torch.tensor(finite_values, dtype=torch.float32).std(unbiased=True).item()
    return float(mean), float(std)


def summarize_histories(histories: list[list[float]]) -> tuple[list[float], list[float]]:
    if not histories:
        return [], []
    max_len = max(len(history) for history in histories)
    mean_history = []
    std_history = []
    for epoch_idx in range(max_len):
        mean, std = mean_std(history[epoch_idx] for history in histories if epoch_idx < len(history))
        mean_history.append(mean)
        std_history.append(std if math.isfinite(std) else 0.0)
    return mean_history, std_history


def format_lr(value: float) -> str:
    return f"{float(value):.3g}"


def uses_bn_conditioning(model_type: Any) -> bool:
    return canonicalize_model_type(model_type) == "resnet18slim"


def strip_known_arch_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in ("_resnet18slim", "_resnet18", "_resnet", "_cnn3", "_cnn"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_arch_family_from_name(name: str) -> str | None:
    lowered = Path(str(name)).name.lower()
    if "resnet18slim" in lowered or lowered.endswith("_resnet18") or lowered.endswith("_resnet"):
        return "resnet18slim"
    if lowered.endswith("_cnn3") or lowered.endswith("_cnn"):
        return "cnn3"
    return None


def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower().replace("_", "-")

    # Downstream callers may pass zoo directory names instead of the raw dataset name.
    if name.startswith("tune-zoo-"):
        name = name[9:]

        for suffix in ("-resnet18slim-05x", "-cnn3", "-cnn", "-uniform", "-kaiming"):
            if name.endswith(suffix):
                return name[: -len(suffix)]

    return name


# Zoo and dataset loading.
def get_zoo_path(dataset_name: str, model_type: str = "cnn3") -> Path | None:
    family = canonicalize_model_type(model_type)
    ds_name = normalize_dataset_name(dataset_name)
    ds_name_underscore = ds_name.replace("-", "_")

    if family == "resnet18slim":
        # Some zoos use dashes in the dataset folder, others use underscores in the tune_zoo suffix.
        for suffix in (ds_name, ds_name_underscore):
            candidate = Path(RESNET18SLIM_ZOO_BASE) / ds_name / "resnet18slim_05x" / f"tune_zoo_{suffix}_resnet18slim_05x"
            if candidate.exists():
                return candidate

        return None

    for suffix in (ds_name, ds_name_underscore):
        candidate = Path(CNN3_ZOO_BASE) / ds_name / "cnn3" / f"tune_zoo_{suffix}_cnn3"
        if candidate.exists():
            return candidate

    return None


def _load_pickle_safe_dataset_dict(dataset_pt: Path) -> dict[str, Any]:
    import __main__

    __main__.CachedDataset = CachedDataset
    __main__.GrayscaleToRGB = GrayscaleToRGB
    return torch.load(str(dataset_pt), weights_only=False)


def load_dataset_from_root_dir(root_dir: str | Path) -> tuple[Any, Any]:
    dataset_pt = resolve_dataset_pt_path(root_dir)

    if not dataset_pt.exists():
        return None, None

    try:
        dataset_dict = _load_pickle_safe_dataset_dict(dataset_pt)
    except Exception:
        return None, None

    return dataset_dict.get("trainset"), dataset_dict.get("testset")


def get_bn_conditioning_loader(root_dir: str | Path, batch_size: int = 128, num_workers: int = 4) -> DataLoader | None:
    dataset_pt = resolve_dataset_pt_path(root_dir)

    if not dataset_pt.exists():
        return None

    try:
        dataset_dict = _load_pickle_safe_dataset_dict(dataset_pt)
    except Exception:
        return None

    parts = [dataset_dict[key] for key in ("trainset", "valset", "testset") if dataset_dict.get(key) is not None]

    if not parts:
        return None

    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_train_test_loaders(dataset_name: str, batch_size: int = 128, num_workers: int = 4, root_dir: str | Path | None = None, model_type: str = "cnn3") -> tuple[DataLoader | None, DataLoader | None]:
    resolved_root = Path(root_dir) if root_dir is not None else get_zoo_path(dataset_name, model_type)

    if resolved_root is None:
        return None, None

    trainset, testset = load_dataset_from_root_dir(resolved_root)

    if trainset is None or testset is None:
        return None, None

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers), DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def detect_model_type(state_dict: dict[str, torch.Tensor]) -> tuple[type[nn.Module], str]:
    if "fc.weight" in state_dict and "layer1.0.conv1.weight" in state_dict:
        return ResNet18Slim, "fc"
    if "module_list.16.bias" in state_dict:
        return CNN3, "module_list.16"
    raise ValueError("Only CNN3 and ResNet18Slim are supported.")


def get_num_classes_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    _, final_layer_key = detect_model_type(state_dict)
    return int(state_dict[f"{final_layer_key}.bias"].shape[0])


def get_channels_in_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    if "module_list.0.weight" in state_dict:
        return int(state_dict["module_list.0.weight"].shape[1])
    if "conv1.weight" in state_dict:
        return int(state_dict["conv1.weight"].shape[1])
    raise ValueError("Cannot infer input channels from state_dict.")


def _infer_width_mult_from_state_dict(state_dict: dict[str, torch.Tensor]) -> float:
    return round(float(state_dict["conv1.weight"].shape[0]) / 64.0, 2)


def get_model_config_from_zoo(root_dir: str | Path | None) -> dict[str, Any] | None:
    if root_dir is None:
        return None

    params_paths = sorted(Path(root_dir).glob("NN_tune_trainable_seed=*/params.json"))

    if not params_paths:
        return None

    config = json.loads(params_paths[0].read_text())
    return config.get("train_loop_config", config)


def get_model_config_cached(root_dir: str | Path | None) -> dict[str, Any] | None:
    if root_dir is None:
        return None

    key = str(root_dir)

    if key not in _MODEL_CONFIG_CACHE:
        _MODEL_CONFIG_CACHE[key] = get_model_config_from_zoo(root_dir)

    return _MODEL_CONFIG_CACHE[key]


# Model construction and finetuning.
def create_model_from_config(model_type: str, channels_in: int, num_classes: int, config: dict[str, Any] | None = None, device: str = "cpu") -> nn.Module:
    cfg = dict(config or {})
    family = canonicalize_model_type(model_type)

    if family == "resnet18slim":
        model_cfg = {
            "model::type": "ResNet18Slim",
            "model::channels_in": channels_in,
            "model::o_dim": num_classes,
            "model::nlin": cfg.get("model::nlin", cfg.get("nlin", "relu")),
            "model::dropout": cfg.get("model::dropout", cfg.get("dropout", 0.0)),
            "model::init_type": cfg.get("model::init_type", cfg.get("init_type", "kaiming_uniform")),
            "model::width_mult": cfg.get("model::width_mult", cfg.get("width_mult", 0.5)),
        }
    else:
        model_cfg = {
            "model::type": "CNN3",
            "model::channels_in": channels_in,
            "model::o_dim": num_classes,
            "model::nlin": cfg.get("model::nlin", cfg.get("nlin", "gelu")),
            "model::dropout": cfg.get("model::dropout", cfg.get("dropout", 0.0)),
            "model::init_type": cfg.get("model::init_type", cfg.get("init_type", "kaiming_uniform")),
        }

    return get_model(model_cfg).to(device)


def create_model_from_state_dict(state_dict: dict[str, torch.Tensor], num_classes: int | None = None, config: dict[str, Any] | None = None, device: str = "cpu") -> nn.Module:
    model_class, _ = detect_model_type(state_dict)
    channels_in = get_channels_in_from_state_dict(state_dict)
    out_dim = get_num_classes_from_state_dict(state_dict) if num_classes is None else int(num_classes)
    resolved_config = dict(config or {})

    # Preserve the architecture family and width multiplier when rebuilding from zoo checkpoints.
    if model_class is ResNet18Slim:
        resolved_config.setdefault("model::type", "ResNet18Slim")
        resolved_config.setdefault("model::width_mult", _infer_width_mult_from_state_dict(state_dict))
    else:
        resolved_config.setdefault("model::type", "CNN3")

    return create_model_from_config(resolved_config["model::type"], channels_in, out_dim, resolved_config, device)


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = "cuda") -> float:
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            total += int(labels.numel())
            correct += int((preds == labels).sum().item())

    return 100.0 * correct / total if total else 0.0


def _get_finetune_defaults(model_type: str) -> tuple[float, float]:
    family = canonicalize_model_type(model_type)
    if family == "resnet18slim":
        return RESNET18SLIM_FINETUNE_LR, RESNET18SLIM_FINETUNE_WD
    return CNN3_FINETUNE_LR, CNN3_FINETUNE_WD


def _make_optimizer(model: nn.Module, model_type: str, lr: float, wd: float) -> torch.optim.Optimizer:
    family = canonicalize_model_type(model_type)
    if family == "resnet18slim":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=RESNET18SLIM_MOMENTUM)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int, device: str, model_type: str, lr: float | None = None, wd: float | None = None) -> tuple[float, list[float], nn.Module]:
    default_lr, default_wd = _get_finetune_defaults(model_type)
    optimizer = _make_optimizer(model, model_type, default_lr if lr is None else lr, default_wd if wd is None else wd)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []
    model = model.to(device)

    for _ in range(int(epochs)):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Keep history in evaluation-space so the caller can report epoch 1 and final accuracy directly.
        history.append(evaluate_model(model, test_loader, device))

    return (history[-1] if history else 0.0), history, model


def finetune_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 1, device: str = "cuda", model_type: str = "cnn3", lr: float | None = None, wd: float | None = None) -> tuple[float, list[float], nn.Module]:
    family = canonicalize_model_type(model_type)
    return _train_model(model, train_loader, test_loader, epochs, device, family, lr=lr, wd=wd)


def finetune_state_dict(state_dict: dict[str, torch.Tensor], num_classes: int, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 1, device: str = "cuda", config: dict[str, Any] | None = None, model_type: str | None = None, lr: float | None = None, wd: float | None = None) -> tuple[float, list[float], nn.Module]:
    family = canonicalize_model_type(model_type or detect_model_type(state_dict)[0].__name__)
    state_num_classes = get_num_classes_from_state_dict(state_dict)
    if state_num_classes != int(num_classes):
        raise ValueError(f"finetune_state_dict expects classifier width {num_classes}, got {state_num_classes}; adapt the decoded state_dict before finetuning")
    model = create_model_from_state_dict(state_dict, num_classes=num_classes, config=config, device=device)
    model.load_state_dict(state_dict, strict=False)

    return _train_model(model, train_loader, test_loader, epochs, device, family, lr=lr, wd=wd)


def train_from_scratch(num_classes: int, channels_in: int, train_loader: DataLoader, test_loader: DataLoader, epochs: int, device: str = "cuda", model_type: str = "cnn3", lr: float | None = None, wd: float | None = None, init_type: str | None = None) -> tuple[float, list[float], nn.Module]:
    model_config = {"model::init_type": init_type} if init_type is not None else None
    model = create_model_from_config(model_type, channels_in, num_classes, config=model_config, device=device)
    return _train_model(model, train_loader, test_loader, epochs, device, model_type, lr=lr, wd=wd)


def load_checkpoint_from_zoo(dataset_name: str, model_type: str = "cnn3") -> dict[str, torch.Tensor] | None:
    zoo_path = get_zoo_path(dataset_name, model_type)

    if zoo_path is None:
        return None

    # Support both plain .pt exports and older nested checkpoint folders.
    checkpoint_files = [path for path in zoo_path.glob("NN_tune_trainable_seed=*/checkpoint_*/*.pt") if path.name != "dataset.pt"]

    if not checkpoint_files:
        checkpoint_files = list(zoo_path.glob("NN_tune_trainable_seed=*/checkpoint_*/checkpoints"))

    if not checkpoint_files:
        return None

    state = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)

    # Zoo checkpoints are not fully standardized, so unwrap the common container keys here.
    if isinstance(state, dict):
        state = state.get("model_state_dict") or state.get("model_state") or state.get("state_dict") or state.get("model") or state

    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    return state if isinstance(state, dict) else None


def get_num_classes(dataset_name: str, model_type: str = "cnn3") -> int:
    normalized = normalize_dataset_name(dataset_name)

    if normalized in DATASET_CLASS_COUNTS:
        return DATASET_CLASS_COUNTS[normalized]

    checkpoint = load_checkpoint_from_zoo(normalized, model_type)
    return get_num_classes_from_state_dict(checkpoint) if checkpoint is not None else 10


def _load_trial_config(trial_dir: Path) -> dict[str, Any]:
    trial_yaml = trial_dir / "trial.yaml"
    if trial_yaml.exists():
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(str(trial_yaml)), resolve=True)
    params_json = trial_dir / "params.json"
    if params_json.exists():
        config = json.loads(params_json.read_text())
        return config.get("train_loop_config", config)
    raise FileNotFoundError(f"No config found in {trial_dir}.")


def _resolve_chkpt_cfg(cfg: Any) -> Any:
    if isinstance(cfg, list):
        return [_resolve_chkpt_cfg(item) for item in cfg]

    if isinstance(cfg, str):
        from omegaconf import OmegaConf

        # String configs in the saved trial point back to the repo config directory.
        config_path = Path(sane.__file__).parent.parent.parent / "config" / "data" / "checkpoints_dataset" / f"{cfg}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Checkpoints dataset config not found: {config_path}")
        raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        raw_cfg.setdefault("name", cfg)
        cfg = raw_cfg

    aug_cfg = cfg.get("augmentations")

    if aug_cfg is not None and isinstance(aug_cfg, dict):
        aug_cfg = from_dict(AugmentationConfig, aug_cfg)

    return from_dict(ZooDatasetConfig, cfg, augmentations=aug_cfg)


def _extract_batch_tensors(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["tokens"], batch["mask"], batch["position"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(f"Unsupported batch type: {type(batch)!r}")


def _infer_n_tokens(loader: DataLoader) -> int:
    batch = next(iter(loader))
    tokens, _, _ = _extract_batch_tensors(batch)
    return int(tokens.shape[-2])


def _infer_max_positions(loader: DataLoader) -> list[int]:
    max_pos: list[int] | None = None

    for batch in loader:
        _, _, position = _extract_batch_tensors(batch)
        batch_max = torch.amax(position, dim=tuple(range(position.ndim - 1))).int().tolist()
        max_pos = batch_max if max_pos is None else [max(a, b) for a, b in zip(max_pos, batch_max)]

    if max_pos is None:
        raise ValueError("Could not infer max_positions from an empty loader.")

    return [int(value) + 1 for value in max_pos]


# SANE checkpoint restore.
def load_sane_bundle(checkpoint_path: str | Path, device: str = "cuda") -> tuple[SANEAutoEncoder, Any, Any, Any, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)

    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / "checkpoint.pt"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    config = state.get("config") or _load_trial_config(checkpoint_file.parent.parent)
    data_cfg = copy.deepcopy(config["data"])
    tokens_cfg = copy.deepcopy(data_cfg["tokens_dataset"])
    tokens_cfg["data_config_for_hash"] = copy.deepcopy(data_cfg)
    tokens_cfg["read_only"] = True

    # Rebuild the original token dataset locally instead of depending on the trainer restore path.
    factory_cfg = DatasetFactoryConfig(
        checkpoints_dataset=_resolve_chkpt_cfg(data_cfg["checkpoints_dataset"]),
        tokenizer=from_dict(TokenizerConfig, data_cfg["tokenizer"]),
        tokens_dataset=from_dict(CachedWindowedDatasetConfig, tokens_cfg),
        split=from_dict(SplitConfig, data_cfg["split"]),
        dataloader=from_dict(DataLoaderConfig, data_cfg),
        splitter=data_cfg.get("splitter", "RandomSplitter"),
    )

    trainloader, valloader, testloader = DatasetFactory().build(factory_cfg)
    trainset = trainloader.dataset
    valset = valloader.dataset if valloader is not None else None
    testset = testloader.dataset if testloader is not None else None
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["device"] = device

    # Older configs may leave these on "auto" and expect them to be inferred from the loader.
    if model_cfg.get("n_tokens") == "auto":
        model_cfg["n_tokens"] = _infer_n_tokens(trainloader)

    if model_cfg.get("max_positions") == "auto":
        model_cfg["max_positions"] = _infer_max_positions(trainloader)

    sane_ae = SANEAutoEncoder.from_config(model_cfg)
    model_state = state.get("model") or state.get("state_dict") or state

    if any(key.startswith("module.") for key in model_state):
        model_state = {key.replace("module.", "", 1): value for key, value in model_state.items()}

    sane_ae.load_state_dict(model_state)
    sane_ae.to(device)
    sane_ae.eval()
    tokenizer = (valset or testset or trainset).tokenizer

    return sane_ae, trainset, valset, tokenizer, config


# Dataset encoder restore.
def load_dataset_encoder(checkpoint_path: str | Path, device: str = "cuda", config: dict | None = None) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state") or checkpoint
    is_new_arch = any(key.startswith("feature_extractor.") or key.startswith("set_encoder.") for key in state_dict)

    if not is_new_arch:
        raise ValueError("Only unified DatasetEncoder checkpoints are supported by the slim downstream script. Legacy external deepsets checkpoints are no longer supported.")

    # External config (e.g. the SANE checkpoint's `dataset_encoder` block) takes precedence over
    # checkpoint-embedded metadata, which is typically missing because dataset_encoder.pt is saved
    # as a bare state_dict.
    cfg = config or {}
    num_classes = state_dict["classifier.weight"].shape[0] if "classifier.weight" in state_dict else 10
    preset = cfg.get("preset") or checkpoint.get("encoder_preset")

    if preset in ENCODER_PRESETS:
        encoder_cfg = dict(ENCODER_PRESETS[preset])
    else:
        # Fall back to inferring the preset from the saved module names.
        has_resnet = any(key.startswith("feature_extractor.backbone.") for key in state_dict)
        has_set_transformer = any(key.startswith("set_encoder.isab") or key.startswith("set_encoder.pma") for key in state_dict)
        preset_name = f"{'set_transformer' if has_set_transformer else 'deepsets'}_{'resnet' if has_resnet else 'conv'}"
        encoder_cfg = dict(ENCODER_PRESETS[preset_name])

    embedding_dim = cfg.get("embedding_dim", checkpoint.get("embedding_dim", state_dict["classifier.weight"].shape[1]))

    encoder_cfg["num_classes"] = num_classes
    encoder_cfg["embedding_dim"] = embedding_dim
    encoder_cfg["normalize_embeddings"] = cfg.get("normalize_embeddings", checkpoint.get("normalize_embeddings", False))
    encoder_cfg["embedding_scale"] = cfg.get("embedding_scale", checkpoint.get("embedding_scale", 1.0))
    
    if "input_channels" in cfg:
        encoder_cfg["input_channels"] = cfg["input_channels"]

    if encoder_cfg.get("feature_extractor") == "resnet":
        encoder_cfg.setdefault("fe_kwargs", {})
        encoder_cfg["fe_kwargs"] = dict(encoder_cfg["fe_kwargs"])
        encoder_cfg["fe_kwargs"]["pretrained"] = False

    model = build_dataset_encoder(encoder_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# Fixed-window token helpers.
def _flatten_tokenized(tokenized: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(tokenized, tuple):
        return tokenized

    # Layer-wise tokenizers return one tuple per layer; stitch them into the full-model view.
    tokens = torch.cat([item[0] for item in tokenized], dim=0)
    mask = torch.cat([item[1] for item in tokenized], dim=0)
    position = torch.cat([item[2] for item in tokenized], dim=0)

    return tokens, mask, position


def _get_sane_window_size(sane_model: SANEAutoEncoder) -> int:
    max_positions = getattr(getattr(sane_model.encoder, "position_embedding", None), "max_positions", None)

    if isinstance(max_positions, list) and max_positions:
        return int(max_positions[0])

    n_tokens = getattr(sane_model.encoder, "n_tokens", None)

    if n_tokens is None:
        raise ValueError("Could not infer SANE window size.")

    return int(n_tokens)


def _slice_sane_window(tokens: torch.Tensor, mask: torch.Tensor, position: torch.Tensor, window_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens.shape[0] <= window_size:
        return tokens, mask, position

    return tokens[:window_size], mask[:window_size], position[:window_size]


def encode_model(model_or_state_dict: Any, sane_model: SANEAutoEncoder, tokenizer: Any, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_dict = model_or_state_dict.state_dict() if isinstance(model_or_state_dict, nn.Module) else model_or_state_dict
    full_tokens, full_mask, full_position = _flatten_tokenized(tokenizer.tokenize(state_dict))

    # Some SANE checkpoints were trained on one fixed token window rather than the full model.
    tokens, mask, position = _slice_sane_window(full_tokens, full_mask, full_position, _get_sane_window_size(sane_model))
    tokens = tokens.unsqueeze(0).to(device)
    position = position.unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = sane_model.encoder(tokens, position)

    return embedding.mean(dim=1).squeeze(0).cpu(), embedding.cpu(), full_tokens.unsqueeze(0).cpu(), full_mask.unsqueeze(0).cpu(), full_position.unsqueeze(0).cpu()


def _adapt_output_layer(state_dict: dict[str, torch.Tensor], target_state_dict: dict[str, torch.Tensor], final_layer_key: str) -> dict[str, torch.Tensor]:
    weight_key = f"{final_layer_key}.weight"
    bias_key = f"{final_layer_key}.bias"
    weight = state_dict[weight_key]
    bias = state_dict[bias_key]
    target_weight = target_state_dict[weight_key].clone()
    target_bias = target_state_dict[bias_key].clone()
    copy_count = min(int(weight.shape[0]), int(target_weight.shape[0]))

    target_weight[:copy_count] = weight[:copy_count]
    target_bias[:copy_count] = bias[:copy_count]
    state_dict[weight_key] = target_weight
    state_dict[bias_key] = target_bias
    return state_dict


def describe_decode_preflight(sane_model: SANEAutoEncoder, anchor_state_dict: dict[str, torch.Tensor], anchor_tokens: torch.Tensor, target_num_classes: int, predicted_seq_len: int) -> dict[str, Any]:
    full_tokens = anchor_tokens[0].cpu() if anchor_tokens.dim() == 3 else anchor_tokens.cpu()
    full_seq_len = int(full_tokens.shape[0])
    sane_window_size = int(_get_sane_window_size(sane_model))
    anchor_num_classes = int(get_num_classes_from_state_dict(anchor_state_dict))
    expands_classifier = int(target_num_classes) > anchor_num_classes
    decode_possible = full_seq_len <= sane_window_size and int(predicted_seq_len) >= full_seq_len

    if target_num_classes < anchor_num_classes:
        output_layer_action = "slice"
    elif target_num_classes == anchor_num_classes:
        output_layer_action = "keep"
    else:
        output_layer_action = "expand_random_init"

    return {
        "predicted_seq_len": int(predicted_seq_len),
        "anchor_seq_len": full_seq_len,
        "sane_window_size": sane_window_size,
        "anchor_num_classes": anchor_num_classes,
        "target_num_classes": int(target_num_classes),
        "decode_possible": decode_possible,
        "anchor_token_reuse": False,
        "random_weights_added": expands_classifier,
        "output_layer_action": output_layer_action,
        "decode_reason": (
            (
                "full anchor token sequence fits the SANE window and is fully covered by the mapper output"
                if not expands_classifier
                else f"full anchor token sequence fits the SANE window and is fully covered by the mapper output; classifier expands from {anchor_num_classes} to {int(target_num_classes)} and extra output rows use model initialization"
            )
            if decode_possible
            else (
                f"anchor token length {full_seq_len} exceeds SANE window size {sane_window_size}"
                if full_seq_len > sane_window_size
                else f"mapper predicts {int(predicted_seq_len)} tokens but target requires {full_seq_len}"
            )
        ),
    }


def _condition_bn_state_dict(state_dict: dict[str, torch.Tensor], model_template: nn.Module, dataloader: DataLoader, device: str, iterations: int = 200) -> dict[str, torch.Tensor]:
    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.train()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= iterations:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = model(images.to(device))

    model.eval()
    return model.cpu().state_dict()


def decode_embedding(sane_model: SANEAutoEncoder, embedding: torch.Tensor, anchor_state_dict: dict[str, torch.Tensor], anchor_tokens: torch.Tensor, anchor_mask: torch.Tensor, anchor_pos: torch.Tensor, tokenizer: Any, num_classes: int, device: str = "cuda", config: dict[str, Any] | None = None, bn_dataloader: DataLoader | None = None) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    if embedding.dim() == 2:
        embedding = embedding.unsqueeze(0)

    full_tokens = anchor_tokens[0].cpu() if anchor_tokens.dim() == 3 else anchor_tokens.cpu()
    full_mask = anchor_mask[0].cpu() if anchor_mask.dim() == 3 else anchor_mask.cpu()
    full_pos = anchor_pos[0].cpu() if anchor_pos.dim() == 3 else anchor_pos.cpu()
    window_size = _get_sane_window_size(sane_model)
    full_seq_len = int(full_tokens.shape[0])

    if full_seq_len > int(window_size):
        raise ValueError(f"Anchor token sequence length {full_seq_len} exceeds SANE window size {int(window_size)}. Refusing to keep anchor-token suffixes during detokenization.")

    if embedding.shape[1] < full_seq_len:
        raise ValueError(f"Predicted embedding length {int(embedding.shape[1])} is shorter than the full anchor token length {full_seq_len}. Refusing to keep anchor-token suffixes during detokenization.")

    if embedding.shape[1] > full_seq_len:
        embedding = embedding[:, :full_seq_len, :]

    decode_pos = full_pos[: embedding.shape[1]]

    with torch.inference_mode():
        decoded_tokens = sane_model.decoder(embedding.to(device), decode_pos.unsqueeze(0).to(device))

    if int(decoded_tokens.shape[1]) != full_seq_len:
        raise ValueError(f"Decoder returned {int(decoded_tokens.shape[1])} tokens, but the anchor requires {full_seq_len}.")

    # The anchor state dict is only a shape/template reference here; all token values come from the decoder.
    decoded_state = tokenizer.detokenize(decoded_tokens[0].cpu(), mask=full_mask, position=full_pos, reference_statedict=anchor_state_dict)
    model_class, final_layer_key = detect_model_type(anchor_state_dict)

    if model_class is ResNet18Slim and bn_dataloader is not None:
        # Recompute BN statistics after decoding so ResNet predictions are not dominated by stale running stats.
        template = create_model_from_state_dict(anchor_state_dict, num_classes=get_num_classes_from_state_dict(anchor_state_dict), config=config, device=device)
        decoded_state = _condition_bn_state_dict(decoded_state, template, bn_dataloader, device)

    model = create_model_from_state_dict(anchor_state_dict, num_classes=num_classes, config=config, device=device)
    decoded_state = _adapt_output_layer(decoded_state, model.state_dict(), final_layer_key)
    model.load_state_dict(decoded_state, strict=False)
    model.eval()

    return model, decoded_state


# Dataset-embedding extraction.
def _get_sampler(subset_size: int, image_size: tuple[int, int] = (32, 32)) -> DatasetPtImageSetSampler:
    subset_size = int(subset_size)
    image_size = (int(image_size[0]), int(image_size[1]))
    key = (subset_size, image_size)

    if key not in _IMAGE_SAMPLERS:
        _IMAGE_SAMPLERS[key] = DatasetPtImageSetSampler(split="trainset", set_size=subset_size, target_image_size=image_size, target_channels=3)

    return _IMAGE_SAMPLERS[key]


def _encode_dataset_batch(dataset_encoder: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    if hasattr(dataset_encoder, "encode"):
        return dataset_encoder.encode(batch)
    return dataset_encoder.get_embeddings(batch)


def extract_dataset_embeddings_cached(dataset_encoder: nn.Module, dataset_name: str, num_subsets: int, subset_size: int, device: str = "cuda", cache_dir: str | Path | None = None, root_dir: str | Path | None = None, image_size: tuple[int, int] = (32, 32), seed: int | None = None) -> torch.Tensor | None:
    image_size = (int(image_size[0]), int(image_size[1]))
    cache_path = None
    if root_dir is not None:
        cache_source = resolve_dataset_pt_path(root_dir)
        cache_key = cache_source.parent.name
    else:
        cache_key = dataset_name

    if cache_dir is not None:
        size_suffix = "" if image_size == (32, 32) else f"_{image_size[0]}x{image_size[1]}"
        seed_suffix = "" if seed is None else f"_seed{int(seed)}"
        cache_path = Path(cache_dir) / f"dataset_emb_{cache_key}_{num_subsets}_{subset_size}{size_suffix}{seed_suffix}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

    resolved_root = Path(root_dir) if root_dir is not None else get_zoo_path(dataset_name)

    if resolved_root is None:
        return None

    sampler = _get_sampler(subset_size, image_size=image_size)
    images = sampler.load_image_pool(str(resolved_root))

    if not images:
        return None

    image_tensor = torch.stack(images[: min(len(images), 10000)])
    total_images = int(image_tensor.shape[0])
    subset_size = min(int(subset_size), total_images)
    outputs: list[torch.Tensor] = []
    batch_size = min(16, max(1, int(num_subsets)))
    dataset_encoder.eval()

    generator = None

    for start in range(0, int(num_subsets), batch_size):
        count = min(batch_size, int(num_subsets) - start)
        subsets = []

        # Each subset is a small sampled image set that stands in for one dataset embedding query.
        for _ in range(count):
            indices = torch.randperm(total_images, generator=generator)[:subset_size] if generator is not None else torch.randperm(total_images)[:subset_size]
            subsets.append(image_tensor[indices])

        batch = torch.stack(subsets).to(device)

        with torch.inference_mode():
            outputs.append(_encode_dataset_batch(dataset_encoder, batch).cpu())

    embeddings = torch.cat(outputs, dim=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)

    return embeddings


def extract_training_embeddings(sane_model: SANEAutoEncoder, trainset: Any, max_per_dataset: int = 50, device: str = "cuda", arch_filter: str | None = None) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, torch.Tensor]]], dict[str, str]]:
    checkpoints_ds = trainset.checkpoints_dataset

    if isinstance(checkpoints_ds, Subset):
        base_dataset = checkpoints_ds.dataset
        available_indices = list(checkpoints_ds.indices)
    else:
        base_dataset = checkpoints_ds
        available_indices = list(range(len(base_dataset)))

    if not isinstance(base_dataset, CombinedCheckpointsDataset):
        raise ValueError("Expected CombinedCheckpointsDataset for training embeddings.")

    tokenizer = trainset.tokenizer
    arch_filter = canonicalize_model_type(arch_filter) if arch_filter else None
    grouped: dict[str, dict[str, Any]] = {}
    cumulative_sizes = list(base_dataset._cumulative_sizes)

    # Group checkpoints by source zoo so the direct decoder is trained on dataset-level pairs.
    for underlying_idx in available_indices:
        zoo_idx = next(idx for idx, end in enumerate(cumulative_sizes) if underlying_idx < end)
        zoo_ds = base_dataset.datasets[zoo_idx]
        family = infer_arch_family_from_name(zoo_ds.root_dir)

        if arch_filter is not None and canonicalize_model_type(family) != arch_filter:
            continue

        ds_name = Path(zoo_ds.root_dir).name.lower()
        entry = grouped.setdefault(ds_name, {"indices": [], "root_dir": zoo_ds.root_dir})

        if len(entry["indices"]) < int(max_per_dataset):
            entry["indices"].append(int(underlying_idx))

    embeddings_by_dataset: dict[str, torch.Tensor] = {}
    state_dicts_by_dataset: dict[str, list[dict[str, torch.Tensor]]] = {}
    root_dir_by_dataset: dict[str, str] = {}

    for ds_name, entry in grouped.items():
        embeddings = []
        state_dicts = []

        for underlying_idx in tqdm(entry["indices"], desc=f"encode:{ds_name}", leave=False):
            # CombinedCheckpointsDataset may return a checkpoint directly or a list of augmented views.
            sample = base_dataset[underlying_idx]
            sample = sample[0] if isinstance(sample, tuple) else sample
            checkpoint = sample[0] if isinstance(sample, list) else sample
            state_dict = checkpoint.model.state_dict()
            _, embedding, _, _, _ = encode_model(state_dict, sane_model, tokenizer, device)
            embeddings.append(embedding.squeeze(0))
            state_dicts.append(state_dict)

        if embeddings:
            embeddings_by_dataset[ds_name] = torch.stack(embeddings)
            state_dicts_by_dataset[ds_name] = state_dicts
            root_dir_by_dataset[ds_name] = str(entry["root_dir"])

    return embeddings_by_dataset, state_dicts_by_dataset, root_dir_by_dataset