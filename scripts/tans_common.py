#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
for _p in (_repo_root / "src", _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ood_utils import (
    canonicalize_model_type,
    create_model_from_state_dict,
    format_lr,
    get_model_config_cached,
    get_num_classes as _get_num_classes,
    get_zoo_path as _get_zoo_path,
    load_dataset_from_root_dir,
    mean_std,
    model_type_display_name,
    summarize_histories,
)

DEFAULT_FINETUNE_LR_CNN3 = 1e-3
DEFAULT_FINETUNE_LR_RESNET = 1e-4
DEFAULT_TEST_SEEDS = [42, 0, 777]

_SPLITS = {
    "cnn3": {
        "metatrain": (
            "proptit-aif-homework", "cactus-aerial", "dl2020", "four-shapes",
            "e4040-assignment", "car-classification", "simpsons-characters",
            "simpsons-challenge", "breakhis", "mushrooms", "artworks",
            "numta-bengali", "lego-bricks", "ads5035", "day3-kaggle",
            "ct-images", "land-cover", "casting", "asl", "cassava-leaf",
        ),
        "metatest": (
            "colorectal-histology", "covid19", "cifar10",
            "speed-limit-signs", "honeybee-pollen", "real-or-drawing",
        ),
    },
    "resnet18slim": {
        "metatrain": (
            "cactus-aerial", "lego-vs-generic", "asl", "breakhis", "artworks",
            "blood-cells", "ct-images", "land-cover", "casting", "cassava-leaf",
        ),
        "metatest": (
            "colorectal-histology", "cifar10",
            
        ),
    },
}


# seed & IO

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mkdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(directory, filename, payload):
    out = _mkdir(directory) / filename
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_torch(directory, filename, payload):
    out = _mkdir(directory) / filename
    torch.save(payload, out)
    return out


def load_torch(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# dataset splits

def _arch(architecture):
    return canonicalize_model_type(architecture or "cnn3")


def get_metatrain_datasets(architecture=None):
    a = _arch(architecture)
    if a not in _SPLITS:
        raise ValueError(f"No TANS split for: {a}")
    return _SPLITS[a]["metatrain"]


def get_metatest_datasets(architecture=None):
    a = _arch(architecture)
    if a not in _SPLITS:
        raise ValueError(f"No TANS split for: {a}")
    return _SPLITS[a]["metatest"]


# zoo data loading

def get_zoo_path(dataset_name, architecture=None):
    return _get_zoo_path(dataset_name, model_type=_arch(architecture))


def get_num_classes(dataset_name, architecture=None):
    return _get_num_classes(dataset_name, model_type=_arch(architecture))


def load_dataset_pt(dataset_name, architecture=None):
    path = get_zoo_path(dataset_name, architecture)
    return (None, None) if path is None else load_dataset_from_root_dir(path)


def load_models_from_zoo(dataset_name, max_models=100, device="cpu", architecture=None):
    arch = _arch(architecture)
    zoo_path = get_zoo_path(dataset_name, arch)
    if zoo_path is None:
        return None, None, None

    dirs = sorted(glob.glob(str(zoo_path / "NN_tune_trainable_seed=*")))
    n_classes = get_num_classes(dataset_name, arch)
    config = get_model_config_cached(zoo_path)
    ms, accs, sds = [], [], []

    for d in dirs[:max_models]:
        ckpts = sorted(glob.glob(os.path.join(d, "checkpoint_*/checkpoints")))
        if not ckpts:
            continue
        try:
            sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
            m = create_model_from_state_dict(sd, num_classes=n_classes, config=config, device=device)
            m.load_state_dict(sd, strict=False)
            m.eval()
        except Exception:
            continue
        ms.append(m)
        sds.append(sd)
        accs.append(_read_acc(Path(d) / "progress.csv"))

    return (ms, accs, sds) if ms else (None, None, None)


def _read_acc(progress_csv):
    if not progress_csv.exists():
        return 0.9
    try:
        import pandas as pd
        df = pd.read_csv(progress_csv)
        for col in ("val_accuracy", "val_acc", "accuracy"):
            if col in df.columns:
                return float(df[col].iloc[-1])
    except Exception:
        pass
    return 0.9


# arg helpers

def get_default_finetune_lr(arch: str) -> float:
    a = canonicalize_model_type(arch or "cnn3")
    return DEFAULT_FINETUNE_LR_RESNET if a == "resnet18slim" else DEFAULT_FINETUNE_LR_CNN3


def resolve_seeds(args):
    seeds = [int(s) for s in (getattr(args, "seeds", None) or DEFAULT_TEST_SEEDS)]
    return seeds or list(DEFAULT_TEST_SEEDS)


# reporting


def print_single_seed_summary(results, arch, n_epochs):
    rows = [(name, result["selected"]) for name, result in sorted(results.items()) if result.get("selected")]
    if not rows:
        return
    print(f"\nTANS summary ({model_type_display_name(arch)})")
    for name, stats in rows:
        print(f"  {name:<25}  ep0={float(stats['zero_shot_acc']) * 100:6.2f}  ep1={float(stats['epoch_1_acc']) * 100:6.2f}  ep{n_epochs}={float(stats['final_acc']) * 100:6.2f}  src={stats['source']}")
    zero_mean = sum(100.0 * float(stats["zero_shot_acc"]) for _, stats in rows) / len(rows)
    epoch_1_mean = sum(100.0 * float(stats["epoch_1_acc"]) for _, stats in rows) / len(rows)
    final_mean = sum(100.0 * float(stats["final_acc"]) for _, stats in rows) / len(rows)
    print(f"  {'Average':<25}  ep0={zero_mean:6.2f}  ep1={epoch_1_mean:6.2f}  ep{n_epochs}={final_mean:6.2f}")


def print_multi_seed_summary(aggregate, arch):
    n_epochs = int(aggregate["n_finetune_epochs"])
    print(f"\nTANS multi-seed summary ({model_type_display_name(arch)}, seeds={len(aggregate['seeds'])})")
    for name, stats in aggregate["per_dataset"].items():
        line = f"  {name:<25}  ep0={stats['zero_shot_mean']:5.2f}±{stats['zero_shot_std']:<5.2f}  ep1={stats['epoch_means'][0]:5.2f}±{stats['epoch_stds'][0]:<5.2f}"
        if len(stats["epoch_means"]) >= n_epochs:
            line += f"  ep{n_epochs}={stats['epoch_means'][n_epochs - 1]:5.2f}±{stats['epoch_stds'][n_epochs - 1]:<5.2f}"
        print(line)
    grand = aggregate["grand_average"]
    line = f"  {'Average':<25}  ep0={grand['zero_shot_mean']:5.2f}±{grand['zero_shot_std']:<5.2f}"
    if grand["epoch_means"]:
        line += f"  ep1={grand['epoch_means'][0]:5.2f}±{grand['epoch_stds'][0]:<5.2f}"
    if len(grand["epoch_means"]) >= n_epochs:
        line += f"  ep{n_epochs}={grand['epoch_means'][n_epochs - 1]:5.2f}±{grand['epoch_stds'][n_epochs - 1]:<5.2f}"
    print(line)


def aggregate_seed_results(results_by_seed, n_finetune_epochs):
    raw = {}
    for seed, payload in results_by_seed.items():
        for name, result in payload.get("datasets", {}).items():
            sel = result.get("selected")
            if sel is None:
                continue
            raw.setdefault(name, {})[int(seed)] = {
                "source": sel["source"],
                "zero_shot": float(sel["zero_shot_acc"]),
                "epoch_1": float(sel["epoch_1_acc"]),
                "final": float(sel["final_acc"]),
                "acc_history": [float(value) for value in sel.get("acc_history", [])],
            }

    per_dataset = {}
    grand_zero, grand_hist = [], []
    for name in sorted(raw):
        runs = list(raw[name].values())
        zeroes = [100.0 * run["zero_shot"] for run in runs]
        histories = [[100.0 * value for value in run["acc_history"]] for run in runs]
        zero_mean, zero_std = mean_std(zeroes)
        epoch_means, epoch_stds = summarize_histories(histories)
        per_dataset[name] = {
            "n_seeds": len(runs),
            "sources": [run["source"] for run in runs],
            "zero_shot_mean": zero_mean,
            "zero_shot_std": zero_std,
            "epoch_means": epoch_means,
            "epoch_stds": epoch_stds,
        }
        grand_zero.extend(zeroes)
        grand_hist.extend(histories)

    zero_mean, zero_std = mean_std(grand_zero)
    epoch_means, epoch_stds = summarize_histories(grand_hist)
    return {
        "seeds": sorted(int(seed) for seed in results_by_seed),
        "n_finetune_epochs": int(n_finetune_epochs),
        "per_dataset": per_dataset,
        "grand_average": {
            "zero_shot_mean": zero_mean,
            "zero_shot_std": zero_std,
            "epoch_means": epoch_means,
            "epoch_stds": epoch_stds,
        },
        "raw_results": raw,
    }


def format_startup_line(label, values, *, format_as_lr=False):
    if isinstance(values, list):
        values = ", ".join(format_lr(float(value)) for value in values) if format_as_lr else ", ".join(str(value) for value in values)
    return f"{label}: {values}"
