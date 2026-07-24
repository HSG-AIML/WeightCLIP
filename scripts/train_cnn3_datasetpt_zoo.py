#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sane.data.datasets.zoo_dataset_models import CNN3


DEFAULT_OUTPUT_ZOO_ROOT = Path("/local/zoos/metatrain_cnn3")
DEFAULT_LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3]


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset | None = None, transform: Any | None = None):
        self.transform = transform
        self.data: torch.Tensor | list[torch.Tensor] = []
        self.targets: torch.Tensor | list[int] = []

        if dataset is not None:
            data: list[torch.Tensor] = []
            targets: list[int] = []
            for idx in range(len(dataset)):
                image, label = dataset[idx]
                data.append(image)
                targets.append(int(label))
            self.data = torch.stack(data) if data else torch.empty(0)
            self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_batch_iterator(self, batch_size: int, shuffle: bool = True, device: str = "cuda"):
        n_items = len(self)
        indices = torch.randperm(n_items) if shuffle else torch.arange(n_items)
        for start in range(0, n_items, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            batch_data = self.data[batch_idx].to(device, non_blocking=True)
            batch_targets = self.targets[batch_idx].to(device, non_blocking=True)
            yield batch_data, batch_targets


class GrayscaleToRGB:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        return tensor


def canonical_dataset_name(dataset_name: str) -> str:
    base_name = dataset_name[:-5] if dataset_name.endswith("_cnn3") else dataset_name
    return base_name.replace("_", "-")


def resolve_output_root(dataset_name: str, output_zoo_root: str | Path) -> Path:
    dash_name = canonical_dataset_name(dataset_name)
    return Path(output_zoo_root) / dash_name / "cnn3" / f"tune_zoo_{dash_name}_cnn3"


def find_dataset_pt_in_root(dataset_name: str, dataset_pt_root: str | Path) -> Path:
    """Locate ``<root>/<name>/dataset.pt`` built by build_dataset_pts.py.

    Tries the canonical dash name, its underscore form, and the full dataset id.
    """
    dash_name = canonical_dataset_name(dataset_name)
    candidates = [dash_name, dash_name.replace("-", "_"), dataset_name]
    root = Path(dataset_pt_root)
    for name in candidates:
        candidate = root / name / "dataset.pt"
        if candidate.exists():
            return candidate
    return root / candidates[0] / "dataset.pt"


def resolve_source_dataset_pt(args: argparse.Namespace) -> Path:
    """Resolve the source dataset.pt from --dataset-pt or --dataset-pt-root."""
    if args.dataset_pt is not None:
        path = Path(args.dataset_pt)
        return path if path.name == "dataset.pt" else path / "dataset.pt"
    if args.dataset_pt_root is not None:
        return find_dataset_pt_in_root(args.dataset, args.dataset_pt_root)
    raise ValueError("Provide --dataset-pt or --dataset-pt-root (build files with scripts/build_dataset_pts.py)")


def register_pickle_compat_types() -> None:
    import __main__

    __main__.CachedDataset = CachedDataset
    __main__.GrayscaleToRGB = GrayscaleToRGB


def load_dataset_dict(dataset_pt: str | Path) -> dict[str, Any]:
    register_pickle_compat_types()
    payload = torch.load(str(dataset_pt), weights_only=False)
    required = ("trainset", "valset", "testset")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"dataset.pt is missing required splits: {missing}")
    return payload


def copy_dataset_pt(source_dataset_pt: str | Path, output_root: str | Path) -> Path:
    target = Path(output_root) / "dataset.pt"
    if not target.exists():
        shutil.copy2(source_dataset_pt, target)
    return target


def _targets_for_dataset(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        base_targets = _targets_for_dataset(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base_targets[indices]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError(f"Dataset of type {type(dataset).__name__} does not expose targets")
    return torch.as_tensor(targets, dtype=torch.long)


def infer_num_classes(dataset_dict: dict[str, Any]) -> int:
    split_targets = [_targets_for_dataset(dataset_dict[key]) for key in ("trainset", "valset", "testset")]
    return int(torch.unique(torch.cat(split_targets)).numel())


def infer_channels_in(dataset_dict: dict[str, Any]) -> int:
    sample, _ = dataset_dict["trainset"][0]
    return int(sample.shape[0])


def dataset_summary(dataset_dict: dict[str, Any]) -> dict[str, int]:
    return {key: int(len(dataset_dict[key])) for key in ("trainset", "valset", "testset")}


def configure_cuda_runtime() -> None:
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def create_model(num_classes: int, channels_in: int, nlin: str, dropout: float, init_type: str) -> nn.Module:
    return CNN3(channels_in=channels_in, o_dim=num_classes, nlin=nlin, dropout=dropout, init_type=init_type)


def build_scheduler(optimizer: optim.Optimizer, num_epochs: int, scheduler_epochs: int | None, warmup_epochs: int):
    effective_scheduler_epochs = int(scheduler_epochs if scheduler_epochs is not None else num_epochs)
    if warmup_epochs > 0:
        def lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / warmup_epochs
            progress = (epoch_idx - warmup_epochs) / max(1, effective_scheduler_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_scheduler_epochs)


def iterate_batches(dataset: Dataset, batch_size: int, device: str, shuffle: bool, num_workers: int):
    if hasattr(dataset, "get_batch_iterator"):
        yield from dataset.get_batch_iterator(batch_size=batch_size, shuffle=shuffle, device=device)
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=max(0, int(num_workers)), pin_memory=device.startswith("cuda"))
    for inputs, targets in loader:
        yield inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def train_one_epoch(model: nn.Module, trainset: Dataset, batch_size: int, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, criterion: nn.Module, device: str, num_workers: int, grad_clip: float) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    start = time.time()

    for inputs, targets in iterate_batches(trainset, batch_size=batch_size, device=device, shuffle=True, num_workers=num_workers):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
        total += int(targets.size(0))
        n_batches += 1

    scheduler.step()
    return total_loss / max(1, n_batches), 100.0 * correct / max(1, total), time.time() - start


def evaluate_dataset(model: nn.Module, dataset: Dataset, batch_size: int, criterion: nn.Module, device: str, num_workers: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in iterate_batches(dataset, batch_size=batch_size, device=device, shuffle=False, num_workers=num_workers):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
            total += int(targets.size(0))
            n_batches += 1

    return total_loss / max(1, n_batches), 100.0 * correct / max(1, total)


def _parse_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_existing_metrics(progress_csv_path: Path, checkpoint_start_epoch: int) -> dict[str, float]:
    metrics = {"best_test_acc": 0.0, "best_val_acc": 0.0, "final_test_acc": float("nan"), "late_window_best_test_acc": float("nan")}
    if not progress_csv_path.exists():
        return metrics

    late_window_best = float("-inf")
    with progress_csv_path.open() as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            test_acc = _parse_float(row.get("test_acc"))
            val_acc = _parse_float(row.get("val_acc"))
            if not math.isnan(test_acc):
                metrics["best_test_acc"] = max(metrics["best_test_acc"], test_acc)
                metrics["final_test_acc"] = test_acc
                if epoch >= checkpoint_start_epoch:
                    late_window_best = max(late_window_best, test_acc)
            if not math.isnan(val_acc):
                metrics["best_val_acc"] = max(metrics["best_val_acc"], val_acc)

    if late_window_best != float("-inf"):
        metrics["late_window_best_test_acc"] = late_window_best
    elif not math.isnan(metrics["best_test_acc"]):
        metrics["late_window_best_test_acc"] = metrics["best_test_acc"]
    return metrics


def format_lr_value(lr: float) -> str:
    return format(lr, ".0e")


def load_completed_summary(exp_dir: Path, checkpoint_start_epoch: int) -> dict[str, Any]:
    params_path = exp_dir / "params.json"
    progress_path = exp_dir / "progress.csv"
    with params_path.open() as handle:
        params = json.load(handle)
    metrics = load_existing_metrics(progress_path, checkpoint_start_epoch)
    late_window_best = metrics["late_window_best_test_acc"]
    final_test_acc = metrics["final_test_acc"]
    late_window_gap = float("nan")
    if not math.isnan(late_window_best) and not math.isnan(final_test_acc):
        late_window_gap = late_window_best - final_test_acc
    return {"seed": int(params["seed"]), "lr": float(params["lr"]), "best_test_acc": float(params.get("best_test_acc", metrics["best_test_acc"])), "final_test_acc": float(params.get("final_test_acc", final_test_acc)), "late_window_best_test_acc": late_window_best, "late_window_gap": late_window_gap, "skipped": True}


def train_single_model(seed: int, lr: float, dataset_dict: dict[str, Any], dataset_meta: dict[str, Any], output_root: Path, device: str, num_epochs: int, scheduler_epochs: int | None, checkpoint_start_epoch: int, 
                       eval_every: int, batch_size: int, weight_decay: float, num_workers: int, nlin: str, dropout: float, init_type: str, warmup_epochs: int, grad_clip: float, save_outputs: bool) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model = create_model(num_classes=int(dataset_meta["num_classes"]), channels_in=int(dataset_meta["channels_in"]), nlin=nlin, dropout=dropout, init_type=init_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs, scheduler_epochs=scheduler_epochs, warmup_epochs=warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    exp_dir = output_root / f"NN_tune_trainable_seed={seed}"
    progress_csv_path = exp_dir / "progress.csv"
    result_json_path = exp_dir / "result.json"
    start_epoch = 0
    best_test_acc = 0.0
    best_val_acc = 0.0
    late_window_best_test_acc = float("nan")

    if save_outputs:
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_checkpoints = sorted(exp_dir.glob("checkpoint_*"))
        if existing_checkpoints:
            last_checkpoint = existing_checkpoints[-1]
            start_epoch = int(last_checkpoint.name.split("_")[1]) + 1
            if start_epoch >= num_epochs:
                return load_completed_summary(exp_dir, checkpoint_start_epoch)

            checkpoint_path = last_checkpoint / "checkpoints"
            optimizer_path = last_checkpoint / "optimizer"
            scheduler_path = last_checkpoint / "scheduler"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
            if scheduler_path.exists():
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
            metrics = load_existing_metrics(progress_csv_path, checkpoint_start_epoch)
            best_test_acc = metrics["best_test_acc"]
            best_val_acc = metrics["best_val_acc"]
            late_window_best_test_acc = metrics["late_window_best_test_acc"]

    final_test_acc = float("nan")
    eval_batch_size = max(batch_size, batch_size * 2)

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc, time_per_epoch = train_one_epoch(model=model, trainset=dataset_dict["trainset"], batch_size=batch_size, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device, num_workers=num_workers, grad_clip=grad_clip)
        should_eval = (epoch % eval_every == 0) or (epoch == num_epochs - 1)
        if should_eval:
            val_loss, val_acc = evaluate_dataset(model=model, dataset=dataset_dict["valset"], batch_size=eval_batch_size, criterion=criterion, device=device, num_workers=num_workers)
            test_loss, test_acc = evaluate_dataset(model=model, dataset=dataset_dict["testset"], batch_size=eval_batch_size, criterion=criterion, device=device, num_workers=num_workers)
            best_val_acc = max(best_val_acc, val_acc)
            best_test_acc = max(best_test_acc, test_acc)
            final_test_acc = test_acc
            if epoch >= checkpoint_start_epoch:
                if math.isnan(late_window_best_test_acc):
                    late_window_best_test_acc = test_acc
                else:
                    late_window_best_test_acc = max(late_window_best_test_acc, test_acc)
        else:
            val_loss = val_acc = test_loss = test_acc = float("nan")

        result = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc, "learning_rate": optimizer.param_groups[0]["lr"], "time_per_epoch": time_per_epoch}
        if save_outputs:
            if epoch >= checkpoint_start_epoch:
                checkpoint_dir = exp_dir / f"checkpoint_{epoch:06d}"
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "checkpoints")
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer")
                torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler")
            with result_json_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            write_header = (epoch == 0 and not progress_csv_path.exists()) or (epoch == start_epoch and not progress_csv_path.exists())
            with progress_csv_path.open("a") as handle:
                if write_header:
                    handle.write(",".join(result.keys()) + "\n")
                handle.write(",".join(str(value) for value in result.values()) + "\n")

    if math.isnan(late_window_best_test_acc):
        late_window_best_test_acc = best_test_acc
    late_window_gap = float("nan")
    if not math.isnan(late_window_best_test_acc) and not math.isnan(final_test_acc):
        late_window_gap = late_window_best_test_acc - final_test_acc

    summary = {"seed": seed, "lr": lr, "best_test_acc": best_test_acc, "final_test_acc": final_test_acc, "late_window_best_test_acc": late_window_best_test_acc, "late_window_gap": late_window_gap, "skipped": False}
    if save_outputs:
        params = {"seed": seed, "num_epochs": num_epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "warmup_epochs": warmup_epochs, "grad_clip": grad_clip, "best_val_acc": best_val_acc, "best_test_acc": best_test_acc, "final_test_acc": final_test_acc, "late_window_best_test_acc": late_window_best_test_acc, "late_window_gap": late_window_gap, 
                  "model::type": "CNN3", "model::channels_in": int(dataset_meta["channels_in"]), "model::o_dim": int(dataset_meta["num_classes"]), "model::nlin": nlin, "model::dropout": dropout, "model::init_type": init_type}
        with (exp_dir / "params.json").open("w") as handle:
            json.dump(params, handle, indent=2)

    return summary


def get_completed_seeds(experiment_dir: Path, expected_epochs: int) -> set[int]:
    completed: set[int] = set()
    for seed_dir in experiment_dir.glob("NN_tune_trainable_seed=*"):
        params_path = seed_dir / "params.json"
        if not params_path.exists():
            continue
        try:
            with params_path.open() as handle:
                params = json.load(handle)
        except json.JSONDecodeError:
            continue
        if int(params.get("num_epochs", 0)) < int(expected_epochs):
            continue
        completed.add(int(seed_dir.name.split("=")[1]))
    return completed


def task_label(task: dict[str, Any]) -> str:
    return f"{task['mode']}:seed={task['seed']}:lr={format_lr_value(task['lr'])}"


def worker_main(worker_id: int, device: str, dataset_pt: str, output_root: str, shared_config: dict[str, Any], tasks: list[dict[str, Any]], result_queue: Any) -> None:
    try:
        torch.set_num_threads(1)
        configure_cuda_runtime()
        if device.startswith("cuda"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        dataset_dict = load_dataset_dict(dataset_pt)
        dataset_meta = {"num_classes": infer_num_classes(dataset_dict), "channels_in": infer_channels_in(dataset_dict)}
    except Exception as exc:  # pragma: no cover - worker init path
        for task in tasks:
            result_queue.put({"ok": False, "worker_id": worker_id, "device": device, "task": task, "error": f"worker init failed: {exc}"})
        return

    for task in tasks:
        try:
            summary = train_single_model(seed=int(task["seed"]), lr=float(task["lr"]), dataset_dict=dataset_dict, dataset_meta=dataset_meta, output_root=Path(output_root), device=device, num_epochs=int(shared_config["epochs"]), scheduler_epochs=shared_config["scheduler_epochs"], checkpoint_start_epoch=int(shared_config["checkpoint_start_epoch"]), eval_every=int(shared_config["eval_every"]), 
                                         batch_size=int(shared_config["batch_size"]), weight_decay=float(shared_config["weight_decay"]), num_workers=int(shared_config["num_workers"]), nlin=str(shared_config["nlin"]), dropout=float(shared_config["dropout"]), init_type=str(shared_config["init_type"]), warmup_epochs=int(shared_config["warmup_epochs"]), grad_clip=float(shared_config["grad_clip"]), save_outputs=bool(task["save_outputs"]))
            result_queue.put({"ok": True, "worker_id": worker_id, "device": device, "task": task, "summary": summary})
        except Exception as exc:  # pragma: no cover - worker task path
            result_queue.put({"ok": False, "worker_id": worker_id, "device": device, "task": task, "error": str(exc)})


def run_tasks_parallel(tasks: list[dict[str, Any]], dataset_pt: Path, output_root: Path, shared_config: dict[str, Any], devices: list[str], models_per_gpu: int, description: str) -> list[dict[str, Any]]:
    if not tasks:
        return []

    slot_devices = [device for device in devices for _ in range(int(models_per_gpu))]
    worker_count = min(len(tasks), len(slot_devices))
    task_buckets: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for idx, task in enumerate(tasks):
        task_buckets[idx % worker_count].append(task)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []
    for worker_id in range(worker_count):
        process = ctx.Process(target=worker_main, args=(worker_id, slot_devices[worker_id], str(dataset_pt), str(output_root), shared_config, task_buckets[worker_id], result_queue))
        process.start()
        processes.append(process)

    results: list[dict[str, Any]] = []
    with tqdm(total=len(tasks), desc=description) as progress:
        for _ in range(len(tasks)):
            result = result_queue.get()
            results.append(result)
            label = task_label(result["task"])
            progress.set_postfix({"last": label, "status": "ok" if result["ok"] else "fail"})
            progress.update(1)

    for process in processes:
        process.join()
    result_queue.close()
    result_queue.join_thread()
    return results


LR_SELECTION_STRATEGY = "robust_final_mean_minus_std_v1"


def build_sweep_candidate(lr: float, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"lr": float(lr), "lr_label": format_lr_value(float(lr)), "successful_runs": 0, "mean_late_window_best_test_acc": float("-inf"), "mean_late_window_gap": float("inf"), 
                "mean_final_test_acc": float("-inf"), "median_final_test_acc": float("-inf"), "min_final_test_acc": float("-inf"), "std_final_test_acc": float("inf"), "robust_final_test_acc": float("-inf"), "runs": []}

    late_window_values = np.asarray([run["late_window_best_test_acc"] for run in runs], dtype=float)
    late_window_gaps = np.asarray([run["late_window_gap"] for run in runs], dtype=float)
    final_values = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
    mean_late_window_best = float(np.mean(late_window_values))
    mean_late_window_gap = float(np.mean(late_window_gaps))
    mean_final = float(np.mean(final_values))
    median_final = float(np.median(final_values))
    min_final = float(np.min(final_values))
    std_final = float(np.std(final_values))
    robust_final = float(mean_final - std_final)
    return {"lr": float(lr), "lr_label": format_lr_value(float(lr)), "successful_runs": len(runs), "mean_late_window_best_test_acc": mean_late_window_best, "mean_late_window_gap": mean_late_window_gap, 
            "mean_final_test_acc": mean_final, "median_final_test_acc": median_final, "min_final_test_acc": min_final, "std_final_test_acc": std_final, "robust_final_test_acc": robust_final, "runs": runs}


def choose_sweep_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successful_candidates = [candidate for candidate in candidates if candidate["successful_runs"] > 0]
    if not successful_candidates:
        raise RuntimeError("LR sweep produced no successful candidates")
    return sorted(successful_candidates, key=lambda candidate: (-candidate["robust_final_test_acc"], -candidate["median_final_test_acc"], -candidate["min_final_test_acc"], -candidate["mean_late_window_best_test_acc"], candidate["mean_late_window_gap"], candidate["lr"]))[0]


def write_sweep_summary(dataset_name: str, candidates: list[dict[str, Any]], summary_path: Path) -> dict[str, Any]:
    chosen = choose_sweep_candidate(candidates)
    summary = {"dataset": dataset_name, "chosen_lr": chosen["lr"], "chosen_lr_label": chosen["lr_label"], "selection_strategy": LR_SELECTION_STRATEGY, "candidates": candidates}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def summarize_sweep_results(dataset_name: str, results: list[dict[str, Any]], lr_candidates: list[float], summary_path: Path) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = {float(lr): [] for lr in lr_candidates}
    for result in results:
        if result["ok"]:
            summary = result["summary"]
            grouped[float(summary["lr"])] .append(summary)

    candidates: list[dict[str, Any]] = []
    for lr in lr_candidates:
        candidates.append(build_sweep_candidate(float(lr), grouped[float(lr)]))

    if not any(candidate["successful_runs"] > 0 for candidate in candidates):
        raise RuntimeError(f"LR sweep failed for dataset {dataset_name}")
    return write_sweep_summary(dataset_name=dataset_name, candidates=candidates, summary_path=summary_path)


def normalize_existing_sweep_summary(dataset_name: str, summary_payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    raw_candidates = summary_payload.get("candidates", [])
    candidates = [build_sweep_candidate(float(candidate["lr"]), candidate.get("runs", [])) for candidate in raw_candidates]
    if not candidates:
        raise RuntimeError(f"Saved LR sweep summary for {dataset_name} has no candidates")
    return write_sweep_summary(dataset_name=dataset_name, candidates=candidates, summary_path=summary_path)


def load_or_run_lr_sweep(dataset_name: str, dataset_pt: Path, output_root: Path, lr_candidates: list[float], lr_sweep_seeds: int, force_lr_sweep: bool, shared_config: dict[str, Any], devices: list[str], models_per_gpu: int) -> dict[str, Any]:
    summary_path = output_root / "lr_sweep" / "summary.json"
    if summary_path.exists() and not force_lr_sweep:
        with summary_path.open() as handle:
            return normalize_existing_sweep_summary(dataset_name=dataset_name, summary_payload=json.load(handle), summary_path=summary_path)

    sweep_tasks: list[dict[str, Any]] = []
    for lr in lr_candidates:
        for seed in range(1, lr_sweep_seeds + 1):
            sweep_tasks.append({"mode": "sweep", "seed": seed, "lr": float(lr), "save_outputs": False})

    results = run_tasks_parallel(tasks=sweep_tasks, dataset_pt=dataset_pt, output_root=output_root, shared_config=shared_config, devices=devices, models_per_gpu=models_per_gpu, description=f"LR sweep {canonical_dataset_name(dataset_name)}")
    failures = [result for result in results if not result["ok"]]
    for failure in failures:
        print(f"[sweep] {task_label(failure['task'])} failed on {failure['device']}: {failure['error']}")
    return summarize_sweep_results(dataset_name=dataset_name, results=results, lr_candidates=lr_candidates, summary_path=summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN3 model zoos from existing dataset.pt files")
    parser.add_argument("--dataset", required=True, help="Dataset id such as ct_images_cnn3")
    parser.add_argument("--dataset-pt", default=None, help="Explicit path to a prebuilt dataset.pt (or a dir containing it)")
    parser.add_argument("--dataset-pt-root", default=None, help="Root of build_dataset_pts.py output; loads <root>/<name>/dataset.pt")
    parser.add_argument("--zoo-root", default=str(DEFAULT_OUTPUT_ZOO_ROOT), help="Root for the regenerated CNN3 zoos")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of final seeds to train")
    parser.add_argument("--lr-candidates", nargs="+", type=float, default=DEFAULT_LR_CANDIDATES, help="Candidate learning rates for the warmup sweep")
    parser.add_argument("--lr-sweep-seeds", type=int, default=2, help="Number of sweep seeds per LR candidate")
    parser.add_argument("--force-lr-sweep", action="store_true", help="Rerun the LR sweep even if a saved summary exists")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for final runs")
    parser.add_argument("--scheduler-epochs", type=int, default=50, help="Scheduler length in epochs")
    parser.add_argument("--checkpoint-start-epoch", type=int, default=20, help="Only save checkpoints starting from this epoch")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Adam weight decay")
    parser.add_argument("--nlin", default="gelu", help="CNN3 nonlinearity")
    parser.add_argument("--dropout", type=float, default=0.0, help="CNN3 dropout")
    parser.add_argument("--init-type", default="uniform", help="CNN3 init type")
    parser.add_argument("--models-per-gpu", type=int, default=2, help="Parallel models to run on each GPU")
    parser.add_argument("--gpu-id", type=int, default=None, help="Use only one GPU id if provided")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for non-cached fallback datasets")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs for the cosine scheduler")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm, or 0 to disable")
    parser.add_argument("--dry-run", action="store_true", help="Train without writing final zoo outputs")
    return parser.parse_args()


def available_devices(gpu_id: int | None) -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trainer")
    if gpu_id is not None:
        return [f"cuda:{gpu_id}"]
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def main() -> None:
    args = parse_args()
    configure_cuda_runtime()
    source_dataset_pt = resolve_source_dataset_pt(args)
    if not source_dataset_pt.exists():
        raise FileNotFoundError(f"dataset.pt not found: {source_dataset_pt}")

    output_root = resolve_output_root(args.dataset, args.zoo_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        copy_dataset_pt(source_dataset_pt, output_root)

    preview_dataset = load_dataset_dict(source_dataset_pt)
    split_sizes = dataset_summary(preview_dataset)
    dataset_meta = {"num_classes": infer_num_classes(preview_dataset), "channels_in": infer_channels_in(preview_dataset)}
    print(f"Dataset: {args.dataset} -> {canonical_dataset_name(args.dataset)}")
    print(f"Source dataset.pt: {source_dataset_pt}")
    print(f"Output root: {output_root}")
    print(f"Splits: train={split_sizes['trainset']} val={split_sizes['valset']} test={split_sizes['testset']}")
    print(f"Model config: channels_in={dataset_meta['channels_in']} num_classes={dataset_meta['num_classes']} nlin={args.nlin} dropout={args.dropout} init_type={args.init_type}")

    devices = available_devices(args.gpu_id)
    print(f"Devices: {devices}")
    shared_config = {"epochs": int(args.epochs), "scheduler_epochs": args.scheduler_epochs, "checkpoint_start_epoch": int(args.checkpoint_start_epoch), "eval_every": int(args.eval_every), "batch_size": int(args.batch_size), "weight_decay": float(args.weight_decay), 
                     "num_workers": int(args.num_workers), "nlin": str(args.nlin), "dropout": float(args.dropout), "init_type": str(args.init_type), "warmup_epochs": int(args.warmup_epochs), "grad_clip": float(args.grad_clip)}

    sweep_summary = load_or_run_lr_sweep(dataset_name=args.dataset, dataset_pt=source_dataset_pt, output_root=output_root, lr_candidates=[float(value) for value in args.lr_candidates], lr_sweep_seeds=int(args.lr_sweep_seeds),
                                         force_lr_sweep=bool(args.force_lr_sweep), shared_config=shared_config, devices=devices, models_per_gpu=int(args.models_per_gpu))
    chosen_lr = float(sweep_summary["chosen_lr"])
    print(f"Chosen LR: {sweep_summary['chosen_lr_label']} ({chosen_lr})")

    completed = set() if args.dry_run else get_completed_seeds(output_root, expected_epochs=args.epochs)
    if not args.dry_run:
        print(f"Completed final seeds: {len(completed)}")

    final_tasks = [{"mode": "final", "seed": seed, "lr": chosen_lr, "save_outputs": not args.dry_run} for seed in range(1, int(args.num_seeds) + 1) if args.dry_run or seed not in completed]
    if not final_tasks:
        print("All final seeds already trained.")
        return

    final_results = run_tasks_parallel(tasks=final_tasks, dataset_pt=source_dataset_pt, output_root=output_root, shared_config=shared_config, devices=devices, models_per_gpu=int(args.models_per_gpu), description=f"Final train {canonical_dataset_name(args.dataset)}")
    failures = [result for result in final_results if not result["ok"]]
    for failure in failures:
        print(f"[final] {task_label(failure['task'])} failed on {failure['device']}: {failure['error']}")

    successes = len(final_results) - len(failures)
    print(f"Completed final runs: {successes} succeeded, {len(failures)} failed")


if __name__ == "__main__":
    main()