#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from hnet_trainer import HyperResNetTrainer
from hnet_utils import RESNET_TRAIN_DATASETS, SCRIPT_DIR, mean_std, normalize_dataset_name, set_seed

from dataset_to_model import extract_dataset_embeddings_cached, get_bn_conditioning_loader
from ood_utils import _adapt_output_layer, create_model_from_state_dict, finetune_state_dict, get_model_config_cached, get_train_test_loaders, uses_bn_conditioning
from tans_utils import get_test_datasets, get_num_classes as get_resnet_num_classes, get_zoo_path as get_resnet_zoo_path


def _resize_output_layer(state: Dict, layer_key: str, num_classes: int) -> Dict:
    weight_key = f"{layer_key}.weight"
    bias_key = f"{layer_key}.bias"
    in_features = state[weight_key].shape[1]
    target = {
        weight_key: torch.zeros(num_classes, in_features, dtype=state[weight_key].dtype),
        bias_key: torch.zeros(num_classes, dtype=state[bias_key].dtype),
    }
    return _adapt_output_layer(state, target, layer_key)


def get_loaders(dataset_name: str, batch_size: int, num_workers: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    root = get_resnet_zoo_path(dataset_name, architecture="resnet")
    return get_train_test_loaders(dataset_name, batch_size=batch_size, num_workers=num_workers, root_dir=root)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100.0 * correct / max(total, 1)


def condition_bn(state_dict: Dict[str, torch.Tensor], bn_loader: Optional[DataLoader], device: torch.device, model_config: Optional[Dict], iterations: int) -> Dict[str, torch.Tensor]:
    """Re-fit BN running statistics on a generated state dict.

    Generated ResNets inherit BN running_mean / running_var from the random
    template, which is garbage at inference time. We materialise the model,
    flip it to train mode (only BN updates running stats), and run a few
    forward passes over the dataset's own images without computing
    gradients. Returns the updated state dict.
    """
    if bn_loader is None:
        return state_dict
    template = create_model_from_state_dict(state_dict, config=model_config, device=str(device))
    model = copy.deepcopy(template)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    with torch.no_grad():
        for idx, batch in enumerate(bn_loader):
            if idx >= iterations:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(images.to(device))
    model.eval()
    return model.cpu().state_dict()


def finetune(state_dict: Dict[str, torch.Tensor], train_loader: DataLoader, test_loader: DataLoader, num_classes: int, epochs: int, lr: float, wd: Optional[float], device: torch.device, model_config: Optional[Dict]) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Single-LR finetune; returns final accuracy and the finetuned state dict."""
    state = OrderedDict((k, v.detach().clone()) for k, v in state_dict.items())
    state = _resize_output_layer(state, "fc", num_classes)
    final_acc, _, model = finetune_state_dict(state, num_classes, train_loader, test_loader, epochs=epochs, lr=lr, wd=wd, device=str(device), config=model_config, model_type="resnet")
    return float(final_acc), model.state_dict()


def evaluate_single_dataset(trainer: HyperResNetTrainer, dataset_name: str, args) -> Optional[Dict]:
    dataset_name = normalize_dataset_name(dataset_name)
    root = get_resnet_zoo_path(dataset_name, architecture="resnet")
    train_loader, test_loader = get_loaders(dataset_name, args.batch_size, args.num_workers)
    if train_loader is None or test_loader is None:
        print(f"  Skipping {dataset_name}: missing loaders")
        return None

    num_classes = get_resnet_num_classes(dataset_name, architecture="resnet")
    if num_classes > trainer.target_classes:
        print(f"  Skipping {dataset_name}: num_classes={num_classes} > target {trainer.target_classes}")
        return None

    model_config = get_model_config_cached(root) if root else None
    bn_loader = None
    if uses_bn_conditioning("resnet") and not args.no_bn_conditioning:
        bn_loader = get_bn_conditioning_loader(root_dir=root, batch_size=args.batch_size, num_workers=args.num_workers)

    # Sample ood-samples candidate dataset embeddings (mirrors dataset_to_model).
    trainer.dataset_encoder.eval()
    candidate_embs = extract_dataset_embeddings_cached(trainer.dataset_encoder, dataset_name, num_subsets=args.ood_samples, subset_size=args.subset_size, device=str(trainer.device), cache_dir=args.embedding_cache_dir, root_dir=root)
    if candidate_embs is None:
        print(f"  Skipping {dataset_name}: could not extract candidate embeddings")
        return None
    print(f"  Extracted {len(candidate_embs)} candidate embeddings")

    # Generate, BN-condition, zero-shot score each candidate.
    pre_accs: List[float] = []
    candidates: List[Tuple[int, float, Dict]] = []
    with torch.no_grad():
        for idx, emb in enumerate(candidate_embs):
            tokens = trainer.hypernetwork(emb.to(trainer.device))
            init_state = trainer.tokens_to_state_dict(tokens)
            conditioned = condition_bn(init_state, bn_loader, trainer.device, model_config, args.bn_iterations)
            eval_state = OrderedDict((k, v.detach().clone()) for k, v in conditioned.items())
            eval_state = _resize_output_layer(eval_state, "fc", num_classes)
            model = create_model_from_state_dict(eval_state, config=model_config, device=str(trainer.device))
            model.load_state_dict(eval_state, strict=True)
            acc = evaluate_model(model, test_loader, trainer.device)
            pre_accs.append(acc)
            candidates.append((idx, acc, eval_state))

    if not candidates:
        return None

    # Top-k by zero-shot, single-LR finetune.
    candidates.sort(key=lambda c: c[1], reverse=True)
    top_k = min(int(args.top_k), len(candidates))
    top = candidates[:top_k]
    pre_pool_mean, _ = mean_std(pre_accs)
    print(f"  pre over {len(pre_accs)} cand: mean={pre_pool_mean:.2f}% best={top[0][1]:.2f}%")

    ft_1ep: List[float] = []
    ft_final: List[float] = []
    for rank, (sample_idx, pre_acc, state) in enumerate(top, start=1):
        print(f"    top-{rank} (sample={sample_idx}, pre={pre_acc:.2f}%) finetune at lr={args.eval_lr}")
        acc1, mid_state = finetune(state, train_loader, test_loader, num_classes, epochs=1, lr=args.eval_lr, wd=args.eval_wd, device=trainer.device, model_config=model_config)
        ft_1ep.append(acc1)
        remaining = max(int(args.n_eps_finetuning) - 1, 0)
        if remaining > 0:
            accN, _ = finetune(mid_state, train_loader, test_loader, num_classes, epochs=remaining, lr=args.eval_lr, wd=args.eval_wd, device=trainer.device, model_config=model_config)
        else:
            accN = acc1
        ft_final.append(accN)

    pre_mean, pre_std = mean_std([c[1] for c in top])
    one_mean, one_std = mean_std(ft_1ep)
    final_mean, final_std = mean_std(ft_final)
    return {"num_classes": int(num_classes), "ood_samples": int(len(candidate_embs)), "top_k": int(top_k), "top_k_pre": [float(c[1]) for c in top], "top_k_1ep": [float(v) for v in ft_1ep], "top_k_final": [float(v) for v in ft_final], "mean_pre": float(pre_mean), "std_pre": float(pre_std), "mean_1ep": float(one_mean), "std_1ep": float(one_std), "mean_final": float(final_mean), "std_final": float(final_std)}


def run_evaluation(trainer: HyperResNetTrainer, args) -> Dict[str, Dict]:
    datasets = list(args.ood_datasets) if args.ood_datasets else list(get_test_datasets("resnet"))
    results: Dict[str, Dict] = {}
    print("=" * 70)
    print("OOD Evaluation")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print(f"ood_samples={args.ood_samples} | top_k={args.top_k} | n_eps_finetuning={args.n_eps_finetuning} | lr={args.eval_lr}")
    for i, ds in enumerate(datasets, start=1):
        ds = normalize_dataset_name(ds)
        print(f"\n[{i}/{len(datasets)}] {ds}")
        res = evaluate_single_dataset(trainer, ds, args)
        if res is None:
            continue
        results[ds] = res
        print(f"  top-{res['top_k']} | pre={res['mean_pre']:.2f}±{res['std_pre']:.2f} | 1ep={res['mean_1ep']:.2f}±{res['std_1ep']:.2f} | {args.n_eps_finetuning}ep={res['mean_final']:.2f}±{res['std_final']:.2f}")
    return results


def print_summary(results: Dict[str, Dict], n_eps: int) -> None:
    if not results:
        print("\nNo valid OOD results.")
        return
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    pre, one, final = [], [], []
    for ds, r in results.items():
        pre.append(float(r["mean_pre"]))
        one.append(float(r["mean_1ep"]))
        final.append(float(r["mean_final"]))
        print(f"{ds:24s} | pre={r['mean_pre']:6.2f}({r['std_pre']:4.2f}) | 1ep={r['mean_1ep']:6.2f}({r['std_1ep']:4.2f}) | {n_eps:2d}ep={r['mean_final']:6.2f}({r['std_final']:4.2f})")
    pm, ps = mean_std(pre)
    om, ostd = mean_std(one)
    fm, fs = mean_std(final)
    print("-" * 70)
    print(f"Mean top-k pre : {pm:.2f}% ± {ps:.2f}%")
    print(f"Mean top-k 1ep : {om:.2f}% ± {ostd:.2f}%")
    print(f"Mean top-k {n_eps:2d}ep: {fm:.2f}% ± {fs:.2f}%")


def aggregate_seed_results(results_by_seed: Dict[int, Dict[str, Dict]]) -> Dict:
    by_dataset: Dict[str, List[Dict]] = {}
    for seed, results in results_by_seed.items():
        for name, r in results.items():
            by_dataset.setdefault(name, []).append(r)

    per_dataset: Dict[str, Dict] = {}
    grand_pre: List[float] = []
    grand_1ep: List[float] = []
    grand_final: List[float] = []
    for name in sorted(by_dataset):
        runs = by_dataset[name]
        pres = [float(run["mean_pre"]) for run in runs]
        ones = [float(run["mean_1ep"]) for run in runs]
        fins = [float(run["mean_final"]) for run in runs]
        pm, ps = mean_std(pres)
        om, ostd = mean_std(ones)
        fm, fs = mean_std(fins)
        per_dataset[name] = {"n_seeds": len(runs), "pre_mean": float(pm), "pre_std": float(ps), "one_mean": float(om), "one_std": float(ostd), "final_mean": float(fm), "final_std": float(fs), "per_seed_pre": pres, "per_seed_1ep": ones, "per_seed_final": fins}
        grand_pre.extend(pres)
        grand_1ep.extend(ones)
        grand_final.extend(fins)

    pm, ps = mean_std(grand_pre)
    om, ostd = mean_std(grand_1ep)
    fm, fs = mean_std(grand_final)
    return {"seeds": sorted(int(s) for s in results_by_seed), "per_dataset": per_dataset, "grand_average": {"pre_mean": float(pm), "pre_std": float(ps), "one_mean": float(om), "one_std": float(ostd), "final_mean": float(fm), "final_std": float(fs)}}


def print_multi_seed_summary(aggregate: Dict, n_eps: int) -> None:
    seeds = aggregate["seeds"]
    print("\n" + "=" * 70)
    print(f"Multi-seed summary (seeds={seeds})")
    print("=" * 70)
    for name, s in aggregate["per_dataset"].items():
        print(f"  {name:<25} | pre={s['pre_mean']:6.2f}±{s['pre_std']:<5.2f} | 1ep={s['one_mean']:6.2f}±{s['one_std']:<5.2f} | {n_eps:2d}ep={s['final_mean']:6.2f}±{s['final_std']:<5.2f}")
    g = aggregate["grand_average"]
    print("-" * 70)
    print(f"  {'Average':<25} | pre={g['pre_mean']:6.2f}±{g['pre_std']:<5.2f} | 1ep={g['one_mean']:6.2f}±{g['one_std']:<5.2f} | {n_eps:2d}ep={g['final_mean']:6.2f}±{g['final_std']:<5.2f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train + evaluate a Text2Model-style hypernetwork on ResNet18Slim 0.5×.")
    p.add_argument("--mode", choices=["train", "test", "all"], default="all")
    p.add_argument("--save-dir", type=str, default=str(SCRIPT_DIR / "outputs" / "hyper_resnet18slim"))
    p.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path for test mode")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42, help="Seed for training initialization.")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 0, 777], help="Test-time seeds. Evaluation is re-run per seed and averaged. Pass a single value to disable averaging.")
    p.add_argument("--save-multiseed", action="store_true", help="Include per-seed raw results in the saved JSON.")

    p.add_argument("--train-datasets", nargs="+", default=list(RESNET_TRAIN_DATASETS))
    p.add_argument("--ood-datasets", nargs="+", default=None, help="OOD datasets to evaluate (default: the Test split).")
    p.add_argument("--width-mult", type=float, default=0.5)
    p.add_argument("--target-classes", type=int, default=None, help="Fixed classifier output width. Defaults to max num_classes over train datasets.")

    # Dataset encoder + warm-start
    p.add_argument("--dataset-encoder-preset", type=str, default="deepsets_conv")
    p.add_argument("--dataset-embedding-dim", type=int, default=128)
    p.add_argument("--dataset-encoder-pretrain-epochs", type=int, default=20)
    p.add_argument("--dataset-encoder-pretrain-steps-per-epoch", type=int, default=50)
    p.add_argument("--dataset-encoder-pretrain-batch-size", type=int, default=32)
    p.add_argument("--dataset-encoder-pretrain-lr", type=float, default=3e-4)
    p.add_argument("--dataset-encoder-pretrain-wd", type=float, default=1e-6)

    # Dataset embedding sampling
    p.add_argument("--n-subsets", type=int, default=1)
    p.add_argument("--subset-size", type=int, default=10)
    p.add_argument("--max-cache-samples", type=int, default=4096)

    # Hypernetwork architecture
    p.add_argument("--hnet-hidden-dim", type=int, default=120)
    p.add_argument("--token-size", type=int, default=288)
    p.add_argument("--token-window-size", type=int, default=256)
    p.add_argument("--token-position-embedding", choices=["learned", "sinusoidal"], default="learned")
    p.add_argument("--tokenizer-ignore-bn", action="store_true")

    # Training
    p.add_argument("--meta-epochs", type=int, default=100)
    p.add_argument("--outer-lr", type=float, default=1e-3)
    p.add_argument("--outer-wd", type=float, default=1e-4)
    p.add_argument("--inner-epochs", type=int, default=5)
    p.add_argument("--inner-max-batches", type=int, default=30, help="Cap on inner-loop batches per epoch; <=0 means full epoch.")
    p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--inner-momentum", type=float, default=0.9)
    p.add_argument("--inner-wd", type=float, default=5e-4)

    # OOD evaluation
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ood-samples", type=int, default=100, help="Number of candidate dataset embeddings to sample per OOD dataset.")
    p.add_argument("--top-k", type=int, default=5, help="Top-K candidates by zero-shot accuracy to finetune.")
    p.add_argument("--embedding-cache-dir", type=str, default=None)
    p.add_argument("--eval-lr", type=float, default=1e-4, help="Single finetuning learning rate (no sweep).")
    p.add_argument("--eval-wd", type=float, default=None, help="Finetune weight decay; defaults to ood_utils ResNet default.")
    p.add_argument("--n-eps-finetuning", type=int, default=10)
    p.add_argument("--no-bn-conditioning", action="store_true")
    p.add_argument("--bn-iterations", type=int, default=200)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    args.train_datasets = [normalize_dataset_name(ds) for ds in args.train_datasets]

    trainer = HyperResNetTrainer(args)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode in ("train", "all"):
        trainer.train()

    if args.mode in ("test", "all"):
        if args.mode == "test":
            trainer.load_checkpoint(args.checkpoint)
        trainer.hypernetwork.eval()
        trainer.dataset_encoder.eval()

        seeds = [int(s) for s in (args.seeds or [args.seed])]
        out_path = Path(args.save_dir) / "hyper_resnet18slim_ood_results.json"

        if len(seeds) == 1:
            set_seed(seeds[0])
            results = run_evaluation(trainer, args)
            print_summary(results, args.n_eps_finetuning)
            with open(out_path, "w") as f:
                json.dump({"seed": seeds[0], "datasets": results}, f, indent=2)
            print(f"\nSaved OOD results to {out_path}")
        else:
            results_by_seed: Dict[int, Dict[str, Dict]] = {}
            for i, seed in enumerate(seeds, 1):
                print("\n" + "=" * 70)
                print(f"Seed {i}/{len(seeds)}: {seed}")
                print("=" * 70)
                set_seed(seed)
                seed_results = run_evaluation(trainer, args)
                print_summary(seed_results, args.n_eps_finetuning)
                results_by_seed[seed] = seed_results

            aggregate = aggregate_seed_results(results_by_seed)
            print_multi_seed_summary(aggregate, args.n_eps_finetuning)

            payload = {"seeds": aggregate["seeds"], "per_dataset": aggregate["per_dataset"], "grand_average": aggregate["grand_average"]}
            if args.save_multiseed:
                payload["raw_results"] = {str(s): r for s, r in results_by_seed.items()}
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved multi-seed OOD results to {out_path}")


if __name__ == "__main__":
    main()
