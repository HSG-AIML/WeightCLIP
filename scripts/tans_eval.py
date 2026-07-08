#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import random
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ood_utils import (
    CNN3_FINETUNE_WD,
    RESNET18SLIM_FINETUNE_WD,
    canonicalize_model_type,
    create_model_from_state_dict,
    detect_model_type,
    evaluate_model as shared_evaluate_model,
    finetune_model as shared_finetune_model,
    format_lr,
    get_model_config_cached,
    get_zoo_path as shared_get_zoo_path,
    model_type_display_name,
)
from tans_common import (
    DEFAULT_FINETUNE_LR_CNN3,
    DEFAULT_FINETUNE_LR_RESNET,
    DEFAULT_TEST_SEEDS,
    aggregate_seed_results,
    format_startup_line,
    get_default_finetune_lr,
    get_metatest_datasets,
    get_metatrain_datasets,
    get_num_classes,
    load_dataset_pt,
    load_models_from_zoo,
    print_multi_seed_summary,
    print_single_seed_summary,
    load_torch,
    resolve_seeds,
    save_json,
    save_torch,
    set_seed,
)
from tans_core import (
    FunctionalEmbeddingGenerator,
    HardNegativeContrastiveLoss,
    ImageFeatureExtractor,
    ModelEncoder,
    PerformancePredictor,
    QueryEncoder,
    compute_recall,
    query_features_for_dataset,
)

_BATCH = 32
_WORKERS = 4
_TRAIN_LOG_INTERVAL = 1000


# Training

def _split_seed(seed, dataset_name, split):
    base = zlib.adler32(dataset_name.encode())
    return (seed * 1000003 + base + (0 if split == "train" else 1)) % (2**32)


def _should_log_epoch(epoch, total_epochs, interval=_TRAIN_LOG_INTERVAL):
    step = int(epoch) + 1
    total = int(total_epochs)
    return step == 1 or step == total or step % int(interval) == 0


def _load_metatrain_data(args, device):
    print("\nLoading meta-train data")
    extractor = ImageFeatureExtractor(device)
    generator = FunctionalEmbeddingGenerator(args.n_noise_samples, device, args.seed, args.architecture)
    datasets, zoo, q_train, q_test, fdim = [], {}, {}, {}, None

    for name in get_metatrain_datasets(args.architecture):
        ms, accs, sds = load_models_from_zoo(name, args.n_nets, device, args.architecture)
        if ms is None:
            print(f"  {name}: no checkpoints, skipping")
            continue
        qtr = query_features_for_dataset(name, extractor, args.n_query_images, _split_seed(args.seed, name, "train"), args.architecture)
        qte = query_features_for_dataset(name, extractor, args.n_query_images, _split_seed(args.seed, name, "test"), args.architecture)
        if qtr is None or qte is None:
            print(f"  {name}: no dataset.pt, skipping")
            continue
        embs = generator.extract_batch(ms, args.use_features)
        zoo[name] = {"embs": embs, "accs": torch.tensor(accs, dtype=torch.float32), "sds": sds}
        q_train[name] = qtr
        q_test[name] = qte
        datasets.append(name)
        fdim = fdim or embs.shape[1]

    if not datasets:
        raise RuntimeError("No meta-train datasets loaded")
    total_models = sum(len(zoo[name]["sds"]) for name in datasets)
    print(f"Loaded {len(datasets)} datasets | {total_models} models | functional_dim={fdim}")
    return datasets, zoo, q_train, q_test, fdim


@torch.no_grad()
def _eval_alignment(datasets, zoo, qfeat, enc_q, enc_m, pred, device, closs, mse):
    enc_q.eval(); enc_m.eval(); pred.eval()
    qs, ms, av = [], [], []
    
    for name in datasets:
        i = random.randint(0, len(zoo[name]["accs"]) - 1)
        qs.append(enc_q([qfeat[name].to(device)]))
        ms.append(enc_m(zoo[name]["embs"][i].unsqueeze(0).to(device)))
        av.append(float(zoo[name]["accs"][i]))
        
    q = torch.cat(qs); m = torch.cat(ms)
    a = torch.tensor(av, device=device).unsqueeze(1)
    loss = (closs(q, m) + mse(pred(q, m), a)).item()
    recalls, medr, meanr = compute_recall(q, m)
    enc_q.train(); enc_m.train(); pred.train()
    return loss, recalls, medr, meanr


def _save_checkpoint(path, enc_q, enc_m, pred, args, fdim, device, is_best, epoch=None, recalls=None, medr=None, meanr=None):
    fname = "saved_model_max_recall.pt" if is_best else "saved_model.pt"
    save_torch(path, fname, {
        "enc_q": enc_q.cpu().state_dict(),
        "enc_m": enc_m.cpu().state_dict(),
        "predictor": pred.cpu().state_dict(),
        "architecture": args.architecture,
        "functional_dim": fdim,
        "n_dims": args.n_dims,
        "n_noise_samples": args.n_noise_samples,
        "use_features": args.use_features,
        "epoch": epoch, "recall": recalls, "medr": medr, "meanr": meanr,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
    })
    enc_q.to(device); enc_m.to(device); pred.to(device)


def run_train(args, device):
    set_seed(args.seed)
    datasets, zoo, q_train, q_test, fdim = _load_metatrain_data(args, device)
    if args.n_groups is not None:
        datasets = datasets[: args.n_groups]

    enc_q = QueryEncoder(args.n_dims).to(device)
    enc_m = ModelEncoder(args.n_dims, fdim).to(device)
    pred = PerformancePredictor(args.n_dims).to(device)
    closs = HardNegativeContrastiveLoss(nmax=len(datasets)).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(list(enc_q.parameters()) + list(enc_m.parameters()) + list(pred.parameters()), lr=args.lr)

    best_r1 = -1.0
    history = []
    print(f"\nTraining for {args.n_epochs} epochs")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        q_feats, func_embs, acc_vals = [], [], []
        for name in datasets:
            i = random.randint(0, len(zoo[name]["accs"]) - 1)
            q_feats.append(q_train[name].to(device))
            func_embs.append(zoo[name]["embs"][i])
            acc_vals.append(float(zoo[name]["accs"][i]))

        func_batch = torch.stack(func_embs).to(device)
        acc_batch = torch.tensor(acc_vals, dtype=torch.float32, device=device).unsqueeze(1)

        opt.zero_grad(set_to_none=True)
        q = enc_q(q_feats)
        m = enc_m(func_batch)
        loss = closs(q, m) + mse(pred(q, m), acc_batch)
        loss.backward()
        opt.step()

        if (epoch + 1) % args.eval_every != 0:
            continue

        val_loss, recalls, medr, meanr = _eval_alignment(datasets, zoo, q_test, enc_q, enc_m, pred, device, closs, mse)
        r1 = float(recalls[1])
        row = {"epoch": epoch + 1, "train_loss": float(loss.item()), "eval_loss": val_loss,
               "r1": r1, "r5": float(recalls[5]), "r10": float(recalls[10]),
               "medr": medr, "meanr": meanr}
        history.append(row)
        if _should_log_epoch(epoch, args.n_epochs):
            print(
                f"ep={epoch + 1:04d} train={row['train_loss']:.3f} eval={val_loss:.3f} "
                f"r1={r1:.1f} r5={row['r5']:.1f} r10={row['r10']:.1f} "
                f"med={medr:.1f} mean={meanr:.1f} t={time.time() - t0:.1f}s"
            )

        if r1 > best_r1:
            best_r1 = r1
            _save_checkpoint(args.save_path, enc_q, enc_m, pred, args, fdim, device, True, epoch + 1, recalls, medr, meanr)

    _save_checkpoint(args.save_path, enc_q, enc_m, pred, args, fdim, device, False, args.n_epochs)
    payload = {"architecture": args.architecture, "seed": args.seed, "n_epochs": args.n_epochs,
               "n_dims": args.n_dims, "functional_dim": fdim, "best_r1": max(best_r1, 0.0), "history": history,
               "args": {k: v for k, v in vars(args).items() if not k.startswith("_")}}
    save_json(args.save_path, "training_results.json", payload)
    print(f"Saved checkpoints and training_results.json to {args.save_path}")
    return payload


# Testing

def _load_checkpoint(args, device):
    load_path = Path(args.load_path)
    path = load_path / "saved_model_max_recall.pt"
    if not path.exists():
        path = load_path / "saved_model.pt"
    ckpt = load_torch(path)
    arch = canonicalize_model_type(ckpt.get("architecture", args.architecture))
    n = ckpt["n_dims"]
    enc_q = QueryEncoder(n).to(device)
    enc_m = ModelEncoder(n, ckpt["functional_dim"]).to(device)
    pred = PerformancePredictor(n).to(device)
    enc_q.load_state_dict(ckpt["enc_q"]); enc_q.eval()
    enc_m.load_state_dict(ckpt["enc_m"]); enc_m.eval()
    pred.load_state_dict(ckpt["predictor"]); pred.eval()
    return {
        "enc_q": enc_q, "enc_m": enc_m, "pred": pred,
        "architecture": arch, "functional_dim": int(ckpt["functional_dim"]),
        "n_noise_samples": int(ckpt.get("n_noise_samples", 16)),
        "use_features": bool(ckpt.get("use_features", True)),
        "path": str(path), "recall": ckpt.get("recall"), "medr": ckpt.get("medr"),
    }


def _build_retrieval_space(args, ckpt, device):
    generator = FunctionalEmbeddingGenerator(ckpt["n_noise_samples"], device, args.seed, ckpt["architecture"])
    datasets, embs, sds, accs = [], [], [], []

    for name in get_metatrain_datasets(ckpt["architecture"]):
        ms, dataset_accs, dataset_sds = load_models_from_zoo(name, args.n_nets, device, ckpt["architecture"])
        if ms is None:
            continue
        func_embs = generator.extract_batch(ms, ckpt["use_features"])
        with torch.no_grad():
            me = ckpt["enc_m"](func_embs.to(device))
        for i, sd in enumerate(dataset_sds):
            datasets.append(name)
            embs.append(me[i])
            sds.append(sd)
            accs.append(float(dataset_accs[i]))

    if not embs:
        raise RuntimeError("Retrieval space is empty: no meta-train checkpoints loaded")
    print(f"\nRetrieval space: {len(sds)} models")
    return {"datasets": datasets, "embs": torch.stack(embs), "sds": sds, "accs": accs}


def initialize_transfer_model(source_dataset, state_dict, num_classes, architecture, device):
    zoo_root = shared_get_zoo_path(source_dataset, model_type=architecture)
    config = get_model_config_cached(zoo_root)
    model = create_model_from_state_dict(state_dict, num_classes=num_classes, config=config, device=device)
    _, cls_prefix = detect_model_type(state_dict)
    body = {k: v for k, v in state_dict.items() if not k.startswith(f"{cls_prefix}.")}
    model.load_state_dict(body, strict=False)
    return model.to(device)


def run_finetune_single_lr(model, train_loader, test_loader, epochs, device, model_type, finetune_lr):
    family = canonicalize_model_type(model_type)
    wd = RESNET18SLIM_FINETUNE_WD if family == "resnet18slim" else CNN3_FINETUNE_WD
    m = copy.deepcopy(model).to(device)
    _, hist_pct, _ = shared_finetune_model(m, train_loader, test_loader, epochs=epochs, device=device, model_type=family, lr=finetune_lr, wd=wd)
    hist = [v / 100.0 for v in hist_pct]
    return {"epoch_1_acc": hist[0] if hist else 0.0, "final_acc": hist[-1] if hist else 0.0, "acc_history": hist}


def _eval_query_dataset(args, dataset_name, ckpt, space, extractor, device):
    trainset, testset = load_dataset_pt(dataset_name, ckpt["architecture"])
    if trainset is None:
        print(f"Skipping {dataset_name}: no dataset.pt")
        return None
    
    n_classes = get_num_classes(dataset_name, ckpt["architecture"])
    if n_classes > 20:
        print(f"Skipping {dataset_name}: {n_classes} classes")
        return None
    
    q_feat = query_features_for_dataset(dataset_name, extractor, args.n_query_images, args.seed, ckpt["architecture"])
    if q_feat is None:
        print(f"Skipping {dataset_name}: feature extraction failed")
        return None

    with torch.no_grad():
        q_emb = ckpt["enc_q"]([q_feat.to(device)])
    sims = torch.mm(q_emb, space["embs"].t()).squeeze(0)
    top_idx = torch.argsort(sims, descending=True)[: args.n_retrievals]
    top_embs = space["embs"][top_idx]
    
    with torch.no_grad():
        pred_scores = ckpt["pred"](q_emb.expand(top_embs.shape[0], -1), top_embs).squeeze(1)

    rank_of = {int(idx): rank + 1 for rank, idx in enumerate(top_idx.tolist())}
    eval_set = [int(top_idx[0].item())]
    
    for local in torch.argsort(pred_scores, descending=True)[: min(3, len(top_idx))].tolist():
        g = int(top_idx[local].item())
        if g not in eval_set:
            eval_set.append(g)

    train_loader = DataLoader(trainset, batch_size=_BATCH, shuffle=True, num_workers=_WORKERS)
    test_loader = DataLoader(testset, batch_size=_BATCH, shuffle=False, num_workers=_WORKERS)
    pred_by_idx = {int(top_idx[i]): float(pred_scores[i]) for i in range(len(top_idx))}
    finetune_lr = args.finetune_lr

    retrievals, selected = [], None
    for ridx in eval_set:
        src = space["datasets"][ridx]
        m = initialize_transfer_model(src, space["sds"][ridx], n_classes, ckpt["architecture"], device)
        zero_acc = shared_evaluate_model(m, test_loader, device=device) / 100.0
        ft = run_finetune_single_lr(m, train_loader, test_loader, args.n_eps_finetuning, device, ckpt["architecture"], finetune_lr)

        row = {"rank": rank_of[ridx], "source": src, "pred_acc": pred_by_idx[ridx],
            "zero_shot_acc": zero_acc, "epoch_1_acc": ft["epoch_1_acc"], "final_acc": ft["final_acc"],
            "acc_history": ft["acc_history"]}

        retrievals.append(row)
        if selected is None or row["final_acc"] > selected["final_acc"]:
            selected = row

    return {"query_dataset": dataset_name, "num_classes": n_classes, "retrievals": retrievals, "selected": selected}


def run_test_once(args, device):
    set_seed(args.seed)
    ckpt = _load_checkpoint(args, device)
    args.architecture = ckpt["architecture"]
    print(f"Loaded model from {ckpt['path']}")
    if ckpt["recall"]:
        print(f"Training R@1={float(ckpt['recall'][1]):.1f} | MedR={float(ckpt['medr']):.1f}")

    space = _build_retrieval_space(args, ckpt, device)
    extractor = ImageFeatureExtractor(device)
    test_datasets = list(args.ood_datasets) if args.ood_datasets else list(get_metatest_datasets(ckpt["architecture"]))
    test_datasets = [d for d in test_datasets if get_num_classes(d, ckpt["architecture"]) <= 20]
    print(f"Testing {len(test_datasets)} datasets")

    results = {}
    for name in test_datasets:
        r = _eval_query_dataset(args, name, ckpt, space, extractor, device)
        if r is not None:
            results[name] = r

    print_single_seed_summary(results, ckpt["architecture"], args.n_eps_finetuning)
    return {
        "architecture": ckpt["architecture"], "seed": args.seed,
        "checkpoint_path": ckpt["path"], "n_retrievals": args.n_retrievals,
        "n_finetune_epochs": args.n_eps_finetuning,
        "finetune_lr": args.finetune_lr, "datasets": results,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
    }


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="TANS cross-modal retrieval for model zoos", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="all", choices=["train", "test", "all"])
    parser.add_argument("--architecture", default="cnn3", choices=["cnn3", "resnet", "resnet18slim"])
    
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--save-multiseed", action="store_true")
    parser.add_argument("--ood-datasets", nargs="*", default=None)
    
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-dims", type=int, default=128)
    parser.add_argument("--n-nets", type=int, default=10)
    parser.add_argument("--n-query-images", type=int, default=100)
    parser.add_argument("--n-groups", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--n-noise-samples", type=int, default=16)
    parser.add_argument("--use-features", action="store_true", default=True)
    parser.add_argument("--no-use-features", dest="use_features", action="store_false")
    parser.add_argument("--n-retrievals", type=int, default=5)
    parser.add_argument("--n-eps-finetuning", type=int, default=10)

    parser.add_argument("--finetune-lr", type=float, default=None, help="Finetuning LR. Defaults to 1e-3 for cnn3, 1e-4 for resnet.")
    parser.add_argument("--save-path", default="/netscratch2/aasefaw/sane_refact/clean/sphere/internal-sane/models/tans_clean")
    parser.add_argument("--load-path", default=None)

    args = parser.parse_args()
    args.architecture = canonicalize_model_type(args.architecture)
    args.n_groups = len(get_metatrain_datasets(args.architecture)) if args.n_groups is None else int(args.n_groups)
    args.load_path = args.save_path if args.load_path is None else args.load_path
    if args.finetune_lr is None:
        args.finetune_lr = get_default_finetune_lr(args.architecture)
    args.seeds = resolve_seeds(args)
    args.ood_datasets = list(get_metatest_datasets(args.architecture)) if args.ood_datasets is None else list(args.ood_datasets)
    return args


def _print_runtime_overview(args, device):
    print("\nTANS cross-modal retrieval")
    print(f"mode={args.mode} arch={model_type_display_name(args.architecture)} device={device}, seed={args.seed} test_seeds={', '.join(str(seed) for seed in args.seeds)}")
    print(f"finetune_epochs={args.n_eps_finetuning}, finetune_lr={format_lr(float(args.finetune_lr))}")
    print(f"datasets={', '.join(args.ood_datasets)}")
    print(f"save={args.save_path} load={args.load_path}")

def run_test(args, device):
    if len(args.seeds) == 1:
        result = run_test_once(args, device)
        save_json(args.save_path, "test_results.json", result)
        print(f"Saved test_results.json to {args.save_path}")
        return result

    results_by_seed = {}
    for i, seed in enumerate(args.seeds, 1):
        print(f"\nSeed {i}/{len(args.seeds)}: {seed}")
        run_args = copy.copy(args)
        run_args.seed = int(seed)
        results_by_seed[int(seed)] = run_test_once(run_args, device)

    aggregate = aggregate_seed_results(results_by_seed, args.n_eps_finetuning)
    print_multi_seed_summary(aggregate, args.architecture)
    payload = {k: v for k, v in aggregate.items() if args.save_multiseed or k != "raw_results"}
    payload["args"] = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    save_json(args.save_path, "test_results.json", payload)
    print(f"Saved test_results.json to {args.save_path}")
    return aggregate


def main():
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _print_runtime_overview(args, device)
    if args.mode in {"train", "all"}:
        run_train(args, device)
    if args.mode in {"test", "all"}:
        run_test(args, device)


if __name__ == "__main__":
    main()
