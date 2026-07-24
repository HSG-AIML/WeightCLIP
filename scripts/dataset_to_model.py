#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
src_dir = repo_root / "src"

for path in (src_dir, repo_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ood_utils import (
    TEST_DATASETS,
    MemoryBankMixtureTranslator,
    canonicalize_model_type,
    decode_embedding,
    detect_model_type,
    describe_decode_preflight,
    encode_model,
    evaluate_model,
    extract_dataset_embeddings_cached,
    extract_training_embeddings,
    finetune_state_dict,
    get_bn_conditioning_loader,
    get_model_config_cached,
    get_num_classes,
    get_num_classes_from_state_dict,
    get_train_test_loaders,
    get_zoo_path,
    load_checkpoint_from_zoo,
    load_dataset_encoder,
    load_sane_bundle,
    model_type_display_name,
    train_memory_bank_translator,
    train_from_scratch,
    uses_bn_conditioning,
)

DEFAULT_FINETUNE_LR_CNN3 = 1.5e-3
DEFAULT_FINETUNE_LR_RESNET = 1.5e-4
SCRATCH_INIT_TYPES = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "uniform", "normal"]


def get_default_finetune_lr(arch: str) -> float:
    a = canonicalize_model_type(arch or "cnn3")
    return DEFAULT_FINETUNE_LR_RESNET if a == "resnet18slim" else DEFAULT_FINETUNE_LR_CNN3


def resolve_checkpoint_file(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path)
    return checkpoint_file / "checkpoint.pt" if checkpoint_file.is_dir() else checkpoint_file


def _get_training_embeddings_cache_path(checkpoint_file: Path, cache_dir: Path, arch: str, max_models_per_dataset: int) -> Path:
    checkpoint_file = checkpoint_file.resolve()
    stat = checkpoint_file.stat()
    cache_key = json.dumps(
        {
            "checkpoint": str(checkpoint_file),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "arch": canonicalize_model_type(arch),
            "max_models_per_dataset": int(max_models_per_dataset),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return cache_dir / f"training_emb_{digest}.pt"


def _load_training_embeddings_cache(cache_path: Path) -> dict[str, object] | None:
    if not cache_path.exists():
        return None

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    if not isinstance(payload.get("embeddings_by_dataset"), dict):
        return None

    if not isinstance(payload.get("root_dir_by_dataset"), dict):
        return None

    return payload


def _save_training_embeddings_cache(cache_path: Path, payload: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def _slice_anchor_to_num_classes(anchor_state: dict[str, torch.Tensor], target_num_classes: int) -> dict[str, torch.Tensor] | None:
    current_num_classes = get_num_classes_from_state_dict(anchor_state)
    target_num_classes = int(target_num_classes)

    if current_num_classes < target_num_classes:
        return None

    if current_num_classes == target_num_classes:
        return anchor_state

    _, final_layer_key = detect_model_type(anchor_state)
    weight_key = f"{final_layer_key}.weight"
    bias_key = f"{final_layer_key}.bias"
    sliced_state = {key: value.clone() for key, value in anchor_state.items()}
    sliced_state[weight_key] = sliced_state[weight_key][:target_num_classes].clone()
    sliced_state[bias_key] = sliced_state[bias_key][:target_num_classes].clone()
    return sliced_state


def _select_largest_fallback_anchor(train_state_dicts: dict[str, list[dict[str, torch.Tensor]]]) -> tuple[dict[str, torch.Tensor] | None, int | None]:
    best_anchor = None
    best_num_classes = None

    for state_dicts in train_state_dicts.values():
        for state_dict in state_dicts:
            num_classes = get_num_classes_from_state_dict(state_dict)
            if best_num_classes is None or num_classes > best_num_classes:
                best_anchor = state_dict
                best_num_classes = num_classes

    return best_anchor, best_num_classes


def _resolve_anchor_state(dataset_name: str, arch: str, target_num_classes: int, shared_context: dict[str, object]) -> dict[str, torch.Tensor] | None:
    dataset_anchor = load_checkpoint_from_zoo(dataset_name, arch)

    if dataset_anchor is not None:
        adapted_dataset_anchor = _slice_anchor_to_num_classes(dataset_anchor, target_num_classes)
        if adapted_dataset_anchor is not None:
            return adapted_dataset_anchor

    fallback_anchor = shared_context.get("fallback_anchor")
    if fallback_anchor is None:
        return None

    return _slice_anchor_to_num_classes(fallback_anchor, target_num_classes)


def _get_mapper_checkpoint_path(checkpoint_file: Path, dataset_encoder_ckpt: Path, cache_dir: Path, arch: str, max_models_per_dataset: int, subset_size: int, mapper_config: dict[str, object]) -> Path:
    checkpoint_file = checkpoint_file.resolve()
    dataset_encoder_ckpt = dataset_encoder_ckpt.resolve()
    checkpoint_stat = checkpoint_file.stat()
    dataset_encoder_stat = dataset_encoder_ckpt.stat()
    cache_key = json.dumps({"checkpoint": str(checkpoint_file), "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns, "checkpoint_size": checkpoint_stat.st_size, "dataset_encoder_ckpt": str(dataset_encoder_ckpt), 
                            "dataset_encoder_mtime_ns": dataset_encoder_stat.st_mtime_ns, "dataset_encoder_size": dataset_encoder_stat.st_size, "arch": canonicalize_model_type(arch), "max_models_per_dataset": int(max_models_per_dataset), "subset_size": int(subset_size), "mapper_config": mapper_config}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return cache_dir / f"memory_bank_{digest}.pt"


def _load_mapper_checkpoint(checkpoint_path: Path, device: str) -> tuple[MemoryBankMixtureTranslator | None, dict[str, object] | None]:
    if not checkpoint_path.exists():
        return None, None

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None

    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict) or not isinstance(payload.get("hparams"), dict):
        return None, None

    state_dict = payload["state_dict"]
    hparams = payload["hparams"]
    memory_bank = state_dict.get("memory_bank")

    if memory_bank is None:
        return None, None

    try:
        model = MemoryBankMixtureTranslator(input_dim=int(hparams["input_dim"]), seq_len=int(hparams["seq_len"]), token_dim=int(hparams["token_dim"]), d_model=int(hparams["d_model"]), n_head=int(hparams["n_head"]), n_layer=int(hparams["n_layer"]))
        model.set_memory_bank(memory_bank.cpu())
        model.load_state_dict(state_dict)
    except Exception:
        return None, None

    model = model.to(device)
    model.memory_bank = model.memory_bank.cpu()
    model.eval()
    return model, payload


def _save_mapper_checkpoint(checkpoint_path: Path, model: MemoryBankMixtureTranslator, train_config: dict[str, object]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": {"d_model": model.d_model, "n_head": model.n_head, "n_layer": model.n_layer, "input_dim": model.input_dim, "seq_len": model.seq_len, "token_dim": model.token_dim}, "train_config": train_config}, checkpoint_path)


def _sample_memory_bank_embeddings(mapper: MemoryBankMixtureTranslator, dataset_embs: torch.Tensor, sample_batch_size: int, device: str, temperature: float, top_k_cap: int) -> torch.Tensor:
    pred_chunks = []
    sample_top_k = min(int(top_k_cap), max(int(mapper.num_memory) // 4, 0)) if int(top_k_cap) > 0 else max(int(mapper.num_memory) // 4, 0)
    sample_top_k = sample_top_k if sample_top_k > 0 else None

    for start in range(0, len(dataset_embs), max(int(sample_batch_size), 1)):
        chunk = dataset_embs[start : start + max(int(sample_batch_size), 1)].to(device)
        pred_chunks.append(mapper.sample(chunk, mode="expectation", temperature=float(temperature), top_k=sample_top_k).cpu())

    return torch.cat(pred_chunks, dim=0).to(device)


def resolve_dataset_encoder_checkpoint(sane_ckpt: str | Path, dataset_encoder_ckpt: str | None) -> Path:
    if dataset_encoder_ckpt is not None:
        checkpoint = Path(dataset_encoder_ckpt)
        if checkpoint.exists():
            return checkpoint

        raise FileNotFoundError(f"Dataset encoder checkpoint not found: {checkpoint}")

    checkpoint_file = resolve_checkpoint_file(sane_ckpt)
    candidate = checkpoint_file.parent / "dataset_encoder.pt"

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not resolve dataset encoder checkpoint. Expected sibling dataset_encoder.pt "
        "or pass --dataset-encoder-ckpt explicitly."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_direct_decoder(x_dataset: torch.Tensor, y_model: torch.Tensor, ridge_scale: float = 1e-4, device: str | torch.device | None = None) -> dict[str, torch.Tensor | int]:
    target_device = device or x_dataset.device or y_model.device
    x_data = x_dataset.float().to(target_device)
    y_data = y_model.float().to(target_device)
    y_flat = y_data.reshape(y_data.shape[0], -1).contiguous()
    ones = torch.ones(x_data.shape[0], 1, dtype=x_data.dtype, device=x_data.device)
    x_aug = torch.cat([x_data, ones], dim=1)
    xtx = x_aug.t().mm(x_aug)
    lam = max(float(ridge_scale * xtx.diagonal().mean().item()), 1e-6)
    xtx = xtx + lam * torch.eye(xtx.shape[0], dtype=x_data.dtype, device=x_data.device)
    rhs = x_aug.t().mm(y_flat)

    try:
        chol = torch.linalg.cholesky(xtx)
        solution = torch.cholesky_solve(rhs, chol)
    except torch.linalg.LinAlgError:
        solution = torch.linalg.lstsq(xtx, rhs, driver="gelsd").solution

    return {"weight": solution[:-1].contiguous(), "bias": solution[-1].contiguous(), "seq_len": int(y_model.shape[1]), "token_dim": int(y_model.shape[2])}


def apply_direct_decoder(x_dataset: torch.Tensor, decoder_state: dict[str, torch.Tensor | int], device: str | torch.device | None = None) -> torch.Tensor:
    weight = decoder_state["weight"]
    bias = decoder_state["bias"]
    seq_len = int(decoder_state["seq_len"])
    token_dim = int(decoder_state["token_dim"])
    target_device = device or weight.device
    pred_flat = x_dataset.float().to(target_device).mm(weight) + bias.unsqueeze(0)
    return pred_flat.view(-1, seq_len, token_dim)


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=False).item()) if len(values) > 1 else 0.0


def summarize_values(values: list[float]) -> dict[str, object]:
    values = [float(value) for value in values]
    return {"best": max(values) if values else 0.0, "mean": mean_or_zero(values), "std": std_or_zero(values), "values": values}


def summarize_histories(histories: list[list[float]]) -> tuple[list[float], list[float]]:
    if not histories:
        return [], []

    max_len = max(len(history) for history in histories)
    mean_history = []
    std_history = []

    for epoch_idx in range(max_len):
        values = [float(history[epoch_idx]) for history in histories if epoch_idx < len(history)]
        mean_history.append(mean_or_zero(values))
        std_history.append(std_or_zero(values))

    return mean_history, std_history

def format_stats_triplet(best: float, mean: float, std: float) -> str:
    return f"{best:>5.2f}/{mean:>5.2f}/{std:>5.2f}"


def format_prefixed_stats(stats: dict[str, object], prefix: str) -> str:
    return format_stats_triplet(float(stats[f"{prefix}_best"]), float(stats[f"{prefix}_mean"]), float(stats[f"{prefix}_std"]))


def format_score_list(values: list[float]) -> str:
    return ", ".join(f"{float(value):.2f}" for value in values) if values else "-"


def format_preflight_line(preflight: dict[str, object]) -> str:
    random_weights = "yes" if bool(preflight["random_weights_added"]) else "no"
    anchor_reuse = "yes" if bool(preflight["anchor_token_reuse"]) else "no"
    decode_possible = "yes" if bool(preflight["decode_possible"]) else "no"
    return (
        f"preflight: mapper_tokens={int(preflight['predicted_seq_len'])}, "
        f"target_tokens={int(preflight['anchor_seq_len'])}, "
        f"window={int(preflight['sane_window_size'])}, "
        f"decode_possible={decode_possible}, "
        f"anchor_token_reuse={anchor_reuse}, "
        f"random_weights_added={random_weights}, "
        f"output_layer_action={preflight['output_layer_action']}"
    )


def resolve_scratch_lrs(args: argparse.Namespace) -> list[float]:
    scratch_lrs = [float(lr) for lr in (getattr(args, "scratch_lrs", None) or [])]
    if scratch_lrs:
        return scratch_lrs
    if getattr(args, "scratch_lr", None) is not None:
        return [float(args.scratch_lr)]
    return []


def resolve_scratch_init_types(args: argparse.Namespace) -> list[str]:
    scratch_init_types = list(getattr(args, "scratch_init_types", None) or [])
    if scratch_init_types:
        return scratch_init_types
    if getattr(args, "scratch_init_type", None) is not None:
        return [str(args.scratch_init_type)]
    return []


def _load_shared_latent_context(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    dataset_encoder_ckpt = resolve_dataset_encoder_checkpoint(args.sane_ckpt, args.dataset_encoder_ckpt)
    sane_model, trainset, _, tokenizer, sane_config = load_sane_bundle(args.sane_ckpt, args.device)
    ds_cfg = sane_config.get("dataset_encoder", {})
    image_size = tuple(ds_cfg.get("image_size", (32, 32)))
    dataset_encoder = load_dataset_encoder(dataset_encoder_ckpt, args.device, config=ds_cfg)

    alignment_cfg = sane_config.get("alignment", {})
    loss_cfg = sane_config.get("loss", {})
    num_classes = int(dataset_encoder.classifier.out_features) if hasattr(dataset_encoder, "classifier") else None
    fe_name = type(dataset_encoder.feature_extractor).__name__ if hasattr(dataset_encoder, "feature_extractor") else "<unknown>"
    se_name = type(dataset_encoder.set_encoder).__name__ if hasattr(dataset_encoder, "set_encoder") else "<unknown>"
    print("=== Runtime behavior (this eval run) ===")
    print(f"  arch:                              {args.arch}")
    print(f"  sane_ckpt:                         {args.sane_ckpt}")
    print(f"  dataset_encoder_ckpt:              {dataset_encoder_ckpt}")
    print(f"  feature_extractor:                 {fe_name}")
    print(f"  set_encoder:                       {se_name}")
    print(f"  embedding_dim:                     {dataset_encoder.embedding_dim}")
    print(f"  num_classes (cls head):            {num_classes}")
    print(f"  normalize_embeddings:              {dataset_encoder.normalize_embeddings}")
    print(f"  embedding_scale:                   {dataset_encoder.embedding_scale}")
    print(f"  image_size (sampling):             {image_size}")
    print(f"  subset_size (sampling, CLI):       {args.subset_size}")
    print(f"  ood_samples (CLI):                 {args.ood_samples}")
    print(f"  max_models_per_dataset (CLI):      {args.max_models_per_dataset}")
    print(f"  top_k (CLI):                       {args.top_k}")
    print(f"  finetune_epochs (CLI):             {args.finetune_epochs}")
    print("=== Training-time hyperparams (informational, from SANE checkpoint config) ===")
    print(f"  dataset_encoder.classification_weight (initial):     {ds_cfg.get('classification_weight', '<default>')}")
    print(f"  dataset_encoder.classification_weight_anneal_epochs: {ds_cfg.get('classification_weight_anneal_epochs', 0)}")
    print(f"  dataset_encoder.set_size (train):                    {ds_cfg.get('set_size', '<default>')}")
    print(f"  alignment.weight:                                    {alignment_cfg.get('weight', '<default>')}")
    print(f"  alignment.objective:                                 {alignment_cfg.get('objective', '<default>')}")
    print(f"  alignment.temperature:                               {alignment_cfg.get('temperature', '<default>')}")
    print(f"  loss.gamma (SANE contrastive):                       {loss_cfg.get('gamma', '<default>')}")
    print("===============================================")
        
    checkpoint_file = resolve_checkpoint_file(args.sane_ckpt)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "embedding_cache"
    training_cache_path = _get_training_embeddings_cache_path(checkpoint_file, cache_dir, args.arch, args.max_models_per_dataset)
    cached_training = _load_training_embeddings_cache(training_cache_path)
    fallback_anchor = None
    fallback_anchor_num_classes = None

    if cached_training is not None:
        print(f"Loading cached training embeddings from {training_cache_path}")
        train_embeddings = cached_training["embeddings_by_dataset"]
        train_root_dirs = cached_training["root_dir_by_dataset"]
        fallback_anchor = cached_training.get("fallback_anchor")
        fallback_anchor_num_classes = cached_training.get("fallback_anchor_num_classes")

        if fallback_anchor is None or fallback_anchor_num_classes is None:
            print("Cached training embeddings missing max-class fallback anchor metadata; rebuilding fallback anchor from training checkpoints.")
            _, train_state_dicts, _ = extract_training_embeddings(sane_model, trainset, max_per_dataset=args.max_models_per_dataset, device=args.device, arch_filter=args.arch)
            fallback_anchor, fallback_anchor_num_classes = _select_largest_fallback_anchor(train_state_dicts)
            if fallback_anchor is not None:
                _save_training_embeddings_cache(
                    training_cache_path,
                    {
                        "embeddings_by_dataset": train_embeddings,
                        "root_dir_by_dataset": train_root_dirs,
                        "fallback_anchor": fallback_anchor,
                        "fallback_anchor_num_classes": int(fallback_anchor_num_classes),
                    },
                )
    else:
        train_embeddings, train_state_dicts, train_root_dirs = extract_training_embeddings(sane_model, trainset, max_per_dataset=args.max_models_per_dataset, device=args.device, arch_filter=args.arch)
        fallback_anchor, fallback_anchor_num_classes = _select_largest_fallback_anchor(train_state_dicts)
        if train_embeddings and fallback_anchor is not None:
            _save_training_embeddings_cache(training_cache_path, {"embeddings_by_dataset": train_embeddings, "root_dir_by_dataset": train_root_dirs, "fallback_anchor": fallback_anchor, "fallback_anchor_num_classes": int(fallback_anchor_num_classes)})
            print(f"Saved training embeddings cache to {training_cache_path}")

    if not train_embeddings:
        raise RuntimeError(f"No training embeddings found for arch={args.arch}")

    max_seq_len = max(int(embs.shape[1]) for embs in train_embeddings.values())
    latent_dim = int(next(iter(train_embeddings.values())).shape[2])

    x_chunks = []
    y_chunks = []

    for ds_name, model_embs in train_embeddings.items():
        dataset_embs = extract_dataset_embeddings_cached(dataset_encoder, ds_name, num_subsets=len(model_embs), subset_size=args.subset_size, device=args.device, cache_dir=cache_dir, root_dir=train_root_dirs[ds_name], image_size=image_size, seed=args.seed)

        if dataset_embs is None:
            print(f"Could not extract dataset embeddings for {ds_name}, skipping training pair.")
            continue

        count = min(len(model_embs), len(dataset_embs))
        if count == 0:
            print(f"No embeddings available for {ds_name}, skipping training pair.")
            continue

        padded_model_embs = model_embs[:count]
        if padded_model_embs.shape[1] < max_seq_len:
            pad = torch.zeros(count, max_seq_len - padded_model_embs.shape[1], latent_dim, dtype=padded_model_embs.dtype)
            padded_model_embs = torch.cat([padded_model_embs, pad], dim=1)

        x_chunks.append(dataset_embs[:count])
        y_chunks.append(padded_model_embs)

    if not x_chunks or fallback_anchor is None:
        raise RuntimeError("Could not assemble cached latent training pairs.")

    return {"dataset_encoder_ckpt": str(dataset_encoder_ckpt), "dataset_encoder": dataset_encoder, "sane_model": sane_model, "tokenizer": tokenizer, "cache_dir": cache_dir,
            "fallback_anchor": fallback_anchor, "x_train": torch.cat(x_chunks, dim=0), "y_train": torch.cat(y_chunks, dim=0), "image_size": image_size}


def retrieve_neighbour_embeddings(query_embeddings: torch.Tensor, training_dataset_embeddings: torch.Tensor, training_model_embeddings: torch.Tensor, metric: str = "euclidean") -> tuple[torch.Tensor, dict[str, object]]:
    metric = str(metric).lower()
    query = query_embeddings.float().cpu()
    train = training_dataset_embeddings.float().cpu()

    if metric == "cosine":
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        train_norm = torch.nn.functional.normalize(train, dim=-1)
        scores = query_norm.mm(train_norm.t())
        indices = scores.argmax(dim=1)
        selected_scores = scores.max(dim=1).values
        score_name = "cosine_similarity"
        
    elif metric == "euclidean":
        query_sq = (query ** 2).sum(dim=1, keepdim=True)
        train_sq = (train ** 2).sum(dim=1).unsqueeze(0)
        distances_sq = query_sq + train_sq - 2 * query.mm(train.t())
        distances = torch.sqrt(distances_sq.clamp(min=1e-8))
        indices = distances.argmin(dim=1)
        selected_scores = distances.min(dim=1).values
        score_name = "euclidean_distance"
    else:
        raise ValueError(f"Unsupported neighbour metric: {metric}")

    pred_embs = training_model_embeddings.index_select(0, indices.to(training_model_embeddings.device))
    return pred_embs, {"retrieval_metric": metric, "retrieval_score_name": score_name, "retrieval_score_mean": float(selected_scores.mean().item()) if selected_scores.numel() else 0.0, 
                       "retrieval_score_min": float(selected_scores.min().item()) if selected_scores.numel() else 0.0, "retrieval_score_max": float(selected_scores.max().item()) if selected_scores.numel() else 0.0, 
                       "retrieved_unique_count": int(torch.unique(indices).numel())}


def _evaluate_predicted_embeddings(args: argparse.Namespace, dataset_name: str, pred_embs: torch.Tensor, shared_context: dict[str, object]) -> dict[str, object] | None:
    root_dir = get_zoo_path(dataset_name, args.arch)
    train_loader, test_loader = get_train_test_loaders(dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, root_dir=root_dir, model_type=args.arch)

    if train_loader is None or test_loader is None:
        print(f"skip {dataset_name}: missing dataset.pt or zoo path")
        return None

    sane_model = shared_context["sane_model"]
    tokenizer = shared_context["tokenizer"]
    fallback_anchor = shared_context["fallback_anchor"]
    num_classes = get_num_classes(dataset_name, args.arch)
    anchor_state = _resolve_anchor_state(dataset_name, args.arch, num_classes, shared_context)

    if anchor_state is None:
        fallback_classes = get_num_classes_from_state_dict(fallback_anchor) if fallback_anchor is not None else None
        print(f"skip {dataset_name}: no anchor checkpoint with classifier size >= {num_classes} (fallback has {fallback_classes})")
        return None

    _, _, anchor_tokens, anchor_mask, anchor_pos = encode_model(anchor_state, sane_model, tokenizer, args.device)
    preflight = describe_decode_preflight(sane_model, anchor_state, anchor_tokens, num_classes, int(pred_embs.shape[1]))
    model_config = get_model_config_cached(root_dir)
    bn_loader = get_bn_conditioning_loader(root_dir, batch_size=args.batch_size, num_workers=args.num_workers) if root_dir is not None and uses_bn_conditioning(args.arch) else None

    print(f"preflight {dataset_name}: {format_preflight_line(preflight)}")
    print(f"preflight {dataset_name} detail: {preflight['decode_reason']}")

    if not bool(preflight["decode_possible"]):
        print(f"skip {dataset_name}: {preflight['decode_reason']}")
        return None

    candidates = []

    for emb in pred_embs:
        try:
            model, state_dict = decode_embedding(sane_model, emb, anchor_state, anchor_tokens, anchor_mask, anchor_pos, tokenizer, num_classes, device=args.device, config=model_config, bn_dataloader=bn_loader)
            pre_acc = evaluate_model(model, test_loader, args.device)
            candidates.append({"pre_acc": float(pre_acc), "state_dict": state_dict})
        except Exception as exc:
            print(f"decode failed for {dataset_name}: {exc}")

    if not candidates:
        print(f"skip {dataset_name}: no decoded candidates")
        return None

    candidates.sort(key=lambda item: item["pre_acc"], reverse=True)
    all_pre_accs = [float(item["pre_acc"]) for item in candidates]
    top_candidates = candidates[: min(args.top_k, len(candidates))]
    pre_stats = summarize_values([float(item["pre_acc"]) for item in top_candidates])

    epoch_1_scores: list[float] = []
    final_scores: list[float] = []
    finetune_lr = float(getattr(args, "finetune_lr", None) or get_default_finetune_lr(args.arch))
    if args.finetune_epochs > 0:
        print(f"Finetuning with lr={finetune_lr:.0e}")
        for item in top_candidates:
            final_acc, history, _ = finetune_state_dict(item["state_dict"], num_classes, train_loader, test_loader, epochs=args.finetune_epochs, device=args.device, config=model_config, model_type=args.arch, lr=finetune_lr)
            epoch_1_scores.append(float(history[0]) if history else 0.0)
            final_scores.append(float(final_acc))

    epoch_1_stats = summarize_values(epoch_1_scores) if epoch_1_scores else {"best": 0.0, "mean": 0.0, "std": 0.0, "values": []}
    final_stats = summarize_values(final_scores) if final_scores else {"best": 0.0, "mean": 0.0, "std": 0.0, "values": []}

    return {
        "preflight": preflight,
        "num_candidates": len(candidates),
        "top_k": len(top_candidates),
        "pre_best": float(pre_stats["best"]),
        "pre_mean": float(pre_stats["mean"]),
        "pre_std": float(pre_stats["std"]),
        "pre_all": all_pre_accs,
        "pre_topk": pre_stats["values"],
        "finetune_lr": finetune_lr,
        "epoch_1_best": float(epoch_1_stats["best"]),
        "epoch_1_mean": float(epoch_1_stats["mean"]),
        "epoch_1_std": float(epoch_1_stats["std"]),
        "post_best": float(final_stats["best"]),
        "post_mean": float(final_stats["mean"]),
        "post_std": float(final_stats["std"]),
        "post_topk": final_stats["values"],
    }


def run_direct_decode(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    shared_context = _load_shared_latent_context(args, output_dir)
    decoder_state = fit_direct_decoder(shared_context["x_train"], shared_context["y_train"], device=args.device)
    results: dict[str, dict[str, object]] = {}

    for dataset_name in args.ood_datasets:
        root_dir = get_zoo_path(dataset_name, args.arch)
        dataset_embs = extract_dataset_embeddings_cached(shared_context["dataset_encoder"], dataset_name, num_subsets=args.ood_samples, subset_size=args.subset_size, device=args.device, cache_dir=shared_context["cache_dir"], root_dir=root_dir, image_size=shared_context["image_size"], seed=args.seed)

        if dataset_embs is None:
            print(f"skip {dataset_name}: could not extract dataset embeddings")
            continue

        pred_embs = apply_direct_decoder(dataset_embs, decoder_state, device=args.device)
        dataset_result = _evaluate_predicted_embeddings(args, dataset_name, pred_embs, shared_context)
        if dataset_result is not None:
            results[dataset_name] = dataset_result
            _print_single_dataset_result(dataset_name, dataset_result, args.finetune_epochs, verbose=bool(args.verbose))

    return {"dataset_encoder_ckpt": shared_context["dataset_encoder_ckpt"], "finetune_epochs": int(args.finetune_epochs), "finetune_lr": float(args.finetune_lr if hasattr(args, "finetune_lr") and args.finetune_lr is not None else get_default_finetune_lr(args.arch)), "datasets": results}


def run_neighbour(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    shared_context = _load_shared_latent_context(args, output_dir)
    results: dict[str, dict[str, object]] = {}
    neighbour_runs = max(int(args.top_k), 1)

    for dataset_name in args.ood_datasets:
        root_dir = get_zoo_path(dataset_name, args.arch)
        dataset_embs = extract_dataset_embeddings_cached(shared_context["dataset_encoder"], dataset_name, num_subsets=neighbour_runs, subset_size=args.subset_size, device=args.device, cache_dir=shared_context["cache_dir"], root_dir=root_dir, image_size=shared_context["image_size"], seed=args.seed)

        if dataset_embs is None:
            print(f"skip {dataset_name}: could not extract dataset embeddings")
            continue

        pred_embs, retrieval_stats = retrieve_neighbour_embeddings(dataset_embs, shared_context["x_train"], shared_context["y_train"], metric=args.neighbour_metric)
        dataset_result = _evaluate_predicted_embeddings(args, dataset_name, pred_embs, shared_context)
        if dataset_result is None:
            continue
        dataset_result.update(retrieval_stats)
        results[dataset_name] = dataset_result
        _print_single_dataset_result(dataset_name, dataset_result, args.finetune_epochs, verbose=bool(args.verbose))

    return {"dataset_encoder_ckpt": shared_context["dataset_encoder_ckpt"], "finetune_epochs": int(args.finetune_epochs), "finetune_lr": float(args.finetune_lr if hasattr(args, "finetune_lr") and args.finetune_lr is not None else get_default_finetune_lr(args.arch)), "neighbour_metric": str(args.neighbour_metric), "datasets": results}


def run_memory_bank(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    shared_context = _load_shared_latent_context(args, output_dir)
    checkpoint_file = resolve_checkpoint_file(args.sane_ckpt)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "embedding_cache"
    dataset_encoder_ckpt = Path(str(shared_context["dataset_encoder_ckpt"]))
    mapper_config = {"epochs": int(args.mapper_epochs), "patience": int(args.mapper_patience), "lr": float(args.mapper_lr), "batch_size": int(args.mapper_batch_size), "d_model": int(args.mapper_d_model), "n_head": int(args.mapper_n_head), 
                     "n_layer": int(args.mapper_n_layer), "leave_one_out": not bool(args.no_mapper_leave_one_out), "use_amp": not bool(args.no_mapper_amp), "token_subsample": int(args.mapper_token_subsample) if args.mapper_token_subsample is not None else None}
    mapper_checkpoint = _get_mapper_checkpoint_path(checkpoint_file, dataset_encoder_ckpt, cache_dir, args.arch, args.max_models_per_dataset, args.subset_size, mapper_config)
    mapper = None
    mapper_payload = None
    mapper_source = "trained"

    if not args.mapper_retrain:
        mapper, mapper_payload = _load_mapper_checkpoint(mapper_checkpoint, args.device)
        if mapper is not None:
            mapper_source = "checkpoint"
            print(f"Loading cached memory-bank mapper from {mapper_checkpoint}")

    if mapper is None:
        mapper, mapper_loss, mapper_history = train_memory_bank_translator(shared_context["x_train"], shared_context["y_train"], epochs=args.mapper_epochs, lr=args.mapper_lr, batch_size=args.mapper_batch_size, d_model=args.mapper_d_model, n_head=args.mapper_n_head, 
                                                                           n_layer=args.mapper_n_layer, device=args.device, leave_one_out=not args.no_mapper_leave_one_out, use_amp=not args.no_mapper_amp, token_subsample=args.mapper_token_subsample, patience=args.mapper_patience, log_every=args.mapper_log_every)
        mapper_payload = {"train_config": {**mapper_config, "loss": float(mapper_loss), "history": mapper_history}}
        _save_mapper_checkpoint(mapper_checkpoint, mapper, mapper_payload["train_config"])
        print(f"Saved memory-bank mapper to {mapper_checkpoint}")

    results: dict[str, dict[str, object]] = {}

    for dataset_name in args.ood_datasets:
        root_dir = get_zoo_path(dataset_name, args.arch)
        dataset_embs = extract_dataset_embeddings_cached(shared_context["dataset_encoder"], dataset_name, num_subsets=args.ood_samples, subset_size=args.subset_size, device=args.device, cache_dir=shared_context["cache_dir"], root_dir=root_dir, image_size=shared_context["image_size"], seed=args.seed)

        if dataset_embs is None:
            print(f"skip {dataset_name}: could not extract dataset embeddings")
            continue

        pred_embs = _sample_memory_bank_embeddings(mapper, dataset_embs, args.mapper_sample_batch_size, args.device, args.mapper_temperature, args.mapper_top_k)
        dataset_result = _evaluate_predicted_embeddings(args, dataset_name, pred_embs, shared_context)
        if dataset_result is None:
            continue
        results[dataset_name] = dataset_result
        _print_single_dataset_result(dataset_name, dataset_result, args.finetune_epochs, verbose=bool(args.verbose))

    return {"dataset_encoder_ckpt": shared_context["dataset_encoder_ckpt"], "finetune_epochs": int(args.finetune_epochs), "finetune_lr": float(args.finetune_lr if hasattr(args, "finetune_lr") and args.finetune_lr is not None else get_default_finetune_lr(args.arch)), "mapper_checkpoint": str(mapper_checkpoint), "mapper_source": mapper_source, "mapper_config": mapper_config,
            "mapper_train_config": mapper_payload.get("train_config") if isinstance(mapper_payload, dict) else None, "mapper_temperature": float(args.mapper_temperature), "mapper_top_k": int(args.mapper_top_k), "datasets": results}


def run_scratch(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, dict[str, object]] = {}
    runs = max(int(getattr(args, "scratch_runs", 1)), 1)
    base_seed = int(getattr(args, "seed", 0))
    scratch_lrs = resolve_scratch_lrs(args)
    scratch_init_types = resolve_scratch_init_types(args)

    for dataset_idx, dataset_name in enumerate(args.ood_datasets):
        root_dir = get_zoo_path(dataset_name, args.arch)
        train_loader, test_loader = get_train_test_loaders(dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, root_dir=root_dir, model_type=args.arch)

        if train_loader is None or test_loader is None:
            print(f"skip {dataset_name}: missing dataset.pt or zoo path")
            continue

        sample_batch = next(iter(train_loader))
        channels_in = int(sample_batch[0].shape[1])
        num_classes = get_num_classes(dataset_name, args.arch)
        scratch_sweep = []

        for init_idx, init_type in enumerate(scratch_init_types):
            for lr_idx, lr in enumerate(scratch_lrs):
                print(f"Running scratch with init={init_type}, lr={lr:.0e}")
                epoch_1_scores = []
                final_scores = []
                histories = []

                for run_idx in range(runs):
                    set_seed(base_seed + dataset_idx * len(scratch_init_types) * len(scratch_lrs) * runs + init_idx * len(scratch_lrs) * runs + lr_idx * runs + run_idx)
                    final_acc, history, _ = train_from_scratch(num_classes, channels_in, train_loader, test_loader, epochs=args.scratch_epochs, device=args.device, model_type=args.arch, lr=float(lr), init_type=init_type)
                    epoch_1_scores.append(float(history[0]) if history else 0.0)
                    final_scores.append(float(final_acc))
                    histories.append([float(value) for value in history])

                epoch_1_stats = summarize_values(epoch_1_scores)
                final_stats = summarize_values(final_scores)
                history_mean, history_std = summarize_histories(histories)
                scratch_sweep.append(
                    {
                        "lr": float(lr),
                        "lr_label": f"{lr:.0e}",
                        "init_type": init_type,
                        "setting_label": f"{init_type} @ {lr:.0e}",
                        "epoch_1_best": float(epoch_1_stats["best"]),
                        "epoch_1_mean": float(epoch_1_stats["mean"]),
                        "epoch_1_std": float(epoch_1_stats["std"]),
                        "epoch_1_values": epoch_1_stats["values"],
                        "final_best": float(final_stats["best"]),
                        "final_mean": float(final_stats["mean"]),
                        "final_std": float(final_stats["std"]),
                        "final_values": final_stats["values"],
                        "history": history_mean,
                        "history_mean": history_mean,
                        "history_std": history_std,
                        "histories": histories,
                    }
                )

        chosen_setting = max(scratch_sweep, key=lambda item: (float(item["final_mean"]), float(item["final_best"]))) if scratch_sweep else None

        results[dataset_name] = {
            "runs": runs,
            "chosen_lr": float(chosen_setting["lr"]) if chosen_setting is not None else None,
            "chosen_lr_label": chosen_setting["lr_label"] if chosen_setting is not None else None,
            "chosen_init_type": chosen_setting["init_type"] if chosen_setting is not None else None,
            "chosen_setting_label": chosen_setting["setting_label"] if chosen_setting is not None else None,
            "epoch_1_acc": float(chosen_setting["epoch_1_mean"]) if chosen_setting is not None else 0.0,
            "epoch_1_mean": float(chosen_setting["epoch_1_mean"]) if chosen_setting is not None else 0.0,
            "epoch_1_std": float(chosen_setting["epoch_1_std"]) if chosen_setting is not None else 0.0,
            "epoch_1_values": chosen_setting["epoch_1_values"] if chosen_setting is not None else [],
            "final_acc": float(chosen_setting["final_mean"]) if chosen_setting is not None else 0.0,
            "final_mean": float(chosen_setting["final_mean"]) if chosen_setting is not None else 0.0,
            "final_std": float(chosen_setting["final_std"]) if chosen_setting is not None else 0.0,
            "final_values": chosen_setting["final_values"] if chosen_setting is not None else [],
            "history": chosen_setting["history_mean"] if chosen_setting is not None else [],
            "history_mean": chosen_setting["history_mean"] if chosen_setting is not None else [],
            "history_std": chosen_setting["history_std"] if chosen_setting is not None else [],
            "histories": chosen_setting["histories"] if chosen_setting is not None else [],
            "scratch_sweep": scratch_sweep,
            "lr_sweep": scratch_sweep,
        }
        _print_single_scratch_result(dataset_name, results[dataset_name], args.scratch_epochs)

    return {"scratch_lrs": [float(lr) for lr in scratch_lrs], "scratch_init_types": scratch_init_types, "runs": runs, "datasets": results}


def _print_single_dataset_result(dataset_name: str, stats: dict[str, object], finetune_epochs: int, verbose: bool = False) -> None:
    show_epoch_1 = int(finetune_epochs) >= 1
    show_final = int(finetune_epochs) > 1
    final_label = f"epoch {int(finetune_epochs)}"
    finetune_lr = stats.get("finetune_lr")
    lr_display = f"{float(finetune_lr):.0e}" if finetune_lr is not None else "-"
    preflight = stats.get("preflight")

    print(f"\n{dataset_name}")
    if isinstance(preflight, dict):
        print(format_preflight_line(preflight))
        print(f"preflight detail: {preflight['decode_reason']}")
    if "retrieval_metric" in stats:
        print(f"retrieval: metric={stats['retrieval_metric']}, score={stats['retrieval_score_name']}, mean={float(stats['retrieval_score_mean']):.4f}, min={float(stats['retrieval_score_min']):.4f}, max={float(stats['retrieval_score_max']):.4f}, unique={int(stats['retrieved_unique_count'])}")
    print(f"finetune lr: {lr_display}")

    print(f"{'stage':<10} {'best':>8} {'mean':>8} {'std':>8}")
    print("-" * 38)
    print(f"{'epoch 0':<10} {float(stats['pre_best']):>8.2f} {float(stats['pre_mean']):>8.2f} {float(stats['pre_std']):>8.2f}")

    if verbose:
        print(f"zero-shot all ({int(stats['num_candidates'])}): {format_score_list(stats.get('pre_all', []))}")

    if show_epoch_1:
        print(f"{'epoch 1':<10} {float(stats['epoch_1_best']):>8.2f} {float(stats['epoch_1_mean']):>8.2f} {float(stats['epoch_1_std']):>8.2f}")

    if show_final:
        print(f"{final_label:<10} {float(stats['post_best']):>8.2f} {float(stats['post_mean']):>8.2f} {float(stats['post_std']):>8.2f}")


def _print_ranked_summary(title: str, results: dict[str, dict[str, object]], arch: str, finetune_epochs: int, verbose: bool = False) -> None:
    if not results:
        return

    print(f"\n{title} summary ({model_type_display_name(arch)})")

    for dataset_name, stats in results.items():
        _print_single_dataset_result(dataset_name, stats, finetune_epochs, verbose=verbose)


def print_direct_summary(results: dict[str, dict[str, object]], arch: str, finetune_epochs: int, verbose: bool = False) -> None:
    _print_ranked_summary("Direct decode", results, arch, finetune_epochs, verbose=verbose)


def print_neighbour_summary(results: dict[str, dict[str, object]], arch: str, finetune_epochs: int, verbose: bool = False) -> None:
    _print_ranked_summary("Neighbour", results, arch, finetune_epochs, verbose=verbose)


def print_memory_bank_summary(results: dict[str, dict[str, object]], arch: str, finetune_epochs: int, verbose: bool = False) -> None:
    _print_ranked_summary("Memory bank", results, arch, finetune_epochs, verbose=verbose)


def _print_single_scratch_result(dataset_name: str, stats: dict[str, object], scratch_epochs: int) -> None:
    final_label = f"epoch {int(scratch_epochs)}"
    print(f"\n{dataset_name}")
    print(f"selected init: {stats.get('chosen_init_type') or '-'}")
    print(f"selected lr: {stats.get('chosen_lr_label') or '-'}")
    print(f"{'stage':<10} {'best':>8} {'mean':>8} {'std':>8}")
    print("-" * 38)
    print(f"{'epoch 1':<10} {max((float(value) for value in stats.get('epoch_1_values', [])), default=0.0):>8.2f} {float(stats['epoch_1_mean']):>8.2f} {float(stats['epoch_1_std']):>8.2f}")
    print(f"{final_label:<10} {max((float(value) for value in stats.get('final_values', [])), default=0.0):>8.2f} {float(stats['final_mean']):>8.2f} {float(stats['final_std']):>8.2f}")

    scratch_sweep = stats.get("scratch_sweep", [])
    if not scratch_sweep:
        return

    print("\nscratch sweep")
    print(f"{'init':<18} {'lr':<8} {'epoch 1 best':>12} {'mean':>8} {'std':>8} {final_label + ' best':>14} {'mean':>8} {'std':>8}")
    print("-" * 93)
    for sweep_stats in scratch_sweep:
        print(
            f"{sweep_stats['init_type']:<18}"
            f" {sweep_stats['lr_label']:<8}"
            f" {float(sweep_stats['epoch_1_best']):>12.2f} {float(sweep_stats['epoch_1_mean']):>8.2f} {float(sweep_stats['epoch_1_std']):>8.2f}"
            f" {float(sweep_stats['final_best']):>14.2f} {float(sweep_stats['final_mean']):>8.2f} {float(sweep_stats['final_std']):>8.2f}"
        )


def print_scratch_summary(results: dict[str, dict[str, object]], arch: str, runs: int, scratch_epochs: int) -> None:
    if not results:
        return
    print(f"\nScratch summary ({model_type_display_name(arch)}, runs={runs})")
    for dataset_name, stats in results.items():
        _print_single_scratch_result(dataset_name, stats, scratch_epochs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slim dataset-to-model evaluation: direct_decode, neighbour, memory_bank, and scratch.")

    # Checkpoints.
    parser.add_argument("--sane-ckpt", required=True, help="Path to checkpoint.pt or its checkpoint directory.")
    parser.add_argument("--dataset-encoder-ckpt", default=None, help="Optional dataset encoder checkpoint. Defaults to sibling dataset_encoder.pt.")

    # Evaluation mode.
    parser.add_argument("--arch", choices=["cnn3", "resnet18slim"], default="cnn3", help="Architecture family to evaluate.")
    parser.add_argument("--mode", choices=["direct_decode", "neighbour", "memory_bank", "scratch", "both"], default="direct_decode", help="Which evaluation mode to run.")
    parser.add_argument("--ood-datasets", nargs="*", default=None, help="Test datasets to evaluate. Defaults to the built-in test list.")

    parser.add_argument("--max-models-per-dataset", type=int, default=200, help="Maximum training checkpoints per train dataset for the direct map.")
    parser.add_argument("--subset-size", type=int, default=10, help="Images per dataset-encoder set.")
    parser.add_argument("--ood-samples", type=int, default=100, help="Dataset-encoder samples per target dataset.")
    parser.add_argument("--neighbour-metric", "--neighbor-metric", dest="neighbour_metric", choices=["euclidean", "cosine"], default="cosine", help="Distance metric for neighbour retrieval.")
    
    parser.add_argument("--mapper-epochs", type=int, default=5000, help="Maximum training epochs for the memory-bank mapper before early stopping.")
    parser.add_argument("--mapper-patience", type=int, default=300, help="Early stopping patience for the memory-bank mapper.")
    parser.add_argument("--mapper-log-every", type=int, default=200, help="Epoch interval for memory-bank training logs.")
    parser.add_argument("--mapper-lr", type=float, default=0.25e-4, help="Learning rate for the memory-bank mapper.")
    parser.add_argument("--mapper-batch-size", type=int, default=80, help="Training batch size for the memory-bank mapper.")
    parser.add_argument("--mapper-d-model", type=int, default=512, help="Transformer width for the memory-bank mapper.")
    parser.add_argument("--mapper-n-head", type=int, default=8, help="Attention heads for the memory-bank mapper.")
    parser.add_argument("--mapper-n-layer", type=int, default=4, help="Transformer layers for the memory-bank mapper.")
    parser.add_argument("--mapper-sample-batch-size", type=int, default=50, help="Chunk size when sampling memory-bank predictions for OOD datasets.")
    parser.add_argument("--mapper-temperature", type=float, default=0.5, help="Sampling temperature for the memory-bank mapper.")
    parser.add_argument("--mapper-top-k", type=int, default=32, help="Maximum memory-bank top-k used during OOD sampling before the old num_memory // 4 cap is applied.")
    parser.add_argument("--mapper-token-subsample", type=int, default=None, help="Optional token subsample count per batch for memory-bank training.")
    parser.add_argument("--no-mapper-leave-one-out", action="store_true", help="Disable leave-one-out soft targets for memory-bank training.")
    parser.add_argument("--no-mapper-amp", action="store_true", help="Disable AMP for memory-bank training.")
    parser.add_argument("--mapper-retrain", action="store_true", help="Ignore any saved memory-bank mapper checkpoint and retrain.")
    
    parser.add_argument("--top-k", type=int, default=5, help="How many decoded candidates to finetune.")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Finetuning epochs for top-k direct_decode candidates.")
    parser.add_argument("--finetune-lr", type=float, default=None, help="Learning rate for finetuning. Defaults to 1e-3 for cnn3, 1e-4 for resnet.")

    # Shared runtime controls.
    parser.add_argument("--scratch-epochs", type=int, default=10, help="Scratch-training epochs for the baseline.")
    parser.add_argument("--scratch-lr", type=float, default=3e-3, help="Fallback learning rate for scratch training when --scratch-lrs is unset.")
    parser.add_argument("--scratch-lrs", nargs="*", type=float, default=None, help="Learning rates to sweep for scratch training. Defaults to --scratch-lr if unset.")
    parser.add_argument("--scratch-init-type", choices=SCRATCH_INIT_TYPES, default="normal", help="Fallback weight initialization for scratch training when --scratch-init-types is unset.")
    parser.add_argument("--scratch-init-types", nargs="*", choices=SCRATCH_INIT_TYPES, default=None, help="Weight initialization schemes to sweep for scratch training. Defaults to --scratch-init-type if unset.")
    parser.add_argument("--scratch-runs", type=int, default=5, help="Number of independently initialized scratch models to train per dataset.")
    
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for dataset loaders and BN conditioning.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for cached dataset embeddings.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for results.json.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed zero-shot results, including every decoded candidate score.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.arch = canonicalize_model_type(args.arch)
    args.ood_datasets = args.ood_datasets or list(TEST_DATASETS)
    set_seed(args.seed)
    if args.finetune_lr is None:
        args.finetune_lr = get_default_finetune_lr(args.arch)
    scratch_lrs = resolve_scratch_lrs(args)
    scratch_init_types = resolve_scratch_init_types(args)

    print(f"Using device: {args.device}")
    print(f"The finetune lr is {args.finetune_lr:.0e}")
    print(f"The scratch lrs are {', '.join(f'{lr:.0e}' for lr in scratch_lrs)}")
    print(f"The scratch init types are {', '.join(scratch_init_types)}")

    checkpoint_file = resolve_checkpoint_file(args.sane_ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_file.parent / f"dataset_to_model_{args.arch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "arch": args.arch,
        "mode": args.mode,
        "sane_ckpt": str(checkpoint_file),
        "ood_datasets": list(args.ood_datasets),
    }

    if args.mode in {"direct_decode", "both"}:
        direct_results = run_direct_decode(args, output_dir)
        payload["direct_decode"] = direct_results
        print_direct_summary(direct_results["datasets"], args.arch, int(direct_results["finetune_epochs"]), verbose=bool(args.verbose))

    if args.mode == "neighbour":
        neighbour_results = run_neighbour(args, output_dir)
        payload["neighbour"] = neighbour_results
        print_neighbour_summary(neighbour_results["datasets"], args.arch, int(neighbour_results["finetune_epochs"]), verbose=bool(args.verbose))

    if args.mode == "memory_bank":
        memory_bank_results = run_memory_bank(args, output_dir)
        payload["memory_bank"] = memory_bank_results
        print_memory_bank_summary(memory_bank_results["datasets"], args.arch, int(memory_bank_results["finetune_epochs"]), verbose=bool(args.verbose))

    if args.mode in {"scratch", "both"}:
        scratch_results = run_scratch(args)
        payload["scratch"] = scratch_results
        print_scratch_summary(scratch_results["datasets"], args.arch, int(scratch_results["runs"]), int(args.scratch_epochs))

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()