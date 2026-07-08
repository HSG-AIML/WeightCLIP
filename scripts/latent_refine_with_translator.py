#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call

from dataset_to_model import _load_shared_latent_context, _resolve_anchor_state, apply_direct_decoder, fit_direct_decoder, resolve_checkpoint_file
from ood_utils import (
    METATEST_DATASETS,
    _adapt_output_layer,
    _get_finetune_defaults,
    _make_optimizer,
    canonicalize_model_type,
    create_model_from_state_dict,
    decode_embedding,
    describe_decode_preflight,
    detect_model_type,
    encode_model,
    evaluate_model,
    extract_dataset_embeddings_cached,
    get_bn_conditioning_loader,
    get_model_config_cached,
    get_num_classes,
    get_train_test_loaders,
    get_zoo_path,
    uses_bn_conditioning,
)
from shell_geometry_utils import compute_sphere_center, mean_token_norm, scale_toward_radius
from sane.data.datasets import ZooDataset

# Datasets the meta-train model zoos were trained on. Used to estimate the
# in-distribution shell radius from re-encoded zoo models for shell scaling.
METATRAIN_DATASETS = [
    "proptit-aif-homework", "cactus-aerial", "dl2020", "four-shapes", "e4040-assignment",
    "car-classification", "simpsons-characters", "simpsons-challenge", "breakhis", "mushrooms",
    "artworks", "numta-bengali", "lego-bricks", "ads5035", "day3-kaggle",
    "ct-images", "land-cover", "casting", "asl", "cassava-leaf",
]


def _strict_seed(seed: int) -> None:
    """Seed every RNG we touch + force cuDNN to deterministic mode.

    Goes beyond ``dataset_to_model.set_seed`` (which misses numpy + cuDNN) so
    baseline finetune / grad refinement are bit-reproducible at fixed --seed.
    """
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — propagates the main-process seed into workers."""
    base = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(base)
    random.seed(base)


def _deterministic_train_loader(train_loader: DataLoader, seed: int) -> DataLoader:
    """Rebuild ``train_loader`` with a fixed generator + worker_init_fn.

    The dataset object is reused, so this is cheap. Reseed the loader's generator
    before each run to make the shuffle sequence reproducible.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        train_loader.dataset,
        batch_size=getattr(train_loader, "batch_size", 1),
        shuffle=True,
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False),
        drop_last=False,
        generator=g,
        worker_init_fn=_seed_worker,
        persistent_workers=False,
    )


def _reseed(loader: DataLoader, seed: int) -> None:
    """Reset global RNGs + the loader's generator so the next iter() reproduces."""
    _strict_seed(seed)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(int(seed))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-decode model latents, then refine them with gradients through the decoder.")
    p.add_argument("--sane-ckpt", "--checkpoint_path", dest="sane_ckpt", required=True, help="Path to checkpoint.pt or its checkpoint directory.")
    p.add_argument("--dataset-encoder-ckpt", default=None, help="Optional dataset encoder checkpoint. Defaults to sibling dataset_encoder.pt.")
    p.add_argument("--datasets", "--dataset", "--ood-datasets", nargs="+", default=list(METATEST_DATASETS), help="Target datasets. Use 'metatest' for the built-in OOD set.")
    p.add_argument("--arch", "--model_type", dest="arch", default="cnn3", choices=["cnn3", "resnet", "resnet18slim"], help="Zoo model family.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--max-models-per-dataset", type=int, default=200, help="Training checkpoints per source dataset when fitting direct decode.")
    p.add_argument("--subset-size", type=int, default=10, help="Images per dataset-encoder query.")
    p.add_argument("--ood-samples", type=int, default=100, help="Target dataset embeddings to direct-decode.")
    p.add_argument("--max-eval-candidates", "--max_eval_samples", dest="max_eval_candidates", type=int, default=100, help="Direct-decode candidates to score before choosing one.")

    p.add_argument("--batch-size", "--train_batch_size", dest="batch_size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--objective-batches", "--train_batches", dest="objective_batches", type=int, default=1, help="Fixed train batches used for candidate scoring/refinement.")
    p.add_argument("--steps", type=int, default=10, help="Gradient refinement steps. Use 0 for direct-decode only.")
    p.add_argument("--grad-lr", "--grad_lr", dest="grad_lr", type=float, default=3e-2, help="Learning rate for latent refinement.")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--test-every", "--test_every", dest="test_every", type=int, default=10, help="If >0, evaluate test acc every N refinement steps.")
    p.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=3, help="Refine the N best candidates independently and report mean±std.")
    p.add_argument("--lr-schedule", "--lr_schedule", dest="lr_schedule", choices=["none"], default="none", help="LR schedule during refinement (only 'none' is supported).")
    p.add_argument("--baseline-finetune", action=argparse.BooleanOptionalAction, default=True, help="Also finetune the direct-decoded model for the same number of steps (reported as a baseline).")

    p.add_argument("--scale-to-shell", action=argparse.BooleanOptionalAction, default=True, help="Scale the selected latent to the ID shell radius before refinement.")
    p.add_argument("--sphere-center", "--sphere_center", dest="sphere_center", choices=["origin", "mean", "pairwise"], default="origin", help="Center used to measure the shell radius for scaling.")
    p.add_argument("--shell-alpha", type=float, default=1.0, help="Interpolation strength for shell scaling.")
    p.add_argument("--id-models-per-dataset", "--id_models_per_dataset", dest="id_models_per_dataset", type=int, default=20, help="Meta-train checkpoints per zoo used to estimate the ID shell radius.")

    # Kept only because dataset_to_model's shared loader prints/reads these fields.
    p.set_defaults(top_k=1, finetune_epochs=0)
    args = p.parse_args()
    args.arch = canonicalize_model_type(args.arch)
    args.datasets = expand_datasets(args.datasets)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def expand_datasets(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(METATEST_DATASETS if name.lower() in {"metatest", "metatest_all", "all", "all_ood"} else [name])
    return list(dict.fromkeys(out))


def load_direct_decoder(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, torch.Tensor | int]:
    state = fit_direct_decoder(context["x_train"], context["y_train"], device=args.device)
    print(f"Fitted direct decoder: weight={tuple(state['weight'].shape)}, seq_len={state['seq_len']}, token_dim={state['token_dim']}")
    return state


def estimate_pairwise_radius(points: np.ndarray, samples: int = 5000, seed: int = 0) -> tuple[float, float]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        raise ValueError("Need at least two points for pairwise radius.")
    rng = np.random.default_rng(int(seed))
    n_pairs = min(int(samples), len(points) * (len(points) - 1) // 2)
    d2 = np.empty(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        i, j = rng.choice(len(points), size=2, replace=False)
        diff = points[i] - points[j]
        d2[k] = float(np.dot(diff, diff))
    radius = float(np.sqrt(0.5 * d2.mean()))
    stderr = float(d2.std() / max(np.sqrt(n_pairs) * 4.0 * max(radius, 1e-8), 1e-8))
    return radius, stderr


def latent_radius(z: torch.Tensor, mode: str, seed: int = 0) -> tuple[float, float]:
    if mode in {"origin", "mean"}:
        return mean_token_norm(z, center_mode=mode), 0.0
    return estimate_pairwise_radius(z.detach().squeeze(0).float().cpu().numpy(), seed=seed)


def compute_id_radius_from_zoos(args: argparse.Namespace, context: dict[str, Any]) -> tuple[float, float, int]:
    """Re-encode meta-train zoo models with the encoder and estimate the ID shell radius.

    For ``sphere_center`` of ``origin``/``mean`` we average per-model token-norms; for
    ``pairwise`` we accumulate per-token points and use the pairwise-distance estimator.
    """
    if int(args.id_models_per_dataset) <= 0:
        raise ValueError("id_models_per_dataset must be > 0")

    sane_model = context["sane_model"]
    tokenizer = context["tokenizer"]
    sphere_center = args.sphere_center
    norms: list[float] = []
    token_points: list[np.ndarray] = []
    n_encoded = 0

    for ds_name in METATRAIN_DATASETS:
        zoo_path = get_zoo_path(ds_name, model_type=args.arch)
        if zoo_path is None or not Path(zoo_path).exists():
            continue
        try:
            zoo_ds = ZooDataset(root_dir=str(zoo_path), epoch_idx=-1, max_checkpoints=int(args.id_models_per_dataset))
        except Exception:
            continue
        if len(zoo_ds) == 0:
            continue
        for i in range(min(len(zoo_ds), int(args.id_models_per_dataset))):
            try:
                checkpoint = zoo_ds[i]
                state_dict = checkpoint.model.state_dict()
                _, z_full, _, _, _ = encode_model(state_dict, sane_model, tokenizer, device=args.device)
                if sphere_center in ("origin", "mean"):
                    norms.append(mean_token_norm(z_full, center_mode=sphere_center))
                else:
                    token_points.append(z_full.detach().squeeze(0).float().cpu().numpy())
                n_encoded += 1
            except Exception:
                continue

    if sphere_center in ("origin", "mean"):
        if not norms:
            raise RuntimeError("Failed to estimate ID radius: no meta-train zoo checkpoints encoded.")
        return float(np.mean(norms)), float(np.std(norms)), len(norms)

    if not token_points:
        raise RuntimeError("Failed to estimate ID pairwise radius: no meta-train zoo checkpoints encoded.")
    pts = np.concatenate(token_points, axis=0)
    r_pairwise, stderr = estimate_pairwise_radius(pts, samples=5000, seed=args.seed)
    return float(r_pairwise), float(stderr), int(n_encoded)


def scale_to_shell(z: torch.Tensor, args: argparse.Namespace, context: dict[str, Any]) -> torch.Tensor:
    print(f"\n=== Scaling latent to ID shell (center={args.sphere_center}) ===")
    radius, radius_std, n_models = compute_id_radius_from_zoos(args, context)
    print(f"  ID shell radius: r_id = {radius:.4f} ± {radius_std:.4f} (n={n_models} models)")

    z_dev = z.to(args.device)
    current_norm = mean_token_norm(z_dev, center_mode=args.sphere_center) if args.sphere_center in ("origin", "mean") else latent_radius(z_dev, args.sphere_center, seed=args.seed)[0]
    print(f"  Current latent norm: {current_norm:.4f}")

    center_mode = "mean" if args.sphere_center == "pairwise" else args.sphere_center
    center = compute_sphere_center(z_dev, center_mode)
    z_scaled = scale_toward_radius(z_dev, target_radius=radius, alpha=float(args.shell_alpha), center=center)
    scaled_norm = mean_token_norm(z_scaled, center_mode=args.sphere_center) if args.sphere_center in ("origin", "mean") else latent_radius(z_scaled, args.sphere_center, seed=args.seed + 1)[0]
    print(f"  Scaled latent norm:  {scaled_norm:.4f}")
    return z_scaled.detach().cpu()


def finetune_for_steps(model: nn.Module, train_loader, test_loader, args: argparse.Namespace, lr: float, wd: float) -> float:
    model = copy.deepcopy(model).to(args.device)
    optimizer = _make_optimizer(model, args.arch, float(lr), float(wd))
    criterion = nn.CrossEntropyLoss()
    iterator = iter(train_loader)
    model.train()
    for _ in range(max(0, int(args.steps))):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(args.device), labels.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        criterion(logits, labels).backward()
        optimizer.step()
    return evaluate_model(model, test_loader, args.device)


def run_baseline_finetune(base_model: nn.Module, train_loader, test_loader, args: argparse.Namespace) -> float | None:
    """Finetune the direct-decoded model for the same number of steps as latent refinement."""
    if not args.baseline_finetune or int(args.steps) <= 0:
        return None
    lr, wd = _get_finetune_defaults(args.arch)
    print(f"Baseline finetune: steps={args.steps}, lr={lr:.2e}, wd={wd:g}")
    _reseed(train_loader, args.seed)
    acc = finetune_for_steps(base_model, train_loader, test_loader, args, lr, wd)
    print(f"  lr={lr:.2e}: testAcc={acc:.2f}%")
    return float(acc)


def fixed_batches(loader, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches, it = [], iter(loader)
    for _ in range(max(1, int(count))):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))
    return batches


def functional_ce(model: nn.Module, params_and_buffers: dict[str, torch.Tensor], batches, device: str) -> torch.Tensor:
    total = torch.zeros((), device=device)
    seen = 0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = functional_call(model, params_and_buffers, (images,))
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        total = total + F.cross_entropy(logits, labels, reduction="sum")
        seen += int(labels.numel())
    return total / max(1, seen)


def decoded_state_for_grad(sane_model, tokenizer, latent: torch.Tensor, anchor_state: dict[str, torch.Tensor], anchor_mask: torch.Tensor, anchor_pos: torch.Tensor, target_state: dict[str, torch.Tensor], final_layer_key: str, device: str) -> dict[str, torch.Tensor]:
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)
    full_mask = anchor_mask[0] if anchor_mask.dim() == 3 else anchor_mask
    full_pos = anchor_pos[0] if anchor_pos.dim() == 3 else anchor_pos
    full_len = int(full_pos.shape[0])
    if int(latent.shape[1]) < full_len:
        raise ValueError(f"Predicted latent has {latent.shape[1]} tokens, but anchor requires {full_len}.")
    decoded = sane_model.decoder(latent[:, :full_len].to(device), full_pos[:full_len].unsqueeze(0).to(device))[0]
    state = tokenizer.detokenize(decoded, mask=full_mask.to(device), position=full_pos.to(device), reference_statedict=anchor_state)
    return _adapt_output_layer(state, target_state, final_layer_key)


def params_and_buffers(model: nn.Module, state: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    out = {name: state[name].to(device) for name, _ in model.named_parameters()}
    out.update({name: state.get(name, buffer).to(device) for name, buffer in model.named_buffers()})
    return out


def score_latent(sane_model, tokenizer, latent, anchor_state, anchor_mask, anchor_pos, template_model, final_layer_key, batches, device: str) -> float:
    with torch.no_grad():
        state = decoded_state_for_grad(sane_model, tokenizer, latent, anchor_state, anchor_mask, anchor_pos, template_model.state_dict(), final_layer_key, device)
        return float(functional_ce(template_model, params_and_buffers(template_model, state, device), batches, device).item())


def refine_latent(sane_model, tokenizer, z0, anchor_state, anchor_mask, anchor_pos, template_model, 
                  final_layer_key, batches, args, *, grad_lr: float | None = None, test_loader=None, eval_decoder=None) -> tuple[torch.Tensor, float, float | None]:
    """Refine latent. Returns (best_z, best_loss, best_test_acc)."""
    
    base_lr = float(grad_lr if grad_lr is not None else args.grad_lr)
    z = z0.detach().clone().to(args.device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=base_lr)
    best_z, best_loss = z.detach().clone(), float("inf")
    best_test_acc: float | None = None
    print(f"Gradient refinement: steps={args.steps}, lr={base_lr:g}")
    test_every = int(getattr(args, "test_every", 0))
    
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        state = decoded_state_for_grad(sane_model, tokenizer, z, anchor_state, anchor_mask, anchor_pos, template_model.state_dict(), final_layer_key, args.device)
        loss = functional_ce(template_model, params_and_buffers(template_model, state, args.device), batches, args.device)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss, best_z = loss_value, z.detach().clone()

        do_test = test_loader is not None and eval_decoder is not None and test_every > 0 and (step % test_every == 0 or step == args.steps)
        if do_test:
            test_model = eval_decoder(best_z)
            acc = evaluate_model(test_model, test_loader, args.device)
            best_test_acc = acc if best_test_acc is None else max(best_test_acc, acc)
            if step == 1 or step == args.steps or (args.print_every > 0 and step % args.print_every == 0):
                print(f"  step {step:03d}/{args.steps}: ce={loss_value:.4f}, testAcc={acc:.2f}%")
            continue
        
        if step == 1 or step == args.steps or (args.print_every > 0 and step % args.print_every == 0):
            print(f"  step {step:03d}/{args.steps}: ce={loss_value:.4f}")
    return best_z, best_loss, best_test_acc


def decode_for_eval(sane_model, tokenizer, latent, anchor_state, anchor_tokens, anchor_mask, anchor_pos, num_classes: int, model_config, bn_loader, args) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    return decode_embedding(sane_model, latent, anchor_state, anchor_tokens, anchor_mask, anchor_pos, tokenizer, num_classes, device=args.device, config=model_config, bn_dataloader=bn_loader)


def _refine_one_sample( *, candidate_idx: int, candidate_ce: float, pred_latents: torch.Tensor, dataset_name: str, args: argparse.Namespace,
    context: dict[str, Any], sane_model, tokenizer, anchor_state, anchor_tokens, anchor_mask, anchor_pos, template_model, final_layer_key: str,
    num_classes: int, model_config, bn_loader, train_loader, test_loader, batches) -> dict[str, Any]:
    
    z0 = pred_latents[candidate_idx : candidate_idx + 1]
    base_model, _ = decode_for_eval(sane_model, tokenizer, z0, anchor_state, anchor_tokens, anchor_mask, anchor_pos, num_classes, model_config, bn_loader, args)
    base_acc = evaluate_model(base_model, test_loader, args.device)
    print(f"{dataset_name}[cand={candidate_idx}]: trainCE={candidate_ce:.4f}, testAcc={base_acc:.2f}%")
    baseline_ft_acc = run_baseline_finetune(base_model, train_loader, test_loader, args)

    z_start = z0
    if args.scale_to_shell:
        z_start = scale_to_shell(z0, args, context)
        start_model, _ = decode_for_eval(sane_model, tokenizer, z_start, anchor_state, anchor_tokens, anchor_mask, anchor_pos, num_classes, model_config, bn_loader, args)
        print(f"{dataset_name}[cand={candidate_idx}]: refinement start testAcc={evaluate_model(start_model, test_loader, args.device):.2f}%")

    refined_acc, refined_ce = None, None
    if int(args.steps) > 0:
        def _eval_decoder(latent):
            model, _ = decode_for_eval(sane_model, tokenizer, latent, anchor_state, anchor_tokens, anchor_mask, anchor_pos, num_classes, model_config, bn_loader, args)
            return model

        _reseed(train_loader, args.seed)
        z_refined, refined_ce, _ = refine_latent(
            sane_model, tokenizer, z_start, anchor_state, anchor_mask, anchor_pos,
            template_model, final_layer_key, batches, args,
            grad_lr=float(args.grad_lr), test_loader=test_loader, eval_decoder=_eval_decoder,
        )
        refined_model, _ = decode_for_eval(sane_model, tokenizer, z_refined, anchor_state, anchor_tokens, anchor_mask, anchor_pos, num_classes, model_config, bn_loader, args)
        refined_acc = evaluate_model(refined_model, test_loader, args.device)
        print(f"{dataset_name}[cand={candidate_idx}]: refined trainCE={refined_ce:.4f}, testAcc={refined_acc:.2f}%")

    return {
        "candidate": candidate_idx,
        "base_ce": candidate_ce,
        "base_acc": base_acc,
        "baseline_ft_acc": baseline_ft_acc,
        "refined_ce": refined_ce,
        "refined_acc": refined_acc,
    }


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None, None, 0
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


def run_dataset(dataset_name: str, args: argparse.Namespace, context: dict[str, Any], direct_decoder: dict[str, torch.Tensor | int]) -> dict[str, Any]:
    root_dir = get_zoo_path(dataset_name, args.arch)
    raw_train_loader, test_loader = get_train_test_loaders(dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, root_dir=root_dir, model_type=args.arch)
    if raw_train_loader is None or test_loader is None:
        raise RuntimeError(f"Missing loaders for {dataset_name}.")

    sane_model, tokenizer = context["sane_model"], context["tokenizer"]
    num_classes = get_num_classes(dataset_name, args.arch)
    anchor_state = _resolve_anchor_state(dataset_name, args.arch, num_classes, context)
    if anchor_state is None:
        raise RuntimeError(f"No anchor checkpoint can support {num_classes} classes for {dataset_name}.")

    _, z_anchor, anchor_tokens, anchor_mask, anchor_pos = encode_model(anchor_state, sane_model, tokenizer, args.device)
    if int(direct_decoder["token_dim"]) != int(z_anchor.shape[2]):
        raise RuntimeError(f"Direct decoder token_dim={direct_decoder['token_dim']} but anchor token_dim={z_anchor.shape[2]}.")
    preflight = describe_decode_preflight(sane_model, anchor_state, anchor_tokens, num_classes, int(direct_decoder["seq_len"]))
    if not bool(preflight["decode_possible"]):
        raise RuntimeError(str(preflight["decode_reason"]))

    model_config = get_model_config_cached(root_dir)
    bn_loader = get_bn_conditioning_loader(root_dir, batch_size=args.batch_size, num_workers=args.num_workers) if root_dir is not None and uses_bn_conditioning(args.arch) else None
    template_model = create_model_from_state_dict(anchor_state, num_classes=num_classes, config=model_config, device=args.device).eval()
    _, final_layer_key = detect_model_type(anchor_state)

    seed = int(args.seed)
    _strict_seed(seed)
    train_loader = _deterministic_train_loader(raw_train_loader, seed)

    ds_embeddings = extract_dataset_embeddings_cached(context["dataset_encoder"], dataset_name, num_subsets=args.ood_samples, subset_size=args.subset_size, device=args.device, cache_dir=context["cache_dir"], root_dir=root_dir, image_size=context["image_size"], seed=seed)
    if ds_embeddings is None:
        raise RuntimeError(f"Could not extract dataset embeddings for {dataset_name}.")

    pred_latents = apply_direct_decoder(ds_embeddings, direct_decoder, device=args.device).detach().cpu()
    batches = fixed_batches(train_loader, args.objective_batches)

    scored: list[tuple[float, int]] = []
    for idx in range(min(int(args.max_eval_candidates), int(pred_latents.shape[0]))):
        try:
            scored.append((score_latent(sane_model, tokenizer, pred_latents[idx : idx + 1], anchor_state, anchor_mask, anchor_pos, template_model, final_layer_key, batches, args.device), idx))
        except Exception as exc:
            print(f"  candidate {idx} failed: {exc}")
    if not scored:
        raise RuntimeError("No direct-decode candidates could be scored.")
    scored.sort()

    selected = scored[: min(int(args.num_samples), len(scored))]
    multi_sample = len(selected) > 1
    if multi_sample:
        print(f"\n{dataset_name}: refining {len(selected)} best candidates by score")
        for rank, (ce, idx) in enumerate(selected):
            print(f"  sample {rank + 1}: candidate={idx}, scoreCE={ce:.4f}")

    all_results: list[dict[str, Any]] = []
    for rank, (cand_ce, cand_idx) in enumerate(selected):
        if multi_sample:
            print(f"\n{'#' * 60}\n# {dataset_name} sample {rank + 1}/{len(selected)} (candidate={cand_idx})\n{'#' * 60}")
        all_results.append(_refine_one_sample(
            candidate_idx=cand_idx, candidate_ce=cand_ce, pred_latents=pred_latents,
            dataset_name=dataset_name, args=args, context=context,
            sane_model=sane_model, tokenizer=tokenizer,
            anchor_state=anchor_state, anchor_tokens=anchor_tokens, anchor_mask=anchor_mask, anchor_pos=anchor_pos,
            template_model=template_model, final_layer_key=final_layer_key,
            num_classes=num_classes, model_config=model_config, bn_loader=bn_loader,
            train_loader=train_loader, test_loader=test_loader, batches=batches,
        ))

    base_mean, base_std, _ = _mean_std([r["base_acc"] for r in all_results])
    refined_mean, refined_std, refined_n = _mean_std([r["refined_acc"] for r in all_results])
    ft_mean, ft_std, _ = _mean_std([r["baseline_ft_acc"] for r in all_results])
    base_ce_mean, _, _ = _mean_std([r["base_ce"] for r in all_results])
    refined_ce_mean, _, _ = _mean_std([r["refined_ce"] for r in all_results])

    if len(all_results) > 1:
        print(f"\n{dataset_name}: aggregated over {len(all_results)} samples (seed={seed})")
        print(f"  base    testAcc: {base_mean:.2f}% ± {base_std:.2f}%")
        if ft_mean is not None:
            print(f"  baseFT  testAcc: {ft_mean:.2f}% ± {ft_std:.2f}%")
        if refined_mean is not None:
            print(f"  refined testAcc: {refined_mean:.2f}% ± {refined_std:.2f}% (n={refined_n})")

    return {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "num_samples": len(all_results),
        "seed": seed,
        "candidates": [r["candidate"] for r in all_results],
        "base_ce": base_ce_mean,
        "base_acc": base_mean,
        "base_acc_std": base_std,
        "baseline_ft_acc": ft_mean,
        "baseline_ft_acc_std": ft_std,
        "refined_ce": refined_ce_mean,
        "refined_acc": refined_mean,
        "refined_acc_std": refined_std,
    }


def _fmt_mean_std(mean: float | None, std: float | None, multi: bool) -> str:
    if mean is None:
        return "-"
    if multi and std is not None:
        return f"{mean:.2f}±{std:.2f}"
    return f"{mean:.2f}"


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else float(a) - float(b)


def _fmt_value(value: float | None, width: int = 12) -> str:
    return f"{'-':>{width}s}" if value is None else f"{value:>{width}.2f}"


def _fmt_signed(value: float | None, width: int = 8) -> str:
    return f"{'-':>{width}s}" if value is None else f"{value:>+{width}.2f}"


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    multi = any(int(r.get("num_samples", 1)) > 1 for r in results)
    header = f"{'dataset':24s} {'cls':>4s} {'n':>3s} {'base_ce':>9s} {'base':>12s} {'ft':>12s} {'refined':>12s} {'delta':>8s}"
    print("\nSummary")
    print(header)

    deltas: list[float] = []
    base_accs: list[float] = []
    ft_accs: list[float] = []
    refined_accs: list[float] = []
    for row in results:
        ce = row["base_ce"] if row["base_ce"] is not None else float("nan")
        delta = _delta(row.get("refined_acc"), row.get("baseline_ft_acc"))
        if delta is not None:
            deltas.append(delta)
        if row.get("base_acc") is not None:
            base_accs.append(float(row["base_acc"]))
        if row.get("baseline_ft_acc") is not None:
            ft_accs.append(float(row["baseline_ft_acc"]))
        if row.get("refined_acc") is not None:
            refined_accs.append(float(row["refined_acc"]))
        print(
            f"{row['dataset'][:24]:24s} "
            f"{row['num_classes']:4d} "
            f"{row.get('num_samples', 1):3d} "
            f"{ce:9.4f} "
            f"{_fmt_mean_std(row['base_acc'], row.get('base_acc_std'), multi):>12s} "
            f"{_fmt_mean_std(row['baseline_ft_acc'], row.get('baseline_ft_acc_std'), multi):>12s} "
            f"{_fmt_mean_std(row['refined_acc'], row.get('refined_acc_std'), multi):>12s} "
            f"{_fmt_signed(delta)}"
        )

    base_avg = float(np.mean(base_accs)) if base_accs else None
    ft_avg = float(np.mean(ft_accs)) if ft_accs else None
    refined_avg = float(np.mean(refined_accs)) if refined_accs else None
    delta_avg = float(np.mean(deltas)) if deltas else None
    print("-" * len(header))
    print(
        f"{'average':24s} {'':>4s} {'':>3s} {'':>9s} "
        f"{_fmt_value(base_avg)} "
        f"{_fmt_value(ft_avg)} "
        f"{_fmt_value(refined_avg)} "
        f"{_fmt_signed(delta_avg)}"
    )


def main() -> None:
    args = parse_args()
    _strict_seed(args.seed)
    checkpoint_file = resolve_checkpoint_file(args.sane_ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_file.parent / f"dataset_to_model_{args.arch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _load_shared_latent_context(args, output_dir)
    direct_decoder = load_direct_decoder(args, context)

    results = []
    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} ===")
        try:
            results.append(run_dataset(dataset_name, args, context, direct_decoder))
        except Exception as exc:
            print(f"skip {dataset_name}: {exc}")
    print_summary(results)


if __name__ == "__main__":
    main()
