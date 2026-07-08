#!/usr/bin/env python3
"""Trainer for the Text2Model EVLayer-based hypernetwork.

Two-phase training:
  1. Warm-start the DeepSets dataset encoder by classifying which meta-train
     ResNet zoo a randomly-sampled image set belongs to.
  2. Meta-train the hypernetwork via Text2Model's outer/inner trick:
       generated init -> short inner SGD adaptation -> outer step via
       autograd.grad(weights, hnet.params, grad_outputs=init - final).
"""

from __future__ import annotations

import copy
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hnet_utils import DatasetTensorCache, TEXT2MODEL_MODELS_PATH, Text2ModelStructuredStateHyperNetwork, build_dataset_encoder_for_hnet, infer_target_classes, mask_logits, mean_std, normalize_dataset_name

from ood_utils import create_model_from_state_dict, get_train_test_loaders
from sane.data.datasets.zoo_dataset_models import ResNet18Slim
from sane.data.tokenizers.sparse import SparseTokenizer
from tans_utils import get_num_classes as get_resnet_num_classes, get_zoo_path as get_resnet_zoo_path


class HyperResNetTrainer:
    """Owns the dataset encoder, hypernetwork, tokenizer, and meta-training loop."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_datasets = [normalize_dataset_name(ds) for ds in args.train_datasets]
        self.target_classes = int(args.target_classes or infer_target_classes(self.train_datasets))
        self.tensor_cache = DatasetTensorCache(max_cache_samples=args.max_cache_samples)
        self.checkpoint_path = Path(args.save_dir) / "hyper_resnet18slim.pt"
        self._initialize_components()

    def _initialize_components(self) -> None:
        self.args.target_classes = int(self.target_classes)

        self.template_model = ResNet18Slim(channels_in=3, o_dim=self.target_classes, dropout=0.0, width_mult=self.args.width_mult).to(self.device)
        self.template_state_dict = copy.deepcopy(self.template_model.state_dict())
        self.learnable_param_names = [name for name, _ in self.template_model.named_parameters()]

        self.tokenizer = SparseTokenizer(tokensize=self.args.token_size, device="cpu", mode="full_model", reference_statedict=self.template_state_dict)
        anchor_tokens, anchor_mask, anchor_pos = self.tokenizer(self.template_state_dict)
        self.anchor_token_shape = tuple(anchor_tokens.shape)

        self.dataset_encoder, self.dataset_encoder_metadata = build_dataset_encoder_for_hnet(self.args, self.device)
        self.hypernetwork = Text2ModelStructuredStateHyperNetwork(input_dim=int(self.dataset_encoder.embedding_dim), anchor_mask=anchor_mask, anchor_pos=anchor_pos, token_dim=int(anchor_tokens.shape[1]), hidden_dim=self.args.hnet_hidden_dim, token_window_size=self.args.token_window_size, position_embedding=self.args.token_position_embedding).to(self.device)

        params = list(self.hypernetwork.parameters()) + list(self.dataset_encoder.parameters())
        self.outer_optimizer = torch.optim.AdamW(params, lr=self.args.outer_lr, weight_decay=self.args.outer_wd)


    def _sample_subset_tensor(self, dataset_name: str, subset_size: int) -> torch.Tensor:
        images = self.tensor_cache.get(dataset_name)
        n = min(int(subset_size), len(images))
        if n <= 0:
            raise ValueError(f"Dataset {dataset_name} has no cached training images")
        return images[torch.randperm(len(images))[:n]]

    def _compute_dataset_embedding(self, dataset_name: str, train_mode: bool) -> torch.Tensor:
        images = self.tensor_cache.get(dataset_name)
        subset_size = min(self.args.subset_size, len(images))
        subsets = [images[torch.randperm(len(images))[:subset_size]] for _ in range(self.args.n_subsets)]
        batch = torch.stack(subsets).to(self.device)
        # DeepSets uses BN; with n_subsets==1 we must use eval-mode BN.
        if train_mode and int(self.args.n_subsets) > 1:
            self.dataset_encoder.train()
        else:
            self.dataset_encoder.eval()
        with torch.set_grad_enabled(train_mode):
            return self.dataset_encoder.encode(batch).mean(dim=0)

    def _pretrain_dataset_encoder(self) -> None:
        epochs = int(self.args.dataset_encoder_pretrain_epochs)
        if epochs <= 0:
            return
        K = len(self.train_datasets)
        batch_size = int(self.args.dataset_encoder_pretrain_batch_size)
        steps = int(self.args.dataset_encoder_pretrain_steps_per_epoch)
        subset_size = int(self.args.subset_size)
        opt = torch.optim.AdamW(self.dataset_encoder.parameters(), lr=self.args.dataset_encoder_pretrain_lr, weight_decay=self.args.dataset_encoder_pretrain_wd)

        print(f"Warm-starting dataset encoder for {epochs} epoch(s) over {K} ResNet zoos")
        self.dataset_encoder.train()
        for p in self.dataset_encoder.parameters():
            p.requires_grad = True

        for ep in range(epochs):
            start = time.time()
            losses, accs = [], []
            for _ in range(steps):
                labels = torch.arange(batch_size, device=self.device) % K
                labels = labels[torch.randperm(batch_size, device=self.device)]
                sets = [self._sample_subset_tensor(self.train_datasets[int(l)], subset_size) for l in labels.cpu().tolist()]
                batch = torch.stack(sets).to(self.device)
                logits = self.dataset_encoder(batch)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    losses.append(float(loss.item()))
                    accs.append(float((logits.argmax(1) == labels).float().mean().item() * 100.0))
            l_m, l_s = mean_std(losses)
            a_m, a_s = mean_std(accs)
            print(f"[ds-pretrain {ep+1:03d}] loss={l_m:.4f}±{l_s:.4f} | acc={a_m:.2f}±{a_s:.2f}% | t={time.time()-start:.1f}s")

        self.dataset_encoder.train()

    def tokens_to_state_dict(self, tokens: torch.Tensor) -> OrderedDict:
        return self.tokenizer.detokenize(tokens, mask=self.hypernetwork.anchor_mask, position=self.hypernetwork.anchor_pos, reference_statedict=self.template_state_dict)

    def build_model_from_state(self, state_dict, model_config=None) -> nn.Module:
        model = create_model_from_state_dict(state_dict, config=model_config, device=str(self.device))
        model.load_state_dict(state_dict, strict=True)
        return model

    def _inner_adapt(self, init_state: Dict[str, torch.Tensor], train_loader: DataLoader, num_classes: int) -> Dict[str, torch.Tensor]:
        model = self.build_model_from_state(init_state)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.args.inner_lr, momentum=self.args.inner_momentum, weight_decay=self.args.inner_wd)
        model.train()
        for _ in range(int(self.args.inner_epochs)):
            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.args.inner_max_batches > 0 and batch_idx >= self.args.inner_max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = mask_logits(model(images), num_classes)
                loss = criterion(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return model.state_dict()

    def _get_loaders(self, dataset_name: str) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        root = get_resnet_zoo_path(dataset_name, architecture="resnet")
        return get_train_test_loaders(dataset_name, batch_size=self.args.batch_size, num_workers=self.args.num_workers, root_dir=root)

    def _meta_step(self, dataset_name: str) -> Dict[str, float]:
        num_classes = get_resnet_num_classes(dataset_name, architecture="resnet")
        train_loader, _ = self._get_loaders(dataset_name)
        if train_loader is None:
            raise FileNotFoundError(f"No train loader for {dataset_name}")

        dataset_embedding = self._compute_dataset_embedding(dataset_name, train_mode=True)
        generated_tokens = self.hypernetwork(dataset_embedding)
        init_state = self.tokens_to_state_dict(generated_tokens)
        final_state = self._inner_adapt(init_state, train_loader, num_classes)

        self.outer_optimizer.zero_grad(set_to_none=True)
        meta_params = list(self.hypernetwork.parameters()) + list(self.dataset_encoder.parameters())

        delta_theta, weight_list = [], []
        for name in self.learnable_param_names:
            tensor = init_state[name]
            weight_list.append(tensor)
            delta_theta.append((tensor.detach() - final_state[name].detach()).to(tensor.dtype))

        grads = torch.autograd.grad(weight_list, meta_params, grad_outputs=delta_theta, allow_unused=True)
        for p, g in zip(meta_params, grads):
            if g is not None:
                p.grad = g
        self.outer_optimizer.step()

        with torch.no_grad():
            delta_norm = float(torch.sqrt(sum((d.float() ** 2).sum() for d in delta_theta)).cpu().item())
        return {"delta_norm": delta_norm}

    def train(self) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        print("=" * 70)
        print("Training ResNet18Slim hypernetwork (Text2Model EVLayer)")
        print("=" * 70)
        print(f"Train datasets ({len(self.train_datasets)}): {self.train_datasets}")
        print(f"Target classes (output width): {self.target_classes}")
        print(f"Device: {self.device}")
        print(f"EVLayer source: {TEXT2MODEL_MODELS_PATH}")
        print(f"Tokenizer: sparse/full_model | seq_len={self.anchor_token_shape[0]} | token_dim={self.anchor_token_shape[1]}")
        print(f"Hypernetwork params: {sum(p.numel() for p in self.hypernetwork.parameters()):,}")
        print(f"Checkpoint path: {self.checkpoint_path}")

        self._pretrain_dataset_encoder()

        for ep in range(int(self.args.meta_epochs)):
            start = time.time()
            order = list(self.train_datasets)
            random.shuffle(order)
            metrics = []
            for ds in order:
                m = self._meta_step(ds)
                metrics.append(m)
                print(f"epoch {ep+1:03d} | {ds:18s} | delta={m['delta_norm']:.2f}")
            d_m, d_s = mean_std(m["delta_norm"] for m in metrics)
            print(f"[epoch {ep+1:03d}] delta={d_m:.2f}±{d_s:.2f} | t={time.time()-start:.1f}s")
            self.save_checkpoint(epoch=ep + 1)

        print("Training complete.")

    def save_checkpoint(self, epoch: int) -> None:
        torch.save({"epoch": int(epoch), "args": vars(self.args), "train_datasets": list(self.train_datasets), "target_classes": int(self.target_classes), "tokenizer": {"tokensize": int(self.args.token_size), "ignore_bn": bool(self.args.tokenizer_ignore_bn), "seq_len": int(self.anchor_token_shape[0]), "token_dim": int(self.anchor_token_shape[1]), "position_embedding": self.args.token_position_embedding}, "dataset_encoder_state_dict": self.dataset_encoder.state_dict(), "dataset_encoder_metadata": self.dataset_encoder_metadata, "hypernetwork_state_dict": self.hypernetwork.state_dict()}, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        path = Path(checkpoint_path or self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.target_classes = int(payload["target_classes"])
        self.train_datasets = list(payload.get("train_datasets", self.train_datasets))
        self.args.train_datasets = list(self.train_datasets)
        saved = payload.get("args", {})
        for k in ("width_mult", "token_size", "token_window_size", "token_position_embedding", "tokenizer_ignore_bn", "hnet_hidden_dim", "dataset_encoder_preset", "dataset_embedding_dim"):
            if k in saved:
                setattr(self.args, k, saved[k])
        self._initialize_components()
        self.dataset_encoder.load_state_dict(payload["dataset_encoder_state_dict"])
        self.hypernetwork.load_state_dict(payload["hypernetwork_state_dict"])
        self.dataset_encoder.to(self.device).eval()
        self.hypernetwork.to(self.device).eval()
        print(f"Loaded checkpoint from {path}")
