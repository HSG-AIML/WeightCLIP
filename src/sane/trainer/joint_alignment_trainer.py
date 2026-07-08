import copy
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import autocast

from sane.loss.alignment import DatasetAlignmentLoss
from sane.loss.deepsets_classification import (
    DatasetEncoderAuxiliaryLosses,
    build_logical_dataset_registry,
)
from sane.model import build_dataset_encoder
from sane.model.dataset_encoder import ENCODER_PRESETS
from sane.trainer.contrastive_trainer import SANEContrastiveTrainer
from sane.trainer.trainer import _load_trial_config

logger = logging.getLogger(__name__)


class SANEJointAlignmentTrainer(SANEContrastiveTrainer):
    """Joint trainer for SANE plus dataset-encoder alignment.

    The existing ``loss`` config remains the SANE-only branch (reconstruction /
    contrastive / etc.). This trainer adds two auxiliary branches on
    top of it:

        total = sane_loss + alignment_loss + deepsets_cls_loss

    Alignment updates both models. Classification only affects the
    dataset encoder. The SANE loss branch remains unchanged.
    """

    def setup(self, config: Dict[str, Any]):
        config = copy.deepcopy(config)
        tokens_cfg = config.setdefault("data", {}).setdefault("tokens_dataset", {})
        tokens_cfg["include_zoo_metadata"] = True

        super().setup(config)

        self._setup_dataset_encoder_stack()
        self._setup_joint_optimizer()
        if self.config.get("scheduler") is not None:
            self._setup_joint_scheduler()
        else:
            self.scheduler = None

    # Setup helpers

    def _resolved_checkpoint_configs(self):
        raw = self.config["data"]["checkpoints_dataset"]
        resolved = self._resolve_chkpt_cfg(raw)
        return resolved if isinstance(resolved, list) else [resolved]

    def _get_n_tokens(self) -> int:
        for data in self.trainloader:
            if isinstance(data, Mapping):
                return int(data["tokens"].shape[-2])
        return super()._get_n_tokens()

    def _get_max_positions(self) -> List[int]:
        max_pos = None
        for data in self.trainloader:
            if isinstance(data, Mapping):
                position = data["position"]
                if position.dim() == 4:
                    position = position[:, 0]
                batch_max_pos = torch.amax(
                    position,
                    dim=tuple(range(position.ndim - 1)),
                ).int().tolist()
                if max_pos is None:
                    max_pos = batch_max_pos
                else:
                    for idx, value in enumerate(batch_max_pos):
                        if value > max_pos[idx]:
                            max_pos[idx] = value
                continue
        if max_pos is not None:
            return [pos + 1 for pos in max_pos]
        return super()._get_max_positions()

    def _resolve_zoo_roots(self) -> list[str]:
        return [cfg.root_dir for cfg in self._resolved_checkpoint_configs()]

    def _setup_dataset_encoder_stack(self) -> None:
        dataset_encoder_cfg = copy.deepcopy(self.config.get("dataset_encoder", {}))
        if not dataset_encoder_cfg:
            raise ValueError(
                "Joint alignment training requires a top-level 'dataset_encoder' config block."
            )

        preset = dataset_encoder_cfg.pop("preset", None)
        model_cfg: Dict[str, Any] = {}
        if preset is not None:
            if preset not in ENCODER_PRESETS:
                raise ValueError(f"Unknown dataset_encoder preset: {preset}")
            model_cfg.update(copy.deepcopy(ENCODER_PRESETS[preset]))

        zoo_roots = self._resolve_zoo_roots()
        self.logical_dataset_to_class_id = build_logical_dataset_registry(zoo_roots)

        num_classes = dataset_encoder_cfg.pop("num_classes", "auto")
        if num_classes == "auto":
            num_classes = len(self.logical_dataset_to_class_id)
        model_cfg["num_classes"] = int(num_classes)

        image_size = dataset_encoder_cfg.pop("image_size", (32, 32))
        input_channels = int(dataset_encoder_cfg.get("input_channels", 3))
        self.dataset_encoder_image_size = tuple(image_size)
        self.dataset_encoder_input_channels = input_channels
        self.dataset_encoder_set_size = int(dataset_encoder_cfg.pop("set_size", 32))
        self.dataset_encoder_classification_weight = float(
            dataset_encoder_cfg.pop("classification_weight", 0.0)
        )
        self.dataset_encoder_classification_weight_anneal_epochs = int(
            dataset_encoder_cfg.pop("classification_weight_anneal_epochs", 0)
        )

        model_cfg.update(dataset_encoder_cfg)
        model_cfg.setdefault("input_channels", input_channels)

        self.dataset_encoder = build_dataset_encoder(model_cfg).to(self.device)

        sane_latent_dim = int(self.config["model"].get("latent_dim", 128))
        if sane_latent_dim != self.dataset_encoder.embedding_dim:
            raise ValueError(
                f"SANE latent_dim ({sane_latent_dim}) must match dataset_encoder embedding_dim "
                f"({self.dataset_encoder.embedding_dim})."
            )

        alignment_cfg = copy.deepcopy(self.config.get("alignment", {}))
        alignment_cfg.setdefault("weight", 0.0)
        alignment_cfg.setdefault("set_size", self.dataset_encoder_set_size)
        alignment_cfg.setdefault("image_resize", self.dataset_encoder_image_size)
        if alignment_cfg.get("freeze_deepsets", False) and (self.dataset_encoder_classification_weight > 0):
            raise ValueError("alignment.freeze_deepsets=True is incompatible with dataset_encoder classification loss.")
        
        self.alignment_loss_fn = DatasetAlignmentLoss(self.dataset_encoder, **alignment_cfg)

        self.dataset_encoder_aux = DatasetEncoderAuxiliaryLosses(
            dataset_encoder=self.dataset_encoder,
            logical_dataset_to_class_id=self.logical_dataset_to_class_id,
            set_size=self.dataset_encoder_set_size,
            image_resize=self.dataset_encoder_image_size,
            classification_weight=self.dataset_encoder_classification_weight,
            target_channels=self.dataset_encoder_input_channels,
        )

    def _collect_param_groups(self, module: torch.nn.Module, recurse: bool = True) -> tuple[list, list]:
        params = list(module.parameters()) if recurse else [param for _, param in module.named_parameters(recurse=False)]
        decay = [p for p in params if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in params if p.requires_grad and p.dim() < 2]
        return decay, nodecay

    def _setup_joint_optimizer(self) -> None:
        optimizer_config = self.config["optimizer"]
        optimizer_class = optimizer_config.get("class", None)
        optimizer_config = {k: v for k, v in optimizer_config.items() if k != "class"}
        if optimizer_class is None:
            raise ValueError("Optimizer class must be specified in the configuration.")
        if isinstance(optimizer_class, str):
            optimizer_class = getattr(torch.optim, optimizer_class)

        if not issubclass(optimizer_class, torch.optim.Optimizer):
            raise TypeError(f"Optimizer class {optimizer_class} must be a subclass of torch.optim.Optimizer.")

        encoder_lr_scaler = self.config.get("encoder_lr_scaler", None)
        optim_groups = []
        if encoder_lr_scaler is not None and hasattr(self.sane_ae, "encoder"):
            enc_ids = {id(p) for p in self.sane_ae.encoder.parameters()}
            base_lr = optimizer_config.get("lr", 1e-4)
            enc_lr = base_lr * encoder_lr_scaler

            enc_decay = [p for p in self.sane_ae.encoder.parameters() if p.requires_grad and p.dim() >= 2]
            enc_nodecay = [p for p in self.sane_ae.encoder.parameters() if p.requires_grad and p.dim() < 2]
            rest_decay = [p for p in self.sane_ae.parameters() if p.requires_grad and p.dim() >= 2 and id(p) not in enc_ids]
            rest_nodecay = [p for p in self.sane_ae.parameters() if p.requires_grad and p.dim() < 2 and id(p) not in enc_ids]
            optim_groups.extend([
                {"params": rest_decay},
                {"params": rest_nodecay, "weight_decay": 0.0},
                {"params": enc_decay, "lr": enc_lr},
                {"params": enc_nodecay, "lr": enc_lr, "weight_decay": 0.0},
            ])
        else:
            sane_decay, sane_nodecay = self._collect_param_groups(self.sane_ae)
            optim_groups.extend([
                {"params": sane_decay},
                {"params": sane_nodecay, "weight_decay": 0.0},
            ])

        ds_decay, ds_nodecay = self._collect_param_groups(self.dataset_encoder)
        optim_groups.extend([
            {"params": ds_decay},
            {"params": ds_nodecay, "weight_decay": 0.0},
        ])

        alignment_decay, alignment_nodecay = self._collect_param_groups(self.alignment_loss_fn, recurse=False)
        if alignment_decay:
            optim_groups.append({"params": alignment_decay})
        if alignment_nodecay:
            optim_groups.append({"params": alignment_nodecay, "weight_decay": 0.0})

        self.optimizer = optimizer_class(optim_groups, **optimizer_config)

    def _setup_joint_scheduler(self) -> None:
        scheduler_config = self.config["scheduler"]
        scheduler_class = scheduler_config.get("class", None)
        scheduler_config = {k: v for k, v in scheduler_config.items() if k != "class"}
        if scheduler_class is None:
            raise ValueError("Scheduler class must be specified in the configuration.")
        if isinstance(scheduler_class, str):
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)

        if not issubclass(scheduler_class, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(f"Scheduler class {scheduler_class} must be a subclass of torch.optim.lr_scheduler._LRScheduler.")

        if scheduler_class == torch.optim.lr_scheduler.OneCycleLR:
            scheduler_config["total_steps"] = self._get_total_steps()
            base_lr = self.config["optimizer"]["lr"]
            scheduler_config["max_lr"] = [group.get("lr", base_lr) for group in self.optimizer.param_groups]

        self.scheduler = scheduler_class(self.optimizer, **scheduler_config)

    # Batch helpers

    def _batch_size(self, batch: Mapping[str, Any] | Tuple[torch.Tensor, ...]) -> int:
        if isinstance(batch, Mapping):
            return int(batch["tokens"].shape[0])
        return int(batch[0].shape[0])

    def _normalize_metric(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def _unpack_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(batch, Mapping):
            raise ValueError("SANEJointAlignmentTrainer expects batches with zoo metadata. Ensure data.tokens_dataset.include_zoo_metadata=true.")
        return dict(batch)

    def _build_batch_dataset_metadata(self,batch: Mapping[str, Any],device: torch.device) -> tuple[torch.Tensor, list[str]]:
        zoo_root_dirs = batch.get("zoo_root_dir")
        if zoo_root_dirs is None:
            raise ValueError("Batch is missing zoo_root_dir metadata required for alignment.")

        zoo_paths: list[str] = []
        raw_to_pos: dict[str, int] = {}
        dataset_indices = []
        for root_dir in zoo_root_dirs:
            if root_dir is None:
                raise ValueError("Encountered batch item without zoo_root_dir metadata.")
            if root_dir not in raw_to_pos:
                raw_to_pos[root_dir] = len(zoo_paths)
                zoo_paths.append(root_dir)
            dataset_indices.append(raw_to_pos[root_dir])

        return torch.tensor(dataset_indices, device=device, dtype=torch.long), zoo_paths

    def _prepare_views(self, batch: Mapping[str, Any], training: bool) -> Dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        mask = batch["mask"]
        position = batch["position"]

        if tokens.dim() == 4:
            if training:
                idx_i = 0 if self.view_1_canon_train else random.randint(0, tokens.shape[1] - 1)
                idx_j = random.randint(0, tokens.shape[1] - 1)
                while idx_j == idx_i and tokens.shape[1] > 1:
                    idx_j = random.randint(0, tokens.shape[1] - 1)
                transform_first = not self.view_1_canon_train
            else:
                idx_i = 0 if self.view_1_canon_valid else random.randint(0, tokens.shape[1] - 1)
                idx_j = random.randint(0, tokens.shape[1] - 1)
                while idx_j == idx_i and tokens.shape[1] > 1:
                    idx_j = random.randint(0, tokens.shape[1] - 1)
                transform_first = not self.view_1_canon_valid

            tokens_i = tokens[:, idx_i]
            mask_i = mask[:, idx_i]
            position_i = position[:, idx_i]
            tokens_j = tokens[:, idx_j]
            mask_j = mask[:, idx_j]
            position_j = position[:, idx_j]

            # Alignment uses the first / canonical view by explicit decision.
            tokens_align = tokens[:, 0]
            mask_align = mask[:, 0]
            position_align = position[:, 0]
        elif tokens.dim() == 3:
            idx_i = 0
            tokens_i, mask_i, position_i = tokens, mask, position
            tokens_j, mask_j, position_j = tokens, mask, position
            tokens_align, mask_align, position_align = tokens, mask, position
            transform_first = training and not self.view_1_canon_train
        else:
            raise ValueError(
                f"Tokens must have 3 or 4 dimensions, got shape {tuple(tokens.shape)}."
            )

        if transform_first:
            tokens_i = self._transform_tokens(tokens_i)
        tokens_j = self._transform_tokens(tokens_j)

        return {
            "tokens_i": tokens_i.to(self.device),
            "mask_i": mask_i.to(self.device),
            "position_i": position_i.to(self.device),
            "tokens_j": tokens_j.to(self.device),
            "mask_j": mask_j.to(self.device),
            "position_j": position_j.to(self.device),
            "tokens_align": tokens_align.to(self.device),
            "mask_align": mask_align.to(self.device),
            "position_align": position_align.to(self.device),
            "alignment_reuses_i": idx_i == 0,
        }

    # Loss assembly

    def _compute_alignment(
        self,
        sane_embeddings: torch.Tensor,
        dataset_indices: torch.Tensor,
        zoo_paths: list[str],
        *,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        if self.alignment_loss_fn.weight <= 0:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "alignment_loss": zero,
                "alignment_acc_fwd": zero,
                "alignment_acc_bwd": zero,
            }

        out = self.alignment_loss_fn(
            sane_embeddings,
            dataset_indices,
            zoo_paths,
        )
        return {
            "alignment_loss": out["alignment_loss"],
            "alignment_acc_fwd": torch.tensor(out["alignment_acc_fwd"], device=self.device),
            "alignment_acc_bwd": torch.tensor(out["alignment_acc_bwd"], device=self.device),
        }

    def _compute_joint_metrics(self, batch: Mapping[str, Any], training: bool) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_views(batch, training=training)
        dataset_indices, zoo_paths = self._build_batch_dataset_metadata(batch, self.device)

        output_i = self.sane_ae(x=prepared["tokens_i"], mask=prepared["mask_i"], p=prepared["position_i"])
        output = {k + "_i": v for k, v in output_i.items()}
        output_j = self.sane_ae(x=prepared["tokens_j"], mask=prepared["mask_j"], p=prepared["position_j"])
        output.update({k + "_j": v for k, v in output_j.items()})
        sane_loss = self.loss_fn(output)

        if prepared["alignment_reuses_i"]:
            alignment_z = output_i["z"]
        else:
            alignment_z = self.sane_ae(x=prepared["tokens_align"], mask=prepared["mask_align"], p=prepared["position_align"])["z"]

        alignment_metrics = self._compute_alignment(alignment_z, dataset_indices, zoo_paths, training=training)
        aux_metrics = self.dataset_encoder_aux.compute(dataset_indices, zoo_paths, training=training)

        total = (
            sane_loss["loss"]
            + alignment_metrics["alignment_loss"]
            + aux_metrics["deepsets_cls_loss"]
        )

        metrics = dict(sane_loss)
        metrics["loss_sane"] = sane_loss["loss"]
        metrics.update(alignment_metrics)
        metrics.update(aux_metrics)
        metrics["loss"] = total
        return metrics

    # Step methods

    def _joint_parameters(self):
        alignment_params = [param for _, param in self.alignment_loss_fn.named_parameters(recurse=False) if param.requires_grad]
        return list(self.sane_ae.parameters()) + list(self.dataset_encoder.parameters()) + alignment_params

    def _step_training(self) -> Dict[str, Any]:
        start_time = __import__("time").time()
        running_sum = None
        total_samples = 0

        self.sane_ae.train()
        self.dataset_encoder.train()

        if self.dataset_encoder_classification_weight_anneal_epochs > 0:
            progress = min(self.epoch / float(self.dataset_encoder_classification_weight_anneal_epochs), 1.0)
            self.dataset_encoder_aux.classification_weight = self.dataset_encoder_classification_weight * max(0.0, 1.0 - progress)

        for batch in self.trainloader:
            batch_result = self._step_training_forward(batch)
            batch_size = self._batch_size(batch)

            if running_sum is None:
                running_sum = {k: 0.0 for k in batch_result.keys()}

            for key, value in batch_result.items():
                running_sum[key] += self._normalize_metric(value) * batch_size
            total_samples += batch_size

        results = {k: v / total_samples for k, v in running_sum.items()}
        results["epoch"] = self.epoch
        results["training_time"] = __import__("time").time() - start_time
        return results

    def _step_training_forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        batch = self._unpack_batch(batch)
        self.optimizer.zero_grad()

        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                loss = self._compute_joint_metrics(batch, training=True)
        else:
            loss = self._compute_joint_metrics(batch, training=True)

        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.scale(loss['loss']).backward()
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._joint_parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                scale_after = self.scaler.get_scale()
                if scale_before <= scale_after:
                    self.scheduler.step()
        else:
            loss['loss'].backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._joint_parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss

    def _step_validation(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        self.sane_ae.eval()
        self.dataset_encoder.eval()

        running_sum = None
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_result = self._step_validation_forward(batch)
                batch_size = self._batch_size(batch)

                if running_sum is None:
                    running_sum = {k: 0.0 for k in batch_result.keys()}

                for key, value in batch_result.items():
                    running_sum[key] += self._normalize_metric(value) * batch_size
                total_samples += batch_size

        results = {k: v / total_samples for k, v in running_sum.items()}
        return results

    def _step_validation_forward(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        batch = self._unpack_batch(batch)
        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                return self._compute_joint_metrics(batch, training=False)
        return self._compute_joint_metrics(batch, training=False)

    # Save & load

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        dataset_encoder_path = checkpoint_dir / "dataset_encoder.pt"

        state = {
            'epoch': self.epoch,
            'config': self.config,
            'model': self.sane_ae.state_dict(),
            'dataset_encoder': self.dataset_encoder.state_dict(),
            'alignment_loss': {
                key: value
                for key, value in self.alignment_loss_fn.state_dict().items()
                if not key.startswith('deepsets_encoder.')
            },
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(state, checkpoint_path)
        torch.save(self.dataset_encoder.state_dict(), dataset_encoder_path)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        dataset_encoder_path = checkpoint_dir / "dataset_encoder.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.epoch = state['epoch']
        self.sane_ae.load_state_dict(state['model'])
        dataset_encoder_state = state.get('dataset_encoder')
        if dataset_encoder_state is None:
            if not dataset_encoder_path.exists():
                raise FileNotFoundError(f"Checkpoint is missing dataset encoder state and no dataset_encoder.pt was found at {dataset_encoder_path}.")
            dataset_encoder_state = torch.load(dataset_encoder_path, map_location=self.device, weights_only=True)
        
        self.dataset_encoder.load_state_dict(dataset_encoder_state)
        alignment_loss_state = state.get('alignment_loss')
        if alignment_loss_state is not None:
            self.alignment_loss_fn.load_state_dict(alignment_loss_state, strict=False)
        try:
            self.optimizer.load_state_dict(state['optimizer'])
        except (ValueError, RuntimeError) as exc:
            logger.warning("Could not restore optimizer state: %s", exc)
        if self.scheduler and state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler and state.get('scaler') is not None:
            self.scaler.load_state_dict(state['scaler'])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)
        checkpoint = checkpoint_dir / "checkpoint.pt"
        dataset_encoder_path = checkpoint_dir / "dataset_encoder.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)

        config = state.get("config") or _load_trial_config(checkpoint.parent.parent)
        trainer = cls()
        trainer.setup(config)
        trainer.epoch = state['epoch']
        trainer.sane_ae.load_state_dict(state['model'])
        dataset_encoder_state = state.get('dataset_encoder')
        if dataset_encoder_state is None:
            if not dataset_encoder_path.exists():
                raise FileNotFoundError(f"Checkpoint is missing dataset encoder state and no dataset_encoder.pt was found at {dataset_encoder_path}.")
            dataset_encoder_state = torch.load(dataset_encoder_path, map_location="cpu", weights_only=False)
        
        trainer.dataset_encoder.load_state_dict(dataset_encoder_state)
        alignment_loss_state = state.get('alignment_loss')
        if alignment_loss_state is not None:
            trainer.alignment_loss_fn.load_state_dict(alignment_loss_state, strict=False)
        try:
            trainer.optimizer.load_state_dict(state['optimizer'])
        except (ValueError, RuntimeError) as exc:
            logger.warning("Could not restore optimizer state: %s", exc)
        if trainer.scheduler and state['scheduler'] is not None:
            try:
                trainer.scheduler.load_state_dict(state['scheduler'])
            except (ValueError, RuntimeError) as exc:
                logger.warning("Could not restore scheduler state: %s", exc)
        if trainer.scaler and state.get('scaler') is not None:
            trainer.scaler.load_state_dict(state['scaler'])
        return trainer