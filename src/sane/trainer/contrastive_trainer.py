import torch
import random

from typing import Optional, Dict, Any, Tuple

from sane.trainer.trainer import SANETrainer

class SANEContrastiveTrainer(SANETrainer):
    def setup(self, config: Dict[str, Any]):
        super().setup(config)

        self.mask_probability = self.config.get('mask_probability', 0.0)
        assert isinstance(self.mask_probability, float) and 0 <= self.mask_probability <= 1, "`mask_probability` must be a float between 0 and 1 but is {}.".format(self.mask_probability)

        self.additive_noise = self.config.get('additive_noise', 0.0)
        assert isinstance(self.additive_noise, float) and self.additive_noise >= 0, "`additive_noise` must be a float >= 0 but is {}.".format(self.additive_noise)

        self.multiplicative_noise = self.config.get('multiplicative_noise', 0.0)
        assert isinstance(self.multiplicative_noise, float) and self.multiplicative_noise >= 0, "`multiplicative_noise` must be a float >= 0 but is {}.".format(self.multiplicative_noise)

    def _step_training_forward(self, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a forward pass and backpropagation for a single batch of training data.

        Args:
            tensors (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing ``(tokens, mask, position)``.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss and any additional metrics.
        """
        if len(tensors) != 3:
            raise ValueError("Tensors must be a tuple of (tokens, mask, position) but has length {} and dimensions {}.".format(len(tensors), [tensor.dim() for tensor in tensors]))

        # If the tokens (or mask, positions) have 4 dimensions, it means we have different views.
        # In that case, dimensions are: (batch_size, num_views, num_tokens, token_size)
        # If we have different views, we select 2 different ones as augmentations.
        if len(tensors[0].shape) == 4:
            idx_i = 0 if self.view_1_canon_train else random.randint(0, tensors[0].shape[1] - 1)
            # Sample idx_j, ensuring it's different from idx_i
            idx_j = random.randint(0, tensors[0].shape[1] - 1)
            while idx_j == idx_i and tensors[0].shape[1] > 1:
                idx_j = random.randint(0, tensors[0].shape[1] - 1)

            tokens_i, mask_i, position_i = tensors[0][:, idx_i], tensors[1][:, idx_i], tensors[2][:, idx_i]
            tokens_j, mask_j, position_j = tensors[0][:, idx_j], tensors[1][:, idx_j], tensors[2][:, idx_j]
        else:
            tokens_i, mask_i, position_i = tensors[0], tensors[1], tensors[2]
            tokens_j, mask_j, position_j = tensors[0], tensors[1], tensors[2]

        if not self.view_1_canon_train:
            tokens_i = self._transform_tokens(tokens_i)
        tokens_j = self._transform_tokens(tokens_j)

        tokens_i, mask_i, position_i = tokens_i.to(self.device), mask_i.to(self.device), position_i.to(self.device)
        tokens_j, mask_j, position_j = tokens_j.to(self.device), mask_j.to(self.device), position_j.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass with optional AMP autocast
        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                output_i = self.sane_ae(x=tokens_i, mask=mask_i, p=position_i)
                output = {k + '_i': v for k, v in output_i.items()}
                output_j = self.sane_ae(x=tokens_j, mask=mask_j, p=position_j)
                output.update({k + '_j': v for k, v in output_j.items()})
                loss = self.loss_fn(output)
        else:
            output_i = self.sane_ae(x=tokens_i, mask=mask_i, p=position_i)
            output = {k + '_i': v for k, v in output_i.items()}
            output_j = self.sane_ae(x=tokens_j, mask=mask_j, p=position_j)
            output.update({k + '_j': v for k, v in output_j.items()})
            loss = self.loss_fn(output)

        # Backward pass with optional gradient scaling
        if self.use_amp:
            # Track scale to detect if optimizer step was skipped due to inf/NaN gradients
            scale_before = self.scaler.get_scale()
            self.scaler.scale(loss['loss']).backward()
            # Optional gradient clipping
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.sane_ae.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Only step scheduler if optimizer actually stepped (scale didn't decrease)
            # see https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
            if self.scheduler is not None:
                scale_after = self.scaler.get_scale()
                if scale_before <= scale_after:
                    self.scheduler.step()
        else:
            loss['loss'].backward()
            # Optional gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.sane_ae.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss

    def _step_validation_forward(self, tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a forward pass for a single batch of validation data.

        Args:
            tensors (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing ``(tokens, mask, position)``.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed loss and any additional metrics.
        """
        # If the tokens (or mask, positions) have 4 dimensions, it means we have different views.
        # In that case, dimensions are: (batch_size, num_views, num_tokens, token_size)
        # If we have different views, we select 2 different ones as augmentations.
        if len(tensors[0].shape) == 4:
            idx_i = 0 if self.view_1_canon_valid else random.randint(0, tensors[0].shape[1] - 1)
            # Sample idx_j, ensuring it's different from idx_i
            idx_j = random.randint(0, tensors[0].shape[1] - 1)
            while idx_j == idx_i and tensors[0].shape[1] > 1:
                idx_j = random.randint(0, tensors[0].shape[1] - 1)

            tokens_i, mask_i, position_i = tensors[0][:, idx_i], tensors[1][:, idx_i], tensors[2][:, idx_i]
            tokens_j, mask_j, position_j = tensors[0][:, idx_j], tensors[1][:, idx_j], tensors[2][:, idx_j]
        else:
            tokens_i, mask_i, position_i = tensors[0], tensors[1], tensors[2]
            tokens_j, mask_j, position_j = tensors[0], tensors[1], tensors[2]

        if not self.view_1_canon_valid:
            tokens_i = self._transform_tokens(tokens_i)
        tokens_j = self._transform_tokens(tokens_j)

        tokens_i, mask_i, position_i = tokens_i.to(self.device), mask_i.to(self.device), position_i.to(self.device)
        tokens_j, mask_j, position_j = tokens_j.to(self.device), mask_j.to(self.device), position_j.to(self.device)

        # Forward pass with optional AMP autocast (no gradient scaling needed for validation)
        if self.use_amp:
            device_type = 'cuda' if 'cuda' in self.device else 'cpu'
            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype_torch):
                output_i = self.sane_ae(x=tokens_i, mask=mask_i, p=position_i)
                output = {k + '_i': v for k, v in output_i.items()}
                output_j = self.sane_ae(x=tokens_j, mask=mask_j, p=position_j)
                output.update({k + '_j': v for k, v in output_j.items()})
                loss = self.loss_fn(output)
        else:
            output_i = self.sane_ae(x=tokens_i, mask=mask_i, p=position_i)
            output = {k + '_i': v for k, v in output_i.items()}
            output_j = self.sane_ae(x=tokens_j, mask=mask_j, p=position_j)
            output.update({k + '_j': v for k, v in output_j.items()})
            loss = self.loss_fn(output)

        return loss

    def _transform_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Applies data transformations to the input tokens.

        Args:
            tokens (torch.Tensor): The input tokens.

        Returns:
            torch.Tensor: The transformed tokens.
        """
        transformed_tokens = tokens.clone()  # Avoid modifying the original tensor

        if self.mask_probability > 0:
            # Randomly mask some tokens with probability `mask_probability`
            mask = torch.rand_like(transformed_tokens) < self.mask_probability
            transformed_tokens[mask] = 0  # or any mask value, e.g., -1 or a learnable mask token

        if self.additive_noise > 0:
            # Add Gaussian noise with mean 0 and std `additive_noise`
            noise = torch.randn_like(transformed_tokens) * self.additive_noise
            transformed_tokens += noise

        if self.multiplicative_noise > 0:
            # Multiply by a random factor centered around 1 with std `multiplicative_noise`
            noise = torch.randn_like(transformed_tokens) * self.multiplicative_noise + 1
            transformed_tokens *= noise

        return transformed_tokens
