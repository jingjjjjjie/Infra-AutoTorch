'''
Loss function registry.
'''
import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for probability or logit inputs.

    Args:
        alpha: Weight assigned to positive examples. Negative examples receive
            ``1 - alpha``. Set to ``None`` to disable class weighting.
        gamma: Focusing strength. ``0`` reduces to (optionally weighted) BCE.
        reduction: One of ``"none"``, ``"mean"``, or ``"sum"``.
        from_logits: Whether inputs are raw logits instead of probabilities.
    """

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        from_logits: bool = False,
    ):
        super().__init__()
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError(f'alpha must be between 0 and 1, got {alpha}')
        if gamma < 0:
            raise ValueError(f'gamma must be non-negative, got {gamma}')
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            raise ValueError(
                f'inputs and targets must have the same shape, got '
                f'{tuple(inputs.shape)} and {tuple(targets.shape)}'
            )

        targets = targets.to(dtype=inputs.dtype)
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probabilities = torch.sigmoid(inputs)
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            probabilities = inputs

        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        loss = (1 - p_t).pow(self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


LOSS_FN_MAP = {
    'bce': nn.BCELoss,
    'bce_with_logits': nn.BCEWithLogitsLoss,
    'binary_focal': BinaryFocalLoss,
}
