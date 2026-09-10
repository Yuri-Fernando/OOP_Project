"""PGD (Madry et al.) sobre classificadores de imagem (torch)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pgd(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.008,
    steps: int = 10,
    clip_min: float | None = None,
    clip_max: float | None = None,
    random_start: bool = True,
) -> torch.Tensor:
    model.eval()
    x_orig = x.clone().detach()
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_orig - epsilon), x_orig + epsilon)
        if clip_min is not None or clip_max is not None:
            x_adv = x_adv.clamp(clip_min, clip_max)

    return x_adv.detach()
