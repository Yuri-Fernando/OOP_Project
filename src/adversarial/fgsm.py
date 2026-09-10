"""FGSM sobre classificadores de imagem (torch).

Opera no espaço do tensor de entrada já normalizado. `epsilon` é o raio da
perturbação em L-infinito nesse espaço. Ataque real: usa o gradiente do
próprio modelo.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> torch.Tensor:
    """Retorna `x_adv = x + epsilon * sign(grad_x CE(model(x), y))`."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    if clip_min is not None or clip_max is not None:
        x_adv = x_adv.clamp(clip_min, clip_max)
    return x_adv.detach()
