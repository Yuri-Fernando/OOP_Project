"""Adversarial patch — otimiza um recorte quadrado que, colado numa posição
fixa da imagem, força o classificador para um alvo (ou apenas maximiza o
erro). Diferente de FGSM/PGD, a perturbação é localizada e visível — modela
adesivos físicos no mundo real.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_patch(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: int | None = None,
    patch_size: int = 8,
    top: int = 2,
    left: int = 2,
    steps: int = 30,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna `(x_patched, patch)`. Se `target` for None, o patch maximiza a
    loss da classe verdadeira (predição atual); senão, minimiza a loss em
    relação a `target` (ataque direcionado)."""
    model.eval()
    b, c, h, w = x.shape
    device = x.device

    with torch.no_grad():
        base_pred = model(x).argmax(1)

    patch = torch.rand(1, c, patch_size, patch_size, device=device, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=lr)

    for _ in range(steps):
        x_patched = x.clone()
        x_patched[:, :, top:top + patch_size, left:left + patch_size] = patch
        logits = model(x_patched)
        if target is None:
            loss = -F.cross_entropy(logits, base_pred)
        else:
            tgt = torch.full((b,), target, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        patch.data.clamp_(0.0, 1.0)

    x_patched = x.clone()
    x_patched[:, :, top:top + patch_size, left:left + patch_size] = patch.detach()
    return x_patched.detach(), patch.detach()
