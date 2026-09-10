"""Corrupções não-adversariais — baseline para separar "o modelo é frágil a
qualquer perturbação" de "o modelo é frágil a perturbação *adversarial*".

Se a acurácia cai igual sob ruído aleatório e sob PGD do mesmo tamanho, o
problema é robustez geral; se cai muito mais sob PGD, é vulnerabilidade
adversarial específica.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_noise(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def brightness(x: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    return x + delta


def gaussian_blur(x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    c = x.shape[1]
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).expand(c, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=c)


BASELINES = {
    "gaussian_noise": gaussian_noise,
    "brightness": brightness,
    "gaussian_blur": gaussian_blur,
}
