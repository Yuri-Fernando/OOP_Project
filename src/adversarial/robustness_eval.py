"""Avaliação de robustez de um classificador de imagem: acurácia limpa vs.
acurácia sob cada ataque/corrupção, sobre o mesmo lote.
"""
from __future__ import annotations

import torch

from src.adversarial.corruptions import BASELINES
from src.adversarial.fgsm import fgsm
from src.adversarial.pgd import pgd


@torch.no_grad()
def _accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean().item())


def evaluate_robustness(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    pgd_steps: int = 10,
) -> dict[str, float]:
    """Retorna um dicionário `{cenário: acurácia}` — `clean`, cada baseline
    de corrupção, `fgsm` e `pgd` no mesmo `epsilon`."""
    results: dict[str, float] = {"clean": _accuracy(model, x, y)}

    for name, fn in BASELINES.items():
        sigma = epsilon if name != "gaussian_blur" else 1.0
        x_corr = fn(x, sigma) if name != "gaussian_blur" else fn(x)
        results[name] = _accuracy(model, x_corr, y)

    results["fgsm"] = _accuracy(model, fgsm(model, x, y, epsilon=epsilon), y)
    results["pgd"] = _accuracy(
        model, pgd(model, x, y, epsilon=epsilon, alpha=epsilon / 3, steps=pgd_steps), y
    )
    return results


def adversarial_gap(results: dict[str, float]) -> float:
    """Quanto pior o PGD é em relação ao ruído gaussiano do mesmo tamanho
    (isola a fragilidade adversarial da fragilidade geral)."""
    return results.get("gaussian_noise", results["clean"]) - results["pgd"]
