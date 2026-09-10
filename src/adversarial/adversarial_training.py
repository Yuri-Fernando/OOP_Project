"""Adversarial training para o classificador (fine-tune curto).

A cada batch, gera exemplos por FGSM/PGD contra o modelo no estado atual e
treina em (limpo + adversarial). É a defesa que, na prática, mais recupera
acurácia robusta.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.adversarial.pgd import pgd


def adversarial_finetune(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 1,
    epsilon: float = 0.03,
    alpha: float = 0.008,
    pgd_steps: int = 5,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pgd(model, x, y, epsilon=epsilon, alpha=alpha, steps=pgd_steps)
            model.train()

            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss = 0.5 * F.cross_entropy(logits_clean, y) + 0.5 * F.cross_entropy(logits_adv, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model
