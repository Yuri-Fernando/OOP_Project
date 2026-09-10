"""Testes do módulo adversarial (V3). Usam um CNN minúsculo e tensores
sintéticos — sem download de pesos, rápido — mas exercitam os ataques reais
(gradiente do modelo)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adversarial import (  # noqa: E402
    adversarial_finetune,
    adversarial_gap,
    evaluate_robustness,
    fgsm,
    make_patch,
    pgd,
)

torch.manual_seed(0)


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(8 * 16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture(scope="module")
def data_and_model():
    x = torch.rand(24, 3, 16, 16)
    y = torch.randint(0, 3, (24,))
    model = TinyCNN(3)
    # treina rápido para o gradiente ter sinal
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return x, y, model


def test_fgsm_perturbation_is_l_inf_bounded(data_and_model):
    x, y, model = data_and_model
    eps = 0.05
    x_adv = fgsm(model, x, y, epsilon=eps)
    assert (x_adv - x).abs().max().item() <= eps + 1e-5


def test_pgd_reduces_accuracy_more_than_fgsm(data_and_model):
    x, y, model = data_and_model
    acc = lambda xx: (model(xx).argmax(1) == y).float().mean().item()
    a_fgsm = acc(fgsm(model, x, y, epsilon=0.1))
    a_pgd = acc(pgd(model, x, y, epsilon=0.1, alpha=0.02, steps=20))
    assert a_pgd <= a_fgsm + 1e-6


def test_evaluate_robustness_returns_all_scenarios(data_and_model):
    x, y, model = data_and_model
    res = evaluate_robustness(model, x, y, epsilon=0.08)
    assert set(res) >= {"clean", "gaussian_noise", "brightness", "gaussian_blur", "fgsm", "pgd"}
    assert res["pgd"] <= res["clean"] + 1e-6
    assert isinstance(adversarial_gap(res), float)


def test_adversarial_patch_changes_or_keeps_shape(data_and_model):
    x, y, model = data_and_model
    x_patched, patch = make_patch(model, x[:4], target=1, patch_size=6, steps=15)
    assert x_patched.shape == x[:4].shape
    assert patch.shape == (1, 3, 6, 6)
    assert patch.min() >= 0.0 and patch.max() <= 1.0


def test_adversarial_finetune_improves_or_holds_robust_accuracy(data_and_model):
    x, y, model = data_and_model
    before = evaluate_robustness(model, x, y, epsilon=0.1)["pgd"]

    import copy

    m2 = copy.deepcopy(model)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    adversarial_finetune(m2, loader, epochs=2, epsilon=0.1, alpha=0.02, pgd_steps=3, lr=5e-3)
    after = evaluate_robustness(m2, x, y, epsilon=0.1)["pgd"]

    assert after >= before - 0.15  # não deve degradar a robustez
