"""Adversarial Computer Vision (V3) — ataques e defesas adversariais sobre a
etapa de classificação ResNet do VisionGuard (`src/core/classifier_v2.py`).

FGSM / PGD / adversarial patch, baselines de corrupção não-adversarial para
comparação, avaliação de robustez (clean vs. adversarial) e adversarial
training.

A detecção YOLO (V1) e a classificação ResNet (V2) permanecem intactas —
este módulo é uma camada de avaliação/hardening por cima.
"""
from __future__ import annotations

from src.adversarial.adversarial_training import adversarial_finetune
from src.adversarial.corruptions import BASELINES
from src.adversarial.fgsm import fgsm
from src.adversarial.patch import make_patch
from src.adversarial.pgd import pgd
from src.adversarial.robustness_eval import adversarial_gap, evaluate_robustness

__all__ = [
    "fgsm",
    "pgd",
    "make_patch",
    "BASELINES",
    "evaluate_robustness",
    "adversarial_gap",
    "adversarial_finetune",
]
