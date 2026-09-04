"""
src/core/classifier_v2.py

V2: adiciona uma etapa de classificação por ResNet sobre cada região
detectada pelo YOLO (src/core/detector.py). A V1 fazia apenas detecção de
objetos (bounding box + label do COCO); a V2 recorta cada bounding box e
passa pela ResNet para extrair uma classificação/refinamento adicional
(feature extraction + classificação de imagem), demonstrando o uso de
arquiteturas residuais (skip connections) ao lado da detecção baseada em
YOLO.
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class ResNetClassifier:
    """Wrapper fino sobre uma ResNet pré-treinada (torchvision), usada como
    segundo estágio do pipeline: dado um crop de imagem (recorte de uma
    detecção do YOLO), devolve o rótulo ImageNet mais provável e a
    confiança associada. Também expõe `extract_features`, que devolve o
    vetor de features da penúltima camada (antes da cabeça de
    classificação) -- útil para tarefas futuras de similaridade/embeddings."""

    def __init__(self, backbone="resnet18", device=None, topk=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.topk = topk

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }
        if backbone not in weights_map:
            raise ValueError(f"Backbone não suportado: {backbone}")

        weights = weights_map[backbone]
        self.model = getattr(models, backbone)(weights=weights).to(self.device)
        self.model.eval()

        self.categories = weights.meta["categories"]
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # torso sem a última camada (fc) -> feature extractor
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def _prepare(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(crop_rgb).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def classify(self, crop_bgr):
        """Classifica um recorte de imagem (BGR, formato OpenCV) e retorna
        uma lista de (label, confidence) com os `topk` resultados mais
        prováveis segundo a ResNet."""
        tensor = self._prepare(crop_bgr)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        top_probs, top_idx = torch.topk(probs, self.topk)
        return [
            (self.categories[idx.item()], float(prob.item()))
            for prob, idx in zip(top_probs, top_idx)
        ]

    @torch.no_grad()
    def extract_features(self, crop_bgr):
        """Retorna o vetor de features (embedding) da penúltima camada da
        ResNet para o recorte informado -- útil para comparação de
        similaridade entre objetos detectados."""
        tensor = self._prepare(crop_bgr)
        features = self.feature_extractor(tensor)
        return features.flatten(1).squeeze(0).cpu().numpy()
