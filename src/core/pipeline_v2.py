"""
src/core/pipeline_v2.py

V2 do VisionPipeline: mantém o fluxo original (Detector -> Processor ->
Visualizer) intacto e adiciona um quarto estágio -- ResNetClassifier -- que
roda sobre o crop de cada detecção-alvo, produzindo uma classificação
refinada (ImageNet) além do label bruto do YOLO/COCO.

    YOLOv8 -> Object Detection
       |
       +-- ResNet -> Image Classification / Feature Extraction
"""

from src.core.pipeline import VisionPipeline


class VisionPipelineV2(VisionPipeline):
    def __init__(self, detector, processor, visualizer, classifier):
        super().__init__(detector, processor, visualizer)
        self.classifier = classifier

    def run(self, frame):
        processed_frame, detections, targets = super().run(frame)

        for target in targets:
            x1, y1, x2, y2 = map(int, target["box"][0])
            crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]

            if crop.size == 0:
                target["resnet_classification"] = []
                continue

            target["resnet_classification"] = self.classifier.classify(crop)

        return processed_frame, detections, targets
