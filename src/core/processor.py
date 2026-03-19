class DetectionProcessor:
    def __init__(self, target_classes=None, ignore_classes=None):
        self.target_classes = target_classes or []
        self.ignore_classes = ignore_classes or []

    def filter(self, result):
        detections = []

        for box in result.boxes:
            label = result.names[int(box.cls)]

            # ignorar classes (ex: person)
            if label in self.ignore_classes:
                continue

            conf = float(box.conf)

            detections.append({
                "label": label,
                "confidence": conf,
                "box": box.xyxy.cpu().numpy().tolist()
            })

        return detections

    def detect_target(self, detections):
        return [d for d in detections if d["label"] in self.target_classes]