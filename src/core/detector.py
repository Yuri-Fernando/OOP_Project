from ultralytics import YOLO

class Detector:
    def __init__(self, model="yolov8n.pt"):
        self.model = YOLO(model)

    def predict(self, frame, conf=0.5):
        results = self.model(frame, conf=conf)
        return results[0]