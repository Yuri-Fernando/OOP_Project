class VisionPipeline:
    def __init__(self, detector, processor, visualizer):
        self.detector = detector
        self.processor = processor
        self.visualizer = visualizer

    def run(self, frame):
        result = self.detector.predict(frame)
        detections = self.processor.filter(result)
        targets = self.processor.detect_target(detections)

        frame = self.visualizer.draw(frame, detections)

        return frame, detections, targets