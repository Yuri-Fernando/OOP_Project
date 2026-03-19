import cv2

class Visualizer:
    def draw(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"][0])
            label = d["label"]
            conf = d["confidence"]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)

        return frame