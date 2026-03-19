import cv2

def load_image(path):
    return cv2.imread(path)


def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    ratio = width / w
    new_dim = (width, int(h * ratio))
    return cv2.resize(frame, new_dim)


def convert_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    return frame