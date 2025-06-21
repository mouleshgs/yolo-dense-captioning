import cv2
import time
import threading

from detector import YOLODetector
from caption_model import CaptionModel
from utils.preprocessor import convert_to_pil
from utils.visualizer import draw_caption
from caption_enhancer import CaptionEnhancer

# Initialize models
detector = YOLODetector()
captioner = CaptionModel()
enhancer = CaptionEnhancer("llama3.2")

# Shared caption and lock for thread safety
last_caption = "Warming up..."
caption_lock = threading.Lock()
CAPTION_INTERVAL = 5  # seconds

# Setup webcam
cap = cv2.VideoCapture(0)

# Thread function for captioning
def caption_worker():
    global last_caption
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)
        detected_labels = detector.get_labels(results)
        pil_frame = convert_to_pil(frame).resize((224, 224))

        raw_caption = captioner.caption(pil_frame)
        enhanced_caption = enhancer.enhance(raw_caption, detected_labels)

        with caption_lock:
            last_caption = enhanced_caption

        time.sleep(CAPTION_INTERVAL)

# Start the background captioning thread
threading.Thread(target=caption_worker, daemon=True).start()

# Main loop: detection and display
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect(frame)
    frame = detector.draw_boxes(frame, results)

    with caption_lock:
        current = last_caption

    draw_caption(frame, current)
    cv2.imshow("YOLO Dense Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
