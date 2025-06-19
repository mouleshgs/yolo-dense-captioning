import cv2
import time
from detector import YOLODetector
from caption_model import CaptionModel
from utils.preprocessor import convert_to_pil
from utils.visualizer import draw_caption

detector = YOLODetector()
captioner = CaptionModel()

cap = cv2.VideoCapture(0)  # Webcam input
last_caption = ""
last_time = 0
CAPTION_INTERVAL = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_time > CAPTION_INTERVAL:
        pil_frame = convert_to_pil(frame)
        last_caption = captioner.caption(pil_frame)
        last_time = current_time

    draw_caption(frame, last_caption)
    cv2.imshow("YOLO Dense Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
