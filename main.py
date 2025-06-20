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
CAPTION_INTERVAL = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect(frame)
    frame = detector.draw_boxes(frame, results)
    detected_labels = detector.get_labels(results)

    print("part 1 working")

    current_time = time.time()
    if current_time - last_time > CAPTION_INTERVAL:
        pil_frame = convert_to_pil(frame)
        pil_frame = pil_frame.resize((224, 224))
        
        if detected_labels:
            yolo_caption = " | Objects: " + ", ".join(detected_labels)
        else:
            yolo_caption = ""

        last_caption = captioner.caption(pil_frame) + yolo_caption
        print(f"Captioning took {time.time() - current_time:.2f} sec")
        last_time = current_time

    print("part 2 working")
    draw_caption(frame, last_caption)
    cv2.imshow("YOLO Dense Captioning", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
