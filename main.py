import cv2
import time
from detector import YOLODetector
from caption_model import CaptionModel
from utils.preprocessor import convert_to_pil
from utils.visualizer import draw_caption
from caption_enhancer import CaptionEnhancer

# Initialize modules
detector = YOLODetector()
captioner = CaptionModel()
enhancer = CaptionEnhancer() 

# Setup webcam
cap = cv2.VideoCapture(0)
last_caption = ""
last_time = 0
CAPTION_INTERVAL = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = detector.detect(frame)
    frame = detector.draw_boxes(frame, results)
    detected_labels = detector.get_labels(results)

    print("[INFO] Detection complete")

    current_time = time.time()
    if current_time - last_time > CAPTION_INTERVAL:
        # Convert frame for caption model
        pil_frame = convert_to_pil(frame).resize((224, 224))

        # GIT-based caption
        raw_caption = captioner.caption(pil_frame)

        # Enhance using LLaMA with object labels
        enhanced_caption = enhancer.enhance(raw_caption, detected_labels)

        last_caption = enhanced_caption
        last_time = current_time

        print(f"[INFO] Captioning + enhancement took {time.time() - current_time:.2f} sec")

    draw_caption(frame, last_caption)
    cv2.imshow("YOLO Dense Captioning", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
