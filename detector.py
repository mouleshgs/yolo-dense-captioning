from ultralytics import YOLO
import cv2


class YOLODetector:
    def __init__(self, model_path="yolov8m.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        return results

    def draw_boxes(self, frame, results):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            text = f"{label} ({conf:.2f})"

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (225, 0, 0), 1)
        return frame

    def get_labels(self, results):
        labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            if conf > 0.5:
                labels.append(label)

        return list(set(labels))  
