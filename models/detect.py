from ultralytics import YOLO


class ShelfDetector:
    """
    Retail Shelf AI Detector
    """

    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, image_path, confidence=0.40):

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False
        )

        detections = []

        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            })

        return {
            "total_products": len(detections),
            "detections": detections
        }
