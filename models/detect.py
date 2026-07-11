from ultralytics import YOLO


class ShelfDetector:
    """
    RetailShelfAI Detection Engine
    Loads a YOLO model and performs object detection on shelf images.
    """

    def __init__(self, model_path="weights/best.pt", confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image_path):
        """
        Run object detection on an image.

        Args:
            image_path (str): Path to image.

        Returns:
            Ultralytics Results object.
        """

        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            verbose=False
        )

        return results
