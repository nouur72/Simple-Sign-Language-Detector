import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class ASLPredictor:
    def __init__(self, model_path, class_names_path, stability_window=4, confidence_threshold=0.6):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.prediction_queue = deque(maxlen=stability_window)
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists() or not self.class_names_path.exists():
            raise FileNotFoundError("Model files are missing. Please train first.")

        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow is required for prediction.") from exc

        self.model = keras.models.load_model(self.model_path)
        with self.class_names_path.open("r", encoding="utf-8") as handle:
            self.class_names = json.load(handle)

    def _extract_hand_region(self, frame):
        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 500:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        return frame[y : y + h, x : x + w]

    def _preprocess_region(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (96, 96))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def predict_frame(self, frame):
        hand_region = self._extract_hand_region(frame)
        if hand_region is None:
            return None, 0.0, None

        batch = self._preprocess_region(hand_region)
        predictions = self.model.predict(batch, verbose=0)[0]
        confidence = float(np.max(predictions))
        index = int(np.argmax(predictions))
        label = self.class_names[index]

        self.prediction_queue.append((label, confidence))
        if len(self.prediction_queue) < self.stability_window:
            return None, confidence, hand_region

        labels = [item[0] for item in self.prediction_queue]
        if len(set(labels)) == 1:
            stable_label = labels[-1]
            stable_confidence = float(np.mean([item[1] for item in self.prediction_queue]))
            if stable_confidence >= self.confidence_threshold:
                return stable_label, stable_confidence, hand_region

        return None, confidence, hand_region
