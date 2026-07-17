from pathlib import Path

import cv2
import numpy as np


def open_camera(camera_index=0):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        return None
    return capture


def read_frame(capture):
    if capture is None:
        return False, None
    return capture.read()


def release_camera(capture):
    if capture is not None:
        capture.release()


def safe_resize(image, size):
    if image is None:
        return np.zeros(size + (3,), dtype=np.uint8)
    return cv2.resize(image, size)
