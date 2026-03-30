import cv2
import numpy as np

TARGET_WIDTH = 640
TARGET_HEIGHT = 360

def preprocess(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    return denoised

def to_hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def crop_region(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    fh, fw = frame.shape[:2]
    x1, y1 = int(x * fw), int(y * fh)
    x2, y2 = int((x + w) * fw), int((y + h) * fh)
    return frame[y1:y2, x1:x2]
