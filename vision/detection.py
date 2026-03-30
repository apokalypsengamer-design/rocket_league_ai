import cv2
import numpy as np
from config import Config
from core.state import GameState
from vision.preprocessing import to_hsv, to_gray, crop_region

class Detector:
    def __init__(self, config: Config):
        self.config = config
        self._lower = np.array(config.ball_hsv_lower)
        self._upper = np.array(config.ball_hsv_upper)

    def detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        state = GameState(frame_raw=raw, frame_processed=processed)
        state.ball_x, state.ball_y, state.ball_visible, state.ball_radius = self._find_ball(processed)
        state.boost = self._read_boost(processed)
        state.phase = self._determine_phase(state)
        return state

    def _find_ball(self, frame: np.ndarray):
        hsv = to_hsv(frame)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return -1.0, -1.0, False, 0.0

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.config.ball_min_area:
            return -1.0, -1.0, False, 0.0

        M = cv2.moments(c)
        if M["m00"] == 0:
            return -1.0, -1.0, False, 0.0

        h, w = frame.shape[:2]
        cx = M["m10"] / M["m00"] / w
        cy = M["m01"] / M["m00"] / h
        radius = (area ** 0.5) / max(w, h)
        return cx, cy, True, radius

    def _read_boost(self, frame: np.ndarray) -> float:
        roi = crop_region(frame, 0.70, 0.92, 0.25, 0.08)
        if roi.size == 0:
            return -1.0
        gray = to_gray(roi)
        bright = np.sum(gray > 180)
        ratio = bright / gray.size
        return round(min(ratio * 400, 100), 1)

    def _determine_phase(self, s: GameState) -> str:
        if s.boost != -1 and s.boost < self.config.boost_low_threshold:
            return "boost_collect"
        if not s.ball_visible:
            return "rotate"
        if s.ball_y > self.config.defense_threshold:
            return "defense"
        if s.ball_y < self.config.attack_threshold:
            return "attack"
        return "rotate"
