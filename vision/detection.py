from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from config import Config
from core.state import GameState
from vision.preprocessing import to_hsv, to_gray, crop_region

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    visible: bool
    radius: float


class Detector:
    def __init__(self, config: Config):
        self._cfg = config.vision
        self._gameplay = config.gameplay
        self._lower = np.array(self._cfg.ball_hsv_lower, dtype=np.uint8)
        self._upper = np.array(self._cfg.ball_hsv_upper, dtype=np.uint8)

    def detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        ball = self._find_ball(processed)
        boost = self._read_boost(processed)
        phase = self._determine_phase(ball, boost)
        return GameState(
            ball_x=ball.x,
            ball_y=ball.y,
            ball_visible=ball.visible,
            ball_radius=ball.radius,
            boost=boost,
            phase=phase,
            frame_raw=raw,
            frame_processed=processed,
        )

    def _find_ball(self, frame: np.ndarray) -> BallDetection:
        hsv = to_hsv(frame)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BallDetection(-1.0, -1.0, False, 0.0)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self._cfg.ball_min_area:
            return BallDetection(-1.0, -1.0, False, 0.0)

        M = cv2.moments(c)
        if M["m00"] == 0:
            return BallDetection(-1.0, -1.0, False, 0.0)

        h, w = frame.shape[:2]
        cx = M["m10"] / M["m00"] / w
        cy = M["m01"] / M["m00"] / h
        radius = (area ** 0.5) / max(w, h)
        return BallDetection(cx, cy, True, radius)

    def _read_boost(self, frame: np.ndarray) -> float:
        x, y, w, h = self._cfg.boost_roi
        roi = crop_region(frame, x, y, w, h)
        if roi.size == 0:
            return -1.0
        gray = to_gray(roi)
        bright_ratio = np.mean(gray > self._cfg.boost_brightness_threshold)
        return round(min(bright_ratio * self._cfg.boost_scale_factor * 100, 100.0), 1)

    def _determine_phase(self, ball: BallDetection, boost: float) -> str:
        g = self._gameplay
        if boost != -1 and boost < g.boost_low_threshold:
            return "boost_collect"
        if not ball.visible:
            return "rotate"
        if ball.y > g.defense_threshold:
            return "defense"
        if ball.y < g.attack_threshold:
            return "attack"
        return "rotate"
