from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from config import Config
from core.state import GameState, ObjectPosition
from vision.preprocessing import to_hsv, to_gray, crop_region

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class BallDetection:
    x:       float
    y:       float
    visible: bool
    radius:  float


class Detector:
    def __init__(self, config: Config):
        self._cfg      = config.vision
        self._gameplay = config.gameplay

        def arr(lst): return np.array(lst, dtype=np.uint8)

        self._ball_lo        = arr(self._cfg.ball_hsv_lower)
        self._ball_hi        = arr(self._cfg.ball_hsv_upper)
        self._goal_own_lo    = arr(self._cfg.goal_own_hsv_lower)
        self._goal_own_hi    = arr(self._cfg.goal_own_hsv_upper)
        self._goal_enemy_lo  = arr(self._cfg.goal_enemy_hsv_lower)
        self._goal_enemy_hi  = arr(self._cfg.goal_enemy_hsv_upper)
        self._pad_lo         = arr(self._cfg.boost_pad_hsv_lower)
        self._pad_hi         = arr(self._cfg.boost_pad_hsv_upper)
        self._enemy_lo       = arr(self._cfg.enemy_hsv_lower)
        self._enemy_hi       = arr(self._cfg.enemy_hsv_upper)
        self._teammate_lo    = arr(self._cfg.teammate_hsv_lower)
        self._teammate_hi    = arr(self._cfg.teammate_hsv_upper)

    # ── Main entry ──────────────────────────────────────────────────────────

    def detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        if self._cfg.dummy_mode:
            return self._dummy_state(raw, processed)
        return self._real_detect(processed, raw)

    # ── Real detection ──────────────────────────────────────────────────────

    def _real_detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        hsv   = to_hsv(processed)
        ball  = self._find_ball(processed, hsv)
        boost = self._read_boost(processed)

        own_goal, enemy_goal = self._find_goals(processed, hsv)
        boost_pads           = self._find_boost_pads(processed, hsv)
        enemies              = self._find_agents(hsv, self._enemy_lo,    self._enemy_hi)
        teammates            = self._find_agents(hsv, self._teammate_lo, self._teammate_hi)
        nearest_boost        = self._nearest_pad(boost_pads, 0.5, 0.5)
        phase                = self._determine_phase(ball, boost)

        return GameState(
            ball_x=ball.x,        ball_y=ball.y,
            ball_visible=ball.visible, ball_radius=ball.radius,
            boost=boost,          phase=phase,
            own_goal=own_goal,    enemy_goal=enemy_goal,
            enemies=enemies,      teammates=teammates,
            boost_pads=boost_pads, nearest_boost=nearest_boost,
            frame_raw=raw,        frame_processed=processed,
        )

    # ── Dummy state (Vision noch nicht kalibriert) ──────────────────────────

    def _dummy_state(self, raw: np.ndarray, processed: np.ndarray) -> GameState:
        """
        Liefert einen vollständig gefüllten GameState mit konfigurierbaren
        Dummy-Werten. Der Bot kann so vollständig getestet werden, ohne dass
        OpenCV etwas im Frame erkennt.
        """
        cfg   = self._cfg
        ball  = BallDetection(cfg.dummy_ball_x, cfg.dummy_ball_y, True, 0.04)
        boost = cfg.dummy_boost
        phase = self._determine_phase(ball, boost)

        dummy_enemies   = [ObjectPosition(0.70, 0.30, True),
                           ObjectPosition(0.30, 0.20, True)]
        dummy_teammates = [ObjectPosition(0.50, 0.80, True)]
        dummy_pads      = [
            ObjectPosition(0.10, 0.50, True),
            ObjectPosition(0.90, 0.50, True),
            ObjectPosition(0.50, 0.10, True),
            ObjectPosition(0.50, 0.90, True),
        ]
        nearest = self._nearest_pad(dummy_pads, 0.5, 0.5)

        return GameState(
            ball_x=ball.x,        ball_y=ball.y,
            ball_visible=ball.visible, ball_radius=ball.radius,
            boost=boost,          phase=phase,
            own_goal=ObjectPosition(0.5, 1.0, True),
            enemy_goal=ObjectPosition(0.5, 0.0, True),
            enemies=dummy_enemies,
            teammates=dummy_teammates,
            boost_pads=dummy_pads,
            nearest_boost=nearest,
            frame_raw=raw,        frame_processed=processed,
        )

    # ── Vision helpers ──────────────────────────────────────────────────────

    def _find_ball(self, frame: np.ndarray, hsv: np.ndarray) -> BallDetection:
        mask = cv2.inRange(hsv, self._ball_lo, self._ball_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BallDetection(-1.0, -1.0, False, 0.0)
        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self._cfg.ball_min_area:
            return BallDetection(-1.0, -1.0, False, 0.0)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return BallDetection(-1.0, -1.0, False, 0.0)
        h, w = frame.shape[:2]
        cx   = M["m10"] / M["m00"] / w
        cy   = M["m01"] / M["m00"] / h
        return BallDetection(cx, cy, True, (area ** 0.5) / max(w, h))

    def _find_goals(self, frame: np.ndarray, hsv: np.ndarray
                    ) -> tuple[ObjectPosition, ObjectPosition]:
        own   = self._largest_blob(hsv, self._goal_own_lo,   self._goal_own_hi,
                                   frame, self._cfg.goal_min_area)
        enemy = self._largest_blob(hsv, self._goal_enemy_lo, self._goal_enemy_hi,
                                   frame, self._cfg.goal_min_area)
        if not own.visible:
            own   = ObjectPosition(0.5, 1.0, True)
        if not enemy.visible:
            enemy = ObjectPosition(0.5, 0.0, True)
        return own, enemy

    def _find_boost_pads(self, frame: np.ndarray, hsv: np.ndarray) -> list[ObjectPosition]:
        return self._all_blobs(
            hsv, self._pad_lo, self._pad_hi,
            frame, self._cfg.boost_pad_min_area, self._cfg.max_boost_pads,
        )

    def _find_agents(self, hsv: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray) -> list[ObjectPosition]:
        h, w = hsv.shape[:2]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: list[ObjectPosition] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:self._cfg.max_agents]:
            if cv2.contourArea(c) < self._cfg.agent_min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            result.append(ObjectPosition(
                M["m10"] / M["m00"] / w,
                M["m01"] / M["m00"] / h,
                True,
            ))
        return result

    def _largest_blob(self, hsv: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray,
                      frame: np.ndarray, min_area: int) -> ObjectPosition:
        h, w = frame.shape[:2]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ObjectPosition()
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return ObjectPosition()
        M = cv2.moments(c)
        if M["m00"] == 0:
            return ObjectPosition()
        return ObjectPosition(M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h, True)

    def _all_blobs(self, hsv: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray,
                   frame: np.ndarray, min_area: int, limit: int) -> list[ObjectPosition]:
        h, w = frame.shape[:2]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: list[ObjectPosition] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:limit]:
            if cv2.contourArea(c) < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            result.append(ObjectPosition(
                M["m10"] / M["m00"] / w,
                M["m01"] / M["m00"] / h,
                True,
            ))
        return result

    def _read_boost(self, frame: np.ndarray) -> float:
        x, y, w, h = self._cfg.boost_roi
        roi = crop_region(frame, x, y, w, h)
        if roi.size == 0:
            return -1.0
        gray         = to_gray(roi)
        bright_ratio = float(np.mean(gray > self._cfg.boost_brightness_threshold))
        return round(min(bright_ratio * self._cfg.boost_scale_factor * 100, 100.0), 1)

    @staticmethod
    def _nearest_pad(pads: list[ObjectPosition],
                     px: float, py: float) -> ObjectPosition | None:
        visible = [p for p in pads if p.visible]
        if not visible:
            return None
        return min(visible, key=lambda p: abs(p.x - px) + abs(p.y - py))

    def _determine_phase(self, ball: BallDetection, boost: float) -> str:
        g = self._gameplay
        if 0 <= boost < g.boost_low_threshold:
            return "boost_collect"
        if not ball.visible:
            return "rotate"
        if ball.y > g.defense_threshold:
            return "defense"
        if ball.y < g.attack_threshold:
            return "attack"
        return "rotate"
