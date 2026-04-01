from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from config import Config
from core.state import GameState, ObjectPosition, BoostPadType
from vision.preprocessing import to_hsv, to_gray, crop_region

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Boostpad-Größenschwelle: Fläche (normiert auf Bildgröße) über der ein Pad
# als LARGE gilt. Muss ggf. per Kalibrierung angepasst werden.
_LARGE_PAD_AREA_THRESHOLD = 0.003   # ~3‰ der Bildfläche

# Schuss-Opportunity: Ball muss näher als dieser Wert am Spieler sein
_SHOT_BALL_DIST   = 0.25
# … und der Spieler muss "hinter" dem Ball Richtung eigenes Tor stehen
_SHOT_ALIGN_DIST  = 0.20


@dataclass(frozen=True)
class BallDetection:
    x:       float
    y:       float
    visible: bool
    radius:  float
    # Konfidenz: 0.0–1.0. Mehrere Kandidaten → nehmen wir den mit höchster Konfidenz.
    confidence: float = 0.0


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

        # Letzter bekannter Ball-Zustand für Tracking (verhindert Flicker)
        self._last_ball: BallDetection = BallDetection(-1.0, -1.0, False, 0.0, 0.0)
        self._ball_lost_frames: int = 0
        # Nach wie vielen Frames ohne Erkennung gilt der Ball als wirklich weg
        self._ball_max_lost: int = 8

    # ── Main entry ──────────────────────────────────────────────────────────

    def detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        if self._cfg.dummy_mode:
            return self._dummy_state(raw, processed)
        return self._real_detect(processed, raw)

    # ── Real detection ──────────────────────────────────────────────────────

    def _real_detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        hsv   = to_hsv(processed)
        ball  = self._find_ball_robust(processed, hsv)
        boost = self._read_boost(processed)

        own_goal, enemy_goal = self._find_goals(processed, hsv)
        boost_pads           = self._find_boost_pads(processed, hsv)
        enemies              = self._find_agents(hsv, self._enemy_lo,    self._enemy_hi)
        teammates            = self._find_agents(hsv, self._teammate_lo, self._teammate_hi)

        nearest_boost       = self._nearest_pad(boost_pads, 0.5, 0.5)
        nearest_large_boost = self._nearest_pad(
            [p for p in boost_pads if p.pad_type == BoostPadType.LARGE], 0.5, 0.5
        )

        phase = self._determine_phase(ball, boost)

        state = GameState(
            ball_x=ball.x,            ball_y=ball.y,
            ball_visible=ball.visible, ball_radius=ball.radius,
            boost=boost,              phase=phase,
            own_goal=own_goal,        enemy_goal=enemy_goal,
            enemies=enemies,          teammates=teammates,
            boost_pads=boost_pads,
            nearest_boost=nearest_boost,
            nearest_large_boost=nearest_large_boost,
            frame_raw=raw,            frame_processed=processed,
        )

        # Schuss-Opportunity berechnen
        state.shot_opportunity, state.ball_to_goal_angle = \
            self._calc_shot_opportunity(state)

        return state

    # ── Ball-Erkennung (robust mit Tracking) ────────────────────────────────

    def _find_ball_robust(self, frame: np.ndarray, hsv: np.ndarray) -> BallDetection:
        """
        Findet den Ball mit Hough-Circle + Farb-Maske kombiniert.
        Nutzt Tracking: wenn der Ball kurz verschwindet, wird die letzte
        bekannte Position noch für _ball_max_lost Frames gehalten.
        """
        candidates: list[BallDetection] = []

        # Methode 1: Farb-Maske (orange)
        color_ball = self._find_ball_by_color(frame, hsv)
        if color_ball.visible:
            candidates.append(color_ball)

        # Methode 2: Hough-Kreise (formbasiert, farb-unabhängig)
        hough_ball = self._find_ball_by_hough(frame)
        if hough_ball.visible:
            candidates.append(hough_ball)

        # Besten Kandidaten wählen
        if candidates:
            best = max(candidates, key=lambda b: b.confidence)
            self._last_ball = best
            self._ball_lost_frames = 0
            return best

        # Kein Kandidat: Tracking-Buffer nutzen
        self._ball_lost_frames += 1
        if self._ball_lost_frames <= self._ball_max_lost and self._last_ball.visible:
            # Letzte Position mit abnehmender Konfidenz zurückgeben
            decay = 1.0 - (self._ball_lost_frames / self._ball_max_lost)
            return BallDetection(
                self._last_ball.x, self._last_ball.y,
                True, self._last_ball.radius,
                self._last_ball.confidence * decay,
            )

        self._last_ball = BallDetection(-1.0, -1.0, False, 0.0, 0.0)
        return self._last_ball

    def _find_ball_by_color(self, frame: np.ndarray, hsv: np.ndarray) -> BallDetection:
        """Orange/weißen Ball per HSV-Maske finden."""
        mask = cv2.inRange(hsv, self._ball_lo, self._ball_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BallDetection(-1.0, -1.0, False, 0.0, 0.0)

        h, w = frame.shape[:2]
        best_score = 0.0
        best: BallDetection | None = None

        for c in contours:
            area = cv2.contourArea(c)
            if area < self._cfg.ball_min_area:
                continue
            # Wie kreisförmig ist der Kontour? (1.0 = perfekter Kreis)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:   # zu unrund → kein Ball
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            radius = (area ** 0.5) / max(w, h)
            # Score = Kreisförmigkeit × Fläche (größerer Ball = wahrscheinlicher)
            score = circularity * min(area / 500.0, 1.0)
            if score > best_score:
                best_score = score
                best = BallDetection(cx, cy, True, radius, min(score, 1.0))

        return best or BallDetection(-1.0, -1.0, False, 0.0, 0.0)

    def _find_ball_by_hough(self, frame: np.ndarray) -> BallDetection:
        """Hough-Kreise als zweite Erkennungsmethode (unabhängig von Farbe)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = frame.shape[:2]

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=60,
            param2=30,
            minRadius=4,
            maxRadius=50,
        )
        if circles is None:
            return BallDetection(-1.0, -1.0, False, 0.0, 0.0)

        circles = np.round(circles[0, :]).astype(int)
        # Nehmen den Kreis dessen Mittelpunkt die orangeste Farbe hat
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_score = 0.0
        best: BallDetection | None = None

        for (cx, cy, r) in circles:
            # Prüfe ob Pixel im Kreis orangefarben sind
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            roi_hsv = hsv[mask > 0]
            if len(roi_hsv) == 0:
                continue
            orange_ratio = float(np.mean(
                (roi_hsv[:, 0] >= self._ball_lo[0]) &
                (roi_hsv[:, 0] <= self._ball_hi[0]) &
                (roi_hsv[:, 1] >= self._ball_lo[1])
            ))
            score = orange_ratio
            if score > best_score and score > 0.2:
                best_score = score
                nx = cx / w
                ny = cy / h
                radius = r / max(w, h)
                best = BallDetection(nx, ny, True, radius, score * 0.8)

        return best or BallDetection(-1.0, -1.0, False, 0.0, 0.0)

    # ── Tore ────────────────────────────────────────────────────────────────

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

    # ── Boostpads mit Typ ───────────────────────────────────────────────────

    def _find_boost_pads(self, frame: np.ndarray, hsv: np.ndarray) -> list[ObjectPosition]:
        """
        Findet alle Boostpads und klassifiziert sie als SMALL oder LARGE
        anhand der Blob-Fläche relativ zur Bildgröße.
        """
        h, w = frame.shape[:2]
        img_area = float(h * w)

        mask = cv2.inRange(hsv, self._pad_lo, self._pad_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result: list[ObjectPosition] = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:self._cfg.max_boost_pads]:
            area = cv2.contourArea(c)
            if area < self._cfg.boost_pad_min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            norm_area = area / img_area
            pad_type = (BoostPadType.LARGE if norm_area >= _LARGE_PAD_AREA_THRESHOLD
                        else BoostPadType.SMALL)
            result.append(ObjectPosition(
                M["m10"] / M["m00"] / w,
                M["m01"] / M["m00"] / h,
                True,
                pad_type=pad_type,
            ))
        return result

    # ── Agenten ─────────────────────────────────────────────────────────────

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

    # ── Schuss-Opportunity ──────────────────────────────────────────────────

    def _calc_shot_opportunity(self, state: GameState) -> tuple[bool, float]:
        """
        Berechnet ob eine Schuss-Möglichkeit besteht.

        Bedingungen:
        1. Ball ist sichtbar und nah am Spieler
        2. Spieler steht zwischen Ball und eigenem Tor (also "hinter" dem Ball)
        3. Ball-zu-Tor Winkel ist akzeptabel (nicht zu seitlich)
        4. Kein Gegner direkt im Weg (optional)

        Gibt zurück: (shot_opportunity, angle_to_goal)
        angle_to_goal: 0.0 = perfekt aufs Tor, 1.0 = komplett seitlich
        """
        if not state.ball_visible:
            return False, 1.0

        # Winkel Ball → gegnerisches Tor
        dx = state.enemy_goal.x - state.ball_x
        dy = state.enemy_goal.y - state.ball_y
        dist_ball_to_goal = (dx * dx + dy * dy) ** 0.5
        if dist_ball_to_goal < 0.01:
            return True, 0.0

        # Normierter horizontaler Versatz (0 = gerade, 1 = komplett seitlich)
        angle = abs(dx) / (abs(dx) + abs(dy) + 1e-6)

        ball_near = state.ball_dist_to_player < _SHOT_BALL_DIST

        # Spieler soll "hinter" dem Ball sein (höhere y = eigene Hälfte)
        player_behind_ball = state.player_y > state.ball_y + 0.05

        # Winkel zum Tor akzeptabel?
        good_angle = angle < 0.5

        shot_ok = ball_near and player_behind_ball and good_angle

        return shot_ok, angle

    # ── Hilfsfunktionen ─────────────────────────────────────────────────────

    def _largest_blob(self, hsv, lower, upper, frame, min_area) -> ObjectPosition:
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
        return min(visible, key=lambda p: p.distance_to(px, py))

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

    # ── Dummy state ─────────────────────────────────────────────────────────

    def _dummy_state(self, raw: np.ndarray, processed: np.ndarray) -> GameState:
        cfg   = self._cfg
        ball  = BallDetection(cfg.dummy_ball_x, cfg.dummy_ball_y, True, 0.04, 1.0)
        boost = cfg.dummy_boost
        phase = self._determine_phase(ball, boost)

        dummy_enemies   = [ObjectPosition(0.70, 0.30, True),
                           ObjectPosition(0.30, 0.20, True)]
        dummy_teammates = [ObjectPosition(0.50, 0.80, True)]
        dummy_pads      = [
            ObjectPosition(0.10, 0.50, True, BoostPadType.LARGE),
            ObjectPosition(0.90, 0.50, True, BoostPadType.LARGE),
            ObjectPosition(0.50, 0.10, True, BoostPadType.LARGE),
            ObjectPosition(0.50, 0.90, True, BoostPadType.LARGE),
            ObjectPosition(0.25, 0.25, True, BoostPadType.SMALL),
            ObjectPosition(0.75, 0.75, True, BoostPadType.SMALL),
        ]
        nearest       = self._nearest_pad(dummy_pads, 0.5, 0.5)
        nearest_large = self._nearest_pad(
            [p for p in dummy_pads if p.pad_type == BoostPadType.LARGE], 0.5, 0.5
        )

        state = GameState(
            ball_x=ball.x,        ball_y=ball.y,
            ball_visible=ball.visible, ball_radius=ball.radius,
            boost=boost,          phase=phase,
            own_goal=ObjectPosition(0.5, 1.0, True),
            enemy_goal=ObjectPosition(0.5, 0.0, True),
            enemies=dummy_enemies,
            teammates=dummy_teammates,
            boost_pads=dummy_pads,
            nearest_boost=nearest,
            nearest_large_boost=nearest_large,
            frame_raw=raw,        frame_processed=processed,
        )
        state.shot_opportunity, state.ball_to_goal_angle = \
            self._calc_shot_opportunity(state)
        return state
