from __future__ import annotations
import numpy as np
from config import Config
from core.state import GameState, ObjectPosition, BoostPadType
from vision.preprocessing import to_gray, crop_region
from vision.detector import GameDetector, TrackerConfig, TrackedObject


_SHOT_BALL_DIST  = 0.25
_SHOT_ALIGN_DIST = 0.20


class Detector:
    """
    Haupt-Detektor für den Game-Loop.
    Delegiert die eigentliche Bild-Erkennung an GameDetector (detector.py),
    übersetzt dessen TrackedObject-Ergebnisse in GameState-Objekte.
    """

    def __init__(self, config: Config):
        self._cfg      = config.vision
        self._gameplay = config.gameplay
        self._game_det = GameDetector(config, TrackerConfig())

    def detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        if self._cfg.dummy_mode:
            return self._dummy_state(raw, processed)
        return self._real_detect(processed, raw)

    def _real_detect(self, processed: np.ndarray, raw: np.ndarray) -> GameState:
        det   = self._game_det.update(processed)
        boost = self._read_boost(processed)

        ball  = det["ball"]
        enemy = det["enemy"]
        goals = det["goals"]
        pads  = det["boosts"]

        ball_visible = ball.found
        ball_x       = ball.nx if ball.found else -1.0
        ball_y       = ball.ny if ball.found else -1.0
        ball_radius  = ball.radius / max(processed.shape[1], processed.shape[0])

        own_goal   = self._tracked_to_pos(goals.get("own"),   0.5, 1.0)
        enemy_goal = self._tracked_to_pos(goals.get("enemy"), 0.5, 0.0)

        enemies = []
        if enemy.found:
            enemies.append(ObjectPosition(enemy.nx, enemy.ny, True))

        boost_pads: list[ObjectPosition] = []
        for p in pads:
            pad_type = BoostPadType.LARGE if p.width == 1 else BoostPadType.SMALL
            boost_pads.append(ObjectPosition(p.nx, p.ny, True, pad_type=pad_type))

        nearest_boost       = self._nearest_pad(boost_pads, 0.5, 0.5)
        nearest_large_boost = self._nearest_pad(
            [p for p in boost_pads if p.pad_type == BoostPadType.LARGE], 0.5, 0.5
        )

        phase = self._determine_phase(ball_visible, ball_y, boost)

        state = GameState(
            ball_x=ball_x,            ball_y=ball_y,
            ball_visible=ball_visible, ball_radius=ball_radius,
            boost=boost,              phase=phase,
            own_goal=own_goal,        enemy_goal=enemy_goal,
            enemies=enemies,          teammates=[],
            boost_pads=boost_pads,
            nearest_boost=nearest_boost,
            nearest_large_boost=nearest_large_boost,
            frame_raw=raw,            frame_processed=processed,
        )
        state.shot_opportunity, state.ball_to_goal_angle = \
            self._calc_shot_opportunity(state)
        return state

    def _dummy_state(self, raw: np.ndarray, processed: np.ndarray) -> GameState:
        cfg   = self._cfg
        boost = cfg.dummy_boost
        phase = self._determine_phase(True, cfg.dummy_ball_y, boost)
        dummy_pads = [
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
            ball_x=cfg.dummy_ball_x, ball_y=cfg.dummy_ball_y,
            ball_visible=True,       ball_radius=0.04,
            boost=boost,             phase=phase,
            own_goal=ObjectPosition(0.5, 1.0, True),
            enemy_goal=ObjectPosition(0.5, 0.0, True),
            enemies=[ObjectPosition(0.70, 0.30, True),
                     ObjectPosition(0.30, 0.20, True)],
            teammates=[ObjectPosition(0.50, 0.80, True)],
            boost_pads=dummy_pads,
            nearest_boost=nearest,
            nearest_large_boost=nearest_large,
            frame_raw=raw, frame_processed=processed,
        )
        state.shot_opportunity, state.ball_to_goal_angle = \
            self._calc_shot_opportunity(state)
        return state

    def _calc_shot_opportunity(self, state: GameState) -> tuple[bool, float]:
        if not state.ball_visible:
            return False, 1.0
        dx = state.enemy_goal.x - state.ball_x
        dy = state.enemy_goal.y - state.ball_y
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 0.01:
            return True, 0.0
        angle      = abs(dx) / (abs(dx) + abs(dy) + 1e-6)
        ball_near  = state.ball_dist_to_player < _SHOT_BALL_DIST
        behind     = state.player_y > state.ball_y + 0.05
        good_angle = angle < 0.5
        return ball_near and behind and good_angle, angle

    @staticmethod
    def _tracked_to_pos(obj, default_x: float, default_y: float) -> ObjectPosition:
        if obj and obj.found:
            return ObjectPosition(obj.nx, obj.ny, True)
        return ObjectPosition(default_x, default_y, True)

    @staticmethod
    def _nearest_pad(pads: list[ObjectPosition],
                     px: float, py: float) -> ObjectPosition | None:
        visible = [p for p in pads if p.visible]
        if not visible:
            return None
        return min(visible, key=lambda p: ((p.x-px)**2 + (p.y-py)**2)**0.5)

    def _read_boost(self, frame: np.ndarray) -> float:
        x, y, w, h = self._cfg.boost_roi
        roi = crop_region(frame, x, y, w, h)
        if roi.size == 0:
            return -1.0
        gray         = to_gray(roi)
        bright_ratio = float(np.mean(gray > self._cfg.boost_brightness_threshold))
        return round(min(bright_ratio * self._cfg.boost_scale_factor * 100, 100.0), 1)

    def _determine_phase(self, ball_visible: bool,
                         ball_y: float, boost: float) -> str:
        g = self._gameplay
        if 0 <= boost < g.boost_low_threshold:
            return "boost_collect"
        if not ball_visible:
            return "rotate"
        if ball_y > g.defense_threshold:
            return "defense"
        if ball_y < g.attack_threshold:
            return "attack"
        return "rotate"
