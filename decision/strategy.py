from __future__ import annotations
from collections import deque
from config import Config
from core.state import GameState


class Strategy:
    def __init__(self, config: Config):
        self._cfg = config.gameplay
        self._history: deque[str] = deque(maxlen=self._cfg.phase_history_len)
        self._committed_phase: str = "rotate"
        self._phase_counter: int = 0

    def refine_phase(self, state: GameState, frame: int) -> str:
        raw_phase = state.phase
        self._history.append(raw_phase)

        if raw_phase == self._committed_phase:
            self._phase_counter += 1
        else:
            self._phase_counter = 1
            if self._phase_counter >= self._cfg.phase_hysteresis:
                self._committed_phase = raw_phase

        if raw_phase != self._committed_phase:
            if self._phase_counter >= self._cfg.phase_hysteresis:
                self._committed_phase = raw_phase
                self._phase_counter = 0

        return self._committed_phase

    def ball_side(self, state: GameState) -> str:
        if not state.ball_visible:
            return "unknown"
        if state.ball_x < 0.33:
            return "left"
        if state.ball_x > 0.66:
            return "right"
        return "center"

    def should_challenge(self, state: GameState) -> bool:
        if not state.ball_visible:
            return False
        if state.boost not in (-1,) and state.boost < 10:
            return False
        return True

    def dominant_phase(self) -> str:
        if not self._history:
            return "rotate"
        return max(set(self._history), key=self._history.count)
