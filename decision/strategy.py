from config import Config
from core.state import GameState

class Strategy:
    def __init__(self, config: Config):
        self.config = config
        self._history: list[str] = []

    def refine_phase(self, state: GameState, frame: int) -> str:
        phase = state.phase
        self._history.append(phase)
        if len(self._history) > 60:
            self._history.pop(0)
        return phase

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
        if state.boost != -1 and state.boost < 10:
            return False
        return True
