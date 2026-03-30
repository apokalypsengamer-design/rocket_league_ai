from core.state import GameState

class RewardCalculator:
    def __init__(self):
        self._prev: GameState | None = None

    def calculate(self, state: GameState) -> float:
        reward = 0.0

        if state.ball_visible:
            reward += 0.1

        if state.boost != -1:
            reward += state.boost * 0.001

        if state.phase == "attack" and state.ball_visible and state.ball_y < 0.4:
            reward += 0.5

        if state.phase == "defense" and state.ball_visible and state.ball_y > 0.6:
            reward -= 0.3

        self._prev = state
        return reward
