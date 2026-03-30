from config import Config
from core.state import GameState
from input.actions import Action

class RuleEngine:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, state: GameState, frame: int) -> Action:
        phase = state.phase

        if phase == "attack":
            return self._attack(state)
        elif phase == "defense":
            return self._defense(state)
        elif phase == "boost_collect":
            return self._boost_collect(frame)
        else:
            return self._rotate(state)

    def _attack(self, state: GameState) -> Action:
        a = Action(forward=True, boost=True)
        return self._steer_to_ball(a, state)

    def _defense(self, state: GameState) -> Action:
        a = Action(forward=True, boost=True)
        a.steer_left  = state.ball_x < 0.45
        a.steer_right = state.ball_x > 0.55
        return a

    def _rotate(self, state: GameState) -> Action:
        a = Action(forward=True)
        return self._steer_to_ball(a, state)

    def _boost_collect(self, frame: int) -> Action:
        a = Action(forward=True)
        if (frame // 20) % 2 == 0:
            a.steer_right = True
        else:
            a.steer_left = True
        return a

    @staticmethod
    def _steer_to_ball(action: Action, state: GameState) -> Action:
        if not state.ball_visible:
            return action
        if state.ball_x < 0.45:
            action.steer_left, action.steer_right = True, False
        elif state.ball_x > 0.55:
            action.steer_right, action.steer_left = True, False
        return action
