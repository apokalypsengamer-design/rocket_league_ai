from config import Config
from core.state import GameState
from input.actions import Action


class RuleEngine:
    def __init__(self, config: Config):
        self._dz = config.gameplay.steer_dead_zone

    def evaluate(self, state: GameState, frame: int) -> Action:
        phase = state.phase
        if phase == "attack":
            return self._attack(state)
        if phase == "defense":
            return self._defense(state)
        if phase == "boost_collect":
            return self._boost_collect(state)
        return self._rotate(state)

    def _attack(self, state: GameState) -> Action:
        return self._steer_to_ball(Action(forward=True, boost=True), state)

    def _defense(self, state: GameState) -> Action:
        a = Action(forward=True, boost=True)
        a.steer_left  = state.ball_x < (0.5 - self._dz)
        a.steer_right = state.ball_x > (0.5 + self._dz)
        return a

    def _rotate(self, state: GameState) -> Action:
        return self._steer_to_ball(Action(forward=True), state)

    def _boost_collect(self, state: GameState) -> Action:
        a = Action(forward=True)
        if state.ball_visible:
            a.steer_left  = state.ball_x < (0.5 - self._dz)
            a.steer_right = state.ball_x > (0.5 + self._dz)
        else:
            a.steer_right = True
        return a

    def _steer_to_ball(self, action: Action, state: GameState) -> Action:
        if not state.ball_visible:
            return action
        action.steer_left  = state.ball_x < (0.5 - self._dz)
        action.steer_right = state.ball_x > (0.5 + self._dz)
        return action
