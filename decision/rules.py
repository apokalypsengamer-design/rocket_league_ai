from __future__ import annotations
from config import Config
from core.state import GameState, ObjectPosition
from input.actions import Action
from utils.math_utils import distance


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
        target = self._ball_pos(state) if state.ball_visible else state.enemy_goal
        a = Action(forward=True, boost=True)
        return self._steer_to(a, state.player_x, target)

    def _defense(self, state: GameState) -> Action:
        a = Action(forward=True, boost=True)
        target = self._ball_pos(state) if state.ball_visible else state.own_goal
        return self._steer_to(a, state.player_x, target)

    def _rotate(self, state: GameState) -> Action:
        if state.ball_visible:
            a = Action(forward=True)
            return self._steer_to(a, state.player_x, self._ball_pos(state))
        return Action(forward=True)

    def _boost_collect(self, state: GameState) -> Action:
        a = Action(forward=True)
        pad = state.nearest_boost
        if pad and pad.visible:
            return self._steer_to(a, state.player_x, pad)
        if state.ball_visible:
            return self._steer_to(a, state.player_x, self._ball_pos(state))
        return a

    def _steer_to(self, action: Action, player_x: float, target: ObjectPosition) -> Action:
        if not target or not target.visible:
            return action
        diff = target.x - player_x
        action.steer_left  = diff < -self._dz
        action.steer_right = diff >  self._dz
        return action

    @staticmethod
    def _ball_pos(state: GameState) -> ObjectPosition:
        return ObjectPosition(state.ball_x, state.ball_y, state.ball_visible)
