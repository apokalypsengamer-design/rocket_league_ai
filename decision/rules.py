from __future__ import annotations
from config import Config
from core.state import GameState, ObjectPosition
from input.actions import Action


class RuleEngine:
    """
    Pure rule-based decision engine.  No state, no history — each call
    receives a GameState and returns a fresh Action.

    Rules (priority order):
      1. boost_collect  → drive toward nearest boost pad
      2. defense        → drive toward ball (to intercept) with boost
      3. attack         → drive toward ball with boost
      4. rotate         → drive toward ball without boost (conserve)
    """

    def __init__(self, config: Config):
        self._dz = config.gameplay.steer_dead_zone

    def evaluate(self, state: GameState, frame: int) -> Action:
        if state.phase == "attack":
            return self._attack(state)
        if state.phase == "defense":
            return self._defense(state)
        if state.phase == "boost_collect":
            return self._boost_collect(state)
        return self._rotate(state)

    # ── Phase handlers ────────────────────────────────────────────────────────

    def _attack(self, state: GameState) -> Action:
        """Drive toward ball with boost.  If ball invisible, aim at enemy goal."""
        target = self._ball_as_obj(state) if state.ball_visible else state.enemy_goal
        return self._steer_to(Action(forward=True, boost=True), state.player_x, target)

    def _defense(self, state: GameState) -> Action:
        """Rush back toward ball / own goal with boost."""
        target = self._ball_as_obj(state) if state.ball_visible else state.own_goal
        return self._steer_to(Action(forward=True, boost=True), state.player_x, target)

    def _rotate(self, state: GameState) -> Action:
        """Drive forward toward ball without wasting boost."""
        if state.ball_visible:
            return self._steer_to(Action(forward=True), state.player_x, self._ball_as_obj(state))
        return Action(forward=True)

    def _boost_collect(self, state: GameState) -> Action:
        """Navigate toward nearest visible boost pad; fall back to ball direction."""
        a = Action(forward=True)
        pad = state.nearest_boost
        if pad and pad.visible:
            return self._steer_to(a, state.player_x, pad)
        if state.ball_visible:
            return self._steer_to(a, state.player_x, self._ball_as_obj(state))
        return a

    # ── Steering helper ───────────────────────────────────────────────────────

    def _steer_to(self, action: Action, player_x: float, target: ObjectPosition) -> Action:
        """Set steer_left / steer_right based on horizontal offset to target.
        Uses a dead-zone to avoid jitter when the target is centred."""
        if not target or not target.visible:
            return action
        diff = target.x - player_x
        action.steer_left  = diff < -self._dz
        action.steer_right = diff >  self._dz
        return action

    @staticmethod
    def _ball_as_obj(state: GameState) -> ObjectPosition:
        return ObjectPosition(state.ball_x, state.ball_y, state.ball_visible)
