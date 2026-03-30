from __future__ import annotations
from config import Config
from core.state import GameState
from input.actions import Action
from decision.rules import RuleEngine
from decision.strategy import Strategy
from core.logger import setup_logger

log = setup_logger("brain")

_PHASE_LABELS = {
    "attack":        "⚽ Angriff",
    "defense":       "🛡  Verteidigung",
    "boost_collect": "⚡ Boost sammeln",
    "rotate":        "🔄 Rotation",
    "unknown":       "❓ Unbekannt",
}


class Brain:
    def __init__(self, config: Config):
        self._config   = config
        self.rules     = RuleEngine(config)
        self.strategy  = Strategy(config)
        self._frame    = 0
        self._last_log = ""

    def decide(self, state: GameState) -> Action:
        self._frame += 1
        state.phase = self.strategy.refine_phase(state, self._frame)
        action = self.rules.evaluate(state, self._frame)
        state.reasoning = self._build_reasoning(state, action)
        self._console_feedback(state, action)
        return action

    def _build_reasoning(self, state: GameState, action: Action) -> str:
        phase = state.phase
        if phase == "attack":
            return f"Ball {state.ball_side} → fahre mit Boost zum Ball"
        if phase == "defense":
            enemy = state.nearest_enemy
            enemy_str = f"Gegner {enemy.side()}" if enemy else "kein Gegner sichtbar"
            return f"Ball in eigener Hälfte ({state.ball_side}), {enemy_str} → zurück zum Tor"
        if phase == "boost_collect":
            pad = state.nearest_boost
            pad_str = f"nächstes Pad bei ({pad.x:.2f},{pad.y:.2f})" if pad else "kein Pad sichtbar"
            return f"Boost niedrig ({state.boost:.0f}%) → {pad_str}"
        return f"Ball {state.ball_side} → Positionierung"

    def _console_feedback(self, state: GameState, action: Action) -> None:
        if self._frame % 10 != 1:
            return

        keys     = ", ".join(action.active_keys()) or "IDLE"
        phase    = _PHASE_LABELS.get(state.phase, state.phase)
        boost    = f"{state.boost:.0f}%" if state.boost >= 0 else "?"
        enemy    = state.nearest_enemy
        enemy_str = f"Gegner {enemy.side()}" if enemy else "kein Gegner"

        lines = [
            f"┌─ Frame {self._frame:>6} ──────────────────────────────",
            f"│  Aktion   : {keys}",
            f"│  Ziel     : Ball {state.ball_side}, {enemy_str}",
            f"│  Strategie: {state.reasoning}",
            f"│  Phase    : {phase}  │  Boost: {boost}  │  Ball sichtbar: {state.ball_visible}",
            f"└────────────────────────────────────────────────────",
        ]
        output = "\n".join(lines)

        if output != self._last_log:
            print(output)
            log.debug(f"Frame={self._frame} Action={keys} Phase={state.phase} Boost={boost}")
            self._last_log = output
