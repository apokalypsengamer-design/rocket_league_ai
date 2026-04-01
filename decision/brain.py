from __future__ import annotations
import time
from config import Config
from core.state import GameState, ObjectPosition
from input.actions import Action
from decision.rules import RuleEngine
from decision.strategy import Strategy
from core.logger import setup_logger

log = setup_logger("brain")

# ── Phase labels ─────────────────────────────────────────────────────────────
_PHASE_LABEL: dict[str, str] = {
    "attack":        "ANGRIFF      ",
    "defense":       "VERTEIDIGUNG ",
    "boost_collect": "BOOST SAMMELN",
    "rotate":        "ROTATION     ",
    "unknown":       "UNBEKANNT    ",
}

# How often (in seconds) the live status line is printed to stdout.
# Set to 0 to print every frame (flood mode).
_PRINT_INTERVAL: float = 0.25


class Brain:
    """
    Orchestrates strategy + rules and emits a readable live debug line.

    Console output format (one line per interval):
        Frame  042 | Ball (0.65,0.35) rechts | Boost  75% | ANGRIFF       | forward+boost+steer_right | Ball rechts → Boost + Fahren zum Ball
    """

    def __init__(self, config: Config):
        self._config     = config
        self.rules       = RuleEngine(config)
        self.strategy    = Strategy(config)
        self._frame      = 0
        self._last_print = 0.0

    # ── Main entry ────────────────────────────────────────────────────────────

    def decide(self, state: GameState) -> Action:
        self._frame    += 1
        state.phase     = self.strategy.refine_phase(state, self._frame)
        action          = self.rules.evaluate(state, self._frame)
        state.reasoning = self._build_reasoning(state)
        self._maybe_print(state, action)
        return action

    # ── Reasoning ─────────────────────────────────────────────────────────────

    def _build_reasoning(self, state: GameState) -> str:
        p = state.phase

        if p == "attack":
            if state.ball_visible:
                return f"Ball {state.ball_side} @ ({state.ball_x:.2f},{state.ball_y:.2f}) → Boost + Fahren zum Ball"
            return "Ball unsichtbar → fahre vorwärts zum gegnerischen Tor"

        if p == "defense":
            en = state.nearest_enemy
            e_str = f"Gegner {en.side()} @ ({en.x:.2f},{en.y:.2f})" if en else "kein Gegner sichtbar"
            return f"Ball {state.ball_side} in eigener Hälfte | {e_str} → zurück zum Tor"

        if p == "boost_collect":
            pad   = state.nearest_boost
            p_str = f"Pad @ ({pad.x:.2f},{pad.y:.2f})" if pad else "kein Pad"
            boost = f"{state.boost:.0f}%" if state.boost >= 0 else "?"
            return f"Boost niedrig ({boost}) → {p_str}"

        if state.ball_visible:
            return f"Ball {state.ball_side} @ ({state.ball_x:.2f},{state.ball_y:.2f}) → Positionierung"
        return "Ball nicht sichtbar → vorwärts"

    # ── Live console output ───────────────────────────────────────────────────

    def _maybe_print(self, state: GameState, action: Action) -> None:
        now = time.monotonic()
        if now - self._last_print < _PRINT_INTERVAL:
            return
        self._last_print = now
        self._print_line(state, action)

    def _print_line(self, state: GameState, action: Action) -> None:
        keys      = "+".join(action.active_keys()) or "IDLE"
        boost_str = f"{state.boost:5.0f}%" if state.boost >= 0 else "    ?"
        phase_str = _PHASE_LABEL.get(state.phase, state.phase)

        if state.ball_visible:
            ball_str = f"({state.ball_x:.2f},{state.ball_y:.2f}) {state.ball_side:<6}"
        else:
            ball_str = "nicht sichtbar      "

        line = (
            f"Frame {self._frame:>5} | "
            f"Ball {ball_str} | "
            f"Boost {boost_str} | "
            f"{phase_str} | "
            f"{keys:<35} | "
            f"{state.reasoning}"
        )
        print(line)
        log.debug(
            f"frame={self._frame} phase={state.phase} "
            f"action={keys} boost={boost_str.strip()} "
            f"ball=({state.ball_x:.2f},{state.ball_y:.2f}) visible={state.ball_visible}"
        )
