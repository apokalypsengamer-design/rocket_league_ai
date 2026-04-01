from __future__ import annotations
import time
from config import Config
from core.state import GameState, ObjectPosition
from input.actions import Action
from decision.rules import RuleEngine
from decision.strategy import Strategy
from core.logger import setup_logger

log = setup_logger("brain")

_PHASE_LABEL: dict[str, str] = {
    "attack":        "ANGRIFF      ",
    "defense":       "VERTEIDIGUNG ",
    "boost_collect": "BOOST SAMMELN",
    "rotate":        "ROTATION     ",
    "unknown":       "UNBEKANNT    ",
}

_PRINT_INTERVAL: float = 0.25


class Brain:
    """
    Orchestriert Strategy + Rules und gibt lesbaren Debug-Output aus.

    Änderung gegenüber Vorversion:
    - rules.evaluate() gibt jetzt (Action, reason) zurück
    - Shot-Opportunity wird in der Ausgabe angezeigt
    - Phase 'shot' wird als Override behandelt (unabhängig von Strategy)
    """

    def __init__(self, config: Config):
        self._config     = config
        self.rules       = RuleEngine(config)
        self.strategy    = Strategy(config)
        self._frame      = 0
        self._last_print = 0.0

    def decide(self, state: GameState) -> Action:
        self._frame += 1

        # Shot-Opportunity überschreibt immer die Strategy
        if state.shot_opportunity:
            state.phase = "shot"
            self.strategy.reset_pending()   # Hysterese nicht durch Schuss korrumpieren
        else:
            state.phase = self.strategy.refine_phase(state, self._frame)

        # Entscheidung + Begründung
        action, reasoning = self.rules.evaluate(state, self._frame)
        state.reasoning   = reasoning

        self._maybe_print(state, action)
        return action

    def _maybe_print(self, state: GameState, action: Action) -> None:
        now = time.monotonic()
        if now - self._last_print < _PRINT_INTERVAL:
            return
        self._last_print = now
        self._print_line(state, action)

    def _print_line(self, state: GameState, action: Action) -> None:
        keys      = "+".join(action.active_keys()) or "IDLE"
        boost_str = f"{state.boost:5.0f}%" if state.boost >= 0 else "    ?"

        if state.phase == "shot":
            phase_str = "*** SCHUSS ***"
        else:
            phase_str = _PHASE_LABEL.get(state.phase, state.phase)

        if state.ball_visible:
            ball_str = f"({state.ball_x:.2f},{state.ball_y:.2f}) {state.ball_side:<6}"
        else:
            ball_str = "nicht sichtbar      "

        shot_flag = " [SHOOT]" if state.shot_opportunity else ""

        line = (
            f"Frame {self._frame:>5} | "
            f"Ball {ball_str}{shot_flag} | "
            f"Boost {boost_str} | "
            f"{phase_str:<15} | "
            f"{keys:<35} | "
            f"{state.reasoning}"
        )
        print(line)
        log.debug(
            f"frame={self._frame} phase={state.phase} "
            f"shot={state.shot_opportunity} action={keys} "
            f"boost={boost_str.strip()} "
            f"ball=({state.ball_x:.2f},{state.ball_y:.2f}) visible={state.ball_visible}"
        )
