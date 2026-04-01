from __future__ import annotations
from config import Config
from core.state import GameState, ObjectPosition
from input.actions import Action
from decision.rules import RuleEngine
from decision.strategy import Strategy
from core.logger import setup_logger

log = setup_logger("brain")

_PHASE_ICON = {
    "attack":        "ANGRIFF",
    "defense":       "VERTEIDIGUNG",
    "boost_collect": "BOOST SAMMELN",
    "rotate":        "ROTATION",
    "unknown":       "UNBEKANNT",
}


class Brain:
    def __init__(self, config: Config):
        self._config  = config
        self.rules    = RuleEngine(config)
        self.strategy = Strategy(config)
        self._frame   = 0

    def decide(self, state: GameState) -> Action:
        self._frame += 1
        state.phase     = self.strategy.refine_phase(state, self._frame)
        action          = self.rules.evaluate(state, self._frame)
        state.reasoning = self._build_reasoning(state)
        self._print_feedback(state, action)
        return action

    # ── Reasoning string ────────────────────────────────────────────────────

    def _build_reasoning(self, state: GameState) -> str:
        p = state.phase
        if p == "attack":
            target = f"Ball {state.ball_side}"
            if state.ball_visible:
                target += f" @ ({state.ball_x:.2f},{state.ball_y:.2f})"
            return f"{target} → Boost + Fahren zum Ball"

        if p == "defense":
            en    = state.nearest_enemy
            e_str = f"Gegner {en.side()} @ ({en.x:.2f},{en.y:.2f})" if en else "kein Gegner sichtbar"
            return (
                f"Ball {state.ball_side} in eigener Hälfte | {e_str} "
                f"→ Rückkehr zum Tor"
            )

        if p == "boost_collect":
            pad   = state.nearest_boost
            p_str = f"Pad @ ({pad.x:.2f},{pad.y:.2f})" if pad else "kein Pad sichtbar"
            boost = f"{state.boost:.0f}%" if state.boost >= 0 else "?"
            return f"Boost niedrig ({boost}) | nächstes {p_str} → Boost aufnehmen"

        ball_str = (
            f"Ball {state.ball_side} @ ({state.ball_x:.2f},{state.ball_y:.2f})"
            if state.ball_visible else "Ball nicht sichtbar"
        )
        return f"{ball_str} → Positionierung"

    # ── Console output (every frame, compact single line) ───────────────────

    def _print_feedback(self, state: GameState, action: Action) -> None:
        keys      = "+".join(action.active_keys()) or "IDLE"
        boost_str = f"{state.boost:.0f}%" if state.boost >= 0 else "?"
        phase_str = _PHASE_ICON.get(state.phase, state.phase)

        ball_str  = (
            f"({state.ball_x:.2f},{state.ball_y:.2f})"
            if state.ball_visible else "nicht sichtbar"
        )

        enemies_str   = (
            f"{len(state.enemies)} Gegner"
            if state.enemies else "keine Gegner"
        )
        teammates_str = (
            f"{len(state.teammates)} Teammates"
            if state.teammates else "keine Teammates"
        )
        pads_str      = f"{len(state.boost_pads)} Pads"

        print(
            f"Frame {self._frame:>5} | "
            f"Ball {ball_str} | "
            f"{enemies_str} | {teammates_str} | {pads_str} | "
            f"Boost {boost_str} | "
            f"Phase: {phase_str:>15} | "
            f"Action: {keys:<30} | "
            f"Grund: {state.reasoning}"
        )

        log.debug(
            f"frame={self._frame} phase={state.phase} "
            f"action={keys} boost={boost_str} "
            f"ball_visible={state.ball_visible}"
        )
