from __future__ import annotations
from collections import deque
from config import Config
from core.state import GameState


class Strategy:
    """
    Applies hysteresis to the raw phase from the Detector so the Bot
    does not flicker between phases on every frame.

    Fix vs. previous version
    ─────────────────────────
    The old code had a logic error: it set _phase_counter = 1 inside the
    `else` branch and *immediately* checked whether that counter reached
    the hysteresis threshold — which it never could (1 < 8).  The counter
    therefore never committed a new phase.

    New logic: track how many *consecutive* frames the raw phase differs
    from the committed phase.  Only switch when that streak reaches the
    threshold.
    """

    def __init__(self, config: Config):
        self._cfg             = config.gameplay
        self._history: deque[str] = deque(maxlen=self._cfg.phase_history_len)
        self._committed_phase: str = "rotate"
        self._pending_phase:   str = "rotate"
        self._pending_count:   int = 0

    def refine_phase(self, state: GameState, frame: int) -> str:
        raw = state.phase
        self._history.append(raw)

        if raw == self._committed_phase:
            # Already in committed phase — reset pending streak.
            self._pending_phase = raw
            self._pending_count = 0
        else:
            if raw == self._pending_phase:
                self._pending_count += 1
            else:
                # New candidate phase — start fresh streak.
                self._pending_phase = raw
                self._pending_count = 1

            if self._pending_count >= self._cfg.phase_hysteresis:
                self._committed_phase = raw
                self._pending_count   = 0

        return self._committed_phase

    def dominant_phase(self) -> str:
        if not self._history:
            return "rotate"
        return max(set(self._history), key=self._history.count)

    def should_challenge(self, state: GameState) -> bool:
        if not state.ball_visible:
            return False
        if state.boost not in (-1,) and state.boost < 10:
            return False
        return True
