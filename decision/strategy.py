from __future__ import annotations
from collections import deque
from config import Config
from core.state import GameState


class Strategy:
    """
    Wendet Hysterese auf die rohe Phase vom Detector an, damit der Bot
    nicht frame-für-frame zwischen Phasen wechselt.

    Neu: reset_pending() für externe Overrides (z.B. Schuss-Phase).
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
            self._pending_phase = raw
            self._pending_count = 0
        else:
            if raw == self._pending_phase:
                self._pending_count += 1
            else:
                self._pending_phase = raw
                self._pending_count = 1

            if self._pending_count >= self._cfg.phase_hysteresis:
                self._committed_phase = raw
                self._pending_count   = 0

        return self._committed_phase

    def reset_pending(self) -> None:
        """
        Setzt den pending-Counter zurück, ohne die committed phase zu ändern.
        Wird aufgerufen wenn ein externer Override (z.B. Schuss) aktiv ist,
        damit die Hysterese nicht durch die Override-Frames korrumpiert wird.
        """
        self._pending_count = 0
        self._pending_phase = self._committed_phase

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
