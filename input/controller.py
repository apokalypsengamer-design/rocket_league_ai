from __future__ import annotations
import threading
from config import Config
from input.actions import Action
from core.logger import setup_logger

log = setup_logger("controller")

# ── pynput import (optional) ────────────────────────────────────────────────
try:
    from pynput.keyboard import Controller as _KB, Key as _Key
    from pynput.mouse    import Controller as _Mouse, Button as _Button
    _PYNPUT = True
except ImportError:
    _PYNPUT = False
    log.warning("pynput nicht installiert — Inputs werden nur simuliert.")
    log.warning("  → pip install pynput")

# ── String → pynput Key mapping ─────────────────────────────────────────────
# All known special-key strings, including both "space" and " ".
_SPECIAL: dict[str, object] = {}
if _PYNPUT:
    _SPECIAL = {
        "space":    _Key.space,
        " ":        _Key.space,
        "shift_l":  _Key.shift_l,
        "shift_r":  _Key.shift_r,
        "ctrl_l":   _Key.ctrl_l,
        "ctrl_r":   _Key.ctrl_r,
        "alt_l":    _Key.alt_l,
        "alt_r":    _Key.alt_r,
        "enter":    _Key.enter,
        "tab":      _Key.tab,
        "esc":      _Key.esc,
        "up":       _Key.up,
        "down":     _Key.down,
        "left":     _Key.left,
        "right":    _Key.right,
        "backspace": _Key.backspace,
        "delete":   _Key.delete,
        "home":     _Key.home,
        "end":      _Key.end,
    }

_MOUSE_BTN: dict[str, object] = {}
if _PYNPUT:
    _MOUSE_BTN = {
        "left":   _Button.left,
        "right":  _Button.right,
        "middle": _Button.middle,
    }


def _resolve_key(raw: str):
    """Convert a key name string to the correct pynput key object.
    Single printable characters are passed through as-is (pynput accepts them).
    Special names (space, shift_l, …) are looked up in _SPECIAL."""
    if not raw:
        return None
    lower = raw.lower()
    if lower in _SPECIAL:
        return _SPECIAL[lower]
    if len(raw) == 1:          # single printable char e.g. "w", "a", "x"
        return raw
    log.warning(f"Unbekannter Key-String '{raw}' — wird ignoriert.")
    return None


class Controller:
    """
    Translates Action dataclass → real pynput keyboard/mouse events.

    Key design decisions
    ────────────────────
    • State-diff approach: we maintain the *currently held* set of keys.
      execute() computes the *desired* set, then only sends press/release
      for the difference.  This means keys are HELD between frames with
      zero extra events — no spam, no stuttering.

    • Thread lock: a single lock guards _held_keys / _held_btns so that
      release_all() called from a signal handler can never race with execute().

    • Graceful degradation: if pynput is absent every event is logged so the
      rest of the pipeline can still be tested without real inputs.
    """

    def __init__(self, config: Config):
        self._keys_cfg = config.keys
        self._lock     = threading.Lock()

        self._kb    = _KB()    if _PYNPUT else None
        self._mouse = _Mouse() if _PYNPUT else None

        self._held_keys: set = set()
        self._held_btns: set = set()

        # Pre-compute the keyboard mapping once so execute() is cheap.
        self._kb_map: dict[str, object] = {}
        for action_field, raw in config.keys.keyboard_map().items():
            resolved = _resolve_key(raw)
            if resolved is not None:
                self._kb_map[action_field] = resolved

        log.info(f"Controller bereit | pynput={_PYNPUT}")
        log.info(f"  Keyboard-Map: {self._kb_map}")
        log.info(f"  Boost via Maus: {config.keys.boost_via_mouse} ({config.keys.boost_mouse_btn})")

    # ── Public API ───────────────────────────────────────────────────────────

    def execute(self, action: Action) -> None:
        want_keys = self._build_key_set(action)
        want_btns = self._build_btn_set(action)
        with self._lock:
            self._sync_keys(want_keys)
            self._sync_btns(want_btns)

    def release_all(self) -> None:
        with self._lock:
            for k in list(self._held_keys):
                self._release_key(k)
            for b in list(self._held_btns):
                self._release_btn(b)
            self._held_keys.clear()
            self._held_btns.clear()
        log.info("Alle Inputs freigegeben.")

    # ── Build desired input sets ─────────────────────────────────────────────

    def _build_key_set(self, action: Action) -> set:
        keys: set = set()
        for field_name, key_obj in self._kb_map.items():
            if getattr(action, field_name, False):
                keys.add(key_obj)
        return keys

    def _build_btn_set(self, action: Action) -> set:
        if not self._keys_cfg.boost_via_mouse:
            return set()
        if not getattr(action, "boost", False):
            return set()
        btn = _MOUSE_BTN.get(self._keys_cfg.boost_mouse_btn)
        return {btn} if btn else set()

    # ── Diff & sync ──────────────────────────────────────────────────────────

    def _sync_keys(self, want: set) -> None:
        for k in want - self._held_keys:
            self._press_key(k)
        for k in self._held_keys - want:
            self._release_key(k)
        self._held_keys = set(want)

    def _sync_btns(self, want: set) -> None:
        for b in want - self._held_btns:
            self._press_btn(b)
        for b in self._held_btns - want:
            self._release_btn(b)
        self._held_btns = set(want)

    # ── Low-level events ─────────────────────────────────────────────────────

    def _press_key(self, key) -> None:
        log.debug(f"KEY↓ {key}")
        if self._kb:
            try:
                self._kb.press(key)
            except Exception as exc:
                log.error(f"KEY press fehlgeschlagen ({key}): {exc}")

    def _release_key(self, key) -> None:
        log.debug(f"KEY↑ {key}")
        if self._kb:
            try:
                self._kb.release(key)
            except Exception as exc:
                log.error(f"KEY release fehlgeschlagen ({key}): {exc}")

    def _press_btn(self, btn) -> None:
        log.debug(f"BTN↓ {btn}")
        if self._mouse:
            try:
                self._mouse.press(btn)
            except Exception as exc:
                log.error(f"MOUSE press fehlgeschlagen ({btn}): {exc}")

    def _release_btn(self, btn) -> None:
        log.debug(f"BTN↑ {btn}")
        if self._mouse:
            try:
                self._mouse.release(btn)
            except Exception as exc:
                log.error(f"MOUSE release fehlgeschlagen ({btn}): {exc}")
