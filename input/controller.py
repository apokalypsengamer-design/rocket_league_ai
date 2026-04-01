from __future__ import annotations
from config import Config
from input.actions import Action
from core.logger import setup_logger

log = setup_logger("controller")

try:
    from pynput.keyboard import Controller as _KB, Key as _Key
    from pynput.mouse    import Controller as _Mouse, Button as _Button
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

# ── Special-key string → pynput Key object ─────────────────────────────────
_SPECIAL: dict[str, object] = {}
if _PYNPUT:
    _SPECIAL = {
        "shift_l":  _Key.shift_l,
        "shift_r":  _Key.shift_r,
        "ctrl_l":   _Key.ctrl_l,
        "ctrl_r":   _Key.ctrl_r,
        "alt_l":    _Key.alt_l,
        "alt_r":    _Key.alt_r,
        "space":    _Key.space,
        "enter":    _Key.enter,
        "tab":      _Key.tab,
        "esc":      _Key.esc,
        "up":       _Key.up,
        "down":     _Key.down,
        "left":     _Key.left,
        "right":    _Key.right,
        " ":        _Key.space,
    }

# ── Mouse-button string → pynput Button ────────────────────────────────────
_MOUSE_BTN: dict[str, object] = {}
if _PYNPUT:
    _MOUSE_BTN = {
        "left":   _Button.left,
        "right":  _Button.right,
        "middle": _Button.middle,
    }


def _to_key(raw: str):
    """Convert a string key name to a pynput-compatible key object."""
    return _SPECIAL.get(raw.lower(), raw)


class Controller:
    """
    Translates Action objects into real keyboard/mouse inputs via pynput.

    Design:
    - Maintains a set of currently pressed keys/buttons.
    - Each call to execute() computes the desired set, then only
      sends press/release events for the DIFF — no repeated presses.
    - release_all() guarantees a clean state on shutdown.
    - In dry-run mode (no pynput) all events are printed to stdout.
    """

    def __init__(self, config: Config):
        self._key_cfg = config.keys
        self._kb      = _KB()    if _PYNPUT else None
        self._mouse   = _Mouse() if _PYNPUT else None

        self._held_keys: set = set()
        self._held_btns: set = set()

        if not _PYNPUT:
            log.warning("pynput nicht installiert – Inputs werden simuliert (kein echtes Spiel-Input).")
            log.warning("  → pip install pynput")

    # ── Public API ──────────────────────────────────────────────────────────

    def execute(self, action: Action) -> None:
        want_keys = self._build_key_set(action)
        want_btns = self._build_btn_set(action)
        self._sync_keys(want_keys)
        self._sync_btns(want_btns)

    def release_all(self) -> None:
        for k in list(self._held_keys):
            self._release_key(k)
        for b in list(self._held_btns):
            self._release_btn(b)
        self._held_keys.clear()
        self._held_btns.clear()
        log.info("Alle Inputs freigegeben.")

    # ── Build desired sets ──────────────────────────────────────────────────

    def _build_key_set(self, action: Action) -> set:
        """
        Map each active Action boolean to the configured key string.
        Uses keyboard_map() which excludes boost when boost_via_mouse=True,
        preventing boost from accidentally firing as a key.
        """
        mapping = self._key_cfg.keyboard_map()
        keys = set()
        for action_field, raw_key in mapping.items():
            if getattr(action, action_field, False):
                resolved = _to_key(raw_key)
                if resolved:
                    keys.add(resolved)
        return keys

    def _build_btn_set(self, action: Action) -> set:
        """Map boost action to mouse button if boost_via_mouse is enabled."""
        if not self._key_cfg.boost_via_mouse:
            return set()
        if not getattr(action, "boost", False):
            return set()
        btn = _MOUSE_BTN.get(self._key_cfg.boost_mouse_btn)
        return {btn} if btn else set()

    # ── Sync pressed state with desired state ───────────────────────────────

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

    # ── Low-level press/release ─────────────────────────────────────────────

    def _press_key(self, key) -> None:
        if self._kb:
            try:
                self._kb.press(key)
            except Exception as e:
                log.error(f"KEY press error ({key}): {e}")
        else:
            log.debug(f"  [SIM] KEY↓ {key}")

    def _release_key(self, key) -> None:
        if self._kb:
            try:
                self._kb.release(key)
            except Exception as e:
                log.error(f"KEY release error ({key}): {e}")
        else:
            log.debug(f"  [SIM] KEY↑ {key}")

    def _press_btn(self, btn) -> None:
        if self._mouse:
            try:
                self._mouse.press(btn)
            except Exception as e:
                log.error(f"MOUSE press error ({btn}): {e}")
        else:
            log.debug(f"  [SIM] BTN↓ {btn}")

    def _release_btn(self, btn) -> None:
        if self._mouse:
            try:
                self._mouse.release(btn)
            except Exception as e:
                log.error(f"MOUSE release error ({btn}): {e}")
        else:
            log.debug(f"  [SIM] BTN↑ {btn}")
