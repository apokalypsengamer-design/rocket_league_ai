from config import Config
from input.actions import Action

try:
    from pynput.keyboard import Controller as KB, Key
    from pynput.mouse import Controller as Mouse, Button
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

_BUTTON_MAP = {"right": Button.right, "left": Button.left, "middle": Button.middle}

class Controller:
    def __init__(self, config: Config):
        self.config = config
        self._kb     = KB()    if _PYNPUT else None
        self._mouse  = Mouse() if _PYNPUT else None
        self._keys: set   = set()
        self._btns: set   = set()

        if not _PYNPUT:
            print("[Controller] pynput fehlt – Inputs werden nur geloggt.  pip install pynput")

    def execute(self, action: Action):
        want_keys = self._resolve_keys(action)
        want_btns = self._resolve_buttons(action)
        self._sync_keys(want_keys)
        self._sync_btns(want_btns)

    def release_all(self):
        for k in list(self._keys):
            self._up(k)
        for b in list(self._btns):
            self._btn_up(b)

    def _resolve_keys(self, action: Action) -> set:
        keys = set()
        for field, key in self.config.keys.items():
            if getattr(action, field, False):
                keys.add(key)
        return keys

    def _resolve_buttons(self, action: Action) -> set:
        btns = set()
        for field, btn_name in self.config.mouse_buttons.items():
            if getattr(action, field, False):
                btns.add(_BUTTON_MAP.get(btn_name))
        return {b for b in btns if b is not None}

    def _sync_keys(self, want: set):
        for k in want - self._keys:
            self._down(k)
        for k in self._keys - want:
            self._up(k)
        self._keys = want

    def _sync_btns(self, want: set):
        for b in want - self._btns:
            self._btn_down(b)
        for b in self._btns - want:
            self._btn_up(b)
        self._btns = want

    def _down(self, key):
        if self._kb:
            self._kb.press(key)
        else:
            print(f"  KEY↓ {key}")

    def _up(self, key):
        if self._kb:
            self._kb.release(key)
        else:
            print(f"  KEY↑ {key}")

    def _btn_down(self, btn):
        if self._mouse:
            self._mouse.press(btn)
        else:
            print(f"  BTN↓ {btn}")

    def _btn_up(self, btn):
        if self._mouse:
            self._mouse.release(btn)
        else:
            print(f"  BTN↑ {btn}")
