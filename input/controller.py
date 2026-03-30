from config import Config
from input.actions import Action

try:
    from pynput.keyboard import Controller as KB, Key
    from pynput.mouse import Controller as Mouse, Button
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

_BUTTON_MAP = {"right": Button.right, "left": Button.left, "middle": Button.middle} if _PYNPUT else {}
_SPECIAL_KEY_MAP = {"shift_l": Key.shift_l, "shift_r": Key.shift_r,
                    "ctrl_l": Key.ctrl_l,   "ctrl_r": Key.ctrl_r,
                    "alt_l": Key.alt_l,     "space": Key.space} if _PYNPUT else {}


def _resolve_key(raw: str):
    return _SPECIAL_KEY_MAP.get(raw, raw)


class Controller:
    def __init__(self, config: Config):
        self._key_cfg  = config.keys
        self._kb       = KB()    if _PYNPUT else None
        self._mouse    = Mouse() if _PYNPUT else None
        self._keys:  set = set()
        self._btns:  set = set()

        if not _PYNPUT:
            print("[Controller] pynput fehlt – Inputs werden nur geloggt.  pip install pynput")

    def execute(self, action: Action):
        self._sync_keys(self._resolve_keys(action))
        self._sync_btns(self._resolve_buttons(action))

    def release_all(self):
        for k in list(self._keys):
            self._up(k)
        for b in list(self._btns):
            self._btn_up(b)
        self._keys.clear()
        self._btns.clear()

    def _resolve_keys(self, action: Action) -> set:
        keys = set()
        for field_name, raw_key in self._key_cfg.as_dict().items():
            if getattr(action, field_name, False):
                keys.add(_resolve_key(raw_key))
        return keys

    def _resolve_buttons(self, action: Action) -> set:
        if getattr(action, "boost", False):
            btn = _BUTTON_MAP.get(self._key_cfg.boost_mouse_button)
            return {btn} if btn else set()
        return set()

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
