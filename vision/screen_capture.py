import numpy as np
from config import Config

try:
    import mss
    _MSS = True
except ImportError:
    _MSS = False

try:
    from PIL import ImageGrab
    _PIL = True
except ImportError:
    _PIL = False

class ScreenCapture:
    def __init__(self, config: Config):
        self.config = config
        if _MSS:
            self._sct = mss.mss()
        elif not _PIL:
            raise ImportError("pip install mss  oder  pip install Pillow")

    def grab(self) -> np.ndarray:
        if _MSS:
            return self._grab_mss()
        return self._grab_pil()

    def _grab_mss(self) -> np.ndarray:
        mon = self.config.region or self._sct.monitors[self.config.monitor_index]
        img = self._sct.grab(mon)
        return np.array(img)[:, :, :3]

    def _grab_pil(self) -> np.ndarray:
        r = self.config.region
        box = (r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"]) if r else None
        img = ImageGrab.grab(bbox=box)
        return np.array(img)[:, :, ::-1]
