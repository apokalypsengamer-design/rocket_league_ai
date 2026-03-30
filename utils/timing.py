import time

class FrameTimer:
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self._frame_time = 1.0 / target_fps
        self._last = time.perf_counter()
        self.actual_fps = 0.0
        self._count = 0
        self._window_start = time.perf_counter()

    def tick(self, frame_start: float):
        elapsed = time.perf_counter() - frame_start
        sleep = self._frame_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

        self._count += 1
        now = time.perf_counter()
        if now - self._window_start >= 1.0:
            self.actual_fps = self._count / (now - self._window_start)
            self._count = 0
            self._window_start = now

class Stopwatch:
    def __init__(self):
        self._start: float | None = None

    def start(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def reset(self):
        self._start = None
