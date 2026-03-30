import time
from config import Config
from core.state import GameState
from core.logger import setup_logger
from vision.screen_capture import ScreenCapture
from vision.preprocessing import preprocess
from vision.detection import Detector
from decision.brain import Brain
from input.controller import Controller
from utils.timing import FrameTimer

log = setup_logger("game_loop")


class GameLoop:
    def __init__(self, config: Config):
        self.config = config
        self.capture    = ScreenCapture(config)
        self.detector   = Detector(config)
        self.brain      = Brain(config)
        self.controller = Controller(config)
        self.timer      = FrameTimer(config.capture.fps)
        self.running    = False

    def run(self):
        self.running = True
        log.info(f"Loop gestartet @ {self.config.capture.fps} FPS")
        while self.running:
            start = time.perf_counter()

            frame     = self.capture.grab()
            processed = preprocess(frame, self.config.vision)
            state     = self.detector.detect(processed, frame)
            action    = self.brain.decide(state)
            self.controller.execute(action)

            self.timer.tick(start)

    def stop(self):
        self.running = False
        self.controller.release_all()
        log.info("Loop gestoppt, alle Inputs freigegeben")
