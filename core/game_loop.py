from __future__ import annotations
import time
from config import Config
from core.logger import setup_logger
from vision.screen_capture import ScreenCapture
from vision.preprocessing import preprocess
from vision.detection import Detector
from decision.brain import Brain
from input.controller import Controller
from learning.trainer import Trainer
from utils.timing import FrameTimer

log = setup_logger("game_loop")


class GameLoop:
    def __init__(self, config: Config):
        self.config     = config
        self.capture    = ScreenCapture(config)
        self.detector   = Detector(config)
        self.brain      = Brain(config)
        self.controller = Controller(config)
        self.trainer    = Trainer()
        self.timer      = FrameTimer(config.capture.fps)
        self.running    = False

    def run(self) -> None:
        self.running = True
        log.info(
            f"Loop gestartet | FPS={self.config.capture.fps} "
            f"dummy_mode={self.config.vision.dummy_mode}"
        )
        print(
            f"\n{'─'*180}\n"
            f"{'Frame':>7} | {'Ball':^20} | {'Gegner/Teammates':^22} | "
            f"{'Pads':^6} | {'Boost':^7} | {'Phase':^15} | "
            f"{'Action':<30} | Grund\n"
            f"{'─'*180}"
        )

        while self.running:
            t0 = time.perf_counter()

            frame     = self.capture.grab()
            processed = preprocess(frame, self.config.vision)
            state     = self.detector.detect(processed, frame)
            action    = self.brain.decide(state)
            self.controller.execute(action)
            self.trainer.record(state, action)

            self.timer.tick(t0)

    def stop(self) -> None:
        self.running = False
        self.controller.release_all()
        self.trainer.reset_episode()
        log.info("Loop gestoppt.")
