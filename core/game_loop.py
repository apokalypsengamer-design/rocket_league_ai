from __future__ import annotations
import time
import traceback
from config import Config
from core.logger import setup_logger, print_header
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
            f"Game-Loop gestartet | FPS={self.config.capture.fps} | "
            f"dummy_mode={self.config.vision.dummy_mode}"
        )
        print_header()

        while self.running:
            t0 = time.perf_counter()
            try:
                frame     = self.capture.grab()
                processed = preprocess(frame, self.config.vision)
                state     = self.detector.detect(processed, frame)
                action    = self.brain.decide(state)
                self.controller.execute(action)
                self.trainer.record(state, action)
            except KeyboardInterrupt:
                log.info("KeyboardInterrupt im Loop.")
                break
            except Exception as exc:
                # Fehler vollständig loggen statt still sterben
                log.error(f"FEHLER im Game-Loop:\n{traceback.format_exc()}")
                print(f"\n[FEHLER] {exc}\nSiehe logs/ für Details.\n")
                break

            self.timer.tick(t0)

    def stop(self) -> None:
        self.running = False
        self.controller.release_all()
        self.trainer.reset_episode()
        log.info("Game-Loop gestoppt.")
