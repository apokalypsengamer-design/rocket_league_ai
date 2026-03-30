from config import Config
from core.game_loop import GameLoop
from core.logger import setup_logger

log = setup_logger("main")

def main():
    log.info("Rocket League AI gestartet")
    loop = GameLoop(Config())
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Gestoppt")
        loop.stop()

if __name__ == "__main__":
    main()
