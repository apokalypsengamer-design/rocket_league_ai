from __future__ import annotations
import sys
from config import Config
from core.game_loop import GameLoop
from core.logger import setup_logger

log = setup_logger("main")


def main() -> None:
    log.info("Rocket League AI wird gestartet …")

    cfg = Config()

    # Quick sanity output so the user can verify settings before the loop runs.
    print("\n=== Konfiguration ===")
    print(f"  FPS         : {cfg.capture.fps}")
    print(f"  Monitor     : {cfg.capture.monitor_index}")
    print(f"  Dummy-Mode  : {cfg.vision.dummy_mode}  "
          f"(False = echte Ball-Erkennung via OpenCV)")
    print(f"  Boost-Maus  : {cfg.keys.boost_via_mouse}  "
          f"({'Maus-' + cfg.keys.boost_mouse_btn if cfg.keys.boost_via_mouse else 'Taste ' + cfg.keys.boost_key})")
    print(f"  Keyboard-Map: {cfg.keys.keyboard_map()}")
    print("=====================\n")
    print("Starte in 3 Sekunden — wechsle jetzt zum Rocket-League-Fenster!")
    import time; time.sleep(3)

    loop = GameLoop(cfg)
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Abbruch durch Nutzer (Ctrl+C).")
    finally:
        loop.stop()
        log.info("Sauber beendet.")
        sys.exit(0)


if __name__ == "__main__":
    main()
