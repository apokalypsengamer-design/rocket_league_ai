"""
test_inputs.py
──────────────
Testet den Controller isoliert ohne Screenshot oder Vision.
Führt nacheinander alle wichtigen Aktionen aus und zeigt was gesendet wird.

Ausführen:
    python test_inputs.py

Du wirst 5 Sekunden Zeit haben, in das RL-Fenster zu wechseln.
Dann werden Aktionen mit kurzer Pause ausgeführt.
"""
from __future__ import annotations
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from input.actions import Action
from input.controller import Controller
from core.logger import setup_logger

log = setup_logger("test_inputs", level="DEBUG")


def test_action(ctrl: Controller, action: Action, label: str, hold_sec: float = 0.5) -> None:
    active = "+".join(action.active_keys()) or "IDLE"
    print(f"\n  [{label}]  Keys: {active}")
    ctrl.execute(action)
    time.sleep(hold_sec)
    ctrl.execute(Action())   # release everything
    time.sleep(0.15)


def main() -> None:
    print("\n=== Controller-Test ===")
    print("Wechsle in 5 Sekunden zum Rocket-League-Fenster …")
    time.sleep(5)

    cfg  = Config()
    ctrl = Controller(cfg)

    print("\nStarte Tests:")

    test_action(ctrl, Action(forward=True),                       "Vorwärts",           hold_sec=1.0)
    test_action(ctrl, Action(backward=True),                      "Rückwärts",          hold_sec=0.5)
    test_action(ctrl, Action(forward=True, steer_left=True),      "Links lenken",       hold_sec=0.8)
    test_action(ctrl, Action(forward=True, steer_right=True),     "Rechts lenken",      hold_sec=0.8)
    test_action(ctrl, Action(forward=True, boost=True),           "Vorwärts + Boost",   hold_sec=0.8)
    test_action(ctrl, Action(forward=True, jump=True),            "Vorwärts + Sprung",  hold_sec=0.3)
    test_action(ctrl, Action(forward=True, boost=True,
                              steer_right=True),                  "Angriff rechts",     hold_sec=1.0)

    ctrl.release_all()
    print("\n✓ Alle Tests abgeschlossen.")


if __name__ == "__main__":
    main()
