from config import Config
from core.state import GameState
from input.actions import Action
from decision.rules import RuleEngine
from decision.strategy import Strategy

class Brain:
    def __init__(self, config: Config):
        self.config = config
        self.rules = RuleEngine(config)
        self.strategy = Strategy(config)
        self._frame = 0

    def decide(self, state: GameState) -> Action:
        self._frame += 1
        state.phase = self.strategy.refine_phase(state, self._frame)
        return self.rules.evaluate(state, self._frame)
