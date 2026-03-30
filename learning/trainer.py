from core.state import GameState
from input.actions import Action
from learning.memory import Memory, Experience
from learning.reward import RewardCalculator
from core.logger import setup_logger

log = setup_logger("trainer")

class Trainer:
    def __init__(self, batch_size: int = 64):
        self.memory = Memory()
        self.reward_calc = RewardCalculator()
        self.batch_size = batch_size
        self._episode_reward = 0.0
        self._step = 0

    def record(self, state: GameState, action: Action):
        reward = self.reward_calc.calculate(state)
        self._episode_reward += reward
        self._step += 1

        exp = Experience(
            state_phase=state.phase,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            boost=state.boost,
            action_keys=action.active_keys(),
            reward=reward,
        )
        self.memory.store(exp)

        if self._step % 300 == 0:
            log.info(f"Step {self._step} | Episode-Reward: {self._episode_reward:.2f} | Memory: {len(self.memory)}")

    def train_step(self):
        if not self.memory.is_ready(self.batch_size):
            return

        batch = self.memory.sample(self.batch_size)
        avg_reward = sum(e.reward for e in batch) / len(batch)
        log.info(f"Train-Step | Batch-Avg-Reward: {avg_reward:.4f}")

    def reset_episode(self):
        log.info(f"Episode beendet | Gesamt-Reward: {self._episode_reward:.2f}")
        self._episode_reward = 0.0
