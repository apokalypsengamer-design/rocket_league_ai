from __future__ import annotations
from core.state import GameState
from input.actions import Action
from learning.memory import Memory, Experience
from learning.reward import RewardCalculator
from core.logger import setup_logger

log = setup_logger("trainer")


class Trainer:
    def __init__(self, batch_size: int = 64):
        self.memory          = Memory()
        self.reward_calc     = RewardCalculator()
        self.batch_size      = batch_size
        self._episode_reward = 0.0
        self._step           = 0

    def record(self, state: GameState, action: Action) -> None:
        reward = self.reward_calc.calculate(state)
        self._episode_reward += reward
        self._step += 1

        pad = state.nearest_boost
        exp = Experience(
            state_phase=state.phase,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            boost=state.boost,
            player_x=state.player_x,
            player_y=state.player_y,
            ball_visible=state.ball_visible,
            enemy_count=len(state.enemies),
            teammate_count=len(state.teammates),
            nearest_pad_x=pad.x if pad else -1.0,
            nearest_pad_y=pad.y if pad else -1.0,
            action_keys=action.active_keys(),
            reward=reward,
            reasoning=state.reasoning,
        )
        self.memory.store(exp)

        if self._step % 300 == 0:
            log.info(
                f"Step {self._step} | Reward: {self._episode_reward:.2f} "
                f"| Memory: {len(self.memory)} | Phase: {state.phase}"
            )

    def train_step(self) -> None:
        if not self.memory.is_ready(self.batch_size):
            return
        batch = self.memory.sample(self.batch_size)
        avg_reward = sum(e.reward for e in batch) / len(batch)
        log.info(f"Train-Step | Batch-Avg-Reward: {avg_reward:.4f}")

    def reset_episode(self) -> None:
        log.info(f"Episode beendet | Gesamt-Reward: {self._episode_reward:.2f} | Steps: {self._step}")
        self._episode_reward = 0.0
