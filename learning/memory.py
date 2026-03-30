import collections
import random
from dataclasses import dataclass
import numpy as np

@dataclass
class Experience:
    state_phase: str
    ball_x: float
    ball_y: float
    boost: float
    action_keys: list
    reward: float

class Memory:
    def __init__(self, max_size: int = 10_000):
        self._buffer: collections.deque = collections.deque(maxlen=max_size)

    def store(self, experience: Experience):
        self._buffer.append(experience)

    def sample(self, n: int) -> list[Experience]:
        n = min(n, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def __len__(self):
        return len(self._buffer)

    def is_ready(self, min_samples: int = 64) -> bool:
        return len(self._buffer) >= min_samples
