from __future__ import annotations
import collections
import random
from dataclasses import dataclass, field


@dataclass
class Experience:
    state_phase:    str
    ball_x:         float
    ball_y:         float
    boost:          float
    player_x:       float
    player_y:       float
    ball_visible:   bool
    enemy_count:    int
    teammate_count: int
    nearest_pad_x:  float
    nearest_pad_y:  float
    action_keys:    list[str]
    reward:         float
    reasoning:      str = ""


class Memory:
    def __init__(self, max_size: int = 10_000):
        self._buffer: collections.deque[Experience] = collections.deque(maxlen=max_size)

    def store(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def sample(self, n: int) -> list[Experience]:
        n = min(n, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_samples: int = 64) -> bool:
        return len(self._buffer) >= min_samples
