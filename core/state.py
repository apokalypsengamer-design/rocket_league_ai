from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ObjectPosition:
    x: float = -1.0
    y: float = -1.0
    visible: bool = False

    def side(self, dead_zone: float = 0.05) -> str:
        if not self.visible:
            return "unknown"
        if self.x < 0.5 - dead_zone:
            return "links"
        if self.x > 0.5 + dead_zone:
            return "rechts"
        return "mitte"


@dataclass
class GameState:
    ball_x: float = -1.0
    ball_y: float = -1.0
    ball_visible: bool = False
    ball_radius: float = 0.0

    player_x: float = 0.5
    player_y: float = 0.5

    own_goal:   ObjectPosition = field(default_factory=lambda: ObjectPosition(0.5, 1.0, True))
    enemy_goal: ObjectPosition = field(default_factory=lambda: ObjectPosition(0.5, 0.0, True))

    enemies:   list[ObjectPosition] = field(default_factory=list)
    teammates: list[ObjectPosition] = field(default_factory=list)

    boost_pads: list[ObjectPosition] = field(default_factory=list)
    nearest_boost: ObjectPosition | None = None

    boost: float = -1.0
    phase: str = "unknown"
    reasoning: str = ""

    frame_raw: np.ndarray | None = None
    frame_processed: np.ndarray | None = None

    @property
    def ball_side(self) -> str:
        if not self.ball_visible:
            return "unsichtbar"
        if self.ball_x < 0.45:
            return "links"
        if self.ball_x > 0.55:
            return "rechts"
        return "mitte"

    @property
    def nearest_enemy(self) -> ObjectPosition | None:
        visible = [e for e in self.enemies if e.visible]
        if not visible:
            return None
        return min(visible, key=lambda e: abs(e.x - self.player_x) + abs(e.y - self.player_y))

    def __repr__(self) -> str:
        boost_str = f"{self.boost:.0f}%" if self.boost >= 0 else "?"
        return (
            f"GameState(ball=({self.ball_x:.2f},{self.ball_y:.2f}) "
            f"sichtbar={self.ball_visible} boost={boost_str} phase={self.phase})"
        )
