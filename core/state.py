from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class BoostPadType(Enum):
    SMALL = "small"   # 12 Boost
    LARGE = "large"   # 100 Boost


@dataclass
class ObjectPosition:
    x: float = -1.0
    y: float = -1.0
    visible: bool = False
    # Für Boostpads: Typ (small/large). Für andere Objekte: None.
    pad_type: BoostPadType | None = None

    def side(self, dead_zone: float = 0.05) -> str:
        if not self.visible:
            return "unknown"
        if self.x < 0.5 - dead_zone:
            return "links"
        if self.x > 0.5 + dead_zone:
            return "rechts"
        return "mitte"

    def distance_to(self, ox: float, oy: float) -> float:
        return ((self.x - ox) ** 2 + (self.y - oy) ** 2) ** 0.5


@dataclass
class GameState:
    # ── Ball ─────────────────────────────────────────────────────────────────
    ball_x:       float = -1.0
    ball_y:       float = -1.0
    ball_visible: bool  = False
    ball_radius:  float = 0.0

    # ── Spieler (geschätzte Position) ────────────────────────────────────────
    player_x: float = 0.5
    player_y: float = 0.5

    # ── Tore ─────────────────────────────────────────────────────────────────
    own_goal:   ObjectPosition = field(default_factory=lambda: ObjectPosition(0.5, 1.0, True))
    enemy_goal: ObjectPosition = field(default_factory=lambda: ObjectPosition(0.5, 0.0, True))

    # ── Agenten ──────────────────────────────────────────────────────────────
    enemies:   list[ObjectPosition] = field(default_factory=list)
    teammates: list[ObjectPosition] = field(default_factory=list)

    # ── Boostpads ─────────────────────────────────────────────────────────────
    boost_pads:    list[ObjectPosition] = field(default_factory=list)
    nearest_boost: ObjectPosition | None = None
    # Nächster GROSSER Boost-Pad (100er)
    nearest_large_boost: ObjectPosition | None = None

    # ── Status ────────────────────────────────────────────────────────────────
    boost:   float = -1.0
    phase:   str   = "unknown"
    reasoning: str = ""

    # ── Schuss-Opportunity ────────────────────────────────────────────────────
    # True wenn Ball nah genug + Winkel zum Tor gut genug für Schuss
    shot_opportunity: bool  = False
    ball_to_goal_angle: float = 0.0   # 0.0 = perfekt aufs Tor, 1.0 = seitlich

    # ── Rohe Frames ──────────────────────────────────────────────────────────
    frame_raw:       np.ndarray | None = None
    frame_processed: np.ndarray | None = None

    # ── Properties ───────────────────────────────────────────────────────────

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
    def ball_dist_to_player(self) -> float:
        if not self.ball_visible:
            return 999.0
        return ((self.ball_x - self.player_x) ** 2 +
                (self.ball_y - self.player_y) ** 2) ** 0.5

    @property
    def ball_dist_to_enemy_goal(self) -> float:
        if not self.ball_visible or not self.enemy_goal.visible:
            return 999.0
        return ((self.ball_x - self.enemy_goal.x) ** 2 +
                (self.ball_y - self.enemy_goal.y) ** 2) ** 0.5

    @property
    def nearest_enemy(self) -> ObjectPosition | None:
        visible = [e for e in self.enemies if e.visible]
        if not visible:
            return None
        return min(visible, key=lambda e: e.distance_to(self.player_x, self.player_y))

    @property
    def enemy_between_ball_and_goal(self) -> bool:
        """Grobe Prüfung ob ein Gegner zwischen Ball und gegnerischem Tor steht."""
        if not self.ball_visible or not self.enemies:
            return False
        for e in self.enemies:
            if not e.visible:
                continue
            # Gegner liegt zwischen Ball und Tor (y-Achse)
            min_y = min(self.ball_y, self.enemy_goal.y)
            max_y = max(self.ball_y, self.enemy_goal.y)
            if min_y <= e.y <= max_y:
                # Und ungefähr auf der X-Linie
                if abs(e.x - self.ball_x) < 0.2:
                    return True
        return False

    def __repr__(self) -> str:
        boost_str = f"{self.boost:.0f}%" if self.boost >= 0 else "?"
        return (
            f"GameState(ball=({self.ball_x:.2f},{self.ball_y:.2f}) "
            f"sichtbar={self.ball_visible} boost={boost_str} "
            f"phase={self.phase} shot={self.shot_opportunity})"
        )
