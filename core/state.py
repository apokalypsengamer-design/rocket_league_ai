from dataclasses import dataclass
import numpy as np

@dataclass
class GameState:
    ball_x: float = -1.0
    ball_y: float = -1.0
    ball_visible: bool = False
    ball_radius: float = 0.0

    player_x: float = 0.5
    player_y: float = 0.5

    boost: float = -1.0
    phase: str = "unknown"

    frame_raw: np.ndarray | None = None
    frame_processed: np.ndarray | None = None

    def __repr__(self):
        return (
            f"GameState(ball=({self.ball_x:.2f},{self.ball_y:.2f}) "
            f"visible={self.ball_visible} boost={self.boost:.0f}% "
            f"phase={self.phase})"
        )
