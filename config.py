from dataclasses import dataclass, field
from pynput.keyboard import Key
 
@dataclass
class Config:
    fps: int = 30
    monitor_index: int = 1
    region: dict | None = None
 
    keys: dict = field(default_factory=lambda: {
        "forward":        "w",
        "backward":       "s",
        "steer_left":     "a",
        "steer_right":    "d",
        "jump":           " ",
        "powerslide":     Key.shift_l,
        "air_roll":       Key.shift_l,
        "air_pitch_up":   "s",
        "air_pitch_down": "w",
        "air_roll_left":  "q",
        "air_roll_right": "e",
    })
 
    mouse_buttons: dict = field(default_factory=lambda: {
        "boost": "right",
    })
 
    boost_low_threshold: float = 20.0
    ball_hsv_lower: list = field(default_factory=lambda: [5, 100, 100])
    ball_hsv_upper: list = field(default_factory=lambda: [25, 255, 255])
    ball_min_area: int = 50
    defense_threshold: float = 0.6
    attack_threshold: float = 0.4
    log_dir: str = "logs"
    log_level: str = "INFO"
