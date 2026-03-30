from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    fps: int = 30
    monitor_index: int = 1
    region: dict | None = None


@dataclass
class KeyConfig:
    forward:        str = "w"
    backward:       str = "s"
    steer_left:     str = "a"
    steer_right:    str = "d"
    jump:           str = " "
    powerslide:     str = "shift_l"
    air_roll:       str = "shift_l"
    air_pitch_up:   str = "s"
    air_pitch_down: str = "w"
    air_roll_left:  str = "q"
    air_roll_right: str = "e"
    boost_mouse_button: str = "right"

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "boost_mouse_button"}


@dataclass
class VisionConfig:
    target_width:   int = 640
    target_height:  int = 360

    ball_hsv_lower: list = field(default_factory=lambda: [5, 100, 100])
    ball_hsv_upper: list = field(default_factory=lambda: [25, 255, 255])
    ball_min_area:  int  = 50

    goal_own_hsv_lower:   list = field(default_factory=lambda: [100, 80, 80])
    goal_own_hsv_upper:   list = field(default_factory=lambda: [130, 255, 255])
    goal_enemy_hsv_lower: list = field(default_factory=lambda: [0, 80, 80])
    goal_enemy_hsv_upper: list = field(default_factory=lambda: [10, 255, 255])
    goal_min_area: int = 200

    boost_pad_hsv_lower: list = field(default_factory=lambda: [20, 100, 150])
    boost_pad_hsv_upper: list = field(default_factory=lambda: [35, 255, 255])
    boost_pad_min_area:  int  = 30
    max_boost_pads:      int  = 6

    enemy_hsv_lower:   list = field(default_factory=lambda: [140, 80, 80])
    enemy_hsv_upper:   list = field(default_factory=lambda: [170, 255, 255])
    teammate_hsv_lower: list = field(default_factory=lambda: [35, 80, 80])
    teammate_hsv_upper: list = field(default_factory=lambda: [85, 255, 255])
    agent_min_area: int = 30
    max_agents: int = 3

    boost_roi:                  tuple = (0.70, 0.92, 0.25, 0.08)
    boost_brightness_threshold: int   = 180
    boost_scale_factor:         float = 4.0

    dummy_mode: bool = True


@dataclass
class GameplayConfig:
    boost_low_threshold: float = 20.0
    defense_threshold:   float = 0.60
    attack_threshold:    float = 0.40
    steer_dead_zone:     float = 0.05
    phase_history_len:   int   = 60
    phase_hysteresis:    int   = 8


@dataclass
class Config:
    capture:  CaptureConfig  = field(default_factory=CaptureConfig)
    keys:     KeyConfig      = field(default_factory=KeyConfig)
    vision:   VisionConfig   = field(default_factory=VisionConfig)
    gameplay: GameplayConfig = field(default_factory=GameplayConfig)
    log_dir:  str = "logs"
    log_level: str = "INFO"
