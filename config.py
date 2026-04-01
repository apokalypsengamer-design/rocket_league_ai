from __future__ import annotations
from dataclasses import dataclass, field

# ── Module-level constant — NOT inside the dataclass ────────────────────────
# This is the authoritative list of Action fields that map to keyboard keys.
# Keeping it here (outside KeyConfig) prevents dataclass from treating it as
# an instance field and accidentally including it in keyboard_map().
_KB_ACTION_FIELDS: tuple[str, ...] = (
    "forward", "backward", "steer_left", "steer_right",
    "jump", "powerslide",
    "air_roll", "air_pitch_up", "air_pitch_down",
    "air_roll_left", "air_roll_right",
)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    fps:           int         = 30
    monitor_index: int         = 1
    region:        dict | None = None


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

@dataclass
class KeyConfig:
    # Ground movement
    forward:     str = "w"
    backward:    str = "s"
    steer_left:  str = "a"
    steer_right: str = "d"

    # Actions
    jump:        str = "space"      # resolved to Key.space in controller
    powerslide:  str = "shift_l"

    # Aerial
    air_roll:       str = "shift_l"
    air_pitch_up:   str = "s"
    air_pitch_down: str = "w"
    air_roll_left:  str = "q"
    air_roll_right: str = "e"

    # Boost — choose ONE:
    #   boost_via_mouse=True  → right-mouse button
    #   boost_via_mouse=False → boost_key keyboard key
    boost_via_mouse: bool = False
    boost_mouse_btn: str  = "right"
    boost_key:       str  = "x"

    def keyboard_map(self) -> dict[str, str]:
        """Return {action_field: raw_key_string} for the controller to resolve."""
        result: dict[str, str] = {}
        for f in _KB_ACTION_FIELDS:      # use module-level constant, NOT self
            val = getattr(self, f, "")
            if val:
                result[f] = val
        if not self.boost_via_mouse and self.boost_key:
            result["boost"] = self.boost_key
        return result


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    target_width:  int = 640
    target_height: int = 360

    # Ball — orange in Rocket League
    ball_hsv_lower: list = field(default_factory=lambda: [5,  150, 150])
    ball_hsv_upper: list = field(default_factory=lambda: [20, 255, 255])
    ball_min_area:  int  = 40

    # Goals
    goal_own_hsv_lower:   list = field(default_factory=lambda: [100, 80, 80])
    goal_own_hsv_upper:   list = field(default_factory=lambda: [130, 255, 255])
    goal_enemy_hsv_lower: list = field(default_factory=lambda: [0,   80, 80])
    goal_enemy_hsv_upper: list = field(default_factory=lambda: [10,  255, 255])
    goal_min_area: int = 200

    # Boost pads — yellow
    boost_pad_hsv_lower: list = field(default_factory=lambda: [20, 100, 150])
    boost_pad_hsv_upper: list = field(default_factory=lambda: [35, 255, 255])
    boost_pad_min_area:  int  = 30
    max_boost_pads:      int  = 6

    # Agents
    enemy_hsv_lower:    list = field(default_factory=lambda: [140, 80, 80])
    enemy_hsv_upper:    list = field(default_factory=lambda: [170, 255, 255])
    teammate_hsv_lower: list = field(default_factory=lambda: [35,  80, 80])
    teammate_hsv_upper: list = field(default_factory=lambda: [85,  255, 255])
    agent_min_area: int = 30
    max_agents:     int = 3

    # Boost bar (screen ROI, normalised coords)
    boost_roi:                  tuple = (0.70, 0.92, 0.25, 0.08)
    boost_brightness_threshold: int   = 180
    boost_scale_factor:         float = 4.0

    # Dummy mode: True = use hard-coded values (no OpenCV needed)
    dummy_mode:   bool  = True
    dummy_ball_x: float = 0.65
    dummy_ball_y: float = 0.35
    dummy_boost:  float = 75.0


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------

@dataclass
class GameplayConfig:
    boost_low_threshold: float = 20.0
    defense_threshold:   float = 0.60
    attack_threshold:    float = 0.40
    steer_dead_zone:     float = 0.08   # wider → less steering jitter
    phase_history_len:   int   = 60
    phase_hysteresis:    int   = 6


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    capture:   CaptureConfig  = field(default_factory=CaptureConfig)
    keys:      KeyConfig      = field(default_factory=KeyConfig)
    vision:    VisionConfig   = field(default_factory=VisionConfig)
    gameplay:  GameplayConfig = field(default_factory=GameplayConfig)
    log_dir:   str = "logs"
    log_level: str = "INFO"
