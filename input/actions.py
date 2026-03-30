from dataclasses import dataclass

@dataclass
class Action:
    forward:        bool = False
    backward:       bool = False
    steer_left:     bool = False
    steer_right:    bool = False
    jump:           bool = False
    boost:          bool = False
    powerslide:     bool = False
    air_roll:       bool = False
    air_pitch_up:   bool = False
    air_pitch_down: bool = False
    air_roll_left:  bool = False
    air_roll_right: bool = False

    def active_keys(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]

    def __repr__(self):
        active = self.active_keys()
        return f"Action({', '.join(active) or 'IDLE'})"
