"""Held-key velocity/height commands for local V5 simulation playback."""

import math


class KeyboardCommand:
    MOTION_KEYS = {"W", "S", "A", "D", "Q", "E", "T", "G"}

    @staticmethod
    def event_fields(event):
        """CHAR events carry text, while KEY events carry an enum or key name."""
        kind = getattr(event.type, "name", str(event.type)).rsplit(".", 1)[-1].upper()
        if kind not in ("KEY_PRESS", "KEY_RELEASE", "KEY_REPEAT"):
            return None, None
        name = getattr(event.input, "name", event.input)
        if not isinstance(name, str):
            return None, None
        return ("SPACE" if name == " " else name.rsplit(".", 1)[-1].upper()), kind

    def __init__(self, vx_acceleration=.8, yaw_acceleration=2.5, *, height_range=(.29, .32)):
        if (not all(math.isfinite(value) and value > 0 for value in (vx_acceleration, yaw_acceleration))
                or len(height_range) != 2 or not all(math.isfinite(value) for value in height_range)
                or not 0 < height_range[0] <= .305 <= height_range[1]):
            raise ValueError("Keyboard rates must be positive and height bounds must contain 0.305 m")
        self.keys = set()
        self.height = .305
        self.vx = 0.
        self.yaw = 0.
        self.vx_limit = .5
        self.yaw_limit = 1.
        self.vx_acceleration = vx_acceleration
        self.yaw_acceleration = yaw_acceleration
        self.height_range = tuple(height_range)

    def key(self, name, pressed):
        if pressed:
            self.keys.add(name)
        else:
            self.keys.discard(name)

    def stop(self):
        self.keys.clear()
        self.vx = self.yaw = 0.

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Keyboard update duration must be positive and finite")
        vx = self.vx_limit * (int("W" in self.keys) - int("S" in self.keys))
        yaw = self.yaw_limit * (int("A" in self.keys) - int("D" in self.keys))
        self.vx += max(-self.vx_acceleration * dt, min(self.vx_acceleration * dt, vx - self.vx))
        self.yaw += max(-self.yaw_acceleration * dt, min(self.yaw_acceleration * dt, yaw - self.yaw))
        up = bool(self.keys & {"Q", "T"})
        down = bool(self.keys & {"E", "G"})
        self.height = max(self.height_range[0], min(self.height_range[1],
                          self.height + .02 * dt * (int(up) - int(down))))
        return [self.vx, self.yaw, self.height]
