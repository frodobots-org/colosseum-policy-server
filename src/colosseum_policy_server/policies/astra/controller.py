from dataclasses import asdict, dataclass
import math

from .planner import Decision, Move


@dataclass(frozen=True)
class ControllerConfig:
    linear: float = 0.15
    angular: float = 0.25
    metres_per_command_second: float = 1.538
    turn_tolerance_deg: float = 8.0
    move_timeout_s: float = 20.0
    heading_sign: int = -1

    def __post_init__(self):
        values = asdict(self)
        values.pop('heading_sign')
        if self.heading_sign not in {-1, 1} or any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Controller settings must be positive finite numbers")
        if self.linear > 1 or self.angular > 1:
            raise ValueError("Commands must be within [-1, 1]")


class Controller:
    def __init__(self, config=ControllerConfig()):
        self.config = config
        self.moves = []
        self.done = False
        self.error = ""
        self.started = None
        self.previous_time = None
        self.previous_heading = None
        self.travelled = 0.0
        self.turned = 0.0
        self.history = []

    def load(self, decision: Decision):
        self.moves = list(decision.plan)
        if self.moves and self.moves[0].action == "look_around":
            self.moves = [Move("left", 30)]
        self.done = decision.done
        self.error = ""
        self._reset_move()

    def _reset_move(self):
        self.started = self.previous_time = self.previous_heading = None
        self.travelled = self.turned = 0.0

    def _complete(self, move):
        self.history.append(move.label())
        self.history = self.history[-12:]
        self.moves.pop(0)
        self._reset_move()
        return (0.0, 0.0)

    def step(self, now, telemetry, *, rear_available=False):
        if self.done or self.error or not self.moves:
            return (0.0, 0.0)
        move = self.moves[0]
        if move.action in {"stop", "observe"}:
            return self._complete(move)
        if move.action == "backward" and not rear_available:
            self.error = "Reverse requires a fresh rear camera"
            return (0.0, 0.0)
        heading = telemetry.get("orientation")
        if heading is not None and not math.isfinite(heading):
            heading = None
        if self.started is None:
            self.started = self.previous_time = now
            self.previous_heading = heading
        elapsed = max(0.0, min(now - self.previous_time, 0.25))
        self.previous_time = now
        if now - self.started > self.config.move_timeout_s:
            self.error = "Motion feedback timed out"
            return (0.0, 0.0)
        if move.action in {"left", "right"}:
            if heading is None or self.previous_heading is None:
                self.error = "Turn requires fresh heading feedback"
                return (0.0, 0.0)
            direction = 1 if move.action == "left" else -1
            self.turned += ((heading - self.previous_heading + 180) % 360 - 180) * direction * self.config.heading_sign
            self.previous_heading = heading
            if self.turned >= move.amount - self.config.turn_tolerance_deg:
                return self._complete(move)
            return (0.0, direction * self.config.angular)
        speed = telemetry.get("speed")
        if speed is None or not math.isfinite(speed):
            speed = self.config.linear * self.config.metres_per_command_second
        self.travelled += abs(speed) * elapsed
        if self.travelled >= move.amount:
            return self._complete(move)
        angular = {"forward_left": self.config.angular, "forward_right": -self.config.angular}.get(move.action, 0.0)
        return (self.config.linear * (-1 if move.action == "backward" else 1), angular)
