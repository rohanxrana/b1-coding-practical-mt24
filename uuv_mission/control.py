"""Feedback controllers for the UUV depth-tracking mission."""


class PDController:
    """Discrete-time PD controller: u[t] = KP*e[t] + KD*(e[t] - e[t-1]), with e[-1] = 0."""

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def reset(self) -> None:
        """Clear the stored error so a new simulation starts from e[-1] = 0."""
        self.prev_error = 0.0

    def __call__(self, reference: float, observation: float) -> float:
        """Return the control action for the current reference and observed depth."""
        error = reference - observation
        derivative = error - self.prev_error  # dt = 1 in Submarine
        self.prev_error = error
        return self.kp * error + self.kd * derivative
