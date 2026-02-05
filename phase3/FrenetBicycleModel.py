import numpy as np
from pal.utilities.math import wrap_to_pi


class FrenetBicycleModel:
    """
    Frenet kinematic bicycle with speed actuator.
    State:  xf = [s, ey, epsi, v]
    Input:  u  = [delta, v_cmd]
    """
    def __init__(self, L: float, Ts: float, tau_v: float = 0.3):
        self.L = L
        self.Ts = Ts
        self.tau_v = tau_v

    def step(self, xf, u, kappa):
        s, ey, epsi, v = xf
        delta, v_cmd = u
        Ts = self.Ts

        denom = 1.0 - kappa * ey
        denom = max(denom, 1e-3)  # avoid singularity

        s_dot = v * np.cos(epsi) / denom
        ey_dot = v * np.sin(epsi)
        epsi_dot = (v / self.L) * np.tan(delta) - kappa * s_dot
        v_dot = (v_cmd - v) / self.tau_v

        xf_next = np.array([
            s + Ts * s_dot,
            ey + Ts * ey_dot,
            wrap_to_pi(epsi + Ts * epsi_dot),
            v + Ts * v_dot
        ], dtype=float)
        return xf_next