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

    def linearize(self, xf_nom, u_nom, kappa_nom, eps=1e-6):
        nx = xf_nom.size
        nu = u_nom.size
        f0 = self.step(xf_nom, u_nom, kappa_nom)

        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))

        for i in range(nx):
            d = np.zeros(nx); d[i] = eps
            A[:, i] = (self.step(xf_nom + d, u_nom, kappa_nom) - f0) / eps

        for j in range(nu):
            d = np.zeros(nu); d[j] = eps
            B[:, j] = (self.step(xf_nom, u_nom + d, kappa_nom) - f0) / eps

        c = f0 - A @ xf_nom - B @ u_nom
        return A, B, c
