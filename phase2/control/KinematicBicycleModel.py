"""
Kinematic bicycle (discrete-time) + MPC (paper-style: tracking + input + input-rate penalty)

State (augmented to handle input-rate constraints cleanly):
    X = [px, py, psi, v, delta_prev, a_prev]
Control:
    U = [delta, a]

Discrete dynamics (forward Euler for kinematic states; exact shift for prev-input states):
    beta_k = atan( (lr/(lf+lr)) * tan(delta_k) )
    px_{k+1}   = px_k   + Ts * v_k * cos(psi_k + beta_k)
    py_{k+1}   = py_k   + Ts * v_k * sin(psi_k + beta_k)
    psi_{k+1}  = psi_k  + Ts * (v_k/lr) * sin(beta_k)
    v_{k+1}    = v_k    + Ts * a_k
    delta_prev_{k+1} = delta_k
    a_prev_{k+1}     = a_k

MPC formulation (matches the structure you wrote from the paper):
    min Σ ||z_k - z_ref_k||_Q^2 + Σ ||u_k||_R^2 + Σ ||u_k - u_{k-1}||_{Rbar}^2
    s.t. z_{k+1} = f(z_k, u_k),
         u_min ≤ u_k ≤ u_max,
         du_min ≤ (u_k - u_{k-1})/Ts ≤ du_max

Implementation details:
- We enforce rate constraints via nonlinear path constraints on:
      delta_rate = (delta - delta_prev)/Ts
      a_rate     = (a     - a_prev)/Ts
- We also include rate penalties in the cost using the same expressions.
"""

import numpy as np

class KinematicBicycleModel:
    """
    Discrete-time kinematic bicycle model with Euler integration.
    State: [px, py, psi, v, delta_prev, a_prev]
    Input: [delta, a]
    """

    def __init__(self, lf: float, lr: float, Ts: float):
        self.lf = lf
        self.lr = lr
        self.Ts = Ts

    def slip_angle(self, delta: float) -> float:
        return np.arctan((self.lr / (self.lf + self.lr)) * np.tan(delta))

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        px, py, psi, v, delta_prev, a_prev = x
        delta, a = u

        beta = self.slip_angle(delta)

        px_n  = px  + self.Ts * v * np.cos(psi + beta)
        py_n  = py  + self.Ts * v * np.sin(psi + beta)
        psi_n = psi + self.Ts * (v / self.lr) * np.sin(beta)
        v_n   = v   + self.Ts * a

        return np.array([
            px_n, py_n, psi_n, v_n,
            delta, a
        ], dtype=float)

    def linearize(self, x_nom: np.ndarray, u_nom: np.ndarray, eps: float = 1e-6):
        """
        Numerical Jacobian:
            x+ ≈ A x + B u + c
        """
        nx = x_nom.size
        nu = u_nom.size

        f0 = self.step(x_nom, u_nom)

        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))

        for i in range(nx):
            dx = np.zeros(nx)
            dx[i] = eps
            A[:, i] = (self.step(x_nom + dx, u_nom) - f0) / eps

        for j in range(nu):
            du = np.zeros(nu)
            du[j] = eps
            B[:, j] = (self.step(x_nom, u_nom + du) - f0) / eps

        c = f0 - A @ x_nom - B @ u_nom
        return A, B, c
