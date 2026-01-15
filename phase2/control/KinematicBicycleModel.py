import numpy as np

class KinematicBicycleModel:
    """
    Velocity-input kinematic bicycle with augmented prev-input states.

    State:  x = [px, py, psi, delta_prev, v_prev]
    Input:  u = [delta, v]
    """
    def __init__(self, lf: float, lr: float, Ts: float):
        self.lf = lf
        self.lr = lr
        self.Ts = Ts

    def _beta(self, delta: float) -> float:
        return np.arctan((self.lr / (self.lf + self.lr)) * np.tan(delta))

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        px, py, psi, v, delta_prev, v_prev = x
        delta, v_cmd = u

        beta = self._beta(delta) # compute slip angle

        px_n  = px  + self.Ts * v * np.cos(psi + beta)
        py_n  = py  + self.Ts * v * np.sin(psi + beta)
        psi_n = psi + self.Ts * (v / self.lr) * np.sin(beta)

        v_n = v_cmd  # 🔥 velocity is now DIRECT input

        return np.array([
            px_n,
            py_n,
            psi_n,
            v_n,
            delta,
            v_cmd
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
            dx = np.zeros(nx); dx[i] = eps
            A[:, i] = (self.step(x_nom + dx, u_nom) - f0) / eps

        for j in range(nu):
            du = np.zeros(nu); du[j] = eps
            B[:, j] = (self.step(x_nom, u_nom + du) - f0) / eps

        c = f0 - A @ x_nom - B @ u_nom
        return A, B, c
