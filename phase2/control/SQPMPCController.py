import numpy as np
from control.LinearizedMPC import LinearizedMPC


class SQPMPCController:
    """
    High-level MPC controller.
    Public interface: compute_control(x, zref)
    """

    def __init__(self, model, N, bounds, weights, sqp_iters=3):
        self.model = model
        self.N = N
        self.sqp_iters = sqp_iters

        self.mpc = LinearizedMPC(model, N, bounds, weights)

        self.nx = 6
        self.nu = 2

        self.X_nom = None
        self.U_nom = None

    def initialize_nominal(self, x0):
        self.X_nom = np.zeros((self.N + 1, self.nx))
        self.U_nom = np.zeros((self.N, self.nu))
        self.X_nom[0] = x0

        delta0, v0 = x0[4], x0[5]
        for k in range(self.N):
            self.U_nom[k] = [delta0, v0]
            self.X_nom[k+1] = self.model.step(self.X_nom[k], self.U_nom[k])

    def compute_control(self, x0, zref, uref, verbose=False):
        if self.X_nom is None:
            self.initialize_nominal(x0)

        for i in range(self.sqp_iters):
            X_opt, U_opt = self.mpc.solve(
                x0, zref, uref, self.X_nom, self.U_nom, verbose
            )
            self.X_nom = X_opt
            self.U_nom = U_opt

        return U_opt[0]   # apply only first control (receding horizon)
