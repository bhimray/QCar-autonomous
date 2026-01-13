import cvxpy as cp
import numpy as np

from control.KinematicBicycleModel import KinematicBicycleModel

class LinearizedMPC:
    """
    Solves ONE convex QP given linearized dynamics.
    """

    def __init__(self, model: KinematicBicycleModel, N: int,
                 bounds: dict, weights: dict):
        self.model = model
        self.N = N
        self.bounds = bounds
        self.weights = weights

        self.nx = 6
        self.nu = 2

    def solve(self, x0, zref, X_nom, U_nom, verbose=False):
        N = self.N
        Ts = self.model.Ts

        # Linearize along nominal
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = self.model.linearize(X_nom[k], U_nom[k])
            A_list.append(A)
            B_list.append(B)
            c_list.append(c)

        # Decision variables
        X = cp.Variable((N + 1, self.nx))
        U = cp.Variable((N, self.nu))

        # Weights
        Q = np.diag(self.weights["Q"])
        R = np.diag(self.weights["R"])
        Rr = np.diag(self.weights["Rrate"])

        # Constraints
        cons = [X[0] == x0]

        dmin, dmax = self.bounds["delta"]
        amin, amax = self.bounds["a"]
        drmin, drmax = self.bounds["delta_rate"]
        armin, armax = self.bounds["a_rate"]
        vmin, vmax = self.bounds["v"]

        for k in range(N):
            cons += [X[k+1] == A_list[k] @ X[k] + B_list[k] @ U[k] + c_list[k]]

            cons += [dmin <= U[k,0], U[k,0] <= dmax]
            cons += [amin <= U[k,1], U[k,1] <= amax]

            # cons += [vmin <= X[k,3], X[k,3] <= vmax]

            cons += [(U[k,0] - X[k,4]) / Ts >= drmin,
                     (U[k,0] - X[k,4]) / Ts <= drmax]

            cons += [(U[k,1] - X[k,5]) / Ts >= armin,
                     (U[k,1] - X[k,5]) / Ts <= armax]

        # Objective
        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[k,0:4] - zref[k], Q)
            cost += cp.quad_form(U[k], R)

            rate = cp.hstack([
                (U[k,0] - X[k,4]) / Ts,
                (U[k,1] - X[k,5]) / Ts
            ])
            cost += cp.quad_form(rate, Rr)

        cost += cp.quad_form(X[N,0:4] - zref[N], Q)

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=verbose)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC QP failed: {prob.status}")

        return X.value, U.value
