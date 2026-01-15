import cvxpy as cp
import numpy as np


class LinearizedMPC:
    """
    Linearized MPC with VELOCITY input.

    State:
        x = [px, py, psi, v, delta_prev, v_prev]   (nx=6)
    Input:
        u = [delta, v]                          (nu=2)

    Model provides:
        A, B, c = model.linearize(x_nom, u_nom)
    where:
        x_{k+1} = A_k x_k + B_k u_k + c_k
    """

    def __init__(self, model, N: int, bounds: dict, weights: dict):
        self.model = model
        self.N = N
        self.bounds = bounds
        self.weights = weights

        self.nx = 6
        self.nu = 2

    def solve(self, x0, zref, uref, X_nom, U_nom, verbose=False):
        """
        Solve the linearized MPC QP.
        """
        N = self.N
        Ts = self.model.Ts
        
        x0 = np.asarray(x0).reshape(-1)
        # ---------- Linearize along nominal ----------
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = self.model.linearize(X_nom[k], U_nom[k])
            A_list.append(A)
            B_list.append(B)
            c_list.append(c)

        # ---------- Decision variables ----------
        X = cp.Variable((N + 1, self.nx))
        U = cp.Variable((N, self.nu))

        # ---------- Weights ----------
        # tracking is for [px, py, psi, v]  -> v comes from X[:,4] (v_prev state at each step)
        Q = np.diag(self.weights["Q"])       # len=4
        R = np.diag(self.weights["R"])       # len=2
        Rr = np.diag(self.weights["Rrate"])  # len=2

        # ---------- Bounds ----------
        dmin, dmax = self.bounds["delta"]
        vmin, vmax = self.bounds["v"]
        drmin, drmax = self.bounds["delta_rate"]
        vrmin, vrmax = self.bounds["v_rate"]

        # ---------- Constraints ----------
        cons = [X[0, :] == x0]

        for k in range(N):
            # dynamics
            cons += [X[k + 1, :] == A_list[k] @ X[k, :] + B_list[k] @ U[k, :] + c_list[k]]

            # input bounds
            cons += [U[k, 0] >= dmin, U[k, 0] <= dmax]
            cons += [U[k, 1] >= vmin, U[k, 1] <= vmax]

            # cons += [(U[k, 0] - X[k, 4]) / Ts >= drmin,
            #          (U[k, 0] - X[k, 4]) / Ts <= drmax]

            # cons += [(U[k, 1] - X[k, 5]) / Ts >= vrmin,
            #          (U[k, 1] - X[k, 5]) / Ts <= vrmax]

            # keep the stored v_prev within bounds too (helps feasibility)
            cons += [X[k, 3] >= vmin, X[k, 3] <= vmax]

        cons += [X[N, 3] >= vmin, X[N, 3] <= vmax]

        # ---------- Objective ----------
        cost = 0
        for k in range(N):
            # build tracking vector [px,py,psi,v]
            zk = cp.hstack([X[k, 0], X[k, 1], X[k, 2], X[k, 3]])
            # cost += cp.quad_form(zk - zref[k, :], Q)

            cost += cp.quad_form(U[k,:] - uref[k,:], R)

            rate = cp.hstack([
                (U[k,0] - X[k,4]) / Ts,
                (U[k,1] - X[k,5]) / Ts
            ])
            cost += cp.quad_form(rate, Rr)

        zN = cp.hstack([X[N, 0], X[N, 1], X[N, 2], X[N, 3]])
        cost += cp.quad_form(zN - zref[N, :], Q)

        # ---------- Solve ----------
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=verbose)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC QP failed: {prob.status}")

        return X.value, U.value
