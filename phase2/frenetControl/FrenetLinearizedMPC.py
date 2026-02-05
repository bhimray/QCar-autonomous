import cvxpy as cp
import numpy as np
from .FrenetBicycleModel import FrenetBicycleModel
import control as ct

class FrenetLinearizedMPC:
    """
    State:  xf = [s, ey, epsi, v]   (nx=4)
    Input:  u  = [delta, v_cmd]     (nu=2)
    """
    def __init__(self, model: FrenetBicycleModel, N: int, bounds: dict, weights: dict):
        self.model = model
        self.N = N
        self.bounds = bounds
        self.weights = weights
        self.nx = 4
        self.nu = 2

    def solve(self, x0, xref, X_nom, U_nom, kappa_seq, verbose=False):
        """
        x0      : (4,) current frenet state
        xref    : (N+1,4) reference [s_ref, 0, 0, v_ref]
        uref    : (N,2)   reference [delta_ref, v_ref] (optional; can be zeros)
        X_nom   : (N+1,4) nominal
        U_nom   : (N,2) nominal
        kappa_seq: (N,) curvature at nominal s
        """
        N = self.N
        Ts = self.model.Ts

        # linearize along nominal
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = self.model.linearize(X_nom[k], U_nom[k], kappa_seq[k])
            A_list.append(A); B_list.append(B); c_list.append(c)
            ctrb_rank = np.linalg.matrix_rank(ct.ctrb(A, B))
            is_ctrl = (ctrb_rank == A.shape[0])
            print("is controllable at step", k, ":", is_ctrl)
        X = cp.Variable((N+1, self.nx)) 
        U = cp.Variable((N, self.nu))
        sigma = cp.Variable(N+1, nonneg=True)
        rho = 1e5  # large penalty
        
        Q  = np.diag(self.weights["Q"])    # nx
        R  = np.diag(self.weights["R"])    # nu
        S  = np.diag(self.weights["Sdu"])  # nu (delta-u smoothness)

        dmin, dmax = self.bounds["delta"]
        vmin, vmax = self.bounds["v_cmd"]
        eymin, eymax = self.bounds["ey"]

        cons = [X[0, :] == x0]

        for k in range(N):
            cons += [X[k+1, :] == A_list[k] @ X[k, :] + B_list[k] @ U[k, :] + c_list[k]]
            cons += [U[k, 0] >= dmin, U[k, 0] <= dmax]
            cons += [U[k, 1] >= vmin, U[k, 1] <= vmax]
            cons += [X[k,1] <= eymax + sigma[k]]
            cons += [X[k,1] >= eymin - sigma[k]]
        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[k, :] - xref[k, :], Q)
            cost += cp.quad_form(U[k, :], R) #not tracking uref for now

            if k > 0:
                du = U[k, :] - U[k-1, :]
                cost += cp.quad_form(du, S)

        cost += cp.quad_form(X[N, :] - xref[N, :], Q)
        cost += rho * cp.sum_squares(sigma)

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=verbose)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Frenet MPC failed: {prob.status}")

        return X.value, U.value
