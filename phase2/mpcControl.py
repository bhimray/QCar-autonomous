from __future__ import annotations
import numpy as np
import cvxpy as cp


# ----------------------------
# 1) Nonlinear model utilities
# ----------------------------
def slip_angle(delta: float, lf: float, lr: float) -> float:
    return np.arctan((lr / (lf + lr)) * np.tan(delta))


def f_disc(x: np.ndarray, u: np.ndarray, lf: float, lr: float, Ts: float) -> np.ndarray:
    """Nonlinear discrete dynamics x_{k+1} = f(x_k, u_k)."""
    px, py, psi, v, delta_prev, a_prev = x
    delta, a = u

    beta = slip_angle(delta, lf, lr)
    px_n  = px  + Ts * v * np.cos(psi + beta)
    py_n  = py  + Ts * v * np.sin(psi + beta)
    psi_n = psi + Ts * (v / lr) * np.sin(beta)
    v_n   = v   + Ts * a

    # store previous inputs for rate constraints
    delta_prev_n = delta
    a_prev_n     = a

    return np.array([px_n, py_n, psi_n, v_n, delta_prev_n, a_prev_n], dtype=float)


def numerical_jacobian(f, x: np.ndarray, u: np.ndarray, eps: float = 1e-6):
    """
    Compute A = df/dx, B = df/du numerically via finite differences.
    f: function returning next state.
    """
    nx = x.size
    nu = u.size
    fx = f(x, u)

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        A[:, i] = (f(x + dx, u) - fx) / eps

    for j in range(nu):
        du = np.zeros(nu)
        du[j] = eps
        B[:, j] = (f(x, u + du) - fx) / eps

    return A, B, fx


# -----------------------------------------
# 2) Build and solve one QP (linearized MPC)
# -----------------------------------------
def solve_lmpc_qp(
    x0: np.ndarray,
    zref: np.ndarray,        # shape (N+1, 4) for [px, py, psi, v] references
    lf: float, lr: float, Ts: float, N: int,
    # bounds
    delta_bounds: tuple[float, float],
    a_bounds: tuple[float, float],
    delta_rate_bounds: tuple[float, float],
    a_rate_bounds: tuple[float, float],
    v_bounds: tuple[float, float] = (0.0, 50.0),
    # weights
    Q_diag: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 1.0),
    R_diag: tuple[float, float] = (1.0, 1.0),
    Rrate_diag: tuple[float, float] = (5.0, 1.0),
    # linearization nominal trajectories
    X_nom: np.ndarray | None = None,  # (N+1, 6)
    U_nom: np.ndarray | None = None,  # (N, 2)
    solver_verbose: bool = False,
):
    """
    Solve one convex QP around (X_nom, U_nom).
    Dynamics: x_{k+1} ≈ f(x_nom, u_nom) + A_k (x_k-x_nom) + B_k (u_k-u_nom)
           => x_{k+1} = A_k x_k + B_k u_k + c_k
    """
    nx = 6
    nu = 2

    if X_nom is None or U_nom is None:
        # default nominal: rollout with holding previous inputs
        X_nom = np.zeros((N + 1, nx))
        U_nom = np.zeros((N, nu))
        X_nom[0] = x0
        # nominal inputs: keep last applied (from augmented state)
        delta0 = x0[4]
        a0 = x0[5]
        for k in range(N):
            U_nom[k] = np.array([delta0, a0])
            X_nom[k + 1] = f_disc(X_nom[k], U_nom[k], lf, lr, Ts)

    # Linearize along nominal
    A_list, B_list, c_list = [], [], []
    for k in range(N):
        xk = X_nom[k]
        uk = U_nom[k]

        def f_local(x, u):
            return f_disc(x, u, lf, lr, Ts)

        A, B, fx = numerical_jacobian(f_local, xk, uk)
        # affine term: c = f(xn,un) - A*xn - B*un
        c = fx - A @ xk - B @ uk

        A_list.append(A)
        B_list.append(B)
        c_list.append(c)

    # Decision vars
    X = cp.Variable((N + 1, nx))
    U = cp.Variable((N, nu))

    # Cost weights
    Q = np.diag(Q_diag)          # for [px,py,psi,v]
    R = np.diag(R_diag)          # for [delta,a]
    Rr = np.diag(Rrate_diag)     # for [delta_rate,a_rate]

    # Constraints
    cons = []
    cons += [X[0, :] == x0]

    dmin, dmax = delta_bounds
    amin, amax = a_bounds
    drmin, drmax = delta_rate_bounds
    armin, armax = a_rate_bounds
    vmin, vmax = v_bounds

    for k in range(N):
        A = A_list[k]
        B = B_list[k]
        c = c_list[k]
        cons += [X[k + 1, :] == A @ X[k, :] + B @ U[k, :] + c]

        # input bounds
        cons += [U[k, 0] >= dmin, U[k, 0] <= dmax]
        cons += [U[k, 1] >= amin, U[k, 1] <= amax]

        # velocity bounds
        cons += [X[k, 3] >= vmin, X[k, 3] <= vmax]

        # rate constraints via augmented previous-input states:
        # delta_rate = (delta - delta_prev)/Ts = (U[k,0] - X[k,4])/Ts
        # a_rate     = (a     - a_prev)/Ts     = (U[k,1] - X[k,5])/Ts
        cons += [(U[k, 0] - X[k, 4]) / Ts >= drmin, (U[k, 0] - X[k, 4]) / Ts <= drmax]
        cons += [(U[k, 1] - X[k, 5]) / Ts >= armin, (U[k, 1] - X[k, 5]) / Ts <= armax]

    # terminal v bound
    cons += [X[N, 3] >= vmin, X[N, 3] <= vmax]

    # Objective: tracking + input + rate
    obj = 0
    for k in range(N):
        zk = X[k, 0:4]
        zref_k = zref[k, :]
        obj += cp.quad_form(zk - zref_k, Q)

        obj += cp.quad_form(U[k, :], R)

        rate_k = cp.hstack([(U[k, 0] - X[k, 4]) / Ts,
                            (U[k, 1] - X[k, 5]) / Ts])
        obj += cp.quad_form(rate_k, Rr)

    # terminal tracking
    obj += cp.quad_form(X[N, 0:4] - zref[N, :], Q)

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=solver_verbose, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP failed: status={prob.status}")

    return {
        "X_opt": X.value,
        "U_opt": U.value,
        "status": prob.status,
        "cost": prob.value,
    }


# ---------------------------------------------------
# 3) SQP loop per timestep (successive linearization)
# ---------------------------------------------------
def mpc_step_sqp(
    x0: np.ndarray,
    zref: np.ndarray,
    lf: float, lr: float, Ts: float, N: int,
    bounds: dict,
    weights: dict,
    sqp_iters: int = 3,
    solver_verbose: bool = False,
):
    """
    Run a few SQP iterations:
        start with nominal rollout
        repeat: linearize -> solve QP -> update nominal
    """
    nx, nu = 6, 2

    # initial nominal from rollout
    X_nom = np.zeros((N + 1, nx))
    U_nom = np.zeros((N, nu))
    X_nom[0] = x0
    delta0, a0 = x0[4], x0[5]
    for k in range(N):
        U_nom[k] = np.array([delta0, a0])
        X_nom[k + 1] = f_disc(X_nom[k], U_nom[k], lf, lr, Ts)

    sol = None
    for _ in range(sqp_iters):
        sol = solve_lmpc_qp(
            x0=x0, zref=zref, lf=lf, lr=lr, Ts=Ts, N=N,
            delta_bounds=bounds["delta"],
            a_bounds=bounds["a"],
            delta_rate_bounds=bounds["delta_rate"],
            a_rate_bounds=bounds["a_rate"],
            v_bounds=bounds.get("v", (0.0, 50.0)),
            Q_diag=weights["Q"],
            R_diag=weights["R"],
            Rrate_diag=weights["Rrate"],
            X_nom=X_nom,
            U_nom=U_nom,
            solver_verbose=solver_verbose,
        )
        X_nom = sol["X_opt"]
        U_nom = sol["U_opt"]

    u0 = sol["U_opt"][0]
    return u0, sol


# -----------------------
# 4) Example / template
# -----------------------
if __name__ == "__main__":
    # Vehicle params
    lf = 0.17
    lr = 0.17
    Ts = 0.1
    N = 20

    # Bounds (edit)
    bounds = {
        "delta": (-0.5, 0.5),          # rad
        "delta_rate": (-1.5, 1.5),     # rad/s
        "v_rate": (-5.0, 5.0),         # m/s^3
        "v": (0.0, 10.0),              # m/s
    }

    # Weights (edit)
    weights = {
        "Q": (30.0, 30.0, 10.0, 2.0),  # px, py, psi, v tracking
        "R": (1.0, 0.5),               # delta, a
        "Rrate": (5.0, 1.0),           # delta_rate, a_rate
    }

    # Initial state (including previous inputs)
    x = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=float)

    # Reference trajectory (straight line demo)
    zref = np.zeros((N + 1, 4))
    for k in range(N + 1):
        zref[k, 0] = x[0] + (x[3] * Ts) * k  # px increases
        zref[k, 1] = 0.0
        zref[k, 2] = 0.0
        zref[k, 3] = 2.0

    # One MPC step
    u0, info = mpc_step_sqp(
        x0=x, zref=zref, lf=lf, lr=lr, Ts=Ts, N=N,
        bounds=bounds, weights=weights,
        sqp_iters=3,
        solver_verbose=False,
    )

    print("u0 = [delta, v] =", u0)

    # Apply to nonlinear plant for next state (like in real loop)
    x_next = f_disc(x, u0, lf, lr, Ts)
    print("x_next =", x_next)
