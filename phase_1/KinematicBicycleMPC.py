import numpy as np
import cvxpy as cp
from pal.utilities.math import wrap_to_pi


class KinematicBicycleMPC:
    def __init__(self, N=15, Ts=0.05, L=0.256):
        self.N = N
        self.Ts = Ts
        self.L = L

        # constraints (tune)
        self.delta_max = np.deg2rad(25)
        self.a_max = 1.0
        self.v_max = 1.5
        self.d_delta_max = np.deg2rad(60) * Ts   # per step
        self.d_a_max = 2.0 * Ts

        # weights (tune)
        self.Qp = 20.0
        self.Qth = 5.0
        self.Qv = 2.0
        self.Ra = 0.2
        self.Rd = 0.5
        self.Rda = 1.0
        self.Rdd = 2.0

        self.u_prev = np.zeros(2)

        # CVXPY variables/params (build once, update params each call)
        nx, nu, N = 4, 2, self.N
        self.X = cp.Variable((nx, N+1))
        self.U = cp.Variable((nu, N))
        self.X0 = cp.Parameter(nx)
        self.Pref = cp.Parameter((2, N+1))   # ref positions
        self.Thref = cp.Parameter(N+1)       # ref heading
        self.Vref = cp.Parameter(N+1)        # ref speed

        # time-varying linear model params
        self.A = [cp.Parameter((nx, nx)) for _ in range(N)]
        self.B = [cp.Parameter((nx, nu)) for _ in range(N)]
        self.c = [cp.Parameter(nx) for _ in range(N)]

        cost = 0
        cons = [self.X[:, 0] == self.X0]

        for k in range(N):
            # dynamics
            cons += [self.X[:, k+1] == self.A[k] @ self.X[:, k] + self.B[k] @ self.U[:, k] + self.c[k]]

            # bounds
            cons += [
                cp.abs(self.U[1, k]) <= self.delta_max,
                cp.abs(self.U[0, k]) <= self.a_max,
                self.X[3, k] >= 0.0,
                self.X[3, k] <= self.v_max
            ]

            # rate constraints
            if k == 0:
                du = self.U[:, k] - self.u_prev
            else:
                du = self.U[:, k] - self.U[:, k-1]
            cons += [
                cp.abs(du[0]) <= self.d_a_max,
                cp.abs(du[1]) <= self.d_delta_max
            ]

            # tracking errors
            ep = self.X[0:2, k] - self.Pref[:, k]
            eth = self.X[2, k] - self.Thref[k]
            ev = self.X[3, k] - self.Vref[k]

            cost += self.Qp * cp.sum_squares(ep) + self.Qth * cp.square(eth) + self.Qv * cp.square(ev)
            cost += self.Ra * cp.square(self.U[0, k]) + self.Rd * cp.square(self.U[1, k])
            cost += self.Rda * cp.square(du[0]) + self.Rdd * cp.square(du[1])

        # terminal tracking
        epN = self.X[0:2, N] - self.Pref[:, N]
        ethN = self.X[2, N] - self.Thref[N]
        evN = self.X[3, N] - self.Vref[N]
        cost += 30.0*self.Qp*cp.sum_squares(epN) + 10.0*self.Qth*cp.square(ethN) + 10.0*self.Qv*cp.square(evN)

        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def linearize_discrete(self, x, u):
        """
        x = [px, py, th, v], u = [a, delta]
        returns (A, B, c) for x+ = A x + B u + c
        """
        Ts, L = self.Ts, self.L
        px, py, th, v = x
        a, delta = u

        ct = np.cos(th)
        st = np.sin(th)
        sec2 = 1.0 / (np.cos(delta)**2 + 1e-8)

        # Continuous jacobians
        A = np.zeros((4,4))
        B = np.zeros((4,2))

        # f = [v cos th, v sin th, v/L tan delta, a]
        A[0,2] = -v*st
        A[0,3] = ct
        A[1,2] =  v*ct
        A[1,3] = st
        A[2,3] = (1.0/L)*np.tan(delta)

        B[2,1] = (v/L)*sec2
        B[3,0] = 1.0

        # Discretize: x+ = x + Ts f(x,u)
        Ad = np.eye(4) + Ts*A
        Bd = Ts*B

        f = np.array([v*ct, v*st, (v/L)*np.tan(delta), a])
        cd = Ts*f + x - Ad@x - Bd@u  # makes equality exact at linearization point
        return Ad, Bd, cd

    def solve(self, x0, ref_pos, ref_th, ref_v):
        """
        x0: (4,)
        ref_pos: (2, N+1), ref_th: (N+1,), ref_v: (N+1,)
        """
        # set parameters
        self.X0.value = x0
        self.Pref.value = ref_pos
        self.Thref.value = ref_th
        self.Vref.value = ref_v

        # build time-varying linear model around predicted trajectory
        # simplest: linearize around (x0, u_prev) for all k
        x_lin = x0.copy()
        u_lin = self.u_prev.copy()
        for k in range(self.N):
            Ad, Bd, cd = self.linearize_discrete(x_lin, u_lin)
            self.A[k].value = Ad
            self.B[k].value = Bd
            self.c[k].value = cd

            # propagate nominal (rough)
            x_lin = Ad @ x_lin + Bd @ u_lin + cd

        # solve QP
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self.U.value is None:
            # fallback
            return 0.0, 0.0

        u0 = self.U.value[:, 0]
        self.u_prev = u0
        a_cmd, delta_cmd = float(u0[0]), float(u0[1])
        return a_cmd, delta_cmd
