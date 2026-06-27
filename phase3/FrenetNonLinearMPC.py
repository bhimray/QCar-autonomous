import numpy as np
import casadi as ca
from pal.utilities.math import wrap_to_pi

class FrenetNonlinearMPC:
    """
    Direct NMPC in Frenet coordinates solved with IPOPT (CasADi).
    States:  [s, ey, epsi, v]
    Inputs:  [delta, a]
    """
    def __init__(
        self,
        path,              # must provide kappa_at_s(s) callable
        Ts=0.02,
        N=20,
        L=0.2,
        # constraints:
        ey_max=0.25,        # lane half-width / lateral constraint
        delta_max=0.52,     # steering angle limit (rad)
        ddelta_max=2.0,     # steering rate limit (rad/s)
        a_min=-1.5, a_max=1.5,
        v_min=0.0, v_max=1.0,
        # weights:
        w_ey=50.0, w_epsi=20.0, w_v=5.0,
        w_delta=1.0, w_a=0.5,
        w_ddelta=10.0, w_da=1.0,
        w_ey_slack=1e4,
        w_ey_slack_lin=100.0,
        # optional curvature speed shaping:
        use_curv_speed_ref=True,
        v_ref_base=0.8,
        kappa_speed_gain=2.0,
    ):
        self.path = path
        self.Ts = Ts
        self.N = N
        self.L = L

        self.ey_max = float(ey_max)
        self.delta_max = float(delta_max)
        self.ddelta_max = float(ddelta_max)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.w_ey = w_ey
        self.w_epsi = w_epsi
        self.w_v = w_v
        self.w_delta = w_delta
        self.w_a = w_a
        self.w_ddelta = w_ddelta
        self.w_da = w_da
        self.w_ey_slack = float(w_ey_slack)
        self.w_ey_slack_lin = float(w_ey_slack_lin)

        self.use_curv_speed_ref = use_curv_speed_ref
        self.v_ref_base = float(v_ref_base)
        self.kappa_speed_gain = float(kappa_speed_gain)

        # build solver once
        self._build_solver()

        # warm start memory
        self.last_sol = None


    def _build_solver(self):
        N = self.N
        Ts = self.Ts
        L = self.L

        # Decision variables
        X = ca.SX.sym("X", 4, N+1)     # states [s, ey, epsi, v]
        U = ca.SX.sym("U", 2, N)       # inputs [delta, a]
        S_ey = ca.SX.sym("S_ey", N+1)  # slack for |ey| <= ey_max + S_ey

        # Parameters
        x0 = ca.SX.sym("x0", 4)              # initial state
        kappa_seq = ca.SX.sym("kappa", N)    # curvature along horizon (precomputed)
        v_ref_param = ca.SX.sym("vref", N+1) # speed reference along horizon (precomputed)

        # dynamics function
        def f(x, u, kappa):
            s, ey, epsi, v = x[0], x[1], x[2], x[3]
            delta, a = u[0], u[1]

            denom = (1 - kappa * ey)
            # avoid division blow-ups numerically (still keep it smooth)
            denom = ca.fmax(denom, 1e-3)

            sdot = v * ca.cos(epsi) / denom
            eydot = v * ca.sin(epsi)
            epsidot = (v / L) * ca.tan(delta) - kappa * sdot
            vdot = a
            return ca.vertcat(sdot, eydot, epsidot, vdot)

        def rk4_step(x, u, kappa):
            k1 = f(x, u, kappa)
            k2 = f(x + 0.5 * Ts * k1, u, kappa)
            k3 = f(x + 0.5 * Ts * k2, u, kappa)
            k4 = f(x + Ts * k3, u, kappa)
            return x + (Ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # constraints list
        g = []
        lbg = []
        ubg = []

        # running cost
        J = 0

        # initial state hard constraint
        g.append(X[:,0] - x0)
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        # Add dynamics constraints
        for k in range(N):
            kappa_k = kappa_seq[k]
            xk = X[:, k]
            uk = U[:, k]
            xnext = X[:, k+1]
            xnext_pred = rk4_step(xk, uk, kappa_k)

            g.append(xnext - xnext_pred)

            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

            # tracking cost (stay on centerline and align)
            ey_k = xk[1]
            epsi_k = xk[2]
            v_k = xk[3]

            v_ref_k = v_ref_param[k]

            J += self.w_ey * (ey_k**2) + self.w_epsi * (epsi_k**2) + self.w_v * ((v_k - v_ref_k)**2)
            J += self.w_delta * (uk[0]**2) + self.w_a * (uk[1]**2)
            J += self.w_ey_slack * (S_ey[k]**2) + self.w_ey_slack_lin * S_ey[k]

            # soft lateral bounds: |ey| <= ey_max + S_ey[k]
            g.append(ey_k - S_ey[k])
            lbg.append(-self.ey_max)
            ubg.append(self.ey_max)
            g.append(-ey_k - S_ey[k])
            lbg.append(-self.ey_max)
            ubg.append(self.ey_max)

            # smoothness penalties (rate)
            if k > 0:
                ddelta = (U[0, k] - U[0, k-1]) / Ts
                da = (U[1, k] - U[1, k-1]) / Ts
                J += self.w_ddelta * (ddelta**2) + self.w_da * (da**2)

                # optional hard rate constraints too
                g.append(ddelta)
                lbg.append(-self.ddelta_max)
                ubg.append(+self.ddelta_max)

        # terminal cost
        eyN = X[1, N]
        epsiN = X[2, N]
        vN = X[3, N]
        v_ref_N = v_ref_param[N]
        J += self.w_ey*(eyN**2) + self.w_epsi*(epsiN**2) + self.w_v*((vN - v_ref_N)**2)
        J += self.w_ey_slack * (S_ey[N]**2) + self.w_ey_slack_lin * S_ey[N]

        g.append(eyN - S_ey[N])
        lbg.append(-np.inf)
        ubg.append(self.ey_max)
        g.append(-eyN - S_ey[N])
        lbg.append(-np.inf)
        ubg.append(self.ey_max)

        # pack variables
        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), ca.reshape(S_ey, -1, 1))
        p = ca.vertcat(x0, kappa_seq, v_ref_param)

        nlp = {"x": z, "f": J, "g": ca.vertcat(*g), "p": p}

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 20,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.linear_solver": "mumps",  # default; keep
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # store sizes and index helpers
        self.nx = 4
        self.nu = 2
        self.nX = self.nx*(self.N+1)
        self.nU = self.nu*self.N
        self.nS = self.N + 1
        self.nz = self.nX + self.nU + self.nS

        # variable bounds
        lbz = -np.inf*np.ones(self.nz)
        ubz = +np.inf*np.ones(self.nz)

        # X bounds by components
        # X layout: [X(:,0), X(:,1), ..., X(:,N)] stacked column-major
        def idx_x(k, i):  # state index i at time k
            return k*self.nx + i

        for k in range(self.N+1):
            # v bounds
            lbz[idx_x(k, 3)] = self.v_min
            ubz[idx_x(k, 3)] = self.v_max

        # U bounds
        offsetU = self.nX
        def idx_u(k, j):  # input j at time k
            return offsetU + k*self.nu + j

        for k in range(self.N):
            # delta bounds
            lbz[idx_u(k, 0)] = -self.delta_max
            ubz[idx_u(k, 0)] = +self.delta_max
            # accel bounds
            lbz[idx_u(k, 1)] = self.a_min
            ubz[idx_u(k, 1)] = self.a_max

        # slack bounds
        offsetS = self.nX + self.nU
        def idx_s(k):
            return offsetS + k

        for k in range(self.N+1):
            lbz[idx_s(k)] = 0.0
            ubz[idx_s(k)] = +np.inf

        self.lbz = lbz
        self.ubz = ubz
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)

    def _build_vref(self, s0, kappa_seq):
        """
        Optional: reduce speed reference with curvature (basic but effective).
        """
        if not self.use_curv_speed_ref:
            return self.v_ref_base * np.ones(self.N+1)

        vref = np.zeros(self.N+1)
        # use |kappa| along horizon to schedule vref
        for k in range(self.N+1):
            kappa_k = kappa_seq[min(k, self.N-1)]
            vref[k] = self.v_ref_base / (1.0 + self.kappa_speed_gain*abs(kappa_k))
        # clamp
        vref = np.clip(vref, self.v_min, self.v_max)
        return vref


    def solve(self, x0_np, kappa_seq_np, delta_prev=0.0, vref_seq_np=None):
        """
        x0_np: [s, ey, epsi, v]
        kappa_seq_np: length N
        returns: (X_opt, U_opt)
        """
        x0_np = np.asarray(x0_np, dtype=float).copy()
        x0_np[2] = wrap_to_pi(x0_np[2])

        kappa_seq_np = np.asarray(kappa_seq_np, dtype=float).reshape(-1)
        assert len(kappa_seq_np) == self.N

        if vref_seq_np is None:
            vref = self._build_vref(x0_np[0], kappa_seq_np)
        else:
            vref = np.asarray(vref_seq_np, dtype=float).reshape(-1)
            if len(vref) != self.N + 1:
                raise ValueError("vref_seq_np must have length N+1.")
            vref = np.clip(vref, self.v_min, self.v_max)
        
        # build initial guess
        if self.last_sol is None:
            # simple guess: hold state constant, hold delta, a=0
            X_guess = np.tile(x0_np.reshape(4,1), (1, self.N+1))
            U_guess = np.zeros((2, self.N))
            U_guess[0, :] = np.clip(delta_prev, -self.delta_max, self.delta_max)
            U_guess[1, :] = 0.0
            S_guess = np.zeros(self.nS)
            z0 = np.concatenate([
                X_guess.reshape(-1, order="F"),
                U_guess.reshape(-1, order="F"),
                S_guess,
            ])
        else:
            # warm start from last solution
            z0 = self.last_sol

        # parameters
        # print("v_ref;", vref[0])
        p = np.concatenate([x0_np, kappa_seq_np, vref])
        sol = self.solver(
            x0=z0,
            lbx=self.lbz, ubx=self.ubz,
            lbg=self.lbg, ubg=self.ubg,
            p=p
        )

        z = np.array(sol["x"]).reshape(-1)
        self.last_sol = z.copy()

        # unpack
        X_flat = z[:self.nX]
        U_flat = z[self.nX:self.nX + self.nU]
        S_flat = z[self.nX + self.nU:]

        X_opt = X_flat.reshape((self.nx, self.N+1), order="F").T
        U_opt = U_flat.reshape((self.nu, self.N), order="F").T
        self.last_slack = S_flat.copy()
        # U_opt[:,0]=delta, U_opt[:,1]=a

        return X_opt, U_opt
