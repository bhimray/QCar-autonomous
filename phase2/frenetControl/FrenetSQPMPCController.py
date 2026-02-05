import numpy as np
from .FrenetBicycleModel import FrenetBicycleModel
from .FrenetLinearizedMPC import FrenetLinearizedMPC
from .Path2D import Path2D

class FrenetSQPMPCController:
    def __init__(self, path: Path2D, model: FrenetBicycleModel, N, bounds, weights, sqp_iters=2):
        self.path = path
        self.model = model
        self.N = N
        self.sqp_iters = sqp_iters
        self.mpc = FrenetLinearizedMPC(model, N, bounds, weights)
        self.X_nom = None
        self.U_nom = None

    def initialize_nominal(self, x0, u0):
        self.X_nom = np.zeros((self.N+1, 4))
        self.U_nom = np.zeros((self.N, 2))
        self.X_nom[0] = x0
        for k in range(self.N):
            self.U_nom[k] = u0
            kappa = self.path.kappa_at_s(self.X_nom[k,0])
            self.X_nom[k+1] = self.model.step(self.X_nom[k], self.U_nom[k], kappa)

    def compute_control(self, x_world, psi, v_meas, delta, verbose=False):
        # project to frenet
        s, ey, epsi, _, _ = self.path.project_frenet(x_world[0], x_world[1], psi, self.model.Ts, v_meas)
        x0 = np.array([s, ey, epsi, v_meas], dtype=float)

        # build references over horizon
        # simple: advance s_ref using v_ref
        Ts = self.model.Ts
        s_ref = np.array([s + k*Ts*v_meas for k in range(self.N+1)], dtype=float)

        xref = np.zeros((self.N+1, 4))
        xref[:,0] = s_ref
        xref[:,1] = 0.0      # ey
        xref[:,2] = 0.0      # epsi
        xref[:,3] = 0.0    # v

        u_current = np.zeros((self.N, 2))
        u_current[:,0] = delta
        u_current[:,1] = v_meas

        # nominal init
        if self.X_nom is None:
            self.initialize_nominal(x0, u_current[0])

        # curvature along nominal
        kappa_seq = np.array([self.path.kappa_at_s(self.X_nom[k,0]) for k in range(self.N)], dtype=float)

        # SQP iterations
        for _ in range(self.sqp_iters):
            X_opt, U_opt = self.mpc.solve(x0, xref, self.X_nom, self.U_nom, kappa_seq, verbose=verbose)
            self.X_nom = X_opt
            self.U_nom = U_opt
            kappa_seq = np.array([self.path.kappa_at_s(self.X_nom[k,0]) for k in range(self.N)], dtype=float)

        # apply first control
        delta_cmd, v_cmd = U_opt[0]
        print("X_OPT:", X_opt[0], "kappa at s:", self.path.kappa_at_s(X_opt[0,0]))
        return float(delta_cmd), float(v_cmd)
