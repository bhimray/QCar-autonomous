import numpy as np

from FrenetNonLinearMPC import FrenetNonlinearMPC
from pal.utilities.math import wrap_to_pi

class FrenetNonlinearMPCController:
    """
    Uses your existing path.project_frenet() for measurement -> (s,ey,epsi),
    then runs NMPC to output (delta_cmd, v_cmd).
    """
    def __init__(self, path, Ts, L, N=20, sqp_iters=1, **nmpc_kwargs):
        self.path = path
        self.Ts = Ts
        self.N = N
        self.nmpc = FrenetNonlinearMPC(
            path=path,
            Ts=Ts,
            N=N,
            L= L,
            **nmpc_kwargs
        )
        self.delta_prev = 0.0


    def compute_control(self, x_world, psi, v_meas, delta, v_max, verbose=False):
        Ts = self.Ts

        # Use world -> frenet only to MEASURE state for MPC
        s, ey, epsi, _, _ = self.path.project_frenet(x_world[0], x_world[1], psi, Ts, v_max)

        x0 = np.array([s, ey, wrap_to_pi(epsi), float(v_meas)], dtype=float)

        # curvature horizon using current s
        # (simple rollout: s_k ≈ s + k*Ts*v, good enough to query kappa)
        s_seq_full = np.array([s + k*Ts*max(v_meas, 0.0) for k in range(self.N + 1)], dtype=float)
        s_seq = s_seq_full[:-1]
        kappa_seq = np.array([self.path.kappa_at_s(sk) for sk in s_seq], dtype=float)
        kappa_ref = float(kappa_seq[0])

        vref_seq = None
        if getattr(self.path, "v_ref", None) is not None:
            vref_seq = np.array([self.path.v_ref_at_s(sk) for sk in s_seq_full], dtype=float)

        X_opt, U_opt = self.nmpc.solve(x0, kappa_seq, vref_seq_np=vref_seq)

        delta_cmd = float(U_opt[0, 0])
        a_cmd = float(U_opt[0, 1])

        # convert accel to velocity command (simple integrator)
        v_cmd = float(np.clip(v_meas + Ts*a_cmd, 0.0, v_max))
        # v_cmd = float(np.clip(v_meas + Ts*a_cmd, -v_max, v_max))

        self.delta_prev = delta_cmd
        return delta_cmd, v_cmd, kappa_ref, ey, epsi
