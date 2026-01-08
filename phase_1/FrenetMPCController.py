# ===== ADD THESE IMPORTS near top =====
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

# from your converted modules
from vehicle_model import vehicle_model
from track_bounds_with_obstacle import track_bounds_with_obstacle
from velocity_profile_gen import velocity_profile_gen


class FrenetMPCController:
    """
    Embed-ready MPC wrapper.
    Does NOT assume how you compute frenet states from (x,y,th).
    """

    def __init__(self, track_file: str, N=50, T=0.5, qp_cond_N=10):
        self.N = N
        self.T = T
        self.dt = T / N

        # --- load model ---
        model = race_car_model_mod()
        self.model = model
        self.nx = int(model["x"].size1())
        self.nu = int(model["u"].size1())

        # --- load track file (same assumption as your MATLAB: LMS_Track.txt columns) ---
        track_data = np.loadtxt(track_file)
        self.Sref = track_data[:, 0]
        self.Xref = track_data[:, 1]
        self.Yref = track_data[:, 2]
        self.Psiref = track_data[:, 3]
        kappa_raw  = track_data[:, 4]

        # curvature smoothing / extension (same idea as before)
        scale_factor = 10.0
        kappa_scaled = kappa_raw / scale_factor
        kappa_smooth = np.convolve(kappa_scaled, np.ones(7)/7, mode="same")

        s_ext = np.concatenate([self.Sref, self.Sref[-1] + self.Sref[1:]])
        kappa_ext = np.concatenate([kappa_smooth, kappa_smooth[1:]])
        self.kapparef_s = ca.interpolant("kapparef_s", "linear", [s_ext], kappa_ext)

        # velocity profile (you had mu=1.2, m=1.8, Ux_max=20)
        track_struct = {
            "station": self.Sref,
            "curvature": np.array(self.kapparef_s(self.Sref)).reshape(-1),
            "total_length": float(self.Sref[-1]),
        }
        self.Ux_final, self.Ux_steady, self.Ux_forward, self.lap_time_min = velocity_profile_gen(
            track_struct, mu=1.2, vehicle_mass=1.8, Ux_max=20.0
        )

        # --- build acados ocp ---
        self.solver = self._build_solver(qp_cond_N=qp_cond_N)

        # store previous input for delta-u penalty
        self.u_prev = np.zeros((self.nu,))

        # obstacle params (keep in object so you can change on fly)
        self.track_width = 0.24
        self.s_obs = 5.0
        self.ye_obs = 0.05
        self.obs_w = 0.05
        self.obs_L = 0.05

    def _build_solver(self, qp_cond_N=10) -> AcadosOcpSolver:
        ocp = AcadosOcp()

        ocp.model.name = self.model["name"]
        ocp.model.x = self.model["x"]
        ocp.model.u = self.model["u"]
        ocp.model.p = self.model["p"]
        ocp.model.f_expl_expr = self.model["f_expl_expr"]
        ocp.model.f_impl_expr = self.model["f_impl_expr"]

        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # constraints (same as your MATLAB)
        delta_min, delta_max = -0.6, 0.6
        a_min, a_max = -20.0, 20.0
        ye_min, ye_max = -0.12, 0.12

        ocp.constraints.idxbu = np.array([0, 1], dtype=int)
        ocp.constraints.lbu = np.array([delta_min, a_min])
        ocp.constraints.ubu = np.array([delta_max, a_max])

        # IMPORTANT: ye is x[4] in Python 0-based indexing
        ocp.constraints.idxbx = np.array([4], dtype=int)
        ocp.constraints.lbx = np.array([ye_min])
        ocp.constraints.ubx = np.array([ye_max])

        # cost: EXTERNAL (same structure as your MATLAB main)
        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        kappa_r = p[0]
        u_prev = p[1:3]
        vx_ref = p[3]
        ye_ref = p[4]
        theta_ref = p[5]
        SoC_ref = p[6]

        e_vx = x[1] - vx_ref
        e_ye = x[4] - ye_ref
        e_th = x[5] - theta_ref
        e_soc = x[6] - SoC_ref

        Delta_u = u - u_prev
        R_c = ca.diag(ca.vertcat(1e-2, 1e-2))

        track_err_cost = 5.0*e_vx**2 + 0.5*e_ye**2 + 0.5*e_th**2 + 0.1*e_soc**2
        stage_cost = track_err_cost + ca.mtimes([Delta_u.T, R_c, Delta_u])

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = track_err_cost

        # solver options (same intent)
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = qp_cond_N

        ocp.solver_options.nlp_solver_tol_stat = 1e-4
        ocp.solver_options.nlp_solver_tol_eq = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.nlp_solver_tol_comp = 1e-4

        solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}.json")
        return solver

    def set_obstacle(self, s_obs, ye_obs, obs_w, obs_L):
        self.s_obs, self.ye_obs, self.obs_w, self.obs_L = s_obs, ye_obs, obs_w, obs_L

    def compute_control(self, x_mpc: np.ndarray, sim_index: int):
        """
        Inputs:
          x_mpc: [s, vx, vy, wz, ye, theta_e, SoC]  shape (7,)
          sim_index: used to pick vx_ref from Ux_final

        Returns:
          delta_cmd, a_cmd
        """
        s0 = float(x_mpc[0])

        # set initial state (hard constraint)
        self.solver.set(0, "x", x_mpc)

        # stage parameter updates + ye bounds updates
        for j in range(self.N):
            # reference progress
            # (this matches your MATLAB idea; you can switch to predicted s if you want later)
            s_curr = s0 + (3.0) * (j / self.N)  # sref_N = 3.0 in your MATLAB
            kappa_val = float(self.kapparef_s(s_curr))

            # vx reference from precomputed profile (needs enough length!)
            idx = min(sim_index + j, len(self.Ux_final) - 1)
            vx_ref_val = float(self.Ux_final[idx])

            # TODO: if you want nonzero refs, set them here
            ye_ref_val = 0.0
            theta_ref_val = 0.0
            SoC_ref_val = 95.0

            p_val = np.array([
                kappa_val,
                self.u_prev[0],
                self.u_prev[1],
                vx_ref_val,
                ye_ref_val,
                theta_ref_val,
                SoC_ref_val
            ])
            self.solver.set(j, "p", p_val)

            ye_min_s, ye_max_s = track_bounds_with_obstacle(
                s_curr, self.track_width, self.s_obs, self.ye_obs, self.obs_w, self.obs_L
            )
            self.solver.set(j, "lbx", np.array([ye_min_s]))
            self.solver.set(j, "ubx", np.array([ye_max_s]))

        status = self.solver.solve()
        if status not in (0, 2):
            raise RuntimeError(f"acados status {status}")

        u0 = self.solver.get(0, "u")  # [delta, a]
        delta_cmd = float(u0[0])
        a_cmd = float(u0[1])

        # update u_prev for next tick (this is IMPORTANT for Δu penalty)
        self.u_prev[:] = u0

        return delta_cmd, a_cmd
