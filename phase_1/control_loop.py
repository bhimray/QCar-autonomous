from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca

# Use your converted Python model + bounds function
from vehicle_model import vehicle_model
from track_bounds_with_obstacle import track_bounds_with_obstacle


class FrenetMPC:
    def __init__(self, path: PathFrenet, N=30, T=0.3):
        self.path = path
        self.N = N
        self.T = T
        self.dt = T / N

        self.model = race_car_model_mod()
        self.nx = int(self.model["x"].size1())
        self.nu = int(self.model["u"].size1())

        self.u_prev = np.zeros((self.nu,))

        # obstacle/track params (tune)
        self.track_width = 0.24
        self.s_obs = 5.0
        self.ye_obs = 0.05
        self.obs_w = 0.05
        self.obs_L = 0.05

        self.sref_N = 3.0  # same as your MATLAB

        self.solver = self._build_solver()

        # init guesses
        # (will be overwritten during loop)
        for j in range(self.N + 1):
            self.solver.set(j, "x", np.zeros((self.nx,)))
        for j in range(self.N):
            self.solver.set(j, "u", np.zeros((self.nu,)))

    def _build_solver(self):
        ocp = AcadosOcp()
        ocp.model.name = self.model["name"]
        ocp.model.x = self.model["x"]
        ocp.model.u = self.model["u"]
        ocp.model.p = self.model["p"]
        ocp.model.f_expl_expr = self.model["f_expl_expr"]
        ocp.model.f_impl_expr = self.model["f_impl_expr"]

        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # bounds
        delta_min, delta_max = -0.6, 0.6
        a_min, a_max = -20.0, 20.0
        ocp.constraints.idxbu = np.array([0, 1], dtype=int)
        ocp.constraints.lbu = np.array([delta_min, a_min])
        ocp.constraints.ubu = np.array([delta_max, a_max])

        # ye bound as state constraint (ye = x[4])
        ocp.constraints.idxbx = np.array([4], dtype=int)
        ocp.constraints.lbx = np.array([-0.12])
        ocp.constraints.ubx = np.array([+0.12])

        # external cost (same structure as your MATLAB main)
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

        track_err = 5.0*e_vx**2 + 0.5*e_ye**2 + 0.5*e_th**2 + 0.1*e_soc**2
        stage_cost = track_err + ca.mtimes([Delta_u.T, R_c, Delta_u])

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = track_err

        # options: virtual-only -> keep it lighter
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 3
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = 10

        ocp.solver_options.nlp_solver_tol_stat = 1e-4
        ocp.solver_options.nlp_solver_tol_eq = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.nlp_solver_tol_comp = 1e-4

        return AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}.json")

    def step(self, x_mpc: np.ndarray, vx_ref_val: float):
        """
        x_mpc = [s, vx, vy, wz, ye, theta_e, SoC]
        vx_ref_val is reference speed for stage 0 (you can extend it across horizon later)
        Returns: (delta_mpc, a_mpc)
        """
        s0 = float(x_mpc[0])

        # initial state constraint
        self.solver.set(0, "x", x_mpc)

        for j in range(self.N):
            s_curr = s0 + self.sref_N * (j / self.N)
            kappa_val = self.path.kappa(s_curr)

            # keep same refs as your MATLAB main (ye_ref=0, theta_ref=0, SoC_ref=95)
            p_val = np.array([
                kappa_val,
                self.u_prev[0],
                self.u_prev[1],
                float(vx_ref_val),
                0.0,
                0.0,
                95.0
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
        self.u_prev[:] = u0

        return float(u0[0]), float(u0[1])
