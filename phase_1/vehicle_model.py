# race_car_model_mod.py
import casadi as ca


def vehicle_model():
    """
    Python conversion of MATLAB: race_car_model_mod.m

    States:
        x = [s, vx, vy, wz, ye, theta_e, SoC]

    Inputs:
        u = [delta, a]

    Parameters (size 7):
        p = [kappa_r, delta_prev, a_prev, p4, p5, p6, p7]
        NOTE: Only p[0:3] are used in the dynamics exactly like your MATLAB.
              The remaining are unused (but allowed).
    """
    model = {}
    model["name"] = "vehicle_model"

    # ----------------
    # States
    # ----------------
    s = ca.SX.sym("s")
    vx = ca.SX.sym("vx")
    vy = ca.SX.sym("vy")
    wz = ca.SX.sym("wz")
    ye = ca.SX.sym("ye")
    theta_e = ca.SX.sym("theta_e")
    SoC = ca.SX.sym("SoC")

    x = ca.vertcat(s, vx, vy, wz, ye, theta_e, SoC)
    nx = int(x.size1())

    # xdot (implicit form)
    xdot = ca.SX.sym("xdot", nx, 1)

    # ----------------
    # Inputs
    # ----------------
    delta = ca.SX.sym("delta")
    a = ca.SX.sym("a")
    u = ca.vertcat(delta, a)

    # ----------------
    # Parameters
    # ----------------
    p = ca.SX.sym("p", 7, 1)
    kappa_r = p[0]      # MATLAB p(1)
    delta_prev = p[1]   # MATLAB p(2) (unused)
    a_prev = p[2]       # MATLAB p(3) (unused)

    # ----------------
    # Vehicle parameters
    # ----------------
    m = 1.8
    Iz = 0.03
    lf = 0.125
    lr = 0.125
    Cf = 68
    Cr = 71
    mu_tyre = 1.2
    g = 9.81

    # Battery / aero parameters (illustrative)
    Cd = 0.35
    rho = 1.225
    Ar = 0.05
    C_alpha = 3000

    # ----------------
    # Nonlinear dynamics
    # ----------------
    eps_vx = 1e-3
    vx_safe = vx + eps_vx

    # Tire lateral forces (bicycle model)
    Fy_f = Cf * (delta - vy / vx_safe + lf * wz / vx_safe)
    Fy_r = Cr * (-vy / vx_safe - lr * wz / vx_safe)

    # Longitudinal, lateral, yaw dynamics
    vx_dot = (
        a
        - Fy_f * ca.sin(delta) / m
        - mu_tyre * g
        + wz * vy
    )

    vy_dot = (
        Fy_f * ca.cos(delta) / m
        + Fy_r / m
        - wz * vx
    )

    wz_dot = (Fy_f * lf * ca.cos(delta) - Fy_r * lr) / Iz

    # Progress s-dot in curvilinear frame
    s_dot = (vx * ca.cos(theta_e) - vy * ca.sin(theta_e)) / (1 - ye * kappa_r)

    # Lateral error & heading error in curvilinear coordinates
    ye_dot = vx * ca.sin(theta_e) + vy * ca.cos(theta_e)

    theta_e_dot = (
        wz
        - ((vx * ca.cos(theta_e) - vy * ca.sin(theta_e)) * kappa_r) / (1 - ye * kappa_r)
    )

    # Battery power and SoC dynamics
    p_alpha = 0.5 * Cd * rho * Ar * (vx**2) + mu_tyre * m * g * vx
    SoC_dot = -(1.0 / C_alpha) * p_alpha

    f_expl = ca.vertcat(s_dot, vx_dot, vy_dot, wz_dot, ye_dot, theta_e_dot, SoC_dot)
    f_impl = xdot - f_expl

    # Fill model dict for acados_template
    model["x"] = x
    model["xdot"] = xdot
    model["u"] = u
    model["p"] = p
    model["f_expl_expr"] = f_expl
    model["f_impl_expr"] = f_impl

    return model
