# ==========================================================
# vehicle_control.py
# Frenet MPC embedded in Quanser QCar control loop (VIRTUAL)
# ==========================================================

import os
import signal
import time
import numpy as np
from threading import Thread

import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images

# === ACADOS / MPC imports ===
import casadi as ca
import acados_setup
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.builders import CMakeBuilder

os.environ["ACADOS_SOURCE_DIR"] = r"C:\Users\rayyu\acados"
if os.name == "nt":
    os.add_dll_directory(r"C:\Users\rayyu\acados\build\acados")  # where your acados.dll is

cmake_builder = CMakeBuilder()
cmake_builder.generator = "NMake Makefiles"     # matches your setup
cmake_builder.build_type = "Release"            # faster / smaller than Debug


from vehicle_model import vehicle_model
from track_bounds_with_obstacles import track_bounds_with_obstacles


# ================= Experiment Configuration =================
tf = 6000
startDelay = 1
controllerUpdateRate = 50   # MPC friendly

enableSteeringControl = True
nodeSequence = [10, 4, 20, 10]

calibrationPose = [0, 2, -np.pi/2]

# ============================================================
#               Path → Frenet Projection
# ============================================================

class PathFrenet:
    def __init__(self, waypoints, cyclic=True):
        self.wp = waypoints
        self.N = waypoints.shape[1]
        self.cyclic = cyclic

        x = waypoints[0, :]
        y = waypoints[1, :]

        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        self.s_wp = np.zeros(self.N)
        self.s_wp[1:] = np.cumsum(ds)
        self.total_length = self.s_wp[-1]

        dx = np.gradient(x)
        dy = np.gradient(y)
        self.psi_wp = np.arctan2(dy, dx)

        dpsi = np.gradient(self.psi_wp)
        ds_safe = np.gradient(self.s_wp + 1e-9)
        self.kappa_wp = dpsi / ds_safe

    def _wrap_s(self, s):
        if not self.cyclic:
            return np.clip(s, 0, self.total_length)
        return s % self.total_length

    def kappa(self, s):
        return np.interp(self._wrap_s(s), self.s_wp, self.kappa_wp)

    def project(self, x, y):
        best_d2 = np.inf
        best = (0, 0, 0)

        for i in range(self.N-1):
            p1 = self.wp[:, i]
            p2 = self.wp[:, i+1]
            v = p2 - p1
            L2 = v @ v
            if L2 < 1e-12:
                continue

            w = np.array([x, y]) - p1
            t = np.clip((w @ v) / L2, 0, 1)
            proj = p1 + t * v
            d = np.array([x, y]) - proj
            d2 = d @ d

            if d2 < best_d2:
                psi = np.arctan2(v[1], v[0])
                n = np.array([-np.sin(psi), np.cos(psi)])
                ye = d @ n
                s = self.s_wp[i] + t * np.sqrt(L2)
                best_d2 = d2
                best = (self._wrap_s(s), ye, wrap_to_pi(psi))

        return best


# ============================================================
#                   Speed PI Controller
# ============================================================

class SpeedController:
    def __init__(self, kp=0.4, ki=0.1):
        self.kp = kp
        self.ki = ki
        self.ei = 0.0
        self.maxThrottle = 0.3

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += e * dt
        u = self.kp * e + self.ki * self.ei
        return float(np.clip(u, -self.maxThrottle, self.maxThrottle))


# ============================================================
#                       Frenet MPC
# ============================================================

class FrenetMPC:
    def __init__(self, path, N=30, T=0.3):
        self.path = path
        self.N = N
        self.T = T
        self.u_prev = np.zeros(2)

        self.model = vehicle_model()
        self.solver = self._build_solver()

    def _build_solver(self):
        ocp = AcadosOcp()
        ocp.model.x = self.model["x"]
        ocp.model.u = self.model["u"]
        ocp.model.p = self.model["p"]
        ocp.model.f_expl_expr = self.model["f_expl_expr"]
        ocp.model.f_impl_expr = self.model["f_impl_expr"]
        ocp.model.name = self.model["name"]

        ocp.parameter_values = np.zeros(7)
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.lbu = np.array([-0.6, -20])
        ocp.constraints.ubu = np.array([+0.6, +20])

        ocp.constraints.idxbx = np.array([4])
        ocp.constraints.lbx = np.array([-0.12])
        ocp.constraints.ubx = np.array([+0.12])

        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        Delta_u = u - p[1:3]
        stage_cost = (
            5*(x[1]-p[3])**2 +
            0.5*x[4]**2 +
            0.5*x[5]**2 +
            ca.mtimes([Delta_u.T, ca.diag([1e-2,1e-2]), Delta_u])
        )
        terminal_cost = (
            5*(x[1]-p[3])**2 +
            0.5*x[4]**2 +
            0.5*x[5]**2
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        return AcadosOcpSolver(
                        ocp,
                        json_file="frenet_mpc.json",
                        cmake_builder=cmake_builder,
                        build=True,
                        generate=True
                    )


    def step(self, x_mpc):
        s0 = x_mpc[0]
        self.solver.set(0, "x", x_mpc)

        for j in range(self.N):
            s = s0 + 3.0*j/self.N
            kappa = self.path.kappa(s)
            p = np.array([kappa, *self.u_prev, x_mpc[1], 0, 0, 95])
            self.solver.set(j, "p", p)

        self.solver.solve()
        u0 = self.solver.get(0, "u")
        x1 = self.solver.get(1, "x")

        self.u_prev[:] = u0
        return u0[0], x1[1]   # delta, anticipative vx


# ============================================================
#                        Control Loop
# ============================================================

def controlLoop():
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf = QCarEKF(x_0=initialPose)
    gps = QCarGPS(initialPose=calibrationPose, calibrate=False)

    speedCtrl = SpeedController()
    mpc = FrenetMPC(path)

    t0 = time.time()
    prev_t = 0

    with qcar, gps:
        while True:
            t = time.time() - t0
            dt = t - prev_t
            prev_t = t

            qcar.read()
            gps.readGPS()

            ekf.update([qcar.motorTach, 0], dt,
                       np.array([gps.position[0], gps.position[1], gps.orientation[2]]),
                       qcar.gyroscope[2])

            x, y, th = ekf.x_hat[:,0]
            v = qcar.motorTach

            if t < startDelay:
                qcar.write(0,0)
                continue

            s, ye, psi = path.project(x, y)
            theta_e = wrap_to_pi(th - psi)

            x_mpc = np.array([s, v, 0, qcar.gyroscope[2], ye, theta_e, 95])
            delta, v_ref = mpc.step(x_mpc)
            u = speedCtrl.update(v, v_ref, dt)

            qcar.write(u, delta)


# ============================================================
#                        Main
# ============================================================

if __name__ == "__main__":

    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()

    path = PathFrenet(waypointSequence)

    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive():
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
