# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
environment_interpretation.py

Skills activity code for environment interpretation lab guide.
Please review the accompanying "Lab Guide - Environment Interpretation" PDF
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logit, expit
from scipy import ndimage
from threading import Thread, Lock
import time
import pyqtgraph as pg
import signal

from hal.utilities.control import PID, StanleyController
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from FrenetBicycleModel import FrenetBicycleModel
from FrenetNonLinearMPCController import FrenetNonlinearMPCController
from Path2D import Path2D

#endregion

# ================ Lightweight Steering PID (Minimal Past/Future Influence) ================
class MinimalInfluencePID:
    """
    PID with optional filtered derivative and tight integral clamping.
    Intended to minimize influence of past (I) and future (D) terms.
    """
    def __init__(
        self,
        Kp=0.5,
        Ki=0.0,
        Kd=0.02,
        u_limits=None,
        i_limit=0.5,
        d_filter_alpha=0.2,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_limits = u_limits
        self.i_limit = abs(i_limit) if i_limit is not None else None
        self.d_filter_alpha = d_filter_alpha

        self._i = 0.0
        self._prev_err = 0.0
        self._d_filt = 0.0

    def reset(self):
        self._i = 0.0
        self._prev_err = 0.0
        self._d_filt = 0.0

    def update(self, setpoint, measurement, dt):
        if dt <= 0:
            return measurement
        err = setpoint - measurement

        # Proportional term
        p = self.Kp * err

        # Integral term (tight clamp)
        self._i += err * dt
        if self.i_limit is not None:
            self._i = np.clip(self._i, -self.i_limit, self.i_limit)
        i = self.Ki * self._i

        # Derivative term with light filtering (short memory)
        d_raw = (err - self._prev_err) / dt
        alpha = self.d_filter_alpha
        self._d_filt = (1.0 - alpha) * self._d_filt + alpha * d_raw
        d = self.Kd * self._d_filt
        self._prev_err = err

        u = p + i + d
        if self.u_limits is not None:
            u = np.clip(u, self.u_limits[0], self.u_limits[1])
        return u

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 300
startDelay = 1
controllerUpdateRate = 60  # Hz

# ===== Vehicle Controller Parameters
# - enableVehicleControl: If true, the QCar will drive through the specified
#   node sequence. If false, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableVehicleControl = True
enableSteeringControl = True
v_ref = 0.5  # m/s
# nodeSequence = [0, 20, 0]
nodeSequence = [9, 14, 9]
# ===== Occupancy Grid Parameters
# - cellWidth: edge length for occupancy grid cells (in meters)
# - r_res: range resolution for polar grid cells (in meters)
# - r_max: maximum range of the lidar (in meters)
# - p_low: likelihood value for vacant cells (according to lidar scan)
# - p_high: likelihood value for occupied cells (according to lidar scan)
cellWidth = 0.02
r_res = 0.05
r_max = 4
p_low = 0.4
p_high = 0.6


#region Initial Setup
lock = Lock()

roadmap = SDCSRoadMap()
waypointSequence = roadmap.generate_path(nodeSequence)
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
x_hat = initialPose
t_hat = 0
print("initial position", x_hat)
if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    hQCar = qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate=False
else:
    calibrate =  'y' in input('do you want to recalibrate?(y/n)')
# Used to enable safe keyboard triggered shutdown
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

gps = QCarGPS(initialPose=initialPose,calibrate=calibrate)
while (not KILL_THREAD) and (gps.readGPS() or  gps.readLidar()):
    pass
#endregion

# W = 0.257         # lane width [m] (example)
W = 0.20  # lane width [m]
b = 0.03       # buffer [m]
ey_min = -(W/2 - b)
ey_max = +(W/2 - b)
lr = lf = 0.128 # meters

bounds = {
        "delta": (-np.pi/6, np.pi/6),
        "v_cmd": (0, v_ref),
        "ey": (ey_min, ey_max)
    }
alpha = 300
Q = alpha * np.array([0.001, 1, 1, 0.001])
beta = 2
R = beta * np.array([3, 1])
gamma = 10
Sdu = gamma * np.array([1, 0.2])
# weights = {
#     # [s, ey, epsi, v]
#     "Q":  Q,
#     # [delta, accel]
#     "R":  R,
#     # [ddelta, daccel]
#     "Sdu": Sdu
# }

weights = {
    # [s, ey, epsi, v]
    "Q":   [ 1e-6,  185.0,  165.0,  10.0 ],
    # [delta, accel]
    "R":   [ 6,   10.0 ],
    # [ddelta, daccel]
    "Sdu": [ 20.0,  15.0 ]
}

Ts = 1.0 / controllerUpdateRate
countMax = controllerUpdateRate / 5
path = Path2D(waypointSequence.T, Ts)  # waypointSequence is (2,N), convert to (N,2)

# Offline speed profile from curvature
a_lat_max = 1.5  # max lateral acceleration [m/s^2]
a_accel = 1.5    # max forward accel [m/s^2]
path.compute_speed_profile(
    a_lat_max=a_lat_max,
    v_min=0.0,
    v_max=v_ref,
    a_accel=a_accel,
    a_decel=a_accel,
)

# DELTA_S_LOG = {"s": [], "delta": [], "delta_ref": []}

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, x_hat, t_hat, DELTA_S_LOG
    u = 0
    v_cmd = 0
    delta = 0
    delta_ref = 0.0
    kappa_ref = 0.0
    # used to limit data sampling to 5 Hz
    count = 0
    last_print_t = 0.0
    mpc_ms_avg = 0.0
    ey = 0.0
    epsi = 0.0
    #endregion

    #region QCar interface setup
    with lock:
        ekf = QCarEKF(x_0=x_hat)
    driveController = QCarDriveController(waypointSequence, cyclic=False)
    pid_controller = PID(
                    Kp= 0.2,
                    Ki= 1,
                    Kd= 0,
                    uLimits= (0, v_ref)
                )
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    # in controlLoop init
    f_controller = FrenetNonlinearMPCController(
        path=path,
        Ts=Ts,
        L=lf+lr,
        N=40,
        sqp_iters=2,
        ey_max=ey_max,
        delta_max=np.pi/6,
        ddelta_max=2.0,
        a_min=-a_accel,
        a_max=a_accel,
        v_min=0.001,
        v_max=v_ref,
        w_ey=weights["Q"][1],
        w_epsi=weights["Q"][2],
        w_v=weights["Q"][3],
        w_delta=weights["R"][0],
        w_a=weights["R"][1],
        w_ddelta=weights["Sdu"][0],
        w_da=weights["Sdu"][1],
        use_curv_speed_ref=True,
        v_ref_base=v_ref,
        kappa_speed_gain= 60.0,
    )
    #endregion

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Update QCar state estimates and drive controller
            qcar.read()
            v = qcar.motorTach
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update(
                    [v, delta],
                    dt,
                    y_gps,
                    qcar.gyroscope[2],
                )
            else:
                ekf.update(
                    [v, delta],
                    Ts,
                    None, # no GPS update
                    qcar.gyroscope[2],
                )
            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.3
            if abs(ey) > abs(ey_max + 0.30):
                print(f"Stopping MPC: lateral offset |ey|={abs(ey):.2f} m exceeds {abs(ey_max):.2f} m")
                qcar.write(0.0, 0.0)
                KILL_THREAD = True
                return

            mpc_dt_ms = 0.0
            if t < startDelay or (not enableVehicleControl):
                u = 0.0000
                delta = 0.0000
                delta_ref = 0.0
            else:
                # --- Controller ---
                mpc_t0 = time.time()
                delta_cmd, v_cmd, kappa_ref, ey, epsi = f_controller.compute_control(
                    x_world=np.array([x, y]),
                    psi=th,
                    v_meas=v,
                    delta=delta,
                    v_max=v_ref,
                    verbose=False
                )
                mpc_dt_ms = (time.time() - mpc_t0) * 1000.0
                mpc_ms_avg = 0.9 * mpc_ms_avg + 0.1 * mpc_dt_ms
                delta_ref = np.arctan((lf + lr) * kappa_ref)
                # Minimal-influence PID to track delta_cmd
                # delta = delta_pid.update(delta_cmd, delta_ref, dt)
                delta = delta_cmd
                u = pid_controller.update(v_cmd, v, dt)
                # u = v_cmd

                # s_now, _, _, _, _ = path.project_frenet(x, y, th, Ts, v_ref)
                # DELTA_S_LOG["s"].append(float(s_now))
                # DELTA_S_LOG["delta"].append(float(delta))
                # DELTA_S_LOG["delta_ref"].append(float(delta_ref))
            # if u < -0.02:
            #     u = 0.0

            qcar.write(u, delta)
            if t - last_print_t >= 0.2:
                mpc_budget_ms = 1000.0 / controllerUpdateRate
                if mpc_dt_ms > mpc_budget_ms:
                    # print(
                    #     f"WARNING: MPC compute time {mpc_dt_ms:.1f} ms exceeds budget "
                    #     f"{mpc_budget_ms:.1f} ms"
                    # )
                    f"kappa={kappa_ref:.2f}, delta_cmd={delta:.2f}, v_mpc/v_cmd/vmeas={v_cmd:.2f}/{u:.2f}/{v:.2f}, "
                print(
                    f"kappa={kappa_ref:.2f}, delta_cmd={delta:.2f}, v_mpc/v_cmd/vmeas={v_cmd:.2f}/{u:.2f}/{v:.2f}, "
                    f"mpc_ms={mpc_dt_ms:.1f}, mpc_avg_ms={mpc_ms_avg:.1f}"
                )
                last_print_t = t
            #endregion
            
            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                # Speed control scope
                # speedScope.axes[0].sample(t_plot, [v, v_ref])
                # speedScope.axes[1].sample(t_plot, [v_ref-v])
                # speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    # x_ref = steeringController.p_ref[0]
                    # y_ref = steeringController.p_ref[1]
                    # th_ref = steeringController.th_ref

                    # x_ref = gps.position[0]
                    # y_ref = gps.position[1]
                    # th_ref = gps.orientation[2]

                    # steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[0].sample(t_plot, [ey])
                    steeringScope.axes[1].sample(t_plot, [epsi])
                    steeringScope.axes[2].sample(t_plot, [u, v])
                    steeringScope.axes[3].sample(t_plot, [delta, delta_ref])

                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion

            if driveController.steeringController.pathComplete:
                return
            continue
        with lock:
            print('Control thread terminated')

def PlotCurvature():

    scale = 0.5  # visual scaling only

    plt.figure()
    plt.plot(path.wp[:,0], path.wp[:,1], 'k-', label="Path")

    for i in range(0, path.N, max(1, path.N // 40)):
        x, y = path.wp[i]
        psi = path.psi[i]
        kappa = path.kappa[i]

        # normal direction
        nx = -np.sin(psi)
        ny =  np.cos(psi)

        plt.arrow(
            x, y,
            scale * kappa * nx,
            scale * kappa * ny,
            head_width=0.03,
            color='r'
        )

    plt.figure()
    plt.axis("equal")
    plt.title("Path with Curvature Normals")
    plt.legend()
    plt.grid(True)
    plt.show()

    psi_unwrapped = np.unwrap(path.psi)
    dpsi_ds = np.gradient(psi_unwrapped, path.s)

    plt.figure()
    plt.plot(path.s, path.kappa, label="κ(s) from code")
    plt.plot(path.s, dpsi_ds, '--', label="dψ/ds (numerical)")
    plt.legend()
    plt.grid(True)
    plt.title("Curvature consistency check")
    plt.show()

#region : Setup and run experiment
if __name__ == '__main__':
#region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
        
    # PlotCurvature()
    # Scope for monitoring speed controller
    # speedScope = MultiScope(
    #     rows=3,
    #     cols=1,
    #     title='Vehicle Speed Control',
    #     fps=fps
    # )
    # speedScope.addAxis(
    #     row=0,
    #     col=0,
    #     timeWindow=tf,
    #     yLabel='Vehicle Speed [m/s]',
    #     yLim=(0, 1)
    # )
    # speedScope.axes[0].attachSignal(name='v_meas', width=2)
    # speedScope.axes[0].attachSignal(name='v_ref')

    # speedScope.addAxis(
    #     row=1,
    #     col=0,
    #     timeWindow=tf,
    #     yLabel='Speed Error [m/s]',
    #     yLim=(-0.5, 0.5)
    # )
    # speedScope.axes[1].attachSignal()

    # speedScope.addAxis(
    #     row=2,
    #     col=0,
    #     timeWindow=tf,
    #     xLabel='Time [s]',
    #     yLabel='Throttle Command [%]',
    #     yLim=(-v_ref, v_ref)
    # )
    # speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

    steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='ey',
            yLim=(-2.5, 2.5)
        )
    steeringScope.axes[0].attachSignal(name='ey')
    # steeringScope.axes[0].attachSignal(name='x_ref')

    steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='epsi',
            yLim=(-1, 5)
        )
    steeringScope.axes[1].attachSignal(name='epsi')
    # steeringScope.axes[1].attachSignal(name='y_ref')
    steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Velocity [m/s]',
            yLim=(-3.5, 3.5)
        )
    steeringScope.axes[2].attachSignal(name='u')
    steeringScope.axes[2].attachSignal(name='v_ref')

    steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
    steeringScope.axes[3].attachSignal(name='delta_cmd')
    steeringScope.axes[3].attachSignal(name='delta_ref')
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

    im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

    steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
    steeringScope.axes[4].images[0].setImage(image=im)

    referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])
    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    # #region
    #  # Scope for path curvature/heading vs s
    # pathScope = MultiScope(
    #         rows=2,
    #         cols=1,
    #         title='Path Curvature/Heading vs s',
    #         fps=fps
    #     )

    # pathScope.addAxis(
    #         row=0,
    #         col=0,
    #         timeWindow=path.s[-1],
    #         yLabel='kappa [1/m]',
    #         yLim=(-2.0, 2.0)
    #     )
    # pathScope.axes[0].attachSignal(name='kappa')

    # pathScope.addAxis(
    #         row=1,
    #         col=0,
    #         timeWindow=path.s[-1],
    #         yLabel='psi [rad]',
    #         yLim=(-np.pi, np.pi)
    #     )
    # pathScope.axes[1].attachSignal(name='psi')
    # pathScope.axes[1].xLabel = 's [m]'

    # # Static plots for kappa/psi vs s
    # kappaPlot = pg.PlotDataItem(
    #         pen={'color': (196,78,82), 'width': 2},
    #         name='kappa(s)'
    #     )
    # pathScope.axes[0].plot.addItem(kappaPlot)
    # kappaPlot.setData(path.s, path.kappa)

    # psiPlot = pg.PlotDataItem(
    #         pen={'color': (85,168,104), 'width': 2},
    #         name='psi(s)'
    #     )
    # pathScope.axes[1].plot.addItem(psiPlot)
    # psiPlot.setData(path.s, path.psi)
    # #endregion

    # PlotCurvature()
    #region : Setup threads, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive():
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        gps.terminate()
    #endregion

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
#endregion



