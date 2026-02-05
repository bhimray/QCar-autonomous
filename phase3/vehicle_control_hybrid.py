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

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 300
startDelay = 1
controllerUpdateRate = 200

# ===== Vehicle Controller Parameters
# - enableVehicleControl: If true, the QCar will drive through the specified
#   node sequence. If false, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableVehicleControl = True
enableSteeringControl = True
v_ref = 0.3
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
b = 0.01       # buffer [m]
ey_min = -(W/2 - b)
ey_max = +(W/2 - b)
lr = lf = 0.128 # meters

bounds = {
        "delta": (-np.pi/6, np.pi/6),
        "v_cmd": (0, v_ref),
        "ey": (ey_min, ey_max)
    }
# weights = {
#         # [s, ey, epsi, v]
#         "Q":   [ 1e-8,  350.0,  200.0,  1.0 ], #s, ey, epsi, v = 1e-8 5.0 3.0 1.0
#         "R":   [ 3.0,  1.0 ],#delta, a, 0.5, 0.5
#         "Sdu": [ 5.0,  3.0 ] #ddelta, da, 1.5, 1
#     }
weights = {
    # [s, ey, epsi, v]
    "Q":   [ 1e-6,  180.0,  90.0,  2.0 ],

    # [delta, accel]
    "R":   [ 6.0,   1.5 ],

    # [ddelta, daccel]
    "Sdu": [ 14.0,  3.0 ]
}


Ts = 1.0 / controllerUpdateRate
countMax = controllerUpdateRate / 5
path = Path2D(waypointSequence.T, Ts)  # waypointSequence is (2,N), convert to (N,2)

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, x_hat, t_hat

    u = 0.0
    delta = 0.0

    # Feedforward from MPC (slow loop)
    delta_ff = 0.0
    u_ff = 0.0

    # used to limit data sampling to 5 Hz (plots)
    countMax = controllerUpdateRate / 5
    count = 0

    last_print_t = 0.0
    mpc_ms_avg = 0.0

    # MPC scheduling: run MPC at mpcRateHz (e.g., 10 or 20)
    mpcRateHz = 15.0
    mpcEvery = max(1, int(round(controllerUpdateRate / mpcRateHz)))
    mpc_counter = 0

    # steering safety / smoothing
    delta_max = np.pi / 6          # keep consistent with your MPC
    ddelta_max = 2.0               # rad/s (same as your MPC constraint)
    tau_steer = 0.25               # sec (for optional simple lag filter on delta_ff)
    delta_ff_filt = 0.0

    #endregion

    #region QCar interface setup
    with lock:
        ekf = QCarEKF(x_0=x_hat)

    driveController = QCarDriveController(waypointSequence, cyclic=False)
    # We'll reuse the internal Stanley (steeringController) as the FAST feedback layer.
    # It usually uses the same waypointSequence/path under the hood.

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    Ts = 1.0 / controllerUpdateRate

    # MPC model + controller (same as yours)
    model_f = FrenetBicycleModel(L=lf + lr, Ts=Ts, tau_v=0.3)
    f_controller = FrenetNonlinearMPCController(
        path=path,
        model=model_f,
        N=30,
        sqp_iters=2,
        ey_max=ey_max,
        delta_max=delta_max,
        ddelta_max=ddelta_max,
        a_min=-1.5,
        a_max=1.5,
        v_min=0.0,
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
        kappa_speed_gain=70.0
    )
    #endregion

    with qcar:
        t0 = time.time()
        t = 0.0

        while (t < tf + startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = max(1e-6, t - tp)
            #endregion

            #region : Read sensors / EKF
            qcar.read()
            v_meas = qcar.motorTach

            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([v_meas, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                # keep dt consistent; use dt not Ts if you can
                ekf.update([v_meas, delta], dt, None, qcar.gyroscope[2])

            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]

            # use lookahead point (same as your Stanley loop)
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            #endregion

            #region : safety check on lateral error (optional)
            _, ey, _, _, _ = path.project_frenet(x, y, th, Ts, v_ref)
            ey_d = 0.5
            if abs(ey) > abs(ey_max + ey_d):
                print(f"Stopping: |ey|={abs(ey):.2f} m exceeds limit {abs(ey_max + ey_d):.2f} m")
                qcar.write(0.0, 0.0)
                KILL_THREAD = True
                return
            #endregion

            #region : Control logic
            mpc_dt_ms = 0.0
            enable_now = (t >= startDelay) and enableVehicleControl

            if not enable_now:
                u = 0.0
                delta = 0.0
                delta_ff = 0.0
                u_ff = 0.0
                delta_ff_filt = 0.0
                delta_fb = 0.0
                kappa_ref = 0.0
            else:
                # ---------- Slow loop: MPC feedforward update ----------
                # Run MPC at lower rate to reduce compute jitter.
                if (mpc_counter % mpcEvery) == 0:
                    # print("MPC")
                    mpc_t0 = time.time()
                    try:
                        # MPC gives steering + speed suggestions (feedforward)
                        delta_cmd, v_cmd, kappa_ref, x_opt = f_controller.compute_control(
                            x_world=np.array([x, y]),
                            psi=th,
                            v_meas=v_meas,
                            delta=delta,      # current applied (best available)
                            v_max=v_ref,
                            verbose=False
                        )
                        mpc_dt_ms = (time.time() - mpc_t0) * 1000.0
                        mpc_ms_avg = 0.9 * mpc_ms_avg + 0.1 * mpc_dt_ms

                        # Use MPC steering as FEEDFORWARD (not final)
                        delta_ff = float(delta_cmd)

                        # Speed command directly from MPC (or clamp)
                        u_ff = float(np.clip(v_cmd, 0.0, v_ref))

                        # Optional: simple first-order filter on feedforward steering
                        alpha = dt / max(tau_steer, 1e-3)
                        alpha = np.clip(alpha, 0.0, 1.0)
                        delta_ff_filt = (1.0 - alpha) * delta_ff_filt + alpha * delta_ff

                    except Exception as e:
                        # MPC failed -> keep last feedforward, rely on Stanley feedback
                        print(f"MPC exception: {e}")
                        # keep delta_ff_filt, u_ff as previous

                mpc_counter += 1

                # ---------- Fast loop: Stanley feedback ----------
                # IMPORTANT: Use your existing Stanley-based steering controller.
                # If driveController.update returns (u, delta) you must adapt:
                # Here we ONLY want delta feedback; speed will come from MPC.
                #
                # Option A (preferred): if you have direct access to steering controller:
                # delta_fb = driveController.steeringController.update(p, th, v_meas, dt)
                #
                # Option B: call driveController.update and discard its speed:
                u_tmp, delta_fb = driveController.update(p, th, v_meas, v_ref, dt, path=path)

                # Combine: feedforward (MPC) + feedback (Stanley)
                delta_des = delta_ff_filt + float(delta_fb)

                # Rate limit steering command (real actuator protection)
                delta_step_max = ddelta_max * dt
                delta = np.clip(delta_des, delta - delta_step_max, delta + delta_step_max)

                # Saturation
                delta = float(np.clip(delta, -delta_max, delta_max))

                # Speed from MPC (you can also blend with PID if you want)
                u = float(u_ff)

            # Write to car
            qcar.write(u, delta)
            #endregion

            #region : periodic print
            if t - last_print_t >= 0.2:
                mpc_budget_ms = 1000.0 / controllerUpdateRate
                print(
                    f"Kappa={kappa_ref:.2f}, ey={ey:.2f}, "
                    f"delta_ff={delta_ff_filt:.2f}, delta_fb={delta_fb:.2f}, delta={delta:.2f}, "
                    f"v_cmd/vmeas={u:.2f}/{v_meas:.2f}, "
                    f"mpc_ms={mpc_dt_ms:.1f}, mpc_avg_ms={mpc_ms_avg:.1f}"
                )
                last_print_t = t
            #endregion

            #region : Update Scopes (kept similar to your MPC loop)
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # # Speed control scope
                # speedScope.axes[0].sample(t_plot, [v_meas, v_ref])
                # speedScope.axes[1].sample(t_plot, [v_ref - v_meas])
                # speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])

                    # current state
                    p_state = np.array([ekf.x_hat[0, 0], ekf.x_hat[1, 0]])

                    # reference (you were using GPS values; keep consistent)
                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p_state[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p_state[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])

                    # show actual delta vs feedforward delta_ff
                    steeringScope.axes[3].sample(t_plot, [delta, delta_ff_filt])

                    arrow.setPos(p_state[0], p_state[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)

                count = 0
            #endregion

            if driveController.steeringController.pathComplete:
                return

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
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')

    steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')

    steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')

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



